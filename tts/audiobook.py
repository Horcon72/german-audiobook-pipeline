"""
audiobook.py — Qwen3-TTS Audiobook Production Pipeline
=======================================================

Responsibilities
----------------
* Model loading with optional SageAttention (sageattn, SM120-compatible fp8 path)
  and torch.compile patches
* Voice-prompt management: create → save as .safetensors → load on subsequent runs
* Text ingestion: EPUB or plain-text → chapters → sentences (NLTK)
* Batch synthesis with a configurable prosodic context window
* Audio composition: sentence WAVs → chapter WAVs → full-book WAV
* Per-sentence logging: RTF, VRAM peak, batch size, context sentences

Environment variables (all have defaults shown)
-------------------------------------------------
MODEL_ID            Qwen/Qwen3-TTS-12Hz-1.7B-Base
MODEL_CACHE         /models
HF_TOKEN            (empty — set in docker-compose.yml)
REFERENCE_WAV       /voices/CB.wav
REFERENCE_TEXT      /voices/CB.txt          (optional; triggers ICL mode)
VOICE_PROMPT_CACHE  /voices/CB_prompt.safetensors
X_VECTOR_ONLY       0                       (1 = x-vector only, faster)
CONTEXT_SENTENCES   3                       (0 = no prosodic context)
TOKEN_BUDGET        800                     (max padded text-tokens per batch call)
MAX_NEW_TOKENS_FACTOR 10                   (audio-token cap = max_text_tokens × factor)
SEGMENT_MODE        paragraph               (paragraph|sentence — segmentation unit for TTS)
PARAGRAPH_MAX_CHARS 500                     (max chars per paragraph; split at last .?! before limit)
CROSSFADE_MS        20                      (crossfade length in ms; 0 = silence gap instead)
PARAGRAPH_MODE      0                       (1 = autoregressive batch=1, max prosody continuity)
VOICE_DESIGN_INSTRUCT  (empty)             (natural-language voice description → enables VoiceDesign mode)
VOICE_DESIGN_MODEL_ID  Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
VOICE_DESIGN_CACHE  /voices/voice_design_profile  (.json + .wav saved here for reuse)
USE_TORCH_COMPILE   0                       (1 = torch.compile, unstable on SM120)
"""
from __future__ import annotations

import gc
import io
import json
import logging
import os
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import torch

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audiobook")

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, str(int(default))).strip().lower()
    return v in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# SageAttention patch (explicit backend — no global --use-sage-attention flag)
# ─────────────────────────────────────────────────────────────────────────────

def _patch_sage_attention() -> bool:
    """
    Replace F.scaled_dot_product_attention with sageattn (high-level dispatcher).

    sageattn auto-selects the best available backend for the current GPU:
      SM75        → Triton int8
      SM80/86/87  → CUDA int8/fp16
      SM89        → CUDA int8/fp8
      SM120/121   → CUDA int8/fp8 (fp8 mma, accurate fp32 accumulator)

    Falls back silently to the original SDPA if SageAttention is unavailable
    or if the inputs are incompatible (non-CUDA, unsupported dtype).
    Returns True if the patch was applied.
    """
    try:
        from sageattention import sageattn  # type: ignore
        import torch.nn.functional as F

        _orig = F.scaled_dot_product_attention

        def _sage_sdpa(
            query, key, value,
            attn_mask=None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale=None,
            **kwargs,
        ):
            # sageattn requires CUDA float16 / bfloat16 tensors without float masks
            if (
                query.is_cuda
                and query.dtype in (torch.float16, torch.bfloat16)
                and attn_mask is None
            ):
                try:
                    return sageattn(
                        query, key, value,
                        is_causal=is_causal,
                        sm_scale=scale,
                    )
                except Exception:
                    pass  # fall through to original
            return _orig(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

        F.scaled_dot_product_attention = _sage_sdpa
        log.info("SageAttention: sageattn (SM120-compatible fp8 path) patch applied")
        return True

    except ImportError:
        log.warning("SageAttention not available — using default SDPA")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SDPA backend selection
# ─────────────────────────────────────────────────────────────────────────────

def _set_sdpa_backend(backend: str) -> None:
    """
    Set the global PyTorch SDPA backend.

    backend values (SDPA_BACKEND env var):
      math      — default PyTorch math kernel (current baseline)
      efficient — memory-efficient attention (xformers-style, built into PyTorch)
      cudnn     — cuDNN fused attention (fastest on SM120 in isolation)
      flash     — FlashAttention (requires flash-attn package)
    """
    b = backend.lower().strip()
    try:
        torch.backends.cuda.enable_flash_sdp(b == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(b == "efficient")
        torch.backends.cuda.enable_math_sdp(b == "math")
        torch.backends.cuda.enable_cudnn_sdp(b == "cudnn")
        log.info("SDPA backend set to: %s", b)
    except Exception as exc:
        log.warning("SDPA backend switch failed (%s) — keeping defaults", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_model_path(cache_dir: str, model_id: str) -> str:
    """
    Return the local model path. Tries the snapshot cache first; if absent,
    returns model_id so transformers can resolve it online.
    """
    cache = Path(cache_dir)
    # HuggingFace snapshot_download stores under models--<org>--<name>/snapshots/<hash>
    slug = "models--" + model_id.replace("/", "--")
    snapshots = cache / slug / "snapshots"
    if snapshots.exists():
        dirs = sorted(snapshots.iterdir())
        if dirs:
            return str(dirs[-1])
    # Flat download (download_model.py uses local_dir not cache_dir)
    flat = cache / model_id.split("/")[-1]
    if flat.exists():
        return str(flat)
    return model_id   # let HF resolve online


def load_model(
    model_id: str | None = None,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> tuple:
    """
    Load Qwen3TTSModel and return (model, sample_rate).

    Loading order:
      1. qwen_tts.Qwen3TTSModel  (official package from QwenLM/Qwen3-TTS)
      2. transformers AutoModel fallback (limited functionality)
    """
    model_id  = model_id  or _env("MODEL_ID",    "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    cache_dir = cache_dir or _env("MODEL_CACHE", "/models")
    hf_token  = hf_token  or _env("HF_TOKEN")

    if hf_token and hf_token.startswith("DEIN_"):
        hf_token = None

    local_path = _find_model_path(cache_dir, model_id)
    log.info("Loading model from: %s", local_path)

    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
        model = Qwen3TTSModel.from_pretrained(
            local_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            token=hf_token,
        )
        sample_rate = getattr(model, "sample_rate", 24_000)
        log.info("Model loaded via qwen_tts (sample_rate=%d)", sample_rate)
        return model, sample_rate

    except ImportError:
        log.warning("qwen_tts package not found — using transformers AutoModel fallback")

    # Transformers fallback (basic — voice cloning API may differ)
    from transformers import AutoModelForTextToWaveform, AutoProcessor  # type: ignore
    processor = AutoProcessor.from_pretrained(local_path, token=hf_token)
    model = AutoModelForTextToWaveform.from_pretrained(
        local_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        token=hf_token,
    )
    model.processor = processor
    sample_rate = getattr(processor, "sampling_rate", 24_000)
    log.info("Model loaded via transformers AutoModel (sample_rate=%d)", sample_rate)
    return model, sample_rate


# ─────────────────────────────────────────────────────────────────────────────
# Voice prompt: safetensors persistence
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tensors(obj) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Extract tensor and scalar fields from *obj*.

    Tries three strategies in order:
      1. vars(obj).__dict__  — works for plain Python objects
      2. dir(obj) + getattr — works for __slots__ objects
      3. Pickle fallback    — stores the entire object as uint8 bytes tensor
                             (decoded transparently by load_voice_prompt)
    Moves tensors to CPU before returning.
    """
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {}

    # Strategy 1: __dict__
    try:
        items = list(vars(obj).items())
    except TypeError:
        items = []

    # Strategy 2: __slots__ / descriptor attributes
    if not items:
        for k in dir(obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(obj, k)
                if callable(v):
                    continue
                items.append((k, v))
            except Exception:
                pass

    for k, v in items:
        if isinstance(v, torch.Tensor):
            tensors[k] = v.cpu()
        elif isinstance(v, (str, int, float, bool)):
            meta[k] = str(v)
        elif isinstance(v, (list, tuple)) and v and all(isinstance(i, (int, float)) for i in v):
            meta[k] = json.dumps(list(v))

    # Strategy 3: pickle fallback — pack the whole object as uint8 tensor
    if not tensors:
        import io
        import pickle
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        raw = torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)
        tensors["__pickle__"] = raw
        meta["__format"] = "pickle"
        log.debug("Voice prompt has no public tensor attrs — stored via pickle fallback")

    return tensors, meta


def save_voice_prompt(prompt, save_path: str | Path, *, source_wav: str, mode: str) -> None:
    """
    Serialise a voice-clone prompt to *save_path* (.safetensors).

    All tensor attributes of *prompt* are stored as safetensors entries.
    Scalar attributes and provenance metadata go into the file's metadata dict.
    A README_voices.txt is written alongside the .safetensors file.
    """
    from safetensors.torch import save_file

    save_path = Path(save_path)
    tensors, scalar_meta = _extract_tensors(prompt)

    if not tensors:
        raise RuntimeError(
            "Voice prompt object has no tensor attributes — cannot save as safetensors. "
            "Check the qwen_tts API version."
        )

    meta = {
        "__source_wav": str(source_wav),
        "__mode": mode,
        "__created": datetime.now().isoformat(),
        "__prompt_class": type(prompt).__qualname__,
        "__tensor_keys": json.dumps(list(tensors.keys())),
        **{f"__scalar_{k}": v for k, v in scalar_meta.items()},
    }

    save_file(tensors, str(save_path), metadata=meta)
    log.info("Voice prompt saved → %s  (%d tensors)", save_path, len(tensors))

    # ── README alongside the .safetensors ─────────────────────────────────────
    readme = save_path.parent / "README_voices.txt"
    lines = [
        "Qwen3-TTS Voice Prompt Cache",
        "=" * 40,
        f"File        : {save_path.name}",
        f"Source WAV  : {source_wav}",
        f"Mode        : {mode}",
        f"Created     : {meta['__created']}",
        f"Class       : {meta['__prompt_class']}",
        "",
        "Tensor fields stored",
        "-" * 40,
    ]
    for k, t in tensors.items():
        lines.append(f"  {k:<30} shape={list(t.shape)}  dtype={t.dtype}")
    lines += [
        "",
        "How to reload (Python)",
        "-" * 40,
        "    from audiobook import load_voice_prompt",
        "    from audiobook import load_model",
        "    model, sr = load_model()",
        f"    prompt = load_voice_prompt(model, '{save_path}')",
        "",
        "The reconstructed prompt object is drop-in compatible with",
        "model.generate_voice_clone(voice_clone_prompt=prompt, ...).",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("README_voices.txt written → %s", readme)


def load_voice_prompt(model, load_path: str | Path):
    """
    Reconstruct a voice-clone prompt from a .safetensors file.

    Strategy:
      1. Load tensors onto CUDA via safetensors.
      2. Detect the original prompt class from metadata.
      3. Instantiate it without calling __init__ (avoids re-encoding the WAV).
      4. Populate all tensor and scalar attributes.
      5. Return the reconstructed object.
    """
    from safetensors import safe_open

    load_path = Path(load_path)
    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {}

    with safe_open(str(load_path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata())
        for key in f.keys():
            t = f.get_tensor(key)
            # __pickle__ is raw uint8 bytes — must not be cast to bfloat16
            tensors[key] = t if key == "__pickle__" else t.to(torch.bfloat16).cuda()

    log.info(
        "Voice prompt loaded ← %s  (%d tensors, mode=%s)",
        load_path, len(tensors), meta.get("__mode", "?"),
    )

    # Pickle fallback path (stored under __scalar___format by save_voice_prompt)
    if "__pickle__" in tensors:
        import pickle
        raw_bytes = tensors["__pickle__"].cpu().numpy().tobytes()
        prompt_obj = pickle.loads(raw_bytes)
        log.info("Voice prompt reconstructed from pickle fallback")
        return prompt_obj

    # Reconstruct the original class (e.g. qwen_tts.VoiceClonePrompt)
    class_name = meta.get("__prompt_class", "")
    prompt_obj = None

    # Try to get the class from the model to do a proper reconstruction
    if class_name:
        for mod in sys.modules.values():
            cls = getattr(mod, class_name.split(".")[-1], None)
            if cls is not None and isinstance(cls, type):
                try:
                    prompt_obj = cls.__new__(cls)
                    break
                except Exception:
                    pass

    if prompt_obj is None:
        prompt_obj = types.SimpleNamespace()

    # Populate tensors
    for k, v in tensors.items():
        setattr(prompt_obj, k, v)

    # Populate scalars
    for k, v in meta.items():
        if k.startswith("__scalar_"):
            attr = k[len("__scalar_"):]
            # Best-effort type restoration
            try:
                setattr(prompt_obj, attr, json.loads(v))
            except (json.JSONDecodeError, ValueError):
                setattr(prompt_obj, attr, v)

    return prompt_obj


def get_or_create_voice_prompt(model, sample_rate: int) -> tuple[object, str]:
    """
    Return (voice_prompt, mode_string).

    * If VOICE_PROMPT_CACHE exists → load from safetensors (no re-encoding).
    * Otherwise → call create_voice_clone_prompt(), save, return.
    """
    cache_path = Path(_env("VOICE_PROMPT_CACHE", "/voices/CB_prompt.safetensors"))
    ref_wav    = _env("REFERENCE_WAV",  "/voices/CB.wav")
    ref_txt    = _env("REFERENCE_TEXT", "/voices/CB.txt")
    x_vec_only = _env_bool("X_VECTOR_ONLY")

    mode = "x_vector" if x_vec_only else "icl"

    if cache_path.exists():
        log.info("Voice prompt cache found — loading %s", cache_path)
        return load_voice_prompt(model, cache_path), mode

    log.info("Creating voice prompt (mode=%s, wav=%s)", mode, ref_wav)

    # Load reference audio
    audio_data, wav_sr = sf.read(ref_wav)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if wav_sr != sample_rate:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=wav_sr, target_sr=sample_rate)
    ref_audio = (audio_data.astype(np.float32), sample_rate)

    # Read reference transcript if available
    ref_text: str | None = None
    txt_path = Path(ref_txt)
    if txt_path.exists() and not x_vec_only:
        ref_text = txt_path.read_text(encoding="utf-8").strip()
        log.info("Reference transcript: %d chars", len(ref_text))
    elif not x_vec_only:
        log.warning("REFERENCE_TEXT not found — falling back to x_vector_only mode")
        x_vec_only = True
        mode = "x_vector"

    # Create prompt via model API
    try:
        prompt = model.create_voice_clone_prompt(
            ref_audio,
            ref_text,
            x_vector_only_mode=x_vec_only,
        )
    except TypeError:
        # Older API without x_vector_only_mode kwarg
        prompt = model.create_voice_clone_prompt(ref_audio, ref_text)

    # Persist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_voice_prompt(prompt, cache_path, source_wav=ref_wav, mode=mode)

    return prompt, mode


# ─────────────────────────────────────────────────────────────────────────────
# VoiceDesign profile persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_voice_design_profile(
    instruct: str,
    model_id: str,
    ref_wav: np.ndarray,
    sample_rate: int,
    save_path: str | Path,
) -> None:
    """
    Persist a VoiceDesign profile so the same voice can be reproduced later.

    Saves two files:
      <save_path>.json   — instruct string + model_id + creation timestamp
      <save_path>.wav    — first 10 s of generated audio as a voice reference

    The .wav can be fed directly to create_voice_clone_prompt() on the Base
    model to produce a voice_clone_prompt for future runs without VoiceDesign.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "instruct": instruct,
        "model_id": model_id,
        "created": datetime.now().isoformat(),
        "note": (
            "To reuse without VoiceDesign: load Base model and call "
            "create_voice_clone_prompt() with the accompanying .wav file."
        ),
    }
    json_path = save_path.with_suffix(".json")
    json_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    wav_path = save_path.with_suffix(".wav")
    max_samples = min(len(ref_wav), sample_rate * 10)
    sf.write(str(wav_path), ref_wav[:max_samples], sample_rate)

    log.info("VoiceDesign profile → %s", json_path)
    log.info("Reference WAV       → %s  (%.1f s)", wav_path, max_samples / sample_rate)


def load_voice_design_profile(path: str | Path) -> dict:
    """Load a saved VoiceDesign profile JSON and return it as a dict."""
    p = Path(path).with_suffix(".json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# VRAM & RTF helpers
# ─────────────────────────────────────────────────────────────────────────────

def vram_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_vram_peak() -> None:
    torch.cuda.reset_peak_memory_stats()


def rtf(audio_duration_s: float, synthesis_time_s: float) -> float:
    """Real-time factor: synthesis_time / audio_duration.  RTF < 1.0 is real-time."""
    if audio_duration_s <= 0:
        return float("inf")
    return synthesis_time_s / audio_duration_s


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wavs_to_numpy(raw, sample_rate: int) -> list[np.ndarray]:
    """
    Normalise whatever generate_voice_clone returns into a list of float32 arrays
    at *sample_rate*.  Handles:
      • list of numpy arrays
      • single numpy array (batch=1)
      • torch.Tensor  [B, T] or [T]
      • tuple (wavs, sr)
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        raw, _sr = raw
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            raw = [raw]
        else:
            raw = list(raw)
    return [np.asarray(w, dtype=np.float32) for w in raw]


def count_text_tokens(model, text: str) -> int:
    """Return the number of text-side input tokens for *text* using the model's tokenizer."""
    ids = model._tokenize_texts([text])[0]
    return int(ids.shape[-1])


def token_aware_batches(
    sentences: list[str],
    token_counts: list[int],
    token_budget: int = 800,
) -> list[list[str]]:
    """
    Partition *sentences* into contiguous batches such that for every batch:

        max(token_counts_in_batch) × len(batch) ≤ token_budget

    This minimises padding waste while maximising sentences per batch call:
    - Short sentences → large batches (low max-token → many fit in budget)
    - Long sentences → smaller batches (high max-token → fewer fit)

    TOKEN_BUDGET = 800 corresponds roughly to batch=12 at the median sentence
    length (~49 tokens) and batch=11 at the longest sentences (72 tokens),
    keeping VRAM below ~8 GB on a 16 GB card.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_max = 0

    for sent, n_tok in zip(sentences, token_counts):
        new_max = max(current_max, n_tok)
        if current and new_max * (len(current) + 1) > token_budget:
            groups.append(current)
            current = []
            current_max = 0
            new_max = n_tok
        current.append(sent)
        current_max = new_max

    if current:
        groups.append(current)
    return groups


def synthesise_batch(
    model,
    sentences: list[str],
    prompt,
    *,
    language: str = "English",
    sample_rate: int = 24_000,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[np.ndarray]:
    """
    Synthesise a batch of sentences with a shared voice prompt.
    Returns a list of float32 numpy arrays (one per sentence).

    *max_new_tokens* overrides the model default (8192).  Pass a tighter value
    derived from the batch's token count to reduce KV-cache overhead and allow
    larger batch sizes within the same VRAM budget.
    """
    langs = [language] * len(sentences)
    extra: dict = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
    if temperature is not None:
        extra["temperature"] = temperature
        extra.setdefault("do_sample", True)
    if top_p is not None:
        extra["top_p"] = top_p
        extra.setdefault("do_sample", True)

    try:
        raw = model.generate_voice_clone(
            text=sentences,
            language=langs,
            voice_clone_prompt=prompt,
            **extra,
        )
    except TypeError:
        raw = model.generate_voice_clone(
            text=sentences,
            voice_clone_prompt=prompt,
            **extra,
        )

    return _wavs_to_numpy(raw, sample_rate)


def _build_context_prompt(model, base_prompt, context_wavs: list[np.ndarray], sr: int):
    """
    Extend *base_prompt* with recently synthesised audio as ICL context.

    Concatenates the last N context WAVs and calls create_voice_clone_prompt()
    again with this audio as the reference — the new x-vector / speech tokens
    capture the current prosodic style without synthesising new audio.

    Falls back to *base_prompt* if the model does not support this pattern.
    """
    if not context_wavs:
        return base_prompt
    try:
        ctx_audio = np.concatenate(context_wavs, axis=0).astype(np.float32)
        extended = model.create_voice_clone_prompt(
            (ctx_audio, sr),
            None,             # no transcript needed for prosodic extension
            x_vector_only_mode=True,  # fast; we only want style, not speech tokens
        )
        return extended
    except Exception as exc:
        log.debug("Context prompt extension skipped (%s); using base prompt", exc)
        return base_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Text ingestion: EPUB and plain-text → chapters → sentences
# ─────────────────────────────────────────────────────────────────────────────

def _html_to_text(html: bytes | str) -> str:
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "lxml").get_text(separator="\n")


def load_epub(path: str) -> list[tuple[str, str]]:
    """
    Parse an EPUB and return [(chapter_title, chapter_text), ...].
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(path)
    chapters: list[tuple[str, str]] = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            title = item.get_name() or f"chapter_{len(chapters)+1}"
            text  = _html_to_text(item.get_content())
            text  = text.strip()
            if len(text) > 50:   # skip nav pages, cover blurbs, etc.
                chapters.append((title, text))
    return chapters


def load_plaintext(path: str) -> list[tuple[str, str]]:
    """Return the entire plain-text file as one chapter."""
    raw = Path(path).read_text(encoding="utf-8")
    return [(Path(path).stem, raw.strip())]


def load_input(path: str) -> list[tuple[str, str]]:
    p = Path(path)
    if p.suffix.lower() == ".epub":
        return load_epub(str(p))
    return load_plaintext(str(p))


def normalize_text(text: str) -> str:
    """
    Normalize text before TTS synthesis.

    Transformations applied:
    - Times like "03:17 Uhr" → "3 Uhr 17"
      Handles HH:MM with optional leading zero and optional "Uhr" suffix.
      E.g. "14:30 Uhr" → "14 Uhr 30", "3:05" → "3 Uhr 5"
    """
    def _expand_time(m: re.Match) -> str:
        h = str(int(m.group(1)))       # strip leading zero
        mi = str(int(m.group(2)))      # strip leading zero
        return f"{h} Uhr {mi}"

    # Match HH:MM with or without trailing " Uhr"
    text = re.sub(r"\b(\d{1,2}):(\d{2})(?:\s*Uhr\b)?", _expand_time, text)

    # Lines ending without sentence punctuation (.!?…"»]) get a period appended.
    # Only applies to non-empty lines; blank lines (paragraph separators) are left untouched.
    text = re.sub(r"([^\s.!?…\"»\n])(\n)", r"\1.\2", text)

    return text


def segment_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    """
    Split *text* into paragraph-level units for TTS synthesis.

    1. Split on blank lines to get raw paragraphs.
    2. Any paragraph longer than *max_chars* is sub-split at the last
       sentence-ending punctuation (. ? !) before the limit.
       Never cuts mid-sentence; falls back to hard-cut only if no boundary exists.
    3. Drops empty fragments and whitespace-only blocks.
    """
    raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[str] = []
    for para in raw:
        if len(para) <= max_chars:
            result.append(para)
            continue
        remaining = para
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            cut = -1
            for i in range(len(window) - 1, -1, -1):
                if window[i] in ".?!" and (i + 1 >= len(window) or window[i + 1] in " \t"):
                    cut = i + 1
                    break
            if cut == -1:
                cut = max_chars
            chunk = remaining[:cut].strip()
            if chunk:
                result.append(chunk)
            remaining = remaining[cut:].strip()
        if remaining:
            result.append(remaining)
    return result


def segment_sentences(text: str) -> list[str]:
    import nltk
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        sents = nltk.sent_tokenize(text)
    # Drop empty / very short fragments
    return [s.strip() for s in sents if len(s.strip()) > 3]


def group_sentences(
    sentences: list[str],
    max_sentences: int = 12,
    max_chars: int = 1500,
) -> list[list[str]]:
    """
    Group sentences into chunks for single-call synthesis.

    Each group is joined into one text and synthesised as a unit, which gives
    the model full context for natural intonation and pacing across sentences.

    A new group starts when **any** hard limit is reached:
    - max_sentences exceeded
    - max_chars exceeded (prevents excessively long sequences)

    A soft break is inserted early when a natural section boundary is detected
    AND the current group is at least half full:
    - Chapter / part headings  (Kapitel, Chapter, Teil, Part, …)
    - Closing dialogue blocks  (sentence ending with ." or !")
    - Exclamatory / rhetorical stand-alone sentences (very short, ≤ 25 chars)
    """
    half = max(2, max_sentences // 2)
    groups: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for sent in sentences:
        sc = len(sent)

        # Hard flush before adding if limits would be exceeded
        if current and (len(current) >= max_sentences or current_chars + sc > max_chars):
            groups.append(current)
            current = []
            current_chars = 0

        current.append(sent)
        current_chars += sc

        # Soft break heuristics (only when group is at least half full)
        if len(current) >= half:
            stripped = sent.rstrip(" .…\"»")
            is_heading = bool(
                re.match(r"^(Kapitel|Chapter|Teil|Part|Abschnitt|Section)\s+\S", stripped, re.I)
            )
            is_dialogue_end = bool(re.search(r'[.!?]["\u00bb]\s*$', sent))
            is_isolated = len(stripped) <= 25 and re.search(r"[.!?]$", sent)

            if is_heading or is_dialogue_end or is_isolated:
                groups.append(current)
                current = []
                current_chars = 0

    if current:
        groups.append(current)

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Audiobook pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SentenceResult:
    __slots__ = ("chapter", "idx", "text", "audio", "rtf_val", "vram_gb", "batch_size", "ctx_count")

    def __init__(self, chapter, idx, text, audio, rtf_val, vram_gb, batch_size, ctx_count):
        self.chapter    = chapter
        self.idx        = idx
        self.text       = text
        self.audio      = audio
        self.rtf_val    = rtf_val
        self.vram_gb    = vram_gb
        self.batch_size = batch_size
        self.ctx_count  = ctx_count


def synthesise_chapter(
    model,
    base_prompt,
    sentences: list[str],
    *,
    chapter_title: str = "chapter",
    context_sentences: int = 3,
    token_budget: int = 800,
    max_new_tokens_factor: int = 10,
    language: str = "English",
    sample_rate: int = 24_000,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[SentenceResult]:
    """
    Synthesise all *sentences* of a chapter and return SentenceResult list.

    Token-aware batch processing
    ----------------------------
    Before synthesis, each sentence is tokenized to get its exact text-token
    count.  Sentences are then grouped into batches so that:

        max(tokens_in_batch) × batch_size ≤ token_budget

    This maximises parallelism for short sentences (large batches) and
    automatically reduces batch size for long sentences — keeping VRAM usage
    predictable regardless of sentence length distribution.

    For each batch, max_new_tokens is capped at:

        max(tokens_in_batch) × max_new_tokens_factor

    This prevents the audio KV-cache from pre-allocating 8192 tokens worth of
    memory when a batch of short sentences only needs ~100 audio tokens each.

    Context window
    --------------
    The last *context_sentences* synthesised WAVs are used as ICL context for
    each batch, anchoring prosodic continuity across batches.
    """
    results: list[SentenceResult] = []
    recent_audio: list[np.ndarray] = []

    # Pre-tokenize all sentences once
    token_counts = [count_text_tokens(model, s) for s in sentences]
    batches = token_aware_batches(sentences, token_counts, token_budget=token_budget)

    sent_idx = 0
    for batch_texts in batches:
        n = len(batch_texts)
        max_tok = max(token_counts[sent_idx : sent_idx + n])
        mnt = max_tok * max_new_tokens_factor

        ctx_wavs  = recent_audio[-context_sentences:] if (context_sentences > 0 and recent_audio) else []
        ctx_count = len(ctx_wavs)
        active_prompt = _build_context_prompt(model, base_prompt, ctx_wavs, sample_rate)

        reset_vram_peak()
        t0 = time.perf_counter()

        wavs = synthesise_batch(
            model, batch_texts, active_prompt,
            language=language, sample_rate=sample_rate,
            max_new_tokens=mnt, temperature=temperature, top_p=top_p,
        )

        wall = time.perf_counter() - t0
        peak = vram_peak_gb()

        total_chars = sum(len(t) for t in batch_texts)

        for j, (text, wav) in enumerate(zip(batch_texts, wavs)):
            audio_dur  = len(wav) / sample_rate
            char_frac  = len(text) / total_chars if total_chars else 1 / n
            synth_time = wall * char_frac
            r = rtf(audio_dur, synth_time)

            result = SentenceResult(
                chapter    = chapter_title,
                idx        = sent_idx + j,
                text       = text,
                audio      = wav,
                rtf_val    = r,
                vram_gb    = peak,
                batch_size = n,
                ctx_count  = ctx_count,
            )
            results.append(result)

            log.info(
                "[%s] sent %3d | RTF %.3f | VRAM %.2f GB | batch %d | tok %d | ctx %d | '%s…'",
                chapter_title, sent_idx + j, r, peak,
                n, max_tok, ctx_count, text[:50],
            )

            recent_audio.append(wav)
            if context_sentences > 0 and len(recent_audio) > context_sentences:
                recent_audio.pop(0)

        sent_idx += n

    return results


def synthesise_chapter_autoregressive(
    model,
    base_prompt,
    sentences: list[str],
    *,
    chapter_title: str = "chapter",
    context_sentences: int = 3,
    max_new_tokens_factor: int = 10,
    language: str = "English",
    sample_rate: int = 24_000,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[SentenceResult]:
    """
    Synthesise all *sentences* one at a time, each with the previous
    *context_sentences* WAVs fed back as ICL context.

    Unlike synthesise_chapter(), there is no batching — every sentence is
    a separate model call.  This maximises prosodic continuity (sentence N
    always benefits from sentence N-1's actual audio) at the cost of
    throughput (expected RTF ~2-3x higher than batch mode).
    """
    results: list[SentenceResult] = []
    recent_audio: list[np.ndarray] = []

    for idx, text in enumerate(sentences):
        n_tok = count_text_tokens(model, text)
        mnt = n_tok * max_new_tokens_factor

        ctx_wavs  = recent_audio[-context_sentences:] if (context_sentences > 0 and recent_audio) else []
        ctx_count = len(ctx_wavs)
        active_prompt = _build_context_prompt(model, base_prompt, ctx_wavs, sample_rate)

        reset_vram_peak()
        t0 = time.perf_counter()

        wavs = synthesise_batch(
            model, [text], active_prompt,
            language=language, sample_rate=sample_rate,
            max_new_tokens=mnt, temperature=temperature, top_p=top_p,
        )

        wall = time.perf_counter() - t0
        peak = vram_peak_gb()
        wav  = wavs[0]
        audio_dur = len(wav) / sample_rate
        r = rtf(audio_dur, wall)

        result = SentenceResult(
            chapter    = chapter_title,
            idx        = idx,
            text       = text,
            audio      = wav,
            rtf_val    = r,
            vram_gb    = peak,
            batch_size = 1,
            ctx_count  = ctx_count,
        )
        results.append(result)

        log.info(
            "[%s] sent %3d | RTF %.3f | VRAM %.2f GB | batch 1 | tok %d | ctx %d | '%s…'",
            chapter_title, idx, r, peak, n_tok, ctx_count, text[:50],
        )

        recent_audio.append(wav)
        if context_sentences > 0 and len(recent_audio) > context_sentences:
            recent_audio.pop(0)

    return results


def synthesise_chapter_voice_design(
    model,
    sentences: list[str],
    instruct: str,
    *,
    chapter_title: str = "chapter",
    token_budget: int = 800,
    max_new_tokens_factor: int = 10,
    language: str = "German",
    sample_rate: int = 24_000,
    temperature: float | None = None,
    top_p: float | None = None,
) -> list[SentenceResult]:
    """
    Token-aware batch synthesis using the VoiceDesign model.

    The *instruct* string (natural-language voice description) is passed to
    every batch call and provides consistent prosody without a context window.
    """
    results: list[SentenceResult] = []
    token_counts = [count_text_tokens(model, s) for s in sentences]
    batches = token_aware_batches(sentences, token_counts, token_budget=token_budget)

    sent_idx = 0
    for batch_texts in batches:
        n = len(batch_texts)
        max_tok = max(token_counts[sent_idx : sent_idx + n])
        mnt = max_tok * max_new_tokens_factor

        langs     = [language] * n
        instructs = [instruct] * n
        extra: dict = {"max_new_tokens": mnt}
        if temperature is not None:
            extra["temperature"] = temperature
            extra["do_sample"] = True
        if top_p is not None:
            extra["top_p"] = top_p
            extra.setdefault("do_sample", True)

        reset_vram_peak()
        t0 = time.perf_counter()

        try:
            raw = model.generate_voice_design(
                text=batch_texts,
                instruct=instructs,
                language=langs,
                **extra,
            )
        except TypeError:
            raw = model.generate_voice_design(
                text=batch_texts,
                instruct=instructs,
                **extra,
            )

        wall = time.perf_counter() - t0
        peak = vram_peak_gb()
        wavs = _wavs_to_numpy(raw, sample_rate)

        total_chars = sum(len(t) for t in batch_texts)
        for j, (text, wav) in enumerate(zip(batch_texts, wavs)):
            audio_dur  = len(wav) / sample_rate
            char_frac  = len(text) / total_chars if total_chars else 1 / n
            synth_time = wall * char_frac
            r = rtf(audio_dur, synth_time)

            result = SentenceResult(
                chapter    = chapter_title,
                idx        = sent_idx + j,
                text       = text,
                audio      = wav,
                rtf_val    = r,
                vram_gb    = peak,
                batch_size = n,
                ctx_count  = 0,
            )
            results.append(result)

            log.info(
                "[%s] sent %3d | RTF %.3f | VRAM %.2f GB | batch %d | tok %d | '%s…'",
                chapter_title, sent_idx + j, r, peak, n, max_tok, text[:50],
            )

        sent_idx += n

    return results


def compose_audio(
    wavs: list[np.ndarray],
    silence_ms: int = 400,
    sample_rate: int = 24_000,
    crossfade_ms: int = 0,
) -> np.ndarray:
    """
    Concatenate WAVs.

    crossfade_ms > 0 — overlap adjacent chunks with a linear crossfade to
        eliminate phase discontinuities and clicks at boundaries.
    crossfade_ms == 0 (default) — insert a silence_ms gap between chunks.
    """
    if not wavs:
        return np.array([], dtype=np.float32)

    if crossfade_ms > 0:
        fade_n = int(sample_rate * crossfade_ms / 1000)
        result = wavs[0].astype(np.float32).copy()
        for wav in wavs[1:]:
            wav = wav.astype(np.float32)
            n = min(fade_n, len(result), len(wav))
            if n > 0:
                fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
                fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)
                overlap  = result[-n:] * fade_out + wav[:n] * fade_in
                result   = np.concatenate([result[:-n], overlap, wav[n:]])
            else:
                result = np.concatenate([result, wav])
        return result

    gap = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, w in enumerate(wavs):
        parts.append(w)
        if i < len(wavs) - 1:
            parts.append(gap)
    return np.concatenate(parts, axis=0)


def run_audiobook(
    input_path: str,
    output_dir: str | None = None,
    *,
    context_sentences: int | None = None,
    token_budget: int | None = None,
    max_new_tokens_factor: int | None = None,
    language: str = "English",
) -> list[SentenceResult]:
    """
    Full pipeline entry point.

    1. Load model + apply SageAttention patch
    2. Load or create voice prompt
    3. Parse input (EPUB or TXT)
    4. Synthesise chapter by chapter
    5. Write chapter WAVs and full-book WAV to output_dir
    6. Return all SentenceResult objects for downstream logging
    """
    # ── Config ────────────────────────────────────────────────────────────────
    output_dir            = output_dir            or _env("OUTPUT_DIR", "/output")
    context_sentences     = context_sentences     if context_sentences     is not None else _env_int("CONTEXT_SENTENCES", 3)
    token_budget          = token_budget          if token_budget          is not None else _env_int("TOKEN_BUDGET", 800)
    max_new_tokens_factor = max_new_tokens_factor if max_new_tokens_factor is not None else _env_int("MAX_NEW_TOKENS_FACTOR", 10)
    language              = _env("LANGUAGE", language)
    use_compile           = _env_bool("USE_TORCH_COMPILE")
    paragraph_mode        = _env_bool("PARAGRAPH_MODE")
    segment_mode          = _env("SEGMENT_MODE", "paragraph")   # paragraph|sentence
    paragraph_max_chars   = _env_int("PARAGRAPH_MAX_CHARS", 500)
    crossfade_ms          = _env_int("CROSSFADE_MS", 20 if segment_mode == "paragraph" else 0)
    voice_design_instruct = _env("VOICE_DESIGN_INSTRUCT", "")
    voice_design_model_id = _env("VOICE_DESIGN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    voice_design_cache    = _env("VOICE_DESIGN_CACHE", "/voices/voice_design_profile")
    _temp_str             = _env("TEMPERATURE", "")
    temperature           = float(_temp_str) if _temp_str else None
    _top_p_str            = _env("TOP_P", "")
    top_p                 = float(_top_p_str) if _top_p_str else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline_t0 = time.perf_counter()
    log.info("Language: %s", language)

    # ── SageAttention ─────────────────────────────────────────────────────────
    _patch_sage_attention()

    # ── SDPA Backend (SDPA_BACKEND=math|efficient|cudnn|flash) ────────────────
    _set_sdpa_backend(_env("SDPA_BACKEND", "math"))

    # ── Model ─────────────────────────────────────────────────────────────────
    if voice_design_instruct:
        os.environ["MODEL_ID"] = voice_design_model_id
        log.info("VoiceDesign mode — loading model: %s", voice_design_model_id)
    model, sr = load_model()
    try:
        model.eval()
    except AttributeError:
        pass

    if use_compile:
        log.warning("torch.compile() is opt-in and currently unstable on SM120. Enabling anyway.")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            log.warning("torch.compile failed: %s", exc)

    # ── Voice prompt (Base mode) or VoiceDesign instruct ─────────────────────
    if not voice_design_instruct:
        prompt, mode = get_or_create_voice_prompt(model, sr)
    else:
        prompt, mode = None, "voice_design"
        log.info("VoiceDesign instruct: %s", voice_design_instruct)

    # ── Parse input ───────────────────────────────────────────────────────────
    chapters = load_input(input_path)
    log.info("Input: %s (%d chapter(s))", input_path, len(chapters))

    all_results: list[SentenceResult] = []
    chapter_wavs: list[np.ndarray] = []

    # ── Synthesise ────────────────────────────────────────────────────────────
    for ch_idx, (title, text) in enumerate(chapters):
        text = normalize_text(text)
        if segment_mode == "paragraph":
            segments = segment_paragraphs(text, max_chars=paragraph_max_chars)
            log.info("Chapter %d/%d — '%s' — %d paragraphs (≤%d chars each)",
                     ch_idx + 1, len(chapters), title, len(segments), paragraph_max_chars)
        else:
            segments = segment_sentences(text)
            log.info("Chapter %d/%d — '%s' — %d sentences", ch_idx + 1, len(chapters), title, len(segments))
        sentences = segments  # alias — rest of the pipeline is unchanged
        if not sentences:
            continue

        if voice_design_instruct:
            ch_results = synthesise_chapter_voice_design(
                model, sentences, voice_design_instruct,
                chapter_title=title,
                token_budget=token_budget,
                max_new_tokens_factor=max_new_tokens_factor,
                language=language,
                sample_rate=sr,
                temperature=temperature,
                top_p=top_p,
            )
        elif paragraph_mode:
            ch_results = synthesise_chapter_autoregressive(
                model, prompt, sentences,
                chapter_title=title,
                context_sentences=context_sentences,
                max_new_tokens_factor=max_new_tokens_factor,
                language=language,
                sample_rate=sr,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            ch_results = synthesise_chapter(
                model, prompt, sentences,
                chapter_title=title,
                context_sentences=context_sentences,
                token_budget=token_budget,
                max_new_tokens_factor=max_new_tokens_factor,
                language=language,
                sample_rate=sr,
                temperature=temperature,
                top_p=top_p,
            )
        all_results.extend(ch_results)

        # ── Write chapter WAV ────────────────────────────────────────────────
        ch_audio = compose_audio([r.audio for r in ch_results], sample_rate=sr,
                                  crossfade_ms=crossfade_ms)
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title)[:60]
        ch_wav = output_path / f"chapter_{ch_idx+1:03d}_{safe_title}.wav"
        sf.write(str(ch_wav), ch_audio, sr)
        log.info("Chapter WAV → %s  (%.1f s)", ch_wav, len(ch_audio) / sr)

        chapter_wavs.append(ch_audio)
        gc.collect()
        torch.cuda.empty_cache()

    # ── VoiceDesign profile save ──────────────────────────────────────────────
    if voice_design_instruct and all_results:
        save_voice_design_profile(
            instruct    = voice_design_instruct,
            model_id    = voice_design_model_id,
            ref_wav     = all_results[0].audio,
            sample_rate = sr,
            save_path   = voice_design_cache,
        )

    # ── Full-book WAV ─────────────────────────────────────────────────────────
    if chapter_wavs:
        book_audio = compose_audio(chapter_wavs, silence_ms=1500, sample_rate=sr)
        book_wav   = output_path / "audiobook_full.wav"
        sf.write(str(book_wav), book_audio, sr)
        total_min = len(book_audio) / sr / 60
        log.info("Full book WAV → %s  (%.1f min)", book_wav, total_min)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        rtfs       = [r.rtf_val for r in all_results]
        vrms       = [r.vram_gb for r in all_results]
        total_wall = time.perf_counter() - pipeline_t0
        audio_dur  = (len(book_audio) / sr) if chapter_wavs else 0.0
        print()
        print("=" * 70)
        print("  SYNTHESE-ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"  Sprache          : {language}")
        print(f"  Sätze            : {len(all_results)}")
        print(f"  Generierungszeit : {total_wall:.1f} s")
        print(f"  Audio-Länge      : {audio_dur:.1f} s  ({audio_dur/60:.2f} min)")
        print(f"  RTF  Mittel      : {np.mean(rtfs):.3f}")
        print(f"  RTF  Min         : {np.min(rtfs):.3f}")
        print(f"  RTF  Max         : {np.max(rtfs):.3f}")
        print(f"  RTF  Gesamt      : {total_wall / audio_dur:.3f}  (Generierungszeit / Audio-Länge)")
        print(f"  VRAM Peak        : {np.max(vrms):.2f} GB")
        print(f"  Ziel RTF ≤ 1.0   : {'✓ erreicht' if total_wall / audio_dur <= 1.0 else '✗ nicht erreicht'}")
        print("=" * 70)
        log.info(
            "Done. Sentences=%d  wall=%.1fs  audio=%.1fs  RTF_total=%.3f  RTF_mean=%.3f  VRAM=%.2fGB",
            len(all_results), total_wall, audio_dur,
            total_wall / audio_dur if audio_dur else float("inf"),
            np.mean(rtfs), np.max(vrms),
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.epub|input.txt> [output_dir]")
        sys.exit(1)
    inp  = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    run_audiobook(inp, outp)
