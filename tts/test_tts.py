"""
test_tts.py — Vergleichstest: Batch-Modus vs. Autoregressiver Modus
====================================================================

Pass 1 — Token-aware Batch (aktueller Produktions-Modus)
    max(tokens_in_batch) × batch_size ≤ TOKEN_BUDGET
    Kontext: letzte CONTEXT_SENTENCES synthetisierte Sätze

Pass 2 — Autoregressiv (ein Satz pro API-Call)
    Jeder Satz erhält das Audio des vorherigen Satzes als ICL-Kontext.
    Maximale Prosodiekontinuität, höherer RTF-Aufwand erwartet.

Output
------
  /output/test_batch.wav
  /output/test_autoregressive.wav

Environment variables
---------------------
  TEST_INPUT           /input/Beispiel.txt
  OUTPUT_DIR           /output
  CONTEXT_SENTENCES    3
  TOKEN_BUDGET         800
  MAX_NEW_TOKENS_FACTOR 10
  LANGUAGE             English
  (+ alle audiobook.py-Variablen für Modell / Voice)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audiobook import (
    _env,
    _env_bool,
    _env_int,
    _patch_sage_attention,
    _set_sdpa_backend,
    compose_audio,
    count_text_tokens,
    get_or_create_voice_prompt,
    load_model,
    segment_sentences,
    synthesise_chapter,
    synthesise_chapter_autoregressive,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [test] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_tts")


def _print_results(label: str, results, total_wall: float, sr: int, wav_path: Path) -> None:
    rtfs  = [r.rtf_val for r in results]
    vrms  = [r.vram_gb for r in results]
    audio = sum(len(r.audio) for r in results) / sr
    rtf_total = total_wall / audio if audio else float("inf")

    print(f"\n  ── {label} ──")
    print(f"    Sätze            : {len(results)}")
    print(f"    Generierungszeit : {total_wall:.1f} s")
    print(f"    Audio-Länge      : {audio:.1f} s  ({audio/60:.2f} min)")
    print(f"    RTF  Mittel      : {np.mean(rtfs):.3f}")
    print(f"    RTF  Min         : {np.min(rtfs):.3f}")
    print(f"    RTF  Max         : {np.max(rtfs):.3f}")
    print(f"    RTF  Gesamt      : {rtf_total:.3f}")
    print(f"    VRAM Peak        : {np.max(vrms):.2f} GB")
    print(f"    Ziel RTF ≤ 1.0   : {'✓ erreicht' if rtf_total <= 1.0 else '✗ nicht erreicht'}")
    print(f"    WAV              : {wav_path}")


def main() -> None:
    input_file            = _env("TEST_INPUT", "/input/Beispiel.txt")
    output_dir            = Path(_env("OUTPUT_DIR", "/output"))
    context_sentences     = _env_int("CONTEXT_SENTENCES", 3)
    token_budget          = _env_int("TOKEN_BUDGET", 800)
    max_new_tokens_factor = _env_int("MAX_NEW_TOKENS_FACTOR", 10)
    language              = _env("LANGUAGE", "English")
    use_compile           = _env_bool("USE_TORCH_COMPILE")

    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        log.info("GPU: %s | SM%d%d | %.0f GB VRAM",
                 dev.name, dev.major, dev.minor, dev.total_memory / 1024**3)
    else:
        log.error("Kein CUDA-GPU gefunden — Abbruch.")
        sys.exit(1)

    _patch_sage_attention()
    _set_sdpa_backend(_env("SDPA_BACKEND", "math"))

    model, sr = load_model()
    try:
        model.eval()
    except AttributeError:
        pass

    if use_compile:
        log.warning("torch.compile ist auf SM120 instabil — aktiviere trotzdem.")
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            log.warning("torch.compile fehlgeschlagen: %s", exc)

    prompt, mode_str = get_or_create_voice_prompt(model, sr)

    inp = Path(input_file)
    if not inp.exists():
        log.error("Eingabedatei nicht gefunden: %s", input_file)
        sys.exit(1)

    sentences = segment_sentences(inp.read_text(encoding="utf-8"))
    if not sentences:
        log.error("Keine Sätze in %s gefunden.", input_file)
        sys.exit(1)

    log.info("Eingabe: %s | %d Sätze | Sprache: %s", input_file, len(sentences), language)
    log.info("Kontext: %d | TOKEN_BUDGET: %d | MNT-Faktor: %d | Voice-Modus: %s",
             context_sentences, token_budget, max_new_tokens_factor, mode_str)

    # ── Pass 1: Token-aware Batch ─────────────────────────────────────────────
    log.info("=== Pass 1: Token-aware Batch (ctx=%d, budget=%d) ===",
             context_sentences, token_budget)
    t0 = time.perf_counter()
    results_batch = synthesise_chapter(
        model, prompt, sentences,
        chapter_title="test_batch",
        context_sentences=context_sentences,
        token_budget=token_budget,
        max_new_tokens_factor=max_new_tokens_factor,
        language=language,
        sample_rate=sr,
    )
    wall_batch = time.perf_counter() - t0

    wav_batch = output_dir / "test_batch.wav"
    sf.write(str(wav_batch), compose_audio([r.audio for r in results_batch], sample_rate=sr), sr)

    torch.cuda.empty_cache()

    # ── Pass 2: Autoregressiv ─────────────────────────────────────────────────
    log.info("=== Pass 2: Autoregressiv (batch=1, ctx=%d) ===", context_sentences)
    t0 = time.perf_counter()
    results_auto = synthesise_chapter_autoregressive(
        model, prompt, sentences,
        chapter_title="test_auto",
        context_sentences=context_sentences,
        max_new_tokens_factor=max_new_tokens_factor,
        language=language,
        sample_rate=sr,
    )
    wall_auto = time.perf_counter() - t0

    wav_auto = output_dir / "test_autoregressive.wav"
    sf.write(str(wav_auto), compose_audio([r.audio for r in results_auto], sample_rate=sr), sr)

    # ── Zusammenfassung ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  VERGLEICH: Batch vs. Autoregressiv")
    print("=" * 70)
    _print_results("Pass 1 — Token-aware Batch", results_batch, wall_batch, sr, wav_batch)
    _print_results("Pass 2 — Autoregressiv (batch=1)", results_auto, wall_auto, sr, wav_auto)
    print()
    print(f"  RTF-Verhältnis Auto/Batch : {(wall_auto / (sum(len(r.audio) for r in results_auto)/sr)) / (wall_batch / (sum(len(r.audio) for r in results_batch)/sr)):.2f}x langsamer")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
