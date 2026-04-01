"""
download_model.py — Model download and first-run cache warm-up.

Run at container start (or standalone) to ensure the model is present in
/models before inference begins.  Safe to call multiple times; HuggingFace Hub
skips already-cached files.

Usage:
    python download_model.py
"""
from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [download] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def download_model(
    model_id: str | None = None,
    cache_dir: str | None = None,
    hf_token: str | None = None,
) -> str:
    """
    Download *model_id* to *cache_dir* via huggingface_hub snapshot_download.

    Returns the local snapshot directory path.
    Skips files that are already cached.
    """
    from huggingface_hub import snapshot_download, HfApi

    model_id  = model_id  or os.environ.get("MODEL_ID",    "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    cache_dir = cache_dir or os.environ.get("MODEL_CACHE", "/models")
    hf_token  = hf_token  or os.environ.get("HF_TOKEN")

    if hf_token and hf_token.startswith("DEIN_"):
        log.warning("HF_TOKEN looks like the placeholder value — download may fail for gated models.")
        hf_token = None

    log.info("Model  : %s", model_id)
    log.info("Cache  : %s", cache_dir)
    log.info("Token  : %s", "set" if hf_token else "not set (public models only)")

    # Quick sanity-check: does the repo exist?
    try:
        api = HfApi(token=hf_token)
        api.repo_info(model_id)
    except Exception as exc:
        log.error("Could not reach HuggingFace repo: %s", exc)
        sys.exit(1)

    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=hf_token,
        # Skip heavy non-essential formats to save bandwidth
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "flax_model*",
            "tf_model*",
            "rust_model.ot",
        ],
        local_files_only=False,
    )

    log.info("Model ready at: %s", local_dir)
    return local_dir


if __name__ == "__main__":
    download_model()
