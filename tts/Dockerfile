###############################################################################
# Qwen3-TTS Audiobook Production Pipeline
# Base: nvidia/cuda:12.8.0-devel-ubuntu22.04
# Target GPU: NVIDIA RTX 5060 Ti (Blackwell SM120, CC 12.0, 16 GB VRAM)
###############################################################################
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# ── Build arguments ────────────────────────────────────────────────────────────
ARG TORCH_CUDA_ARCH_LIST="12.0"

# ── Persistent env (also passed at runtime via docker-compose) ─────────────────
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 1. System packages ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        git wget curl ca-certificates \
        libsndfile1 ffmpeg sox \
        build-essential ninja-build cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ── 2. PyTorch Nightly + CUDA 12.8 ────────────────────────────────────────────
RUN pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# ── 3. Triton (SM120 / Blackwell compatible) ───────────────────────────────────
RUN pip install triton

# ── 4. SageAttention 2.2.0 (SM120 support) ────────────────────────────────────
#  • First: try prebuilt wheel from woct0rdho/SageAttention for cu128 + nightly
#  • Fallback: compile from source with TORCH_CUDA_ARCH_LIST=12.0
RUN TORCH_VER=$(python -c \
        "import torch; v=torch.__version__; print(v.split('+')[0].replace('dev','').rstrip('.'))") \
 && echo "Detected PyTorch: ${TORCH_VER}" \
 && WHEEL="https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0/sageattention-2.2.0+cu128torch${TORCH_VER}-cp311-cp311-linux_x86_64.whl" \
 && pip install "${WHEEL}" 2>/dev/null \
 && echo "=== SageAttention: prebuilt wheel installed ===" \
 || (echo "=== SageAttention: prebuilt wheel unavailable, building from woct0rdho fork (SM120 patches) ===" \
  && git clone --depth 1 https://github.com/woct0rdho/SageAttention.git /tmp/sa \
  && cd /tmp/sa \
  && TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation . \
  && rm -rf /tmp/sa)

# ── 5. Qwen3-TTS Python package ───────────────────────────────────────────────
#  The official qwen_tts package from QwenLM; falls back gracefully if unavailable
RUN pip install git+https://github.com/QwenLM/Qwen3-TTS.git 2>/dev/null \
 && echo "=== qwen_tts package installed ===" \
 || echo "=== qwen_tts source unavailable; transformers AutoModel fallback will be used ==="

# ── 6. Runtime Python dependencies ────────────────────────────────────────────
RUN pip install \
        "transformers>=4.51.0" \
        accelerate \
        "huggingface_hub[hf_transfer]" \
        safetensors \
        soundfile \
        librosa \
        numpy \
        scipy \
        tqdm \
        ebooklib \
        beautifulsoup4 \
        lxml \
        nltk

# NLTK sentence tokeniser data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# ── 7. Application ─────────────────────────────────────────────────────────────
WORKDIR /app
COPY download_model.py audiobook.py test_tts.py ./

RUN mkdir -p /models /voices /input /output

# ── 8. Runtime defaults (all overridable via docker-compose / -e) ──────────────
ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV MODEL_CACHE=/models
ENV REFERENCE_WAV=/voices/CB.wav
ENV REFERENCE_TEXT=/voices/CB.txt
ENV VOICE_PROMPT_CACHE=/voices/CB_prompt.safetensors
ENV CONTEXT_SENTENCES=3
ENV BATCH_SIZE=4
ENV X_VECTOR_ONLY=0
ENV USE_TORCH_COMPILE=0
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "test_tts.py"]
