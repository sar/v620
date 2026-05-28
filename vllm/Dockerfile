# syntax=docker/dockerfile:1.4
# =============================================================================
# vLLM v0.21.x on AMD gfx1030 (Radeon Pro V620) — ROCm 7.2.3
# =============================================================================
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.3-complete
FROM ${BASE_IMAGE} AS base

# =============================================================================
# Environment configuration
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    ROCM_PATH=/opt/rocm \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# =============================================================================
# Stage 1: System build dependencies
# =============================================================================
FROM base AS builder

# Install build tools + ROCm CMake modules (required for vLLM build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    python3.12 python3.12-dev python3.12-venv \
    wget ca-certificates \
    rocm-cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip/wheel/setuptools inside venv (no --break-system-packages needed)
RUN pip install --upgrade pip wheel setuptools

# =============================================================================
# Stage 2: Install PyTorch ROCm 7.2.3 wheels from AMD repo
# =============================================================================
# Wheel listing: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/

RUN mkdir -p /tmp/rocm-wheels && \
    cd /tmp/rocm-wheels && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torch-2.9.1+rocm7.2.3.lw.gitebc02d69-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torchvision-0.24.0+rocm7.2.3.gitb919bd0c-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torchaudio-2.9.0+rocm7.2.3.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/triton-3.5.1+rocm7.2.3.gita272dfa8-cp312-cp312-linux_x86_64.whl"

# Install ROCm PyTorch wheels (inside venv, no --break-system-packages)
RUN pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true && \
    pip install --no-cache-dir \
        /tmp/rocm-wheels/torch-2.9.1+rocm7.2.3.lw.gitebc02d69-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchvision-0.24.0+rocm7.2.3.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchaudio-2.9.0+rocm7.2.3.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/triton-3.5.1+rocm7.2.3.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    rm -rf /tmp/rocm-wheels

# ✅ Sanity check: PyTorch import + ROCm detection (no LD_LIBRARY_PATH hack needed)
RUN python -c "import torch; print('torch:', torch.__version__); print('ROCm available:', torch.cuda.is_available())"

# =============================================================================
# Stage 3: Build and install vLLM v0.20.0 from source
# =============================================================================
FROM builder AS vllm-build

# Pre-install vLLM build dependencies
RUN pip install --no-cache-dir setuptools_scm wheel ninja cmake

# Pre-install vLLM Python dependencies (pinned numpy to avoid conflicts)
RUN pip install --no-cache-dir \
    "transformers>=4.53.0" \
    "ray[default]>=2.40.0" \
    "pydantic>=2.10.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "aiohttp>=3.11.0" \
    "tiktoken>=0.9.0" \
    "sentencepiece>=0.2.0" \
    "prometheus-client>=0.21.0" \
    "huggingface-hub>=0.28.0" \
    "tokenizers>=0.21.0" \
    "numpy>=1.26,<2.1" \
    "xgrammar>=0.1.10" \
    "lm-format-enforcer>=0.10.0"

# Clone VLLM_VERSION
ARG VLLM_VERSION=v0.21.0
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src

# ✅ Build vLLM with correct ROCm flags and ROCM_PATH
# --no-build-isolation uses pre-installed PyTorch ROCm headers in venv
RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /workspace/vllm_build.log

# Verify vLLM build succeeded (import from outside source dir)
RUN cd / && python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Checking logs:"; \
    grep -iE "error|fatal|CMake Error|hipcc" /workspace/vllm_build.log | tail -50; \
    exit 1; }

# =============================================================================
# Stage 4: Final minimal inference image
# =============================================================================
FROM base AS final

# Remove PEP 668 restriction for pip installs in final image
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Copy the Python venv from builder stage
COPY --from=vllm-build /opt/venv /opt/venv

# Ensure venv is prioritized in PATH + ROCm path is set
ENV PATH="/opt/venv/bin:${PATH}" \
    ROCM_PATH=/opt/rocm

# Copy vLLM source and build log for debugging/reference
COPY --from=vllm-build /workspace/vllm-src /workspace/vllm-src
COPY --from=vllm-build /workspace/vllm_build.log /workspace/vllm_build.log

# Set runtime environment variables
ENV HIP_VISIBLE_DEVICES=all \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0

# Create necessary directories for models and cache
RUN mkdir -p /workspace/models /workspace/.cache/huggingface

EXPOSE 8000

# Default command: launch vLLM OpenAI-compatible API server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]