# syntax=docker/dockerfile:1.4
# =============================================================================
# vLLM v0.20.0 on AMD gfx1030 (Radeon Pro V620) — ROCm 7.2.3
# =============================================================================
# Minimal base image, use -complete for full img
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2.3
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
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    # Critical: Ensure ROCm libraries are found at runtime [[16]][[19]]
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# =============================================================================
# Stage 1: System build dependencies
# =============================================================================
FROM base AS builder

# Install only what the base image doesn't provide
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    python3.12 python3.12-dev python3.12-venv \
    wget ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install ROCm runtime libraries missing from minimal dev image
# These provide: libhipfft.so.0, libMIOpen.so.1, librocblas.so, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core ROCm runtime
    rocm-hip-libraries \
    # Individual libraries PyTorch needs
    hipfft \
    miopen-hip \
    rocblas \
    hipblas \
    hipblaslt \
    rocrand \
    rccl \
    comgr \
    hip-runtime-amd \
    # Optional: dev headers if building from source later
    rocm-dev \
    && ldconfig /opt/rocm/lib \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a venv and make it the default python/pip
RUN python3.12 -m venv /opt/venv

# Ensure the venv is on PATH for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip inside the venv (no --break-system-packages needed in venv)
RUN pip install --upgrade pip wheel setuptools

# =============================================================================
# Stage 2: Install PyTorch ROCm 7.2.3 from AMD wheels
# =============================================================================
# ✅ UPDATED: Using ROCm 7.2.3 wheels to match base image [[1]]
# Wheel listing: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/

RUN mkdir -p /tmp/rocm-wheels && \
    cd /tmp/rocm-wheels && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torch-2.9.1+rocm7.2.3.lw.gitebc02d69-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torchvision-0.24.0+rocm7.2.3.gitb919bd0c-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/torchaudio-2.9.0+rocm7.2.3.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.3/triton-3.5.1+rocm7.2.3.gita272dfa8-cp312-cp312-linux_x86_64.whl"

# Install ROCm wheels (removed --break-system-packages; not needed in venv)
RUN pip uninstall -y torch torchvision torchaudio triton 2>/dev/null || true && \
    pip install --no-cache-dir \
        /tmp/rocm-wheels/torch-2.9.1+rocm7.2.3.lw.gitebc02d69-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchvision-0.24.0+rocm7.2.3.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchaudio-2.9.0+rocm7.2.3.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/triton-3.5.1+rocm7.2.3.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    rm -rf /tmp/rocm-wheels

# ✅ Sanity check with explicit LD_LIBRARY_PATH (redundant but safe)
RUN LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} python -c "import torch; print('torch:', torch.__version__); print('ROCm available:', torch.cuda.is_available())"

# =============================================================================
# Stage 3: Install vLLM v0.20.0 from Git
# =============================================================================
FROM builder AS vllm-build

# Pre-install vLLM build dependencies
RUN pip install --no-cache-dir setuptools_scm wheel ninja

# Pre-install vLLM's Python dependencies (pinned numpy to avoid conflicts)
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

# Clone vLLM v0.20.0
ARG VLLM_VERSION=v0.20.0
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src

# Build vLLM with ROCm flags
RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /workspace/vllm_build.log

# Verify the build succeeded
RUN cd / && LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH} python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Compiler errors:"; \
    grep -i "error\|fatal\|hipcc: error" /workspace/vllm_build.log | tail -30; \
    exit 1; }

# =============================================================================
# Stage 4: Final minimal inference image
# =============================================================================
FROM base AS final

# Remove PEP 668 restriction
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Copy the Python venv from builder
COPY --from=vllm-build /opt/venv /opt/venv

# Ensure venv is prioritized in PATH and ROCm libs are findable
ENV PATH="/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

# Copy vLLM source and build log for debugging
COPY --from=vllm-build /workspace/vllm-src /workspace/vllm-src
COPY --from=vllm-build /workspace/vllm_build.log /workspace/vllm_build.log

# Set runtime environment
ENV HIP_VISIBLE_DEVICES=all \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/.cache/huggingface

EXPOSE 8000

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]