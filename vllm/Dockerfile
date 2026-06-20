# syntax=docker/dockerfile:1.4
# =============================================================================
# Version Control Args
# =============================================================================
ARG BASE_IMAGE=rocm-base
ARG ROCM_VERSION=7.2.4

# --- Python3 Wheel version strings ---
ARG PYTORCH_WHEEL_VERSION=2.10.0+rocm7.2.4.lw.git3d3aa833
ARG TORCHVISION_WHEEL_VERSION=0.25.0+rocm7.2.4.git82df5f59
ARG TORCHAUDIO_WHEEL_VERSION=2.10.0+rocm7.2.4.git5047768f
ARG TRITON_WHEEL_VERSION=3.6.0+rocm7.2.4.git4ed88892
ARG VLLM_VERSION=v0.23.0

# --- Build dependencies version strings ---
ARG CMAKE_VERSION=">=3.26.1"
ARG NINJA_VERSION=""
ARG PACKAGING_VERSION=">=24.2"
ARG SETUPTOOLS_VERSION=">=77.0.3,<81.0.0"
ARG SETUPTOOLS_SCM_VERSION=">=8.0"
ARG SETUPTOOLS_RUST_VERSION=">=1.9.0"
ARG WHEEL_VERSION=""
ARG JINJA2_VERSION=""

# --- vLLM Python dependencies version strings ---
ARG TRANSFORMERS_VERSION=">=4.53.0"
ARG RAY_VERSION="[default]>=2.40.0"
ARG PYDANTIC_VERSION=">=2.10.0"
ARG FASTAPI_VERSION=">=0.115.0"
ARG UVICORN_VERSION="[standard]>=0.34.0"
ARG AIOHTTP_VERSION=">=3.11.0"
ARG TIKTOKEN_VERSION=">=0.9.0"
ARG SENTENCEPIECE_VERSION=">=0.2.0"
ARG PROMETHEUS_CLIENT_VERSION=">=0.21.0"
ARG HUGGINGFACE_HUB_VERSION=">=0.28.0"
ARG TOKENIZERS_VERSION=">=0.21.0"
ARG NUMPY_VERSION=">=1.26,<2.1"
ARG XGRAMMAR_VERSION=">=0.1.10"
ARG LM_FORMAT_ENFORCER_VERSION=">=0.10.0"

# =============================================================================
# Base Image
# =============================================================================
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
# Stage 1: Build vLLM (Merged to save BuildKit temporary storage)
# =============================================================================
FROM base AS build

# Redeclare ALL ARGs needed for the build stage
ARG ROCM_VERSION
ARG PYTORCH_WHEEL_VERSION
ARG TORCHVISION_WHEEL_VERSION
ARG TORCHAUDIO_WHEEL_VERSION
ARG TRITON_WHEEL_VERSION
ARG VLLM_VERSION
ARG CMAKE_VERSION
ARG NINJA_VERSION
ARG PACKAGING_VERSION
ARG SETUPTOOLS_VERSION
ARG SETUPTOOLS_SCM_VERSION
ARG SETUPTOOLS_RUST_VERSION
ARG WHEEL_VERSION
ARG JINJA2_VERSION
ARG TRANSFORMERS_VERSION
ARG RAY_VERSION
ARG PYDANTIC_VERSION
ARG FASTAPI_VERSION
ARG UVICORN_VERSION
ARG AIOHTTP_VERSION
ARG TIKTOKEN_VERSION
ARG SENTENCEPIECE_VERSION
ARG PROMETHEUS_CLIENT_VERSION
ARG HUGGINGFACE_HUB_VERSION
ARG TOKENIZERS_VERSION
ARG NUMPY_VERSION
ARG XGRAMMAR_VERSION
ARG LM_FORMAT_ENFORCER_VERSION

# Create virtual environment
RUN python3.12 -m venv /opt/venv

# Pre-install vLLM build dependencies
RUN pip install --no-cache-dir \
    "cmake${CMAKE_VERSION}" \
    "ninja${NINJA_VERSION}" \
    "packaging${PACKAGING_VERSION}" \
    "setuptools${SETUPTOOLS_VERSION}" \
    "setuptools-scm${SETUPTOOLS_SCM_VERSION}" \
    "setuptools-rust${SETUPTOOLS_RUST_VERSION}" \
    "wheel${WHEEL_VERSION}" \
    "jinja2${JINJA2_VERSION}"

# Download and install PyTorch ROCm wheels
RUN mkdir -p /tmp/rocm-wheels && \
    cd /tmp/rocm-wheels && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/torch-${PYTORCH_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/torchvision-${TORCHVISION_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/torchaudio-${TORCHAUDIO_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl" && \
    wget -q "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/triton-${TRITON_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl"

RUN pip install --no-cache-dir \
        /tmp/rocm-wheels/torch-${PYTORCH_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchvision-${TORCHVISION_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/torchaudio-${TORCHAUDIO_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl \
        /tmp/rocm-wheels/triton-${TRITON_WHEEL_VERSION}-cp312-cp312-linux_x86_64.whl && \
    rm -rf /tmp/rocm-wheels

# Sanity check
RUN python -c "import torch; print('torch:', torch.__version__); print('ROCm available:', torch.cuda.is_available())"

# Install vLLM Python dependencies
RUN pip install --no-cache-dir \
    "transformers${TRANSFORMERS_VERSION}" \
    "ray${RAY_VERSION}" \
    "pydantic${PYDANTIC_VERSION}" \
    "fastapi${FASTAPI_VERSION}" \
    "uvicorn${UVICORN_VERSION}" \
    "aiohttp${AIOHTTP_VERSION}" \
    "tiktoken${TIKTOKEN_VERSION}" \
    "sentencepiece${SENTENCEPIECE_VERSION}" \
    "prometheus-client${PROMETHEUS_CLIENT_VERSION}" \
    "huggingface-hub${HUGGINGFACE_HUB_VERSION}" \
    "tokenizers${TOKENIZERS_VERSION}" \
    "numpy${NUMPY_VERSION}" \
    "xgrammar${XGRAMMAR_VERSION}" \
    "lm-format-enforcer${LM_FORMAT_ENFORCER_VERSION}"

# Clone and build vLLM
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src

RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /workspace/vllm_build.log

# Verify build
RUN cd / && python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Checking logs:"; \
    grep -iE "error|fatal|CMake Error|hipcc" /workspace/vllm_build.log | tail -50; \
    exit 1; }

# Clean up build-only dependencies to shrink the venv for the final image
RUN pip uninstall -y cmake ninja

# =============================================================================
# Stage 2: Final minimal inference image
# =============================================================================
FROM base AS final

# Remove PEP 668 restriction for pip installs in final image
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Copy the Python venv from build stage
COPY --from=build /opt/venv /opt/venv

# Copy ONLY the build log for debugging (DO NOT copy vllm-src, it contains GBs of compiled object files)
COPY --from=build /workspace/vllm_build.log /workspace/vllm_build.log

# Ensure venv is prioritized in PATH + ROCm path is set
ENV PATH="/opt/venv/bin:${PATH}" \
    ROCM_PATH=/opt/rocm

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