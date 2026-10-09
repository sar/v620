# Add ARG to accept the base image dynamically
ARG BASE_IMAGE=rocm10-base
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Fix GitHub Issue #25807: 
# Recent ROCm packaging sometimes fails to register /opt/rocm/lib with the dynamic loader.
ENV LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:$LD_LIBRARY_PATH

# Install required system dependencies for Strata and llama.cpp builds
# Strata requires CMake 3.24+ and a C++20 host compiler (GCC 11+ / Clang 14+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake ninja-build python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1. Clone llama.cpp at the EXACT pinned commit required by Strata
# Strata's CPU expert offloading breaks if you use an arbitrary newer commit from 'main'
RUN git clone https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp && \
    cd /workspace/llama.cpp && \
    git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d

# 2. Clone Strata
RUN git clone --depth 1 https://github.com/Niko1221/Strata.git /workspace/Strata

WORKDIR /workspace/Strata

# 3. Install Python dependencies (required for Strata's model packing tools like iq_pack.py)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt || \
    pip3 install --no-cache-dir -r requirements.txt

# 4. Build Strata with HIP support for AMD RDNA2 (gfx1030)
# We force GGML_NATIVE=OFF and STRATA_PORTABLE=ON to ensure no AVX-512 instructions
# are injected by the cloud runner's CPU, gating runner on Haswell/Broadwell.
RUN LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:$LD_LIBRARY_PATH \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    HIP_PLATFORM=amd \
    HIP_COMPILER=clang \
    HIP_RUNTIME=rocclr \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    cmake -S . -B build-hip \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_HIP=ON \
        -DSTRATA_ENABLE_CUDA=OFF \
        -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_PREFILL_MMQ=ON \
        -DSTRATA_PORTABLE=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode" \
        -DSTRATA_GGML_DIR=/workspace/llama.cpp \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_AVX=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_BMI2=ON \
        -DGGML_AVX512=OFF \
        -DGGML_STATIC=OFF \
    && cmake --build build-hip --target strata -- -j $(nproc)

# Set PATH and environment variables for running Strata
ENV PATH="/workspace/Strata/build-hip:${PATH}"
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_PLATFORM=amd
ENV HIP_COMPILER=clang
ENV HIP_RUNTIME=rocclr
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

CMD ["/bin/bash"]
