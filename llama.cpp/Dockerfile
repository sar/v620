# Add ARG to accept the base image dynamically
ARG BASE_IMAGE=rocm-base
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Fix GitHub Issue #25807: 
# Recent ROCm packaging sometimes fails to register /opt/rocm/lib with the dynamic loader.
# Prepending this ensures libhipblas.so and other ROCm libs are found at runtime.
ENV LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LD_LIBRARY_PATH

WORKDIR /app

# Clone llama.cpp source code
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /app/llama.cpp

WORKDIR /app/llama.cpp

# Build llama.cpp with HIP support
# untested on gfx1030: -DGGML_HIP_ROCWMMA_FATTN=ON
# Update LD PATH to reference ROCM_VERSION
RUN LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib:$LD_LIBRARY_PATH \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGPU_TARGETS=gfx1030 \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CURL=OFF \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --config Release --target llama-server llama-cli llama-bench -- -j $(nproc)

# Set PATH
ENV PATH="/app/llama.cpp/build/bin:${PATH}"

WORKDIR /app

CMD ["/bin/bash"]