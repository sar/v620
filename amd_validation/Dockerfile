# Add ARG to accept the base image dynamically
ARG BASE_IMAGE=rocm-dev
FROM ${BASE_IMAGE}

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies, MPI for rccl-tests, RVS dependencies, OpenCL headers, Vulkan tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools & compilers
    build-essential \
    cmake \
    ninja-build \
    ccache \
    pkg-config \
    autoconf \
    automake \
    libtool \
    # Debugging, profiling, and static analysis
    gdb \
    valgrind \
    lcov \
    cppcheck \
    clang-format \
    clang-tidy \
    # Networking & Diagnostic Tools
    nmap \
    curl \
    wget \
    git \
    ca-certificates \
    htop \
    lsof \
    strace \
    tcpdump \
    iputils-ping \
    dnsutils \
    net-tools \
    # Modern CLI Tools
    ripgrep \
    fd-find \
    tmux \
    screen \
    vim \
    nano \
    zsh \
    # JSON & YAML processors
    jq \
    # C++ Development Libraries
    nlohmann-json3-dev \
    libfmt-dev \
    libspdlog-dev \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libbenchmark-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \
    libsqlite3-dev \
    # Multimedia / FFmpeg
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    # Graphics / Vulkan
    libsdl2-dev \
    libglew-dev \
    libglfw3-dev \
    libvulkan-dev \
    vulkan-tools \
    glslang-dev \
    glslang-tools \
    lld \
    # Python 3
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # ROCm / AMD GPU Testing & Benchmarking Dependencies
    rocm-cmake \
    libgomp1 \
    libopenmpi-dev \
    openmpi-bin \
    libpci3 \
    libpci-dev \
    doxygen \
    libyaml-cpp-dev \
    libnuma-dev \
    opencl-headers \
    ocl-icd-opencl-dev \
    libhwloc-dev \
    unzip \
    glslc \
    && rm -rf /var/lib/apt/lists/*

# FIX FOR: fatal error: 'cstdint' file not found
# ROCm's clang++ scans for GCC installations. If it finds newer GCC versions (like 15/16) 
# but the -dev packages (which contain the C++ standard headers) are missing, compilation fails. 
# This attempts to install the libstdc++ headers for any GCC versions present on the system.
RUN apt-get update && \
    for ver in 12 13 14 15 16 17; do \
        apt-get install -y --no-install-recommends libstdc++-${ver}-dev 2>/dev/null || true; \
    done && \
    rm -rf /var/lib/apt/lists/*

# Install the Go version of yq
RUN wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 && \
    chmod +x /usr/local/bin/yq

# Set up ccache
ENV CCACHE_DIR=/ccache
ENV PATH="/usr/lib/ccache:${PATH}"
RUN mkdir -p /ccache && chmod 777 /ccache

# 2 Install Bun
ENV BUN_INSTALL=/usr/local/bun
ENV PATH="/usr/local/bun/bin:${PATH}"
RUN curl -fsSL https://bun.sh/install | bash

# 3 Install Rust
# Fix for "Invalid cross-device link" (os error 18) in Docker BuildKit by avoiding /root overlayfs quirks
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

# 4 Set Environment Variables for Headless Build (RDNA2 gfx1030)
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV AMDGPU_TARGETS=gfx1030
ENV GPU_TARGETS="gfx1030"
ENV ROCM_PATH=/opt/rocm

# Update PATH for Bun, Rust, and ROCm
ENV PATH="/usr/local/cargo/bin:/usr/local/bun/bin:${ROCM_PATH}/bin:${ROCM_PATH}/lib:${PATH}"

# Fix the undefined variable warning for LD_LIBRARY_PATH by setting a clean base path
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu

# Create workspace to keep source code
RUN mkdir -p /workspace

# ==============================================================================
# 1. memtest_vulkan - Vulkan compute tool for testing video memory stability
# ==============================================================================
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/GpuZelenograd/memtest_vulkan.git && \
    cd memtest_vulkan && \
    cargo build --release && \
    cp target/release/memtest_vulkan /usr/local/bin/memtest_vulkan

# ==============================================================================
# 2. rccl-tests - NCCL equivalent for AMD RCCL (GPU communication bandwidth)
# ==============================================================================
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/ROCm/rccl-tests.git && \
    cd rccl-tests && \
    make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi HIP_HOME=${ROCM_PATH} NCCL_HOME=${ROCM_PATH} GPU_TARGETS="gfx1030" -j $(nproc)

# ==============================================================================
# 3. rocm_bandwidth_test - ROCm Bandwidth Test (CPU <-> GPU & GPU <-> GPU)
# ==============================================================================
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/ROCm/rocm_bandwidth_test.git && \
    cd rocm_bandwidth_test && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. \
        -DAMD_APP_STANDALONE_BUILD_PACKAGE=OFF \
        -DAMD_APP_ROCM_BUILD_PACKAGE=ON \
        -DGPU_TARGETS="gfx1030" && \
    make -j $(nproc)

# ==============================================================================
# 4. ROCmValidationSuite (RVS) - System validation, stress testing & diagnostics
# ==============================================================================
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/ROCm/ROCmValidationSuite.git && \
    cd ROCmValidationSuite && \
    cmake -B ./build -DROCM_PATH=${ROCM_PATH} -DCMAKE_INSTALL_PREFIX=${ROCM_PATH} -DCPACK_PACKAGING_INSTALL_PREFIX=${ROCM_PATH} -DGPU_TARGETS="gfx1030" && \
    make -C ./build -j $(nproc)

# ==============================================================================
# 5. gpu-burn - GPU FLOPS stress test for AMD HIP (Fixed Source)
# ==============================================================================
WORKDIR /workspace
# Use the AMD/HIP fork of gpu-burn since the original wilicc repo lacks the HIP Makefile
RUN git clone https://github.com/KerwinTsaiii/amd-gpu-burn.git && \
    cd amd-gpu-burn && \
    make COMPUTE=gfx1030 && \
    cp gpu_burn /usr/local/bin/gpu_burn

# ==============================================================================
# 6. mixbench - Performance benchmarking for OpenCL and HIP
# ==============================================================================
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/ekondis/mixbench.git && \
    cd mixbench && \
    mkdir build_hip && cd build_hip && \
    cmake ../mixbench-hip -DCMAKE_PREFIX_PATH=${ROCM_PATH} -DGPU_TARGETS="gfx1030" && \
    cmake --build ./ -j $(nproc) && \
    cd ../ && \
    mkdir build_opencl && cd build_opencl && \
    cmake ../mixbench-opencl && \
    cmake --build ./ -j $(nproc)

# ==============================================================================
# 7. clpeak - OpenCL performance benchmark
# ==============================================================================
WORKDIR /workspace
# Disable Vulkan backend to avoid strict 'glslc' dependency, focus on OpenCL/HIP for AMD
RUN git clone --recurse-submodules https://github.com/krrishnarraj/clpeak.git && \
    cd clpeak && \
    mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=${ROCM_PATH} -DCLPEAK_ENABLE_VULKAN=OFF -DCLPEAK_ENABLE_GUI=OFF .. && \
    make -j $(nproc) && \
    cp clpeak /usr/local/bin/clpeak

# Reset workdir and default command
WORKDIR /app
CMD ["/bin/bash"]
