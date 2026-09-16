# AMD GPU RDNA2 Testing & Benchmarking Suite

A comprehensive, containerized environment for stress-testing, validating, and benchmarking AMD GPUs, specifically tailored for the **RDNA2 architecture (`gfx1030`)**, such as the Radeon Pro W6800.

This image compiles and retains the source code for all major AMD/ROCm ecosystem testing tools in `/workspace`, allowing for easy recompilation or modification. It is designed to build successfully in headless CI environments (like GitHub Actions) without a GPU attached.

## 📦 Included Tools

| Tool | Category | Description |
| :--- | :--- | :--- |
| **[memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan)** | VRAM Stress | Vulkan compute tool for testing video memory stability and catching hardware/VRAM errors. |
| **[rccl-tests](https://github.com/ROCm/rccl-tests)** | Comm. Bandwidth | The AMD equivalent to `nccl-tests`. Tests multi-GPU communication bandwidth (All-Reduce, Broadcast, etc.). |
| **[rocm_bandwidth_test](https://github.com/ROCm/rocm_bandwidth_test)** | PCIe/Interconnect | Tests CPU-to-GPU and GPU-to-GPU data transfer bandwidth over PCIe or xGMI. |
| **[ROCmValidationSuite (RVS)](https://github.com/ROCm/ROCmValidationSuite)** | Diagnostics | AMD's official system validation tool. Includes `gst` (GPU Stress), `iet` (Input Energy), and `pbt` (PCIe Bandwidth) modules. |
| **[gpu-burn](https://github.com/wilicc/gpu-burn)** | FLOPs Stress | A HIP-ported heavy GEMM stress test to push GPUs to 100% compute and thermal limits. |
| **[mixbench](https://github.com/ekondis/mixbench)** | Roofline Modeling | Benchmarking tool to evaluate the performance limits (compute vs. memory bound) of HIP and OpenCL kernels. |
| **[rocm-examples](https://github.com/ROCm/rocm-examples)** | HIP/Library | A collection of examples for rocBLAS, rocFFT, hipSOLVER, and other core ROCm libraries. |
| **[clpeak](https://github.com/krrishnarraj/clpeak)** | OpenCL Perf | Measures peak OpenCL performance metrics (global/local memory bandwidth, compute SP/DP/HP). |

## Headless Build & Environment Variables

Because RDNA2 (`gfx1030`) is officially an "experimental" or consumer architecture in many ROCm releases, the ROCm runtime and compiler will often throw `HSA_STATUS_ERROR_INVALID_ISA` if they don't detect a supported datacenter GPU. 

To allow this Docker image to compile and run on headless CI runners (like GitHub Actions) and correctly target RDNA2, the following environment variables are baked into the image:

```bash
# Tricks the ROCm runtime into accepting RDNA2 (gfx1030) instructions
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Instructs the HIP/Clang compiler to generate ISA specifically for RDNA2
ENV AMDGPU_TARGETS=gfx1030
ENV GPU_TARGETS="gfx1030"
```

## Building the Image

You can build the image using your base image. The Dockerfile accepts a `BASE_IMAGE` argument.

```bash
docker build \
  --build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:6.2.2-complete \
  -t rocm-rdna2-testing:latest .
```
*(Note: Adjust the `BASE_IMAGE` to match the specific `rocm/dev-ubuntu-xx.xx` tag you are using in your environment).*

## Running the Container

To use the GPU inside the container, you must expose the AMD DRM devices, add the `video` group, and allocate sufficient shared memory (`--ipc=host` and `--shm-size`), which is strictly required by ROCm and RCCL.

```bash
docker run -it --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  -v /path/to/your/local/workspace:/app \
  rocm-rdna2-testing:latest
```

## Usage Cheatsheet

All source code is preserved in `/workspace` if you need to recompile. Binaries that are used frequently are symlinked or copied to `/usr/local/bin`.

### 1. VRAM Stability Test (memtest_vulkan)
```bash
# Run standard 6+ minute VRAM stress test
memtest_vulkan
```

### 2. GPU Compute Stress Test (gpu-burn)
```bash
# Stress test the GPU for 60 seconds
gpu_burn 60
```

### 3. PCIe / Interconnect Bandwidth (rocm_bandwidth_test)
```bash
cd /workspace/rocm_bandwidth_test/build
./rocm_bandwidth_test
```

### 4. Multi-GPU Communication (rccl-tests)
```bash
cd /workspace/rccl-tests

# Run All-Reduce performance test (sizes from 8 Bytes to 128MB)
./all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

### 5. ROCm Validation Suite (RVS)
```bash
cd /workspace/ROCmValidationSuite/build/rvs

# Run all default validation modules (gst, iet, pbt)
./rvs -g all

# Run a specific module (e.g., GPU Stress Test) in verbose mode
./rvs -m gst -d 3
```

### 6. OpenCL Peak Performance (clpeak)
```bash
clpeak
```

### 7. Recompiling Tools
If you want to change compile flags or update a repository, simply navigate to the `/workspace` directory:

```bash
cd /workspace/rccl-tests
make clean
make MPI=1 HIP_HOME=/opt/rocm NCCL_HOME=/opt/rocm/lib GPU_TARGETS="gfx1030" -j $(nproc)
```

## 📂 Directory Structure

```text
/workspace/
├── memtest_vulkan/         # Source & Cargo build artifacts
├── rccl-tests/             # MPI RCCL bandwidth tests
├── rocm_bandwidth_test/    # Next-gen RBT-NG source & build dir
├── ROCmValidationSuite/    # RVS source & build dir
├── gpu-burn/               # HIP GEMM stress test source
├── rocm-examples/          # Official ROCm library examples
├── mixbench/               # Roofline benchmarks (HIP & OpenCL)
├── hip-tests/              # ROCm HIP unit tests
└── clpeak/                 # OpenCL peak perf source

/usr/local/bin/
├── memtest_vulkan          # Executable
├── gpu_burn                # Executable
└── clpeak                  # Executable
```