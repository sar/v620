# Llama.cpp ROCm Docker (gfx1030 Optimized)

This repository provides a highly optimized Docker environment for compiling and running `llama.cpp` with AMD ROCm support, specifically targeted at **`gfx1030` (RDNA2)** architecture GPUs (e.g., Radeon PRO v620, Radeon RX 6800 / 6900 XT).

## How the Compilation Works

Standard `llama.cpp` builds default to CPU or NVIDIA CUDA. This Dockerfile injects specific flags into the CMake configuration to maximize performance on AMD hardware:

*   **`-DGGML_HIP=ON`**: This enables the HIP backend, replacing the standard CUDA implementation with AMD-compatible compute calls [[10]].
*   **`-DGPU_TARGETS=gfx1030`**: Forces the ROCm compiler (`hipcc`) to generate machine code specifically for the RDNA2 architecture.
*   **`-DGGML_CUDA_FA_ALL_QUANTS=ON`**: Despite the "CUDA" naming convention, this flag compiles Flash Attention kernels for *all* KV cache quantization types (e.g., q4_0, q8_0, f16) [[1]]. This ensures maximum compatibility and performance regardless of the GGUF model you load.
*   **`-DBUILD_SHARED_LIBS=ON`**: Builds shared libraries (`.so`) instead of statically linking them. This keeps the final binaries lean but requires the ROCm libraries to be discoverable at runtime.
*   **`-DGGML_CURL=OFF`**: Disables network fetching inside `llama.cpp` to keep the build lean and secure.

### The Issue #25807 Fix
Recent ROCm packaging (specifically version 7.14) sometimes fails to register `/opt/rocm/lib` with the dynamic loader, resulting in `error while loading shared libraries` when trying to run the compiled binaries [[10]]. 

This Dockerfile prepends `/opt/rocm/core-7.14/lib` to the `LD_LIBRARY_PATH` during both the build and execution phases to ensure `libhipblas.so` and other ROCm libraries are always found.

## Building the Image

The GitHub Actions workflow in this repository automatically builds and pushes the image to GHCR on every push to `llama.cpp/`. To build it manually on a local machine with ROCm installed:

```bash
docker build -t ghcr.io/YOUR_USERNAME/llama-cpp-rocm:gfx1030 -f llama.cpp/Dockerfile ./llama.cpp/
```

## Running Inference

To run inference, the container requires direct access to the host's AMD GPU devices (`/dev/kfd` and `/dev/dri`) and the `video` group. 

*(Note: While the Dockerfile sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` at build time, you should also pass it at runtime to ensure the driver correctly identifies the device).*

### 1. Interactive CLI Inference (`llama-cli`)
Use `llama-cli` for interactive chat or one-off prompts.

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v /path/to/models:/models \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ghcr.io/YOUR_USERNAME/llama-cpp-rocm:gfx1030 \
  llama-cli -m /models/model.gguf -p "Your prompt here" -ngl 99
```

### 2. Local Server (`llama-server`)
Use `llama-server` to expose an OpenAI-compatible REST API.

```bash
docker run -d --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -p 8080:8080 \
  -v /path/to/models:/models \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ghcr.io/YOUR_USERNAME/llama-cpp-rocm:gfx1030 \
  llama-server -m /models/model.gguf --host 0.0.0.0 --port 8080 -ngl 99
```

### 3. Benchmarking (`llama-bench`)
Test the token generation speed of your specific hardware.

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v /path/to/models:/models \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ghcr.io/YOUR_USERNAME/llama-cpp-rocm:gfx1030 \
  llama-bench -m /models/model.gguf -ngl 99
```

## Critical Runtime Arguments Explained

*   **`-ngl 99`**: (GPU Layers) Essential for performance. This offloads all transformer layers to your GPU's VRAM. If omitted, inference will fall back to the CPU and be incredibly slow.
*   **`--device` & `--group-add video`**: Grants the Docker container low-level permissions to communicate with the AMD kernel-mode driver.
*   **`HSA_OVERRIDE_GFX_VERSION=10.3.0`**: Overrides hardware detection to force the ROCm driver to treat the card as a supported `gfx1030` device, preventing `hipBLAS` initialization errors.