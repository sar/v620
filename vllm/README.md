# 🐳 vLLM ROCm 7.x.x Docker Image (v0.2x.x)

Optimized Docker image for running **vLLM v0.2x.x** on AMD GPUs using **ROCm 7.x.x**. Built for headless CI environments (GitHub Actions, GitLab CI) with zero GPU dependencies during compilation. Ready for inference on RDNA2/3 and CDNA architectures.

## 📋 Table of Contents
- [AMD GPU Compatibility](#-amd-gpu-compatibility)
- [Prerequisites](#-prerequisites)
- [Build Instructions](#-build-instructions)
- [Run Instructions](#-run-instructions)
- [How to Drop into `/bin/bash`](#-how-to-drop-into-binbash)
- [Environment Variables Reference](#-environment-variables-reference)
- [Troubleshooting & AMD Limitations](#-troubleshooting--amd-limitations)

---

## 🟢 AMD GPU Compatibility

| Architecture | GPU Examples | Status | Notes |
|--------------|--------------|--------|-------|
| **RDNA2** (`gfx1030`) | Radeon Pro V620, RX 6800/6900 | ✅ Supported | Requires `HSA_OVERRIDE_GFX_VERSION=10.3.0` (pre-configured) |
| **RDNA3** (`gfx1100/1101/1102`) | RX 7900 XTX, Pro W7900 | ✅ Supported | Full FP32/FP16/BF16. Triton fallback for attention |
| **CDNA1/2** (`gfx906/90a`) | MI100, MI250/X | ⚠️ Legacy | Older kernels, may require vLLM `<0.18` |
| **CDNA3** (`gfx942`) | MI300A/X | ✅ Optimized | Native FP8 W8A8, FlashAttention, multi-GPU NCCL |

> 🔹 **vLLM on ROCm** disables NVIDIA-only kernels automatically. FlashAttention is replaced with Triton/hipblasLt fallbacks. FP8 quantization is only hardware-accelerated on CDNA3 (MI300 series); RDNA falls back to BF16/FP16.

---

## 🛠 Prerequisites

- **Host OS**: Ubuntu 24.04 (recommended)
- **Docker**: `24.0+` with `buildx` enabled
- **GPU**: AMD ROCm-compatible GPU (see matrix above)
- **Drivers**: Host must have ROCm drivers installed (`amdgpu-install` or distro packages)
- **Disk**: ~50 GB for build, ~20 GB for image
- **RAM**: 32 GB+ recommended for model loading

---

## 🏗 Build Instructions

### Local Build
```bash
DOCKER_BUILDKIT=1 docker build -t vllm-rocm:7.x.x .   # Replace Version
```

### GitHub Actions (CI/CD)
Add to `.github/workflows/build.yml`:
```yaml
name: Build vLLM ROCm Image
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: vllm-rocm:7.x.x   # Replace Version
          cache-from: type=gha
          cache-to: type=gha,mode=max
```
> ✅ **No GPU access required during build.** All HIP/C++ kernels are cross-compiled for `gfx1030`.

---

## 🚀 Run Instructions

### Start vLLM API Server
```bash
docker run -d --rm \
  --name vllm-rocm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/workspace/models \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  vllm-rocm:7.x.x \   # Replace Version
  --model /workspace/models/your-model \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### Verify Server is Running
```bash
curl http://localhost:8000/v1/models
```

---

## 💻 How to Drop into `/bin/bash`

You have two options depending on your workflow:

### Option 1: Runtime Override (Recommended)
Keep the Dockerfile unchanged. Launch with `--entrypoint`:
```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint /bin/bash \
  vllm-rocm:7.x.x
```

### Option 2: Bake into Dockerfile
Replace the last two lines of your `Dockerfile`:
```dockerfile
# Change this:
EXPOSE 8000
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]

# To this:
EXPOSE 8000
CMD ["/bin/bash"]
```
Then run normally: `docker run -it --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host vllm-rocm:7.x.x`

> 💡 **Tip**: Use Option 1 for development/debugging. It preserves the production `CMD` while giving you an interactive shell on demand.

---

## 🔧 Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLLM_TARGET_DEVICE` | `rocm` | Force ROCm backend |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Bypass official support checks for RDNA2 |
| `HIP_VISIBLE_DEVICES` | `all` | GPU selection (e.g., `0,1`) |
| `PYTORCH_ROCM_ARCH` | `gfx1030` | Target architecture for kernel compilation |
| `VLLM_USE_TRITON_FLASH_ATTN` | `0` | Use Triton attention (stable on RDNA) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Optimized VRAM allocation |
| `ROCM_PATH` | `/opt/rocm` | CMake/HIP toolchain path |
| `HF_HOME` | `/workspace/.cache/huggingface` | Model cache directory |

---

## ⚠️ Troubleshooting & AMD Limitations

| Issue | Solution |
|-------|----------|
| `ImportError: libMIOpen.so.1` / `libhipfft.so.0` | Use `rocm/dev-ubuntu-24.04:7.x.x-complete` base (already fixed in Dockerfile) |
| `ROCm kernel failed to compile` | Clear cache: `rm -rf ~/.cache/torch_extensions ~/.triton` inside container |
| `CUDA out of memory` (shows as HIP) | Reduce `--gpu-memory-utilization 0.8`, lower `--max-model-len`, or use quantization (`--quantization awq`) |
| First run is extremely slow | Triton/hipcc compiles kernels on first load. Subsequent runs use cached `.so` files in `~/.cache/torch_extensions` |
| Multi-GPU not visible | Ensure `--ipc=host` and `--group-add video` are passed. Run `rocm-smi` inside container to verify detection |
| `HSA status 0x3003` | Host ROCm driver mismatch. Update host to ROCm 7.2+ or use `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

### 📦 Persisting Kernel Cache
Triton/PyTorch cache compiled kernels. Mount a volume to avoid recompilation on container restart:
```bash
-v /path/to/cache:/root/.cache/torch_extensions
```

---

## 📜 License & Credits
- **vLLM**: Apache 2.0 ([vllm-project/vllm](https://github.com/vllm-project/vllm))
- **PyTorch ROCm**: BSD 3-Clause ([pytorch/pytorch](https://github.com/pytorch/pytorch))
- **ROCm Base Image**: AMD Public License
- **Triton**: Apache 2.0

> Built for headless CI + AMD inference. Test thoroughly before production deployment.

