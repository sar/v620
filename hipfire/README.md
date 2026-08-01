# Hipfire Docker Build Target: gfx1030

This folder contains a Dockerized build environment for the [Hipfire](https://github.com/Kaden-Schutt/hipfire) project. It is designed to compile the project **on a machine without an attached GPU** (e.g., a cloud build VM) for the `gfx1030` architecture (AMD RX 6000 series).

## Why this exists?
The standard Hipfire install script requires an AMD GPU to be present (`/dev/kfd`) to detect architecture and verify the driver. This setup bypasses that requirement by:
1.  Using a ROCm Docker image that contains all build tools (headers/libs) without needing the hardware.
2.  Force-setting the target architecture to `gfx1030`.
3.  Disabling runtime hardware checks during the build process.

## Files

*   **`Dockerfile`**: Defines the build environment using `rocm/dev-ubuntu-24.04:7.2.1-complete`. It installs Rust, Bun, and sets up the necessary environment variables.
*   **`install_headless.sh`**: The modified install script. It clones the Hipfire repo, builds the binaries, and installs them inside the container image.
*   **`deps.sh`**: (Optional) A dependency script for non-Docker installations. **Not used by the Docker build.**

## Prerequisites

*   **Docker**: Must be installed and running on your build machine.
*   **Disk Space**: Ensure you have ~10GB+ free for the Docker image and build artifacts.

## Usage

### 1. Build the Docker Image
Run this command from inside this folder (`gfx1030/`):

```bash
docker build -t hipfire-gfx1030 .
```

*Note: The first run will take several minutes as it downloads the ROCm base image and compiles the project from source.*

### 2. Extract the Binaries
Once the build completes, the binaries are stored inside the container image at `/root/.hipfire`. Use the following commands to copy them to your local machine:

```bash
# 1. Create a temporary container from the image
docker create --name temp-hipfire hipfire-gfx1030

# 2. Copy the compiled folder to your current directory
docker cp temp-hipfire:/root/.hipfire ./hipfire_dist

# 3. Clean up the temporary container
docker rm temp-hipfire
```

You will now have a folder named `hipfire_dist` containing the `bin`, `cli`, and `models` directories.

### 3. Deploy to Target Machine
1.  Zip or transfer the `hipfire_dist` folder to your target machine (the one with the AMD GPU).
2.  Move the contents to your home directory:
    ```bash
    mv hipfire_dist ~/.hipfire
    ```
3.  Add the binary folder to your PATH:
    ```bash
    echo 'export PATH="$HOME/.hipfire/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

## Running on the Target Machine

Ensure your target machine has the ROCm runtime installed (`rocm-hip-runtime` or full ROCm stack).

You can now run Hipfire:

```bash
hipfire list
hipfire run <model.hfq> "Hello world"
```

## Architecture Notes

This build targets **`gfx1030`** (e.g., RX 6800, RX 6900 XT).
If you need to target a different architecture (like RDNA3 `gfx1100`), edit the `Dockerfile` and `install_headless.sh` to change:
*   `HSA_OVERRIDE_GFX_VERSION=10.3.0`
*   `AMDGPU_TARGETS=gfx1030`
