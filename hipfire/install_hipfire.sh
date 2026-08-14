#!/bin/bash
# install_hipfire.sh — Builds Hipfire from local source for GFX1030
set -euo pipefail

HIPFIRE_DIR="$HOME/.hipfire"
BIN_DIR="$HIPFIRE_DIR/bin"
REPO_DIR="/app/hipfire" # Path where Dockerfile cloned the source

# ─── HARDCODED CONFIG ───────────────────────────────────────
TARGET_ARCH="gfx1030"
# ───────────────────────────────────────────────────────────

echo "=== Hipfire Build Process (Local Source) ==="
echo "Source Dir: $REPO_DIR"
echo "Target Arch: $TARGET_ARCH"

# Verify source exists
if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: Source directory $REPO_DIR not found."
    exit 1
fi

# ─── Build Everything ───────────────────────────────────────
echo ""
echo "Starting Full Build..."

# 1. Set Build Flags (Correct for RX 6000 series RDNA2)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# 2. Build the workspace
cd "$REPO_DIR"

echo "Building Engine (Daemon + Infer)..."
# FIX: Changed '-p engine' to '-p hipfire-runtime' and used workspace feature syntax
cargo build --release \
    --features hipfire-runtime/deltanet \
    --example daemon \
    --example infer \
    --example infer_hfq \
    -p hipfire-runtime

echo "Building Quantizer and Tools..."
cargo build --release -p hipfire-quantize

# Optional: Build the new Rust CLI if it's in the workspace
# (Uncomment if you need the actual hipfire CLI tool, it will end up in target/release)
cargo build --release -p hipfire-cli 

# ─── Install Binaries (Auto-Detect) ─────────────────────────
echo ""
echo "Installing binaries to $BIN_DIR..."
mkdir -p "$BIN_DIR"

# 1. Copy binaries from target/release (catches 'hipfire-quantize' and potentially 'hipfire')
find target/release -maxdepth 1 -type f -executable -exec cp -f {} "$BIN_DIR/" \;

# 2. Copy binaries from target/release/examples (catches 'daemon', 'infer')
if [ -d "target/release/examples" ]; then
    find target/release/examples -maxdepth 1 -type f -executable -exec cp -f {} "$BIN_DIR/" \;
fi

echo "  Binaries installed."
ls -la "$BIN_DIR"

# ─── Install Kernels ────────────────────────────────────────
echo ""
echo "Setting up kernels for $TARGET_ARCH..."
KERNEL_DEST="$BIN_DIR/kernels/compiled/$TARGET_ARCH"
mkdir -p "$KERNEL_DEST"

if [ -d "kernels/compiled/$TARGET_ARCH" ]; then
    cp kernels/compiled/$TARGET_ARCH/*.hsaco "$KERNEL_DEST/" 2>/dev/null || true
    echo "  Kernels copied."
else
    echo "  No pre-compiled kernels found in repo. (Will JIT compile on first run)"
fi

# ─── Apply Configuration Files ─────────────────────────────
echo ""
echo "Applying configuration from /app/configs/..."

if [ -f "/app/configs/config.json" ]; then
    cp /app/configs/config.json "$HIPFIRE_DIR/config.json"
    echo "  config.json applied."
else
    echo "  WARNING: /app/configs/config.json not found."
fi

if [ -f "/app/configs/models.json" ]; then
    cp /app/configs/models.json "$HIPFIRE_DIR/models.json"
    echo "  models.json applied."
else
    echo "  WARNING: /app/configs/models.json not found."
fi

echo ""
echo "=== Build Complete ==="
