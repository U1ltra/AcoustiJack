#!/bin/bash
# env.sh — Set up environment variables for each new terminal session
# Usage: source env.sh

REPO_ROOT="/root/AcoustiJack"
PLUGINS_DIR="$REPO_ROOT/plugins"

# ─────────────────────────────────────────────────────────────────────────────
# XLA / CUDA shim (recreate symlink since /tmp is cleared on reboot)
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p /tmp/fake_cuda/nvvm/libdevice
ln -sf /acoustiJack/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc \
    /tmp/fake_cuda/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/fake_cuda

# ─────────────────────────────────────────────────────────────────────────────
# Gazebo plugin paths
# ─────────────────────────────────────────────────────────────────────────────
for plugin_dir in "$PLUGINS_DIR"/*/; do
    plugin_name=$(basename "$plugin_dir")
    lib_path="${plugin_dir%/}/install/$plugin_name/lib"
    export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:+$GZ_SIM_SYSTEM_PLUGIN_PATH:}$lib_path"
done

# ─────────────────────────────────────────────────────────────────────────────
# Gazebo resource path
# ─────────────────────────────────────────────────────────────────────────────
export GZ_SIM_RESOURCE_PATH="$HOME/.gz/models"

echo "Environment variables set:"
echo "  XLA_FLAGS=$XLA_FLAGS"
echo "  GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH"
echo "  GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
