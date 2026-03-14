#!/bin/bash
# setup_env.sh — Prepare Gazebo plugins, models, and worlds for UAV attack simulations
# Usage: source setup_env.sh

REPO_ROOT="/root/uav-attacks"
PLUGINS_DIR="$REPO_ROOT/plugins"
OBJECTS_DIR="$REPO_ROOT/config/objects"
WORLDS_DIR="$REPO_ROOT/config/worlds"

PX4_GZ_MODELS="/root/PX4-Autopilot/Tools/simulation/gz/models"
PX4_GZ_WORLDS="/root/PX4-Autopilot/Tools/simulation/gz/worlds"
GZ_MODELS_DIR="$HOME/.gz/models"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Install MAVSDK (skip if already installed)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [0/3] Checking installations ==="

if find /usr -name "MAVSDKConfig.cmake" 2>/dev/null | grep -q .; then
    echo "  MAVSDK already installed, skipping."
else
    echo "  Installing MAVSDK..."
    MAVSDK_DEB="/tmp/libmavsdk-dev_1.4.16_ubuntu20.04_amd64.deb"
    wget -O "$MAVSDK_DEB" https://github.com/mavlink/MAVSDK/releases/download/v1.4.16/libmavsdk-dev_1.4.16_ubuntu20.04_amd64.deb \
        || { echo "ERROR: Failed to download MAVSDK"; return 1; }
    sudo dpkg -i "$MAVSDK_DEB" \
        || { echo "ERROR: Failed to install MAVSDK"; rm -f "$MAVSDK_DEB"; return 1; }
    rm -f "$MAVSDK_DEB"
    echo "  MAVSDK installed."
fi

# check if tomli_w is installed, if not, install it
if python3 -c "import tomli_w" &> /dev/null; then
    echo "  tomli_w already installed, skipping."
else
    echo "  Installing tomli_w..."
    pip install tomli_w \
        || { echo "ERROR: Failed to install tomli_w"; return 1; }
    echo "  tomli_w installed."
fi

if python3 -c "import pysocialforce" &> /dev/null; then
    pip uninstall -y pysocialforce \
        || { echo "ERROR: Failed to uninstall existing pysocialforce"; return 1; }
else
    echo "  pysocialforce not found, skipping uninstall."
fi
git clone https://github.com/U1ltra/PySocialForce.git
cd PySocialForce
pip install -e . \
    || { echo "ERROR: Failed to install PySocialForce"; return 1; }
cd ..

cd /root/uav-attacks
mkdir -p /tmp/fake_cuda/nvvm/libdevice
ln -sf /acoustiJack/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc /tmp/fake_cuda/nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/tmp/fake_cuda

# ─────────────────────────────────────────────────────────────────────────────
# 1. Compile Gazebo plugins
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [1/3] Building Gazebo plugins ==="

for plugin_dir in "$PLUGINS_DIR"/*/; do
    plugin_name=$(basename "$plugin_dir")
    echo "  Building $plugin_name ..."
    (
        cd "$plugin_dir"
        colcon build
    ) || { echo "ERROR: Failed to build $plugin_name"; return 1; }
    echo "  Done: $plugin_name"
done

# Add each plugin's lib directory to GZ_SIM_SYSTEM_PLUGIN_PATH
for plugin_dir in "$PLUGINS_DIR"/*/; do
    plugin_name=$(basename "$plugin_dir")
    lib_path="${plugin_dir%/}/install/$plugin_name/lib"
    export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:+$GZ_SIM_SYSTEM_PLUGIN_PATH:}$lib_path"
done

echo "  GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare Gazebo models
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [2/3] Preparing Gazebo models ==="

mkdir -p "$GZ_MODELS_DIR"

for model_dir in "$OBJECTS_DIR"/*/; do
    model_name=$(basename "$model_dir")
    echo "  Copying $model_name -> $GZ_MODELS_DIR/"
    cp -r "$model_dir" "$GZ_MODELS_DIR/" || { echo "ERROR: Failed to copy $model_name"; return 1; }
done

export GZ_SIM_RESOURCE_PATH="$HOME/.gz/models"
echo "  GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Prepare Gazebo worlds
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [3/3] Preparing Gazebo worlds ==="

# Copy gimbal_model.sdf as the PX4 gimbal model.sdf
GIMBAL_SDF="$WORLDS_DIR/gimbal_model.sdf"
GIMBAL_DEST="$PX4_GZ_MODELS/gimbal/model.sdf"
echo "  Copying gimbal_model.sdf -> $GIMBAL_DEST"
cp "$GIMBAL_SDF" "$GIMBAL_DEST" || { echo "ERROR: Failed to copy gimbal_model.sdf"; return 1; }

# Copy the remaining .sdf world files to PX4 worlds directory
for sdf_file in "$WORLDS_DIR"/*.sdf; do
    filename=$(basename "$sdf_file")
    if [ "$filename" != "gimbal_model.sdf" ]; then
        echo "  Copying $filename -> $PX4_GZ_WORLDS/"
        cp "$sdf_file" "$PX4_GZ_WORLDS/" || { echo "ERROR: Failed to copy $filename"; return 1; }
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment variables set in this shell:"
echo "  GZ_SIM_SYSTEM_PLUGIN_PATH=$GZ_SIM_SYSTEM_PLUGIN_PATH"
echo "  GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"
