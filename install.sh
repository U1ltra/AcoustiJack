#!/bin/bash
# install.sh — One-time setup: install dependencies, compile plugins, stage models/worlds
# Usage: bash install.sh

set -e

REPO_ROOT="/root/AcoustiJack"
PLUGINS_DIR="$REPO_ROOT/plugins"
OBJECTS_DIR="$REPO_ROOT/config/objects"
WORLDS_DIR="$REPO_ROOT/config/worlds"

PX4_GZ_MODELS="/root/PX4-Autopilot/Tools/simulation/gz/models"
PX4_GZ_WORLDS="/root/PX4-Autopilot/Tools/simulation/gz/worlds"
GZ_MODELS_DIR="$HOME/.gz/models"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [0/3] Checking installations ==="

if find /usr -name "MAVSDKConfig.cmake" 2>/dev/null | grep -q .; then
    echo "  MAVSDK already installed, skipping."
else
    echo "  Installing MAVSDK..."
    MAVSDK_DEB="/tmp/libmavsdk-dev_1.4.16_ubuntu20.04_amd64.deb"
    wget -O "$MAVSDK_DEB" https://github.com/mavlink/MAVSDK/releases/download/v1.4.16/libmavsdk-dev_1.4.16_ubuntu20.04_amd64.deb \
        || { echo "ERROR: Failed to download MAVSDK"; exit 1; }
    sudo dpkg -i "$MAVSDK_DEB" \
        || { echo "ERROR: Failed to install MAVSDK"; rm -f "$MAVSDK_DEB"; exit 1; }
    rm -f "$MAVSDK_DEB"
    echo "  MAVSDK installed."
fi

if python3 -c "import tomli_w" &> /dev/null; then
    echo "  tomli_w already installed, skipping."
else
    echo "  Installing tomli_w..."
    pip install tomli_w \
        || { echo "ERROR: Failed to install tomli_w"; exit 1; }
    echo "  tomli_w installed."
fi

if python3 -c "import pysocialforce" &> /dev/null; then
    pip uninstall -y pysocialforce \
        || { echo "ERROR: Failed to uninstall existing pysocialforce"; exit 1; }
else
    echo "  pysocialforce not found, skipping uninstall."
fi
if [ ! -d "PySocialForce" ]; then
    git clone https://github.com/U1ltra/PySocialForce.git \
        || { echo "ERROR: Failed to clone PySocialForce"; exit 1; }
fi
cd PySocialForce
pip install -e . \
    || { echo "ERROR: Failed to install PySocialForce"; exit 1; }
cd ..

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
    ) || { echo "ERROR: Failed to build $plugin_name"; exit 1; }
    echo "  Done: $plugin_name"
done

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prepare Gazebo models
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [2/3] Preparing Gazebo models ==="

mkdir -p "$GZ_MODELS_DIR"

for model_dir in "$OBJECTS_DIR"/*/; do
    model_name=$(basename "$model_dir")
    echo "  Copying $model_name -> $GZ_MODELS_DIR/"
    cp -r "$model_dir" "$GZ_MODELS_DIR/" || { echo "ERROR: Failed to copy $model_name"; exit 1; }
done

# ─────────────────────────────────────────────────────────────────────────────
# 3. Prepare Gazebo worlds
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [3/3] Preparing Gazebo worlds ==="

GIMBAL_SDF="$WORLDS_DIR/gimbal_model.sdf"
GIMBAL_DEST="$PX4_GZ_MODELS/gimbal/model.sdf"
echo "  Copying gimbal_model.sdf -> $GIMBAL_DEST"
cp "$GIMBAL_SDF" "$GIMBAL_DEST" || { echo "ERROR: Failed to copy gimbal_model.sdf"; exit 1; }

for sdf_file in "$WORLDS_DIR"/*.sdf; do
    filename=$(basename "$sdf_file")
    if [ "$filename" != "gimbal_model.sdf" ]; then
        echo "  Copying $filename -> $PX4_GZ_WORLDS/"
        cp "$sdf_file" "$PX4_GZ_WORLDS/" || { echo "ERROR: Failed to copy $filename"; exit 1; }
    fi
done

echo ""
echo "=== Install complete. Run 'source env.sh' in each new terminal. ==="
