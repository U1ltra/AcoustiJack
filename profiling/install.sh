#!/bin/bash
set -euo pipefail

sudo apt update
sudo apt install -y \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    python3-pip \
    python3-venv \
    python3-usb \
    libusb-1.0-0 \
    python3-serial \
    cmake

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip3 install --upgrade \
    pip \
    setuptools \
    wheel \
    pyaudio \
    spidev \
    gpiozero \
    numpy \
    smbus2 \
    adafruit-blinka \
    adafruit-circuitpython-ds3502 \
    pyvisa \
    pyvisa-py \
    mavsdk \
    pymavlink \
    -e ./AD9833_spi -e ./BMI160_i2c -e ./RIGOL_scpi

ALIAS_CMD="alias labenv='cd $(pwd) && . .venv/bin/activate'"
echo "$ALIAS_CMD" >> ~/.bashrc

python3 - <<'EOF'
try:
    import pyaudio, spidev, gpiozero, smbus2, numpy, board, adafruit_ds3502, wave, pyvisa, mavsdk, pymavlink
    from AD9833_spi import AD9833
    from BMI160_i2c import BMI160
    from RIGOL_scpi import DG1022
    print("Installation verified")
except Exception as e:
    print("Verification failed with error ", e)
EOF
