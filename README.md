# Acoustic Hijacking Attack on Gimbal-Stabilized Tracking Systems

## Installation

Pull our docker image with all necessary environment

```bash
docker pull jiaruili0000/acoustijack:v1
```

Launch the container with or without GPU. GUI is also not required. 

```bash
# with NVIDIA GPU support
sudo docker run -it --rm \
  --network host \
  --device /dev/dri \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash

# without NVIDIA GPU support
sudo docker run -it --rm \
  --network host \
  --device /dev/dri \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash
```

In the docker container

```bash
cd ~
git clone https://github.com/U1ltra/AcoustiJack/
cd AcoustiJack
source setup_env.sh
```

## Reproduction

We run the artifact on  Intel i9-14900K CPU and RTX 4080 GPU. The project only uses 10% of one CPU core and less then 1GB of GPU memory. It also consume less then 3GB of memory. So we expect it to run on less capable machines.

Download the tracker model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1lIzxy_oZuaZ_CmRa1xSjX1fd9fFWlJrb?usp=sharing). Put the directory and the file under onboard/models/. 

To validate claim 1, refer to docs/claim1.md and run the following command
```bash
bash bash/claim1.sh
```

To validate claim 2, refer to docs/claim2.md and run the following command
```bash
bash bash/claim2.sh
```



