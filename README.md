# Acoustic Hijacking Attack on Gimbal-Stabilized Tracking Systems

## Installation

Pull our docker image with all necessary environment

```bash
docker pull jiaruili0000/acoustijack:v1
```

Launch the container. 

```bash
sudo docker run -it --rm \
  --network host \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash
```

In the docker container

```bash
cd ~
git clone https://github.com/U1ltra/AcoustiJack/
cd AcoustiJack
bash install.sh
source env.sh
```

Test PX4 and Gazebo setup
```bash
cd ~/PX4-Autopilot
GZ_IP=127.0.0.1 PX4_GZ_WORLD=field PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="10.0,0.0,1.0,0.0,0.0,0.0" PX4_SIM_MODEL=x500_gimbal ./build/px4_sitl_default/bin/px4
```
If you see `INFO  [init] Gazebo world is ready`, the key PX4-Autopilot and Gazebo environments are ready.

## Reproduction

We run the artifact on  Intel i9-14900K CPU and RTX 4080 GPU. The project only uses 10% of one CPU core and less then 1GB of GPU memory. It also consume less then 3GB of memory. So we expect it to run on less capable machines.

Download the directory (models/) from [Google Drive](https://drive.google.com/drive/folders/1lIzxy_oZuaZ_CmRa1xSjX1fd9fFWlJrb?usp=sharing). Put the directory (models/) under AcoustiJack/onboard/. 

To validate claim 1, refer to docs/claim1.md and run the following command
```bash
bash bash/claim1.sh
```

To validate claim 2, refer to docs/claim2.md and run the following command
```bash
bash bash/claim2.sh
```

## Known Issues
- `bash install.sh` may fail when building the plugins. Simplely rerun can fit it.
- Rarely, when you test PX4 and Gazebo setup, it may stuck at `INFO  [init] Waiting for Gazebo world...` for a noticably long time. Our experience is that it eventually goes through after waiting (1-3 mins) or rerun. 
