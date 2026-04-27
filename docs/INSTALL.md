
# Installation

## Table of Contents
- [Requirements](#requirements)
  - [Option 1: With GPU (Recommended)](#with-gpu)
  - [Option 2: CPU-only](#cpu-only)
- [Prepare Environment](#prepare-environment)
- [Prepare Models](#prepare-models)
- [Cloud Setup (EC2)](#cloud-setup-ec2)
- [Tested Platforms](#tested-platforms)

## Requirements

We recommend running the artifact on a machine with a CUDA-capable GPU. A display is not required.

### With GPU

Install [Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Pull our pre-built Docker image (all dependencies included), or build it yourself from the Dockerfile in the `gz-px4-ros2-docker` submodule:

```bash
docker pull jiaruili0000/acoustijack:v1
```

Create the host mount directory to store experiment results before running the container:

```bash
mkdir -p ~/Documents/exp/docker
```

Launch the container:

```bash
sudo docker run -it --rm \
  --network host \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash
```

**Docker flag notes:**
| Flag | Purpose |
|------|---------|
| `--network host` | Required for ROS 2 and PX4 MAVLINK communication across container processes |
| `--gpus all` | Passes GPU(s) into the container for simulation rendering and tracker inference |
| `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` | Expose all GPU devices and capabilities (graphics, compute, utility) |
| `TF_FORCE_GPU_ALLOW_GROWTH=true` | Prevents TensorFlow from pre-allocating all GPU memory |
| `-v ~/Documents/exp/docker:/root/exp` | Mounts a host directory at `/root/exp` inside the container, preserving experiment logs if the container exits unexpectedly |



### CPU-only

> **Important**: CPU-only execution is less tested and may not produce valid end-to-end outputs in some environments. If you encounter issues on CPU-only hardware, we recommend using a cloud GPU instance (see [Cloud Setup (EC2)](#cloud-setup-ec2)).

Launch the container without GPU support:

```bash
sudo docker run -it --rm \
  --network host \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash
```

The simulation will run, but expect significantly slower execution and possible timing instabilities due to the real-time demands of the Gazebo/PX4 integration.

## Prepare Environment

Run the following inside the Docker container:

```bash
cd ~
git clone https://github.com/U1ltra/Banshee/
cd Banshee
bash install.sh
source env.sh
```

`install.sh` installs Python dependencies, compiles the Gazebo plugins, and stages world and model files into the PX4 and Gazebo directories. `source env.sh` must be run in every new terminal session before launching experiments.

**Verify the PX4 and Gazebo setup:**

```bash
cd ~/PX4-Autopilot
GZ_IP=127.0.0.1 HEADLESS=1 PX4_GZ_WORLD=field PX4_SYS_AUTOSTART=4002 \
  PX4_GZ_MODEL_POSE="10.0,0.0,1.0,0.0,0.0,0.0" PX4_SIM_MODEL=x500_gimbal \
  ./build/px4_sitl_default/bin/px4
```

If you see `INFO  [init] Gazebo world is ready`, the environment is correctly configured. Press `Ctrl-C` to exit — the experiment scripts will launch PX4 and Gazebo automatically.

## Prepare Models

Model checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1lIzxy_oZuaZ_CmRa1xSjX1fd9fFWlJrb?usp=sharing)

**Making checkpoints available inside the container.** Place the downloaded `models/` directory into the host mount point (`~/Documents/exp/docker`) on the machine that runs Docker. Because that directory is bind-mounted to `/root/exp` inside the container (see the `-v` flag in the `docker run` command above), the files become visible inside the container without any copy step:

```bash
# On the Docker host — move the downloaded models/ into the mount point
mv /local/path/to/models ~/Documents/exp/docker/

# Then, inside the Docker container, copy them into the repo
cp -r /root/exp/models /root/Banshee/onboard/
```

If the Docker host is a remote machine (e.g. an EC2 instance), see [Cloud Setup (EC2)](#cloud-setup-ec2) for how to upload the checkpoints with `scp` first.

The checkpoints must be organized as follows inside the container:

```
onboard/
├── models/
│   ├── siamrpn_alex_dwxcorr/
│   │   ├── model.pth
│   │   └── config.yaml
│   └── SiamRPNBIG.model
```

## Cloud Setup (EC2)

We provide instructions for reproducing the results on an Amazon EC2 GPU instance (tested on `g4dn.2xlarge`).

**Step 1 — Connect to the instance (from your local machine):**

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

**Step 2 — Prepare the host directory (on the instance):**

```bash
mkdir -p ~/Documents/exp/docker
```

**Step 3 — Upload model checkpoints (from your local machine):**

```bash
# First download the models/ directory from Google Drive to your local machine
scp -i your-key.pem -r /local/path/to/models ubuntu@<instance-ip>:~/Documents/exp/docker/
```

**Step 4 — Launch the container and run experiments (on the instance):**

```bash
sudo docker run -it \
  --network host \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v ~/Documents/exp/docker:/root/exp \
  jiaruili0000/acoustijack:v1 /bin/bash
```

Inside the container:

```bash
cd ~
git clone https://github.com/U1ltra/Banshee/
cd Banshee
cp -r /root/exp/models onboard/
bash install.sh
source env.sh

bash bash/claim1.sh
bash bash/claim2.sh

python utils/eval/offline_eval.py --claim 1
python utils/eval/offline_eval.py --claim 2
```

## Tested Platforms

We ran all experiments on:
- Intel i9-14900K CPU, RTX 4080 GPU, Ubuntu 24.04

We also verified that the code is functional on:
- Amazon AWS EC2 `g4dn.2xlarge`
- Intel i9-13900KF CPU, 16 GB RAM, Ubuntu 24.04 (GPU instance)
