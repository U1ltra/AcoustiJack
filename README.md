# Acoustic Hijacking Attack on Gimbal-Stabilized Tracking Systems

## Installation

Pull our docker image with all necessary environment directly or build from the Dockerfile within the submodule `gz-px4-ros2-docker`. 

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
GZ_IP=127.0.0.1 HEADLESS=1 PX4_GZ_WORLD=field PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="10.0,0.0,1.0,0.0,0.0,0.0" PX4_SIM_MODEL=x500_gimbal ./build/px4_sitl_default/bin/px4
```
If you see `INFO  [init] Gazebo world is ready`, the key PX4-Autopilot and Gazebo environments are ready. Please then hit `ctrl-c` to terminate this PX4 and Gazebo setup test since the project will spawn them automatically. 

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
- **Intermittent build failures during installation**. Running `bash install.sh` may occasionally fail when building certain plugins. This appears to be non-deterministic (e.g., due to transient build or dependency issues). In our experience, simply rerunning the installation script resolves the problem.
- **Occasional delay during PX4–Gazebo initialization**. During startup, the system may appear to stall at `INFO [init] Waiting for Gazebo world...` for an extended period. In most cases, the process proceeds successfully after waiting (typically 1–3 minutes). If the delay persists, rerunning the setup usually resolves the issue.
- **System-level sensitivity and instability**. Our system involves tight coupling between multiple components, including the Gazebo simulator, PX4-Autopilot, ROS 2, and supporting Python scripts. As a result, execution can be sensitive to system configuration (e.g., compute resources, GPU drivers), which may manifest in several ways:
  - **Performance variability**. On machines with limited resources, the simulator may run slower than real time, introducing timing inconsistencies across components (e.g., delayed message passing). This can lead to variations in experimental results, though we do not expect it to affect the overall conclusions.
  - **Occasional integration instability**. The cross-component interaction may intermittently lead to failures in specific operations (e.g., the takeoff command), causing an individual experiment trial to fail. Based on our observations, this behavior arises from the integration of underlying open-source systems rather than the attack pipeline itself. In such cases, users can rerun the affected trial using the `--only_run` option in `launch.py`. We recommand running the project as foreground job, since we noticed that running as background job (e.g. `nohup`) increases the chance of failures in specific operations (e.g., the takeoff command). 
  - **Rare runtime crashes**. In a small number of environments, we observed occasional crashes during execution, which we attribute to GPU hardware or driver compatibility issues based on our diagnosis. To prevent data loss, we recommend mounting /root/exp in the Docker container as a shared directory with the host machine following the `README.md` instructions, so that logs are preserved in case of unexpected termination.
