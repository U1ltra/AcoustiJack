# Banshee: Target Switch Attacks on Gimbal-Stabilized Visual Tracking Systems via Acoustic Injection

## About
Gimbal-stabilized visual tracking is widely used in autonomous systems such as UAVs. While prior work shows that acoustic signals can disturb gimbal internals, their impact on end-to-end tracking applications remains unclear. We present AcoustiJack, the first physically realizable attack that induces target switching in UAV visual tracking by exploiting acoustic vulnerabilities in gimbal-camera systems. AcoustiJack generates acoustic signals that cause directionally biased camera motion, breaking inter-frame target associations and steering the tracker toward an attacker-chosen object, leading to target switch or target loss. We achieve 93.6% success in simulation and 95.5% in real-world benchtop and in-flight experiments. Our results reveal a practical cross-domain vulnerability between acoustics and vision, highlighting the need for more robust system designs.

## Table of Contents

- [Installation](#installation)
- [Reproduction](#reproduction)
  - [Offline Profiling](#offline-profiling)
    - [Acoustic Injection](#acoustic-injection)
    - [Spectrum Analysis](#spectrum-analysis)
  - [Online Attack](#online-attack)
    - [Simulation Evaluation](#simulation-evaluation)
    - [Physical Evaluation](#physical-evaluation)
- [Known Issues](#known-issues)
- [Acknowledgments](#acknowledgments)

## Installation
Please refer to [docs/INSTALL.md](docs/INSTALL.md) for the full installation instructions.

## Reproduction

### Offline Profiling

The offline profiling pipeline characterizes the acoustic vulnerability of a target gimbal system. It runs on a Raspberry Pi connected to the injection hardware and produces IMU traces used to parameterize the motion model in simulation. The full hardware schematic and wiring guide are in [profiling/README.md](profiling/README.md).

#### Acoustic Injection

**Hardware.** The injection rig consists of:
- AD9833 programmable signal generator (SPI)
- DS3502 digital potentiometer (I²C) — controls signal amplitude
- TDA8932 mono amplifier — drives the PZT transducer
- BMI160 6-DoF IMU (I²C) — records gimbal response
- SG90 servo — positions the transducer relative to the gimbal
- RIGOL DG1022 function generator (optional, via USB/SCPI) — alternative signal source

Wire the components as described in [profiling/README.md](profiling/README.md).

**Running the injection controller.** On the Raspberry Pi, inside `profiling/`:

```bash
cd profiling
python controller.py --help          # see all options
python controller.py                 # run with defaults
```

The controller sweeps a configurable frequency range, drives the transducer at each frequency, and records simultaneous IMU data to CSV files. Collected traces are saved in the output directory (default: `./data/`).

#### Spectrum Analysis

After collecting IMU traces with the injection controller, run the spectrum analysis script to identify resonant frequencies and fit the gimbal motion model:

```bash
cd profiling
python spectrum_analysis.py <path/to/data/folder>
```

The script reads the CSV files produced by `controller.py`, applies bandpass filtering, and generates a PDF report with amplitude–frequency curves and model fits. The fitted parameters are then used to configure `attack/profiled_motion_model.py` for simulation.

---

### Online Attack

The online attack pipeline runs inside the Docker environment described in [docs/INSTALL.md](docs/INSTALL.md). It requires completing the [Prepare Environment](docs/INSTALL.md#prepare-environment) and [Prepare Models](docs/INSTALL.md#prepare-models) steps first.

Two claims from the paper are evaluated; each has a dedicated documentation file:

| Claim | Description | Details |
|-------|-------------|---------|
| Claim 1 | Attack effectiveness across different tracker types | [docs/claim1.md](docs/claim1.md) |
| Claim 2 | Attack generalizability across UAV gimbal systems | [docs/claim2.md](docs/claim2.md) |

#### Simulation Evaluation

**Claim 1 — Attack effectiveness** (~4 hours with GPU)

Evaluates the attack against DaSiamRPN (appearance-based) and UCMCTrack (motion-based) trackers on a HighEndDrone platform. Compare results to Figure 10 (*HighEndDrone* column) in the paper.

```bash
# Inside the Docker container, from /root/AcoustiJack
bash bash/claim1.sh
python utils/eval/offline_eval.py --claim 1
```

**Claim 2 — Attack generalizability** (~4 hours with GPU)

Evaluates the same attack on a MidEndDrone platform (different gimbal characteristics). Compare results to Figure 10 (*MidEndDrone* column) in the paper.

```bash
bash bash/claim2.sh
python utils/eval/offline_eval.py --claim 2
```

**Interpreting the output.** The evaluation script prints a summary like:

```
Tracker performance summary:
dasiam: Target switch rate: 0.90, Target loss rate: 0.10
ucmc: Target switch rate: 0.78, Target loss rate: 0.22
```

- **Target switch rate** — fraction of trials where the tracker successfully switched to the attacker-chosen target (the primary attack success metric, corresponding to Figure 10).
- **Target loss rate** (DoS) — fraction of trials where tracking was lost entirely without a successful target switch. This is a secondary effect; a high value alongside a low target switch rate indicates the attack disrupts tracking but does not reliably redirect it.

If a trial appears to be skipped (logged as `Failed to load simulation state`), it means that trial did not start correctly due to integration instability (see [Known Issues](#known-issues)). The evaluation script handles this gracefully and continues. The affected trial can be rerun with:

```bash
GZ_IP=127.0.0.1 python -u launch.py --only_run <trial_number> <other original args>
```

#### Physical Evaluation

The full physical attack implementation is not released for ethical and safety reasons. The offline profiling pipeline (`profiling/`) covers the hardware characterization step (Section 4.1 of the paper). The simulation experiments validate the attack logic end-to-end within a realistic Gazebo/PX4 environment.

---

## Known Issues
- **Intermittent build failures during installation**. Running `bash install.sh` may occasionally fail when building certain plugins. This appears to be non-deterministic (e.g., due to transient build or dependency issues). In our experience, simply rerunning the installation script resolves the problem.
- **Occasional delay during PX4–Gazebo initialization**. During startup, the system may appear to stall at `INFO [init] Waiting for Gazebo world...` for an extended period. In most cases, the process proceeds successfully after waiting (typically 1–3 minutes). If the delay persists, rerunning the setup usually resolves the issue.
- **GPU memory pre-allocation by TensorFlow**. By default, TensorFlow pre-allocates all available GPU memory. If you observe abnormally high GPU memory usage, set the following environment variable before running experiments:
  ```bash
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  ```
  This can also be passed directly when launching the Docker container with `-e TF_FORCE_GPU_ALLOW_GROWTH=true`.
- **System-level sensitivity and instability**. Our system involves tight coupling between multiple components, including the Gazebo simulator, PX4-Autopilot, ROS 2, and supporting Python scripts. As a result, execution can be sensitive to system configuration (e.g., compute resources, GPU drivers), which may manifest in several ways:
  - **Performance variability**. On machines with limited resources, the simulator may run slower than real time, introducing timing inconsistencies across components (e.g., delayed message passing). This can lead to variations in experimental results, though we do not expect it to affect the overall conclusions.
  - **Occasional integration instability**. The cross-component interaction may intermittently lead to failures in specific operations (e.g., the takeoff command), causing an individual experiment trial to fail. Based on our observations, this behavior arises from the integration of underlying open-source systems rather than the attack pipeline itself. In such cases, users can rerun the affected trial using the `--only_run` option in `launch.py`. We recommend running the project as a foreground job, since running as a background job (e.g. `nohup`) increases the chance of failures in specific operations (e.g., the takeoff command).
  - **Rare runtime crashes**. In a small number of environments, we observed occasional crashes during execution, which we attribute to GPU hardware or driver compatibility issues based on our diagnosis. To prevent data loss, we recommend mounting `/root/exp` in the Docker container as a shared directory with the host machine following the [docs/INSTALL.md](docs/INSTALL.md) instructions, so that logs are preserved in case of unexpected termination.

## Acknowledgments

We thank the following projects:

- [PySOT](https://github.com/STVIR/pysot)
- [kcf](https://github.com/vojirt/kcf)
- [gz-sim-docker](https://github.com/brean/gz-sim-docker)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [UCMCTrack](https://github.com/corfyi/UCMCTrack)