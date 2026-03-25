import os
import sys
import time
import argparse
import subprocess
import numpy as np

from utils.exp_utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_exps", type=int, default=18, help="Number of experiments to run"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="Experiment name suffix",
    )
    parser.add_argument(
        "--world_file",
        type=str,
        default="default",
        help="World file to use for the simulation",
    )
    parser.add_argument(
        "--tracker_type",
        type=str,
        default="SORT",
        choices=["SORT", "SiamRPN", "KCF", "DaSiamRPN", "UCMCTrack"],
        help="Type of tracker to use. SORT, SiamRPN, KCF, DaSiamRPN, UCMCTrack",
    )
    parser.add_argument(
        "--surrogate_model",
        type=str,
        default="SORT",
        choices=["SORT", "SiamRPN", "KCF", "DaSiamRPN", "UCMCTrack"],
        help="Type of surrogate model to use. SORT, SiamRPN, KCF, DaSiamRPN, UCMCTrack",
    )
    parser.add_argument(
        "--atk_cycle",
        type=float,
        default=4.0,
        help="Attack cycle frequency in hertz."
    )
    parser.add_argument(
        "--gim_max_speed",
        type=float,
        nargs=3,
        default=[0.1, 2.0, 2.0],
        help="Gimbal maximum speed (injected by the acoustic signal) in pitch, roll, yaw (rad/s)",
    )
    parser.add_argument(
        "--atk_duration",
        type=int,
        default=50,
        help=(
            "Duration of the attack in seconds."
            "This is the duration of the attack trajectory."
        ),
    )
    parser.add_argument(
        "--atker_distance_range",
        type=float,
        nargs=2,
        default=None,
        help="Min and max distance between the attacker and the victim in meters",
    )
    parser.add_argument(
        "--atker_angle",
        type=float,
        default=90.0,
        help="Angle between attacker and victim in degrees",
    )
    parser.add_argument(
        "--uav_height_range",
        type=float,
        nargs=2,
        default=None,
        help="Min and max height of the UAV in meters",
    )
    parser.add_argument(
        "--SF_toml",
        type=str,
        default="./config/socialForce/social_force.toml",
        help="Path to the social force motion model parameters toml file",
    )
    parser.add_argument(
        "--attack_interval_method",
        type=str,
        default="simdt",
        choices=["simdt", "1overfps"],
        help="Method to determine the attack interval. 'simdt': use simulation time delta; '1overfps': use 1/fps",
    )
    parser.add_argument(
        "--same_appearance",
        action='store_true',
        help="Use the same appearance for the attacker and victim UAVs",
    )
    parser.add_argument(
        "--realtime_factor",
        type=float,
        default=1.0,
        help="Gazebo real-time factor",
    )
    parser.add_argument(
        "--camera_fps",
        type=int,
        default=30,
        help="Camera frames per second",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=400,
        help="Timeout for the experiment in seconds",
    )
    parser.add_argument(
        "--recover_from",
        type=int,
        default=0,
        help="Recover from the specified experiment number",
    )
    parser.add_argument(
        "--run_until",
        type=int,
        default=None,
        help="Run until the specified experiment number",
    )
    parser.add_argument(
        "--only_run",
        type=int,
        nargs='+',
        default=None,
        help="Only run the specified experiment numbers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--PX4_PATH",
        type=str,
        default="~/PX4-Autopilot",
        help="Path to the PX4 Firmware directory",
    )
    parser.add_argument(
        "--vir_env",
        type=str,
        default="/acoustiJack",
        help="Path to the Python virtual environment",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default="~/AcoustiJack",
        help="Path to the code directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/exp",
        help="Directory to save the experiment results",
    )
    
    return parser.parse_args()


def main():
    args = parse_args() # configs shared by all experiments

    # Expand the home directory path
    args.save_dir = os.path.expanduser(args.save_dir)
    args.PX4_PATH = os.path.expanduser(args.PX4_PATH)
    args.vir_env = os.path.expanduser(args.vir_env)
    args.code_dir = os.path.expanduser(args.code_dir)

    curr_time = time.strftime("%Y%m%d-%H%M%S")
    args.save_dir = os.path.join(args.save_dir, f"{curr_time}-{args.exp_name}")
    px4_world_file = os.path.join(
        args.PX4_PATH, f"Tools/simulation/gz/worlds/{args.world_file}.sdf"
    )
    px4_gimbal_file = os.path.join(
        args.PX4_PATH, "Tools/simulation/gz/models/gimbal/model.sdf"
    )

    # Create the save directory if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "log"), exist_ok=True)

    # generate random coordinates
    np.random.seed(args.seed)
    
    scenario_configs = experiment_configs(args)
    trial_configs = generate_trial_configs(args, scenario_configs)

    log_experiment_params(args, trial_configs)
    if args.tracker_type in ["SORT"]:
        toggle_cuda_line(
            os.path.join(args.code_dir, "attack", "gimbal_atk.py"),
            enable_cuda=False,
        )
    elif args.tracker_type in ["SiamRPN", "KCF", "DaSiamRPN", "UCMCTrack"]:
        toggle_cuda_line(
            os.path.join(args.code_dir, "attack", "gimbal_atk.py"),
            enable_cuda=True,
        )
    if args.surrogate_model in ["SiamRPN"]:
        toggle_cuda_line(
            os.path.join(args.code_dir, "attack", "gimbal_atk.py"),
            enable_cuda=True,
        )
    select_motion_model_setting(
        os.path.join(args.code_dir, "attack", "profiled_motion_model.py"),
        args.gim_max_speed,
    )

    print("Starting the experiments... Time: ", time.strftime("%Y%m%d-%H%M%S"))
    for i in range(args.num_exps):
        if i < args.recover_from:
            print(f"Skipping experiment {i+1}.")
            continue
        if args.only_run is not None and i not in args.only_run:
            print(f"Skipping experiment {i+1} as it is not in the only_run list.")
            continue
        if args.run_until is not None and i >= args.run_until:
            print(f"Stopping experiments at {i+1} as per run_until parameter.")
            break

        start_time = time.time()
        print(f"***Running experiment {i+1}...***")
        update_world_sdf(args, trial_configs, i, px4_world_file)
        update_gimbal_sdf(args, px4_gimbal_file)
        modify_px4_params(
            os.path.join(args.code_dir, "bash", "run_sim_suite.sh"),
            world=args.world_file,
            uav_pose=trial_configs["UAV_poses"][i],
        )
        print(
            f"Victim | pose {trial_configs['victim_poses'][i]}, velocity {trial_configs['victim_velocity'][i]}, "
            f"model {trial_configs['gz_models'][i][0]}"
        )
        print(
            f"Attacker | distance {trial_configs['attacker_distances'][i]}, "
            f"model {trial_configs['gz_models'][i][1]}"
        )
        print(f"UAV | pose {trial_configs['UAV_poses'][i]}, height {trial_configs['heights'][i]}")

        for retry in range(2):
            # Run the simulation
            exp_dir = os.path.join(args.save_dir, f"trial_{i+1:03d}")
            # delete the exp_dir if it exists
            if os.path.exists(exp_dir):
                print(f"Experiment directory {exp_dir} already exists, deleting it...")
                subprocess.run(["rm", "-rf", exp_dir])
            os.makedirs(exp_dir, exist_ok=True)
            args.exp_dir = exp_dir
            # cp the world file exp directory
            subprocess.run(["cp", px4_world_file, exp_dir])
            subprocess.run(["cp", px4_gimbal_file, exp_dir])

            # Launch the simulation
            px4_process, atk_process = launch_scripts(args, trial_configs, i)

            term_reason, return_code = monitor_proc(atk_process, timeout_seconds=args.timeout)

            if term_reason == TerminationReason.TIMEOUT:
                print("Attack timed out!")
            elif term_reason == TerminationReason.PROCESS_ENDED:
                print("Attack exited with code", return_code)

            print("Attack completed!")
            print("Check the results at ", exp_dir)
            
            # When atk_process is done, stop the simulation
            # atk_process.wait()
            px4_process.terminate()

            # cleanup
            print("Cleaning up...")
            clean_up(
                keywords=[
                    "gz",
                    "px4_sitl_default",
                    "object-follower.py",
                    "gimbal-control.py",
                    "motion_atk.py",
                    "gimbal_atk.py",
                    "image_saver.py",
                    # "mavsdk_server",
                ],
                tmux_session="simulation",
            )
            
            # check if box_video.mp4 exists
            if os.path.exists(os.path.join(exp_dir, "box_video.mp4")):
                break
            else:
                print("Box video not found, retrying...")
                if retry == 1:
                    print("Experiment failed after 2 tries.")

        # for dir_name in ["images", "raw_images"]:
        #     for img_name in ["image"]:
        #         create_video_from_images(
        #             os.path.join(exp_dir, dir_name, f"{img_name}_*.jpg"),
        #             os.path.join(exp_dir, f"{dir_name}.mp4"),
        #             fps=5,
        #         )

        print(f"Simulation completed! {time.time()-start_time:.2f} seconds\n")

    print("All experiments completed! Time: ", time.strftime("%Y%m%d-%H%M%S"))


if __name__ == "__main__":
    main()
