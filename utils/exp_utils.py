import re
import os
import cv2
import glob
import time
import pickle
import tomllib
import tomli_w
import subprocess
import numpy as np
import xml.etree.ElementTree as ET

from enum import Enum
from itertools import combinations, product
from datetime import datetime, timedelta

class SDFManager:
    def __init__(self, file_path, root_tag="world"):
        self.file_path = os.path.expanduser(file_path)
        self.tree = ET.parse(self.file_path)
        self.root = self.tree.getroot()
        self.world = self.root.find(root_tag)

    def get_element(self, xpath):
        """Get any element using xpath-style string."""
        return self.world.find(xpath)

    def get_element_text(self, xpath):
        """Get text value of any element using xpath-style string."""
        element = self.get_element(xpath)
        if element is not None:
            return element.text
        return None

    def set_element_text(self, xpath, value):
        """Set text value of any element using xpath-style string."""
        element = self.get_element(xpath)
        if element is not None:
            element.text = str(value)
            return True
        return False

    def save(self):
        """Save changes back to file."""
        self.tree.write(self.file_path, encoding="UTF-8", xml_declaration=True)


class TerminationReason(Enum):
    TIMEOUT = "timeout"
    PROCESS_ENDED = "process_ended"


def generate_random_velocity(n, min_speed=0.5, max_speed=2.0):
    cat = {
        "slow": (0.5, 1.0),
        "medium": (1.0, 1.5),
        "fast": (1.5, 2.0),
    }

    # Generate random angles between 0 and 2π
    angles = np.random.uniform(0, 2 * np.pi, n)
    # Generate random speeds between min_speed and max_speed
    speeds = np.random.uniform(min_speed, max_speed, n)
    # Convert to x,y components
    velocity_xy = np.column_stack([speeds * np.cos(angles), speeds * np.sin(angles)])
    # x always negative
    # velocity_xy[:, 0] = -np.abs(velocity_xy[:, 0])

    speeds_cat = np.zeros(n)
    speeds_cat[(speeds >= cat["slow"][0]) & (speeds < cat["slow"][1])] = 1
    speeds_cat[(speeds >= cat["medium"][0]) & (speeds < cat["medium"][1])] = 2
    speeds_cat[(speeds >= cat["fast"][0]) & (speeds < cat["fast"][1])] = 3

    # calculate percentage of each category
    slow = speeds_cat[speeds_cat == 1].size
    medium = speeds_cat[speeds_cat == 2].size
    fast = speeds_cat[speeds_cat == 3].size
    stats = {
        "slow": slow / n,
        "medium": medium / n,
        "fast": fast / n,
    }
    return velocity_xy, speeds_cat, stats


def generate_random_num(n, min_dist=1.0, max_dist=5.0):
    return np.random.uniform(min_dist, max_dist, n)


def log_experiment_params(args, trial_configs):
    """
    Log the experiment parameters to a file.
    """

    # Create the save directory if it does not exist
    with open(os.path.join(args.save_dir, "log", "args.txt"), "w") as f:
        f.write(str(args))

    # pickle dump the trial_configs
    trial_configs_path = os.path.join(args.save_dir, "log", "trial_configs.pkl")
    with open(trial_configs_path, "wb") as f:
        pickle.dump(trial_configs, f)


def update_world_sdf(
    args,
    trial_configs=None,
    trial_index=0,
    file_path="~/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf",
):
    """
    Update the poses of the actors in the given SDF file. The default.sdf file in
    PX4-Autopilot/Tools/simulation/gz/worlds/ directory contains the world description
    for Gazebo simulation. The function updates the pose of the actor_walking and
    actor_following elements.
    """

    actor_walking_pose = trial_configs["victim_poses"][trial_index]
    actor_following_pose = trial_configs["attacker_poses"][trial_index]
    actor_extra_pose = None
    if "extra_poses" in trial_configs:
        actor_extra_pose = trial_configs["extra_poses"][trial_index]
    actor_walking_model = trial_configs["gz_models"][trial_index][2]
    actor_following_model = trial_configs["gz_models"][trial_index][3]
    velocity_xy = trial_configs["victim_velocity"][trial_index]
    gz_realtime_factor = args.realtime_factor  # Real-time factor from command line arguments

    # Expand the home directory path
    file_path = os.path.expanduser(file_path)

    # Create an SDFManager object
    sdf_manager = SDFManager(file_path, root_tag="world")
    
    sdf_manager.set_element_text(
        "./include[@name='actor_walking']/uri",
        trial_configs["gz_models"][trial_index][0],
    )
    sdf_manager.set_element_text(
        "./include[@name='actor_following']/uri",
        trial_configs["gz_models"][trial_index][1],
    )

    x, y, z, roll, pitch, yaw = actor_walking_pose
    sdf_manager.set_element_text(
        "./include[@name='actor_walking']/pose",
        f"{x} {y} {z} {roll} {pitch} {yaw}",
    )
    x, y, z, roll, pitch, yaw = actor_following_pose
    sdf_manager.set_element_text(
        "./include[@name='actor_following']/pose",
        f"{x} {y} {z} {roll} {pitch} {yaw}",
    )
    if actor_extra_pose is not None:
        x, y, z, roll, pitch, yaw = actor_extra_pose
        sdf_manager.set_element_text(
            "./include[@name='actor_extra']/pose",
            f"{x} {y} {z} {roll} {pitch} {yaw}",
        )

    # Update actor_walking velocity
    sdf_manager.set_element_text(
        "./include[@name='actor_walking']/plugin[@filename='libActorControllerPlugin.so']/velocity_x",
        velocity_xy[0],
    )
    sdf_manager.set_element_text(
        "./include[@name='actor_walking']/plugin[@filename='libActorControllerPlugin.so']/velocity_y",
        velocity_xy[1],
    )
    sdf_manager.set_element_text(
        "./physics[@type='ode']/real_time_factor",
        gz_realtime_factor,
    )

    # Save the changes back to file
    sdf_manager.save()
    

def update_gimbal_sdf(
    args,
    file_path="~/PX4-Autopilot/Tools/simulation/gz/models/gimbal/model.sdf",
):
    """ Update the gimbal model SDF file with the specified parameters. """
    print("path to gimbal model SDF file:", file_path)
    # Expand the home directory path
    file_path = os.path.expanduser(file_path)
    # Create an SDFManager object
    sdf_manager = SDFManager(file_path, root_tag="model")
    # Update the gimbal model SDF file with the specified parameters
    sdf_manager.set_element_text(
        "./link[@name='camera_link']/sensor[@name='camera']/update_rate",
        args.camera_fps,
    )
    sdf_manager.set_element_text(
        "./link[@name='camera_link']/sensor[@name='visible_2d']/update_rate",
        args.camera_fps,
    )
    sdf_manager.set_element_text(
        "./link[@name='camera_link']/sensor[@name='3d']/update_rate",
        args.camera_fps,
    )
    
    sdf_manager.save()


def modify_px4_params(file_path, world, uav_pose):
    """
    Modify PX4_GZ_WORLD and PX4_GZ_MODEL_POSE in the bash simulation file.
    
    Args:
        file_path (str): Path to the bash file
        world (str, optional): New world value (e.g., 'default')
        model_pose (str, optional): New model pose (e.g., '308.96,-118.0,2.90, 0, 0, -0.7')
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the PX4 command line
    pattern = r"(tmux send-keys -t \$SESSION_NAME:3 'HEADLESS=1 )(PX4_GZ_WORLD=\w+ )?(PX4_SYS_AUTOSTART=4002 )(PX4_GZ_MODEL_POSE=\"[^\"]+\" )?(PX4_SIM_MODEL=\w+ \.\/build\/px4_sitl_default\/bin\/px4')"
    
    def replace_params(match):
        prefix = match.group(1)
        autostart = match.group(3)
        suffix = match.group(5)
        
        world_part = f"PX4_GZ_WORLD={world} "
        x, y, z, roll, pitch, yaw = uav_pose
        pose_part = f'PX4_GZ_MODEL_POSE="{x},{y},{z},{roll},{pitch},{yaw}" '

        return f"{prefix}{world_part}{autostart}{pose_part}{suffix}"
    
    content = re.sub(pattern, replace_params, content)
    
    with open(file_path, 'w') as f:
        f.write(content)


def toggle_cuda_line(file_path="attack/gimbal_atk.py", enable_cuda=True):
    """Toggle CUDA_VISIBLE_DEVICES line in the specified file.
    
    Args:
        file_path: Path to the Python file
        enable_cuda: True to comment out (enable CUDA), False to uncomment (disable CUDA)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'
    
    for i, line in enumerate(lines):
        if target in line:
            if enable_cuda and not line.strip().startswith('#'):
                lines[i] = f"# {line}"  # Comment out
            elif not enable_cuda and line.strip().startswith('#'):
                lines[i] = line.lstrip('# ')  # Uncomment
            break
    
    with open(file_path, 'w') as f:
        f.writelines(lines)


def select_motion_model_setting(file_path, gim_max_speed):
    """Select the profiled motion model setting based on gim_max_speed.

    gim_max_speed [0.1, 2.0, 2.0] -> setting 1 (three resonant freqs)
    gim_max_speed [1.0, 4.0, 1.0] -> setting 2 (single resonant freq at 30 kHz)
    """
    with open(file_path, 'r') as f:
        content = f.read()

    if list(gim_max_speed) == [0.1, 0.1, 2.0]:
        new_init_body = (
            "        self.resonant_freqs = [23231.0]  # Hz\n"
            "        self.aliased_frequencies = [2.0332]  # Hz\n"
            "        self.velocity_amplitudes = [  # roll, pitch, yaw for each resonant frequency\n"
            "        [math.radians(2.0032), math.radians(15.6742), math.radians(134.5029)],\n"
            "        ]\n"
            "        # self.resonant_freqs = [30000.0]  # Hz\n"
            "        # self.aliased_frequencies = [4.0]  # Hz\n"
            "        # self.velocity_amplitudes = [  # roll, pitch, yaw for each resonant frequency\n"
            "        #     [math.radians(2.0032), math.radians(134.5029), math.radians(15.6742)],\n"
            "        # ]\n"
        )
        print("Selected motion model setting 1 (gim_max_speed=[0.1, 2.0, 2.0])")
    elif list(gim_max_speed) == [1.0, 4.0, 1.0]:
        new_init_body = (
            "        # self.resonant_freqs = [23231.0]  # Hz\n"
            "        # self.aliased_frequencies = [2.0332]  # Hz\n"
            "        # self.velocity_amplitudes = [  # roll, pitch, yaw for each resonant frequency\n"
            "        #     [math.radians(2.0032), math.radians(15.6742), math.radians(134.5029)],\n"
            "        # ]\n"
            "\n"
            "        self.resonant_freqs = [30000.0]  # Hz\n"
            "        self.aliased_frequencies = [4.0]  # Hz\n"
            "        self.velocity_amplitudes = [  # roll, pitch, yaw for each resonant frequency\n"
            "            [math.radians(2.0032), math.radians(134.5029), math.radians(15.6742)],\n"
            "        ]\n"
        )
        print("Selected motion model setting 2 (gim_max_speed=[1.0, 4.0, 1.0])")
    else:
        raise ValueError(
            f"Unknown gim_max_speed {gim_max_speed}. "
            "Expected [0.1, 2.0, 2.0] (setting 1) or [1.0, 4.0, 1.0] (setting 2)."
        )

    pattern = r'(    def __init__\(self\):\n).*?(?=        self\.resonant_freqs = self\.resonant_freqs\[)'
    new_content = re.sub(pattern, r'\g<1>' + new_init_body, content, flags=re.DOTALL)

    if new_content == content:
        print("Warning: profiled_motion_model.py settings block unchanged — pattern may not have matched.")

    with open(file_path, 'w') as f:
        f.write(new_content)


def experiment_configs(args):
    """ Generate experiment configurations that are fixed for the given scenario (world file). """
    scenario = args.world_file
    tracker_type = args.tracker_type
    configs = {}
    
    if scenario == "urban":
        configs["victim_pose_ranges"] = [(2, -25.5, 0.5, 0.0, 0, -np.pi), (2, -25.5, 0.5, 0, 0, np.pi)] # victim: x, y, z, roll, pitch, yaw ranges
        configs["attacker_pose_ranges"] = [(6, -24, 0.5, 0, 0, -np.pi), (6, -27, 0.5, 0, 0, np.pi)]
        configs["victim_velocity_range"] = (0.5, 1.5)
        configs["UAV_pose"] = (-4, -25, 1, 0.0, 0.0, 0.0)  # x, y, z, groll, gpitch, gyaw
        configs["UAV_height_range"] = (10, 10)  # following height range of the UAV
        # configs["attacker_distance_range"] = (1.5, 1.5)  # attacker following distance range
        configs["gz_models"] = {
            "pedestrian1": "model://casual_female",
            "pedestrian2": "model://casual_female_green",
            "pedestrian3": "model://casual_female_red",
            # "pedestrian2": "model://actor_stand",
            # "pedestrian3": "model://male_visitor",
        }
    elif scenario == "field":
        configs["victim_pose_ranges"] = [(16, 0.0, 0.5, 0, 0, -np.pi), (16, 0, 0.5, 0, 0, np.pi)]
        configs["attacker_pose_ranges"] = [(19, -2, 0.5, 0, 0, -np.pi), (19, 2, 0.5, 0, 0, np.pi)]
        configs["victim_velocity_range"] = (0.5, 1.5)
        configs["UAV_pose"] = (10, 0, 1, 0.0, 0.0, 0.0)
        configs["UAV_height_range"] = (10, 10)
        # configs["attacker_distance_range"] = (1.5, 1.5)
        configs["gz_models"] = {
            "pedestrian1": "model://casual_female",
            "pedestrian2": "model://casual_female_green",
            "pedestrian3": "model://casual_female_red",
            # "pedestrian2": "model://actor_stand",
            # "pedestrian3": "model://male_visitor",
        }
    elif scenario == "warehouse":
        configs["victim_pose_ranges"] = [(1, 0, 0.3, 0, 0, -np.pi), (1, 0, 0.3, 0, 0, np.pi)]
        configs["attacker_pose_ranges"] = [(2, 0, 0.3, 0, 0, -np.pi), (4, 2, 0.3, 0, 0, np.pi)]
        configs["victim_velocity_range"] = (0.0, 0.5)
        configs["UAV_pose"] = (-4, 0, 0.5, 0.0, -0.85, 0.0)
        configs["UAV_height_range"] = (10, 10)
        # configs["attacker_distance_range"] = (1.5, 1.5)
        configs["gz_models"] = {
            "pedestrian1": "model://casual_female",
            "pedestrian2": "model://casual_female_green",
            "pedestrian3": "model://casual_female_red",
            # "pedestrian2": "model://actor_stand",
            # "pedestrian3": "model://male_visitor",
        }
    elif scenario == "raceway":
        configs["victim_poses"] = [
            (431.0, -278.0, 0.0, 0.0, 0.0, -0.0),
            (230.0, 11.0, 3.0, 0.0, 0.0, -0.0),
        ]
        configs["attacker_poses"] = [
            (431.0, -273.0, 0.0, 0.0, 0.0, -0.0),
            (230.0, 6.0, 3.0, 0.0, 0.0, -0.0),
        ]
        configs["UAV_poses"] = [
            (416.0, -278.0, 3.0, 0.0, 0.0, 0.0),
            (216.0, 7.0, 4.0, 0.0, 0.0, 0.0),
        ]
        configs["extra_poses"] = [
            (431.0, -260.0, 0.0, 0.0, 0.0, -0.0),
            (230.0, 29.0, 3.0, 0.0, 0.0, -0.0),
        ]
        configs["victim_velocity_range"] = (2.24, 6.71) # m/s. equals 5 to 15 mph
        configs["UAV_height_range"] = (20.0, 20.0)
        configs["attacker_distance_range"] = (5.0, 5.0)
        configs["gz_models"] = {
            "car1": "model://hatchback",
            "car2": "model://hatchback_red",
            "car3": "model://hatchback_blue",
        }

    elif scenario == "urban_car":
        configs["victim_pose_ranges"] = [(-59.0, -52.0, 0.5, 0.0, 0, 0.0), (-59.0, -52.0, 0.5, 0.0, 0, 0.0)] # victim: x, y, z, roll, pitch, yaw ranges
        configs["attacker_pose_ranges"] = [(-59.0, -45.0, 0.5, 0.0, 0, 0.0), (-59.0, -45.0, 0.5, 0.0, 0, 0.0)]
        configs["victim_velocity_range"] = (2.24, 6.71)
        configs["UAV_pose"] = (-75.0, -52.0, 0.0, 0.0, 0.0, 0.0)  # x, y, z, groll, gpitch, gyaw
        configs["UAV_height_range"] = (20.0, 20.0)  # following height range of the UAV
        configs["attacker_distance_range"] = (5.0, 5.0)  # attacker following distance range
        configs["gz_models"] = {
            "car1": "model://hatchback",
            "car2": "model://hatchback_red",
            "car3": "model://hatchback_blue",
        }

    else:
        raise ValueError(f"Invalid world file name: {scenario}. Must be 'urban', 'field', 'warehouse', or 'raceway'.")

    if args.uav_height_range is not None:
        configs["UAV_height_range"] = args.uav_height_range
    if args.atker_distance_range is not None:
        configs["attacker_distance_range"] = args.atker_distance_range
        print("Using custom attacker distance range:", configs["attacker_distance_range"])
    else:
        if scenario == "raceway":
            pass
        elif scenario in ["urban", "field", "warehouse"]:
            if tracker_type in ["SORT", "UCMCTrack"]:
                configs["attacker_distance_range"] = (1.5, 1.5)
            elif tracker_type in ["SiamRPN", "DaSiamRPN", "KCF"]:
                configs["attacker_distance_range"] = (1.5, 1.5)

    if args.same_appearance:
        if scenario in ["urban", "field", "warehouse"]:
            first_model = list(configs["gz_models"].values())[2]
            configs["gz_models"] = {
                "pedestrian1": first_model,
                "pedestrian2": first_model,
                "pedestrian3": first_model,
            }
        elif scenario == "raceway":
            first_model = list(configs["gz_models"].values())[2]
            configs["gz_models"] = {
                "car1": first_model,
                "car2": first_model,
            }


    return configs


def generate_trial_configs(args, scenario_configs):
    """ Generate trial configurations based on the scenario configurations. """
    num_exps = args.num_exps
    configs = {}

    if args.world_file in ["urban", "field", "warehouse"]:
        victim_poses = np.random.uniform(
            scenario_configs["victim_pose_ranges"][0],
            scenario_configs["victim_pose_ranges"][1],
            (num_exps, 6), # x, y, z, roll, pitch, yaw
        )
        attacker_poses = np.random.uniform(
            scenario_configs["attacker_pose_ranges"][0],
            scenario_configs["attacker_pose_ranges"][1],
            (num_exps, 6),
        )
        uav_poses = np.tile(
            scenario_configs["UAV_pose"],
            (num_exps, 1)
        )
        vic_velocity, _, _ = generate_random_velocity(
            num_exps,
            min_speed=scenario_configs["victim_velocity_range"][0],
            max_speed=scenario_configs["victim_velocity_range"][1],
        )
        heights = np.random.uniform(
            scenario_configs["UAV_height_range"][0],
            scenario_configs["UAV_height_range"][1],
            num_exps,
        )
        attacker_distances = np.random.uniform(
            scenario_configs["attacker_distance_range"][0],
            scenario_configs["attacker_distance_range"][1],
            num_exps,
        )
        item_pairs = list(product(scenario_configs["gz_models"].keys(), repeat=2))
        model_pairs = []
        for i in range(num_exps//len(item_pairs)):
            for (m1, m2) in item_pairs:
                model_pairs.append((scenario_configs["gz_models"][m1], scenario_configs["gz_models"][m2], m1, m2))
        model_pairs = model_pairs[:num_exps]  # Ensure we have exactly num_exps pairs
        
        configs["victim_poses"] = victim_poses
        configs["attacker_poses"] = attacker_poses
        configs["UAV_poses"] = uav_poses
        configs["victim_velocity"] = vic_velocity
        configs["heights"] = heights
        configs["attacker_distances"] = attacker_distances
        configs["gz_models"] = model_pairs
        
    elif args.world_file == "raceway":
        victim_poses, attacker_poses, uav_poses = [], [], []
        extra_poses = []
        assert num_exps % len(scenario_configs["victim_poses"]) == 0, \
            "num_exps must be a multiple of the number of predefined poses."
        for vic, atk, uav, ext in zip(
            scenario_configs["victim_poses"],
            scenario_configs["attacker_poses"],
            scenario_configs["UAV_poses"],
            scenario_configs["extra_poses"],
        ):
            for j in range(num_exps // len(scenario_configs["victim_poses"])):
                victim_poses.append(vic)
                attacker_poses.append(atk)
                uav_poses.append(uav)
                extra_poses.append(ext)
        victim_poses = np.array(victim_poses)
        attacker_poses = np.array(attacker_poses)
        uav_poses = np.array(uav_poses)
        extra_poses = np.array(extra_poses)

        vic_mags = np.random.uniform(
            scenario_configs["victim_velocity_range"][0],
            scenario_configs["victim_velocity_range"][1],
            num_exps,
        )
        yaw_angles = victim_poses[:, -1]
        vic_vx = vic_mags * np.cos(yaw_angles)
        vic_vy = vic_mags * np.sin(yaw_angles)
        vic_velocity = np.column_stack((vic_vx, vic_vy))

        heights = np.random.uniform(
            scenario_configs["UAV_height_range"][0],
            scenario_configs["UAV_height_range"][1],
            num_exps,
        )
        attacker_distances = np.random.uniform(
            scenario_configs["attacker_distance_range"][0],
            scenario_configs["attacker_distance_range"][1],
            num_exps,
        )
        item_pairs = list(product(scenario_configs["gz_models"].keys(), repeat=2))
        model_pairs = []
        assert num_exps % len(item_pairs) == 0, \
            "num_exps must be a multiple of the number of model pairs."
        for i in range(num_exps//len(item_pairs)):
            for (m1, m2) in item_pairs:
                model_pairs.append(
                    (scenario_configs["gz_models"][m1], scenario_configs["gz_models"][m2], m1, m2)
                )
        model_pairs = model_pairs[:num_exps]  # Ensure we have exactly num_exps pairs

        configs["victim_poses"] = victim_poses
        configs["attacker_poses"] = attacker_poses
        configs["UAV_poses"] = uav_poses
        configs["extra_poses"] = extra_poses
        configs["victim_velocity"] = vic_velocity
        configs["heights"] = heights
        configs["attacker_distances"] = attacker_distances
        configs["gz_models"] = model_pairs
    
    elif args.world_file == "urban_car":
        victim_poses = np.random.uniform(
            scenario_configs["victim_pose_ranges"][0],
            scenario_configs["victim_pose_ranges"][1],
            (num_exps, 6), # x, y, z, roll, pitch, yaw
        )
        attacker_poses = np.random.uniform(
            scenario_configs["attacker_pose_ranges"][0],
            scenario_configs["attacker_pose_ranges"][1],
            (num_exps, 6),
        )
        uav_poses = np.tile(
            scenario_configs["UAV_pose"],
            (num_exps, 1)
        )
        
        vic_mags = np.random.uniform(
            scenario_configs["victim_velocity_range"][0],
            scenario_configs["victim_velocity_range"][1],
            num_exps,
        )
        yaw_angles = victim_poses[:, -1]
        vic_vx = vic_mags * np.cos(yaw_angles)
        vic_vy = vic_mags * np.sin(yaw_angles)
        vic_velocity = np.column_stack((vic_vx, vic_vy))
        
        heights = np.random.uniform(
            scenario_configs["UAV_height_range"][0],
            scenario_configs["UAV_height_range"][1],
            num_exps,
        )
        attacker_distances = np.random.uniform(
            scenario_configs["attacker_distance_range"][0],
            scenario_configs["attacker_distance_range"][1],
            num_exps,
        )
        item_pairs = list(product(scenario_configs["gz_models"].keys(), repeat=2))
        model_pairs = []
        assert num_exps % len(item_pairs) == 0, \
            "num_exps must be a multiple of the number of model pairs."
        for i in range(num_exps//len(item_pairs)):
            for (m1, m2) in item_pairs:
                model_pairs.append((scenario_configs["gz_models"][m1], scenario_configs["gz_models"][m2], m1, m2))
        model_pairs = model_pairs[:num_exps]  # Ensure we have exactly num_exps pairs
        
        configs["victim_poses"] = victim_poses
        configs["attacker_poses"] = attacker_poses
        configs["UAV_poses"] = uav_poses
        configs["victim_velocity"] = vic_velocity
        configs["heights"] = heights
        configs["attacker_distances"] = attacker_distances
        configs["gz_models"] = model_pairs
        
    else:
        raise ValueError(f"Invalid world file name: {args.world_file}. Must be 'urban', 'warehouse', 'raceway', and 'urban_car'.")

    return configs


def launch_scripts(args, trial_configs, trial_index):
    """
    Launch a Python script in a new tmux session
    """
    
    print("Launching the simulation...")
    
    # GZ_IP=127.0.0.1 HEADLESS=1 PX4_GZ_WORLD=urban_car PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="-75.0,-52.0,0.0,0.0,0.0,0.0" PX4_SIM_MODEL=x500_gimbal ./build/px4_sitl_default/bin/px4
    env = os.environ.copy()
    env["GZ_IP"] = "127.0.0.1"
    env["HEADLESS"] = "1"
    env["PX4_GZ_WORLD"] = args.world_file
    env["PX4_SYS_AUTOSTART"] = "4002"
    env["PX4_SIM_MODEL"] = "x500_gimbal"
    env["PX4_GZ_MODEL_POSE"] = (
        f"{trial_configs['UAV_poses'][trial_index][0]},"
        f"{trial_configs['UAV_poses'][trial_index][1]},"
        f"{trial_configs['UAV_poses'][trial_index][2]},"
        f"{trial_configs['UAV_poses'][trial_index][3]},"
        f"{trial_configs['UAV_poses'][trial_index][4]},"
        f"{trial_configs['UAV_poses'][trial_index][5]}"
    )

    command = ["./build/px4_sitl_default/bin/px4"]

    print(
        "PX4 command:",
        f"GZ_IP={env['GZ_IP']} "
        f"HEADLESS={env['HEADLESS']} "
        f"PX4_GZ_WORLD={env['PX4_GZ_WORLD']} "
        f"PX4_SYS_AUTOSTART={env['PX4_SYS_AUTOSTART']} "
        f"PX4_GZ_MODEL_POSE=\"{env['PX4_GZ_MODEL_POSE']}\" "
        f"PX4_SIM_MODEL={env['PX4_SIM_MODEL']} "
        + " ".join(command)
    )

    px4_process = subprocess.Popen(
        command,
        cwd="/root/PX4-Autopilot",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if args.realtime_factor == 1.0 or args.realtime_factor == 0.5:
        if args.world_file in ["field", "urban", "urban_car"]:
            time.sleep(25)
        else:
            time.sleep(25)
    else:
        raise ValueError("Unsupported realtime factor. Only 1.0 and 0.5 are supported.")
    print("Launching object-follower.py...")
    command = [f"{args.vir_env}/bin/python", "object-follower.py"]
    command.append("--height")
    command.append(str(trial_configs["heights"][trial_index]))
    print("Object follower command:", " ".join(command))
    follower_process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if args.realtime_factor == 1.0 or args.realtime_factor == 0.5:
        if args.world_file in ["raceway", "urban_car"]:
            time.sleep(25)
        else:
            time.sleep(15)
        # if trial_configs["heights"][trial_index] > 20.0:
        #     time.sleep(40)
        # else:
        #     time.sleep(30)
    else:
        raise ValueError("Unsupported realtime factor. Only 1.0 and 0.5 are supported.")

    print("Launching gimbal_atk.py...")
    env = os.environ.copy()
    env["GZ_IP"] = "127.0.0.1"
    command = [
        f"{args.vir_env}/bin/python",
        "attack/gimbal_atk.py",
        "--tracker_type",
        args.tracker_type,
        "--surrogate_model",
        args.surrogate_model,
        "--world_file",
        args.world_file,
        "--victim_velocity",
        str(trial_configs["victim_velocity"][trial_index][0]),
        str(trial_configs["victim_velocity"][trial_index][1]),
        "--atker_dist",
        str(trial_configs["attacker_distances"][trial_index]),
        "--atker_angle",
        str(args.atker_angle),
        "--atk_cycle",
        str(args.atk_cycle),
        "--gim_max_speed",
        str(args.gim_max_speed[0]),
        str(args.gim_max_speed[1]),
        str(args.gim_max_speed[2]),
        "--atk_duration",
        str(args.atk_duration),
        "--victim_model",
        str(trial_configs["gz_models"][trial_index][2]),
        "--atker_model",
        str(trial_configs["gz_models"][trial_index][3]),
        "--save_dir",
        args.exp_dir,
        "--uav_height",
        str(trial_configs["heights"][trial_index]),
        "--real_time_factor",
        str(args.realtime_factor),
        "--attack_interval_method",
        args.attack_interval_method,
    ]
    print("Gimbal attack command:", "GZ_IP=127.0.0.1 ", " ".join(command))
    atk_process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("All scripts launched")
    return px4_process, atk_process


def monitor_proc(atk_process, timeout_seconds=100):
    start_time = datetime.now()
    time_delta = timedelta(seconds=timeout_seconds)

    while datetime.now() - start_time < time_delta:
        time.sleep(10)
        if atk_process.poll() is not None:
            # stdout, stderr = atk_process.communicate()
            # print("Standard output:", stdout)
            # print("Error output:", stderr)
            return TerminationReason.PROCESS_ENDED, atk_process.returncode

    return TerminationReason.TIMEOUT, None


def clean_up(keywords=["gz", "px4_sitl_default"], tmux_session="simulation"):
    """
    Clean up the non-terminated processes and threads
    """
    # Kill the remaining processes
    for keyword in keywords:
        subprocess.run(["pkill", "-9", "-f", keyword])
        time.sleep(0.5)


def create_video_from_images(image_pattern, output_filename, fps=30):
    """
    Create a video from a sequence of images

    Args:
        image_pattern (str): Pattern to match image files (e.g., 'test_*.png')
        output_filename (str): Name of the output video file
        fps (int): Frames per second for the output video
    """

    def natural_sort_key(s):
        """Sort strings containing numbers in natural order"""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

    # Get list of all matching image files and sort them naturally
    image_files = glob.glob(image_pattern)
    image_files.sort(key=natural_sort_key)
    # drop the first two files
    image_files = image_files[2:]

    if not image_files:
        print(f"No images found matching pattern: {image_pattern}")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error reading image: {image_files[0]}")
        return

    height, width = first_image.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Process each image
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is not None:
            out.write(img)
        else:
            print(f"Error reading image: {img_file}")

    # Release the video writer
    out.release()
    print(f"Video saved as {output_filename}")

    # Remove the images
    for img_file in image_files:
        os.remove(img_file)
    # Remove the empty directory
    # os.rmdir(os.path.dirname(image_files[0]))


def delete_files_in_dir(dir_path):
    """
    Delete all files in the given directory
    """
    path = os.path.expanduser(dir_path)
    dirs = os.listdir(path)

    for i in range(len(dirs) - 2):
        trial_path = os.path.join(path, f"trial_{i+1:03d}")
        img_path = os.path.join(trial_path, "images")
        for file in os.listdir(img_path):
            os.remove(os.path.join(img_path, file))
        print(f"Removed all files in {img_path}")


if __name__ == "__main__":
    dir_path = os.path.expanduser("/home/jiaruili/Documents/exp/uavAtk/trial_002/images")
    image_pattern = os.path.join(dir_path, "image_*.jpg")
    output_video = os.path.join(dir_path, "..", "output_raw.mp4")
    fps = 5  # Frames per second

    create_video_from_images(image_pattern, output_video, fps)

    # update_social_force_params()

    # clean_up(
    #     keywords=[
    #         "gz",
    #         "px4_sitl_default",
    #         "px4-venv-sysPy",
    #     ],
    #     tmux_session="simulation",
    # )

    # delete_files_in_dir(
    #     dir_path="/home/jiaruili/Documents/exp/gzFolMe2/20250312-085407-vicRand0_atkInt1_noLate_sameSync",
    # )