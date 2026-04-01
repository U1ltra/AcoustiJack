import os
import re
import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from utils.eval.helper import list_dirs, images_to_video, convert_bbox, get_iou, reconstruct_args


class SimulationState:
    """
    Class to store and manage simulation state from experiment traces.
    Provides indexed access to aligned simulation data.
    """

    def __init__(self, trial_path):
        """
        Initialize the simulation state from a trial path.
        Args:
            trial_path (str): Path to the trial data directory.
        """
        self.trial_path = trial_path
        self.id2name = {"10": "victim", "20": "attacker"}
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.vic_id = self.name2id["victim"]
        self.atk_id = self.name2id["attacker"]

        self.exp_args = reconstruct_args(os.path.join(self.trial_path, "..", "log", "args.txt"))
        self.trial_settings = self._load_coords_data()
        
        # Load all trace data from the trial path
        self.data, self.mapping = self._load_trace_data()

        # Initialize current index
        self.current_idx = 0

    def _load_trace_data(self):
        """Load all trace data from the trial path."""
        if not os.path.exists(os.path.join(self.trial_path, "raw_video.mp4")):
            raise FileNotFoundError(f"Trial path {os.path.join(self.trial_path, "raw_video.mp4")} does not exist.")
        data = {}
        try:
            mapping = {
                # gz
                "camera0_pose": "camera0_pose_history.npy",
                "atker_pose": "atk_pose_history.npy",
                "vic_pose": "vic_pose_history.npy",
                # atk algorithm
                "main_loop_timestamp": "main_timestamp_history.npy",
                "omega": "omega_history.npy",
                "it_trace": "iteration_trace_history.npy",
                "mal_cam_offset": "malicious_camera_offset_history.npy",
                "surrogate_pred": "surrogate_pred_history.npy",
                # tracking system
                # "gimbal_error": "vic_vehicle/gimbal_error_history.npy",
                # "uav_error": "vic_vehicle/uav_error_history.npy",
            }
            for obj_id in self.name2id.values():
                mapping[f"{obj_id}_det_uav"] = f"vic_vehicle/{obj_id}_history.npy"
                mapping[f"{obj_id}_pred_uav"] = f"vic_vehicle/{obj_id}_pred_history.npy"
                mapping[f"{obj_id}_pred_w_conf_uav"] = f"vic_vehicle/{obj_id}_pred_w_conf_history.npy"
                mapping[f"{obj_id}_world_uav"] = (
                    f"vic_vehicle/{obj_id}_world_history.npy"
                )
            for key, filename in mapping.items():
                file_path = os.path.join(self.trial_path, filename)
                data[key] = np.load(file_path, allow_pickle=True)
            
            with open(os.path.join(self.trial_path, "vic_vehicle", "all_pred_trks.pkl"), 'rb') as f:
                data["pred_trks"] = pickle.load(f)
                
            if len(data["pred_trks"]) != len(data["main_loop_timestamp"]): # in SORT experiments, I have 20 warmup frames
                # drop the initial items
                extra_items = len(data["pred_trks"]) - len(data["main_loop_timestamp"])
                data["pred_trks"] = data["pred_trks"][extra_items:]
                for obj_id in self.name2id.values():
                    data[f"{obj_id}_det_uav"] = data[f"{obj_id}_det_uav"][extra_items:]
                    data[f"{obj_id}_pred_uav"] = data[f"{obj_id}_pred_uav"][extra_items:]
                    data[f"{obj_id}_pred_w_conf_uav"] = data[f"{obj_id}_pred_w_conf_uav"][extra_items:]
                    data[f"{obj_id}_world_uav"] = data[f"{obj_id}_world_uav"][extra_items:]

            video_path = os.path.join(self.trial_path, "raw_video.mp4")
            self.cap = cv2.VideoCapture(video_path)
                
        except Exception as e:
            raise

        self.timestamp_categories = {
            "main_loop": [
                "omega",
                "it_trace",
                "mal_cam_offset",
                "gimbal_error",
                "uav_error",
                "surrogate_pred",
                "pred_trks",
            ],
            "gz": ["camera0_pose", "atker_pose", "vic_pose"],
        }
        
        self.valid_indices = np.arange(len(data["main_loop_timestamp"]))
        self.max_idx = len(self.valid_indices) - 1
        
        return data, mapping
    
    def _load_coords_data(self):
        """Load and parse coordinates data from log/trial_configs.pkl."""
        coords_path = os.path.join(self.trial_path, "..", "log", "trial_configs.pkl")
        if not os.path.exists(coords_path):
            return None

        trial_configs_path = os.path.join(self.trial_path, "..", "log", "trial_configs.pkl")
        with open(trial_configs_path, "rb") as f:
            trial_configs = pickle.load(f)
        
        return trial_configs
    
    def get_settings_for_trial(self, trial_num):
        if self.trial_settings is None:
            return None
        if trial_num < 1 or trial_num > len(self.trial_settings['victim_velocity']):
            raise ValueError(f"Trial number {trial_num} is out of bounds")
        
        trial_setting = {}
        for key, value in self.trial_settings.items():
            trial_setting[key] = value[trial_num - 1]
    
        return trial_setting

    def get_frame(self, idx=None):
        """
        Get all data for a specific frame index.
        Args:
            idx (int, optional): Frame index. If None, use current index.
        Returns:
            dict: Frame data with aligned indices.
        """
        if idx is None:
            idx = self.current_idx
        if idx < 0 or idx > self.max_idx:
            raise IndexError(f"Index {idx} is out of bounds [0, {self.max_idx}]")
        idx = self.valid_indices[idx]

        frame_data = {}
        for key, value in self.data.items():
            closest_idx = idx

            # if key is in gz timestamp categories, find the closest that aligns with main loop timestamp
            if key in self.timestamp_categories["gz"]:
                main_ts = self.data["main_loop_timestamp"][idx][1]  # 0: ros timestamp, 1: simulation time
                closest_idx = np.argmin(
                    np.abs(value[:, -1] - main_ts)
                )  # -2 is the ros timestamp, -1 is the simulation time

            if (isinstance(value, np.ndarray) or isinstance(value, list)):
                # Make sure we don't exceed the bounds of any array
                if closest_idx < len(value):
                    frame_data[key] = value[closest_idx]
                elif len(value) == 0:
                    # If the array is empty, we can't access any index
                    frame_data[key] = None
                else:
                    # Use last available value if index exceeds length
                    frame_data[key] = value[-1]
            else:
                # For non-array data, just copy the value
                frame_data[key] = value

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                frame_data["img"] = frame
            else:
                raise ValueError(f"Failed to read frame {idx} from video")
        else:
            raise ValueError("Video capture is not opened")

        return frame_data

    def delta_time_to_index(self, idx, delta):
        """Find the index of the frame closest to a given time delta."""
        timestamps = self.data["main_loop_timestamp"]
        if idx < 0 or idx >= len(timestamps):
            raise IndexError(f"Index {idx} is out of bounds")
        target_time = timestamps[idx] + delta * 1e9
        closest_idx = np.argmin(np.abs(timestamps - target_time))
        return closest_idx  # TODO: always round up?

    def __len__(self):
        """Return the number of frames in the simulation state."""
        return len(self.valid_indices)


def draw_bounding_boxes(sim_state):
    images = []
    for i in range(len(sim_state)):
        frame = sim_state.get_frame(i)
        vic_det_uav = frame[f"{sim_state.vic_id}_det_uav"]
        atk_det_uav = frame[f"{sim_state.atk_id}_det_uav"]
        vic_pred_uav = frame[f"{sim_state.vic_id}_pred_uav"]
        atk_pred_uav = frame[f"{sim_state.atk_id}_pred_uav"]
        img = frame["img"]
        # draw bounding boxes on the image
        bboxes = [
            vic_det_uav,
            atk_det_uav,
            vic_pred_uav,
            atk_pred_uav,
        ]
        labels = [
            "vic det",
            "atk det",
            "vic pred",
            "atk pred",
        ]
        colors = [
            (255, 0, 0),  # Blue for victim detection
            (0, 255, 0),  # Green for attacker detection
            (0, 0, 255),  # Red for victim prediction
            (255, 255, 0),  # Cyan for attacker prediction
        ]
        img_copy = img.copy()
        for bbox, label, color in zip(bboxes, labels, colors):
            if bbox is not None and len(bbox) == 4 and bbox[0] is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img_copy, f"Frame {i+20}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        images.append(img_copy)
    video_output_path = "debug_tracking_output.mp4"
    images_to_video(images, video_output_path, fps=5)

def parse_hijacking_log(log_path):
    """Parse log.txt and return list of (start_frame, end_frame) hijacking intervals.
    end_frame is exclusive (the frame where hijacking was lost), or inf if never lost.
    """
    intervals = []
    start = None
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'Hijacking successful at frame (\d+)', line)
            if m:
                start = int(m.group(1))
                continue
            m = re.search(r'Hijacking lost at frame (\d+)', line)
            if m and start is not None:
                intervals.append((start, int(m.group(1))))
                start = None
    if start is not None:
        intervals.append((start, float(70)))
    return intervals


def check_success(sim_state, idx):
    if idx < 0 or idx >= len(sim_state):
        return False

    if not hasattr(sim_state, '_hijacking_intervals'):
        log_path = os.path.join(sim_state.trial_path, 'log.txt')
        if os.path.exists(log_path):
            sim_state._hijacking_intervals = parse_hijacking_log(log_path)
        else:
            sim_state._hijacking_intervals = None

    if sim_state._hijacking_intervals is not None:
        return any(start <= idx < end for start, end in sim_state._hijacking_intervals)

    # Fallback to frame-level detection if no log available
    frame = sim_state.get_frame(idx)
    vic_pred_uav = frame[f"{sim_state.vic_id}_pred_uav"]
    atk_det_uav = frame[f"{sim_state.atk_id}_det_uav"]
    vic_det_uav = frame[f"{sim_state.vic_id}_det_uav"]

    if vic_pred_uav[0] is None or atk_det_uav[0] is None or vic_det_uav[0] is None:
        return False

    iou_atk = get_iou(vic_pred_uav.reshape(-1, 4), atk_det_uav.reshape(-1, 4))[0]
    iou_vic = get_iou(vic_pred_uav.reshape(-1, 4), vic_det_uav.reshape(-1, 4))[0]

    atk_det_xywh = convert_bbox(atk_det_uav, "x1y1x2y2", "xywh")
    vic_x1, vic_y1, vic_x2, vic_y2 = vic_pred_uav
    atk_x, atk_y, atk_w, atk_h = atk_det_xywh

    # Check if the attacker's bounding box overlaps with the victim's predicted bounding box
    if atk_x >= vic_x1 and atk_x <= vic_x2 and atk_y >= vic_y1 and atk_y <= vic_y2 and iou_atk > iou_vic:
        return True

    return False

def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation of tracking experiments")
    parser.add_argument("--result_path", type=str, default="/root/exp", help="Path to the directory containing experiment results")
    parser.add_argument("--claim", type=int, default=0, help="Claim 0: all experiments, 1: claim 1 experiments, 2: claim 2 experiments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    result_path = os.path.expanduser(result_path)
    # list all experiment directories in the result path
    exp_names = list_dirs(result_path)
    eval_exps = []
    for exp_name in exp_names:
        if args.claim == 1 and "uav2" not in exp_name:
            eval_exps.append(exp_name)
        elif args.claim == 2 and "uav2" in exp_name:
            eval_exps.append(exp_name)
        elif args.claim == 0:
            eval_exps.append(exp_name)

    trackers = ["siamr", "dasiam", "sort", "ucmc", "kcf"]
    tracker_results = {tracker: {"success": 0, "dos": 0, "total": 0} for tracker in trackers}

    for exp_name in eval_exps:
        print(f"Processing experiment: {exp_name}")
        exp_path = os.path.join(result_path, exp_name)
        dirs = list_dirs(exp_path)
        success = 0
        dos_count = 0
        skip_count = 0
        suc_trials = []
        longest_streaks = []

        for trial_num in tqdm(range(len(dirs) - 1)):
            # print(f"Processing trial {trial_num + 1}/{len(dirs) - 1}")
            trial_path = os.path.join(exp_path, f"trial_{trial_num+1:03d}")
            try:
                sim_state = SimulationState(trial_path)
            except Exception as e:
                skip_count += 1
                continue
            
            success_flag = False
            current_streak = 0
            max_streak = 0
            check_success(sim_state, 0)
            if sim_state._hijacking_intervals is not None:
                streaks = sim_state._hijacking_intervals
            else:
                streaks = [(0,0)]
            for streak in streaks:
                length = streak[1] - streak[0]
                if length > max_streak:
                    max_streak = length
                if length >= 4:
                    success_flag = True


            longest_streaks.append(max_streak)

            trial_settings = sim_state.get_settings_for_trial(trial_num + 1)

            try:
                last_frame = sim_state.get_frame(len(sim_state) - 1)
            except Exception as e:
                skip_count += 1
                continue
            vic_det_uav = last_frame[f"{sim_state.vic_id}_det_uav"]
            atk_det_uav = last_frame[f"{sim_state.atk_id}_det_uav"]
            vic_pred_uav = last_frame[f"{sim_state.vic_id}_pred_uav"]
            atk_pred_uav = last_frame[f"{sim_state.atk_id}_pred_uav"]
            vic_pred_w_conf_uav = last_frame[f"{sim_state.vic_id}_pred_w_conf_uav"]
            # Check tracking loss and DoS on the last frame
            loss_flag = False
            if vic_pred_uav[0] is None:
                loss_flag = True
                if not success_flag:
                    dos_count += 1
            else:
                bboxes1 = vic_pred_uav.reshape(-1, 4)
                bboxes2 = vic_det_uav.reshape(-1, 4)
                ious = get_iou(bboxes1, bboxes2)
                if ious[0] < 0.01 or vic_pred_w_conf_uav[-1] < 0.15:
                    loss_flag = True
                    if not success_flag:
                        dos_count += 1

        for tracker in trackers:
            if tracker in exp_name:
                tracker_results[tracker]["success"] += sum([1 for streak in longest_streaks if streak >= 5])
                tracker_results[tracker]["dos"] += dos_count
                tracker_results[tracker]["total"] += (len(dirs) - 1 - skip_count)
    print("Tracker performance summary:")
    for tracker, results in tracker_results.items():
        if results["total"] == 0:
            continue
        success_rate = results["success"] / results["total"] if results["total"] > 0 else 0
        dos_rate = results["dos"] / results["total"] if results["total"] > 0 else 0
        print(f"{tracker}: Success rate: {success_rate:.2f}, DoS rate: {dos_rate:.2f}")