import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from utils.eval.tracker_objects import KCFTracker
from utils.eval.helper import list_dirs, images_to_video, convert_bbox, visualize_bbox_trace_video, get_iou, calculate_angular_velocity, plot_angular_velocity, reconstruct_args, get_direction_region


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
                # if key == "it_trace":
                    # data[key] = np.squeeze(data[key])  # Remove empty dimensions
                    # data[key] = data[key][0]
                # print(f"Loaded {key} from {file_path} with shape {data[key].shape}")
            
            with open(os.path.join(self.trial_path, "vic_vehicle", "all_pred_trks.pkl"), 'rb') as f:
                data["pred_trks"] = pickle.load(f)
            # with open(os.path.join(self.trial_path, "vic_vehicle", "all_matched.pkl"), 'rb') as f:
            #     data["matched"] = pickle.load(f)
                
            if len(data["pred_trks"]) != len(data["main_loop_timestamp"]): # in SORT experiments, I have 20 warmup frames
                # drop the initial items
                extra_items = len(data["pred_trks"]) - len(data["main_loop_timestamp"])
                data["pred_trks"] = data["pred_trks"][extra_items:]
                # data["matched"] = data["matched"][extra_items:]
                for obj_id in self.name2id.values():
                    data[f"{obj_id}_det_uav"] = data[f"{obj_id}_det_uav"][extra_items:]
                    data[f"{obj_id}_pred_uav"] = data[f"{obj_id}_pred_uav"][extra_items:]
                    data[f"{obj_id}_pred_w_conf_uav"] = data[f"{obj_id}_pred_w_conf_uav"][extra_items:]
                    data[f"{obj_id}_world_uav"] = data[f"{obj_id}_world_uav"][extra_items:]

            video_path = os.path.join(self.trial_path, "raw_video.mp4")
            self.cap = cv2.VideoCapture(video_path)
                
        except Exception as e:
            # print(f"Error loading data from {self.trial_path}: {str(e)}")
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
        
        # print("Loaded data shapes:")
        # print(f"data['10_det_uav']", data["10_det_uav"].shape)
        # print(f"data['omega']", data["omega"].shape)
        # print(f"data['it_trace']", data["it_trace"].shape)
        # print(f"data['mal_cam_offset']", data["mal_cam_offset"].shape)
        # print(f"data['surrogate_pred']", data["surrogate_pred"].shape)
        # print(data["main_loop_timestamp"].shape)

        # self.valid_indices = np.where(data["omega"][:, 0] != None)[0]
        # start_idx = self.valid_indices[0]
        # self.valid_indices = [i for i in range(start_idx, len(data["omega"]))]
        # self.max_idx = len(self.valid_indices) - 1
        
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
                # print(f"Aligning {key} with main loop timestamp at index {closest_idx}")

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

def check_success(sim_state, idx):
    if idx < 0 or idx >= len(sim_state):
        return False

    frame = sim_state.get_frame(idx)
    vic_pred_uav = frame[f"{sim_state.vic_id}_pred_uav"]
    atk_det_uav = frame[f"{sim_state.atk_id}_det_uav"]
    
    if vic_pred_uav[0] is None or atk_det_uav[0] is None:
        return False

    atk_det_xywh = convert_bbox(atk_det_uav, "x1y1x2y2", "xywh")
    vic_x1, vic_y1, vic_x2, vic_y2 = vic_pred_uav
    atk_x, atk_y, atk_w, atk_h = atk_det_xywh

    # Check if the attacker's bounding box overlaps with the victim's predicted bounding box
    if atk_x >= vic_x1 and atk_x <= vic_x2 and atk_y >= vic_y1 and atk_y <= vic_y2:
        return True

    return False

    # bboxes1 = vic_pred_uav.reshape(-1, 4)
    # bboxes2 = atk_det_uav.reshape(-1, 4)
    # ious = get_iou(bboxes1, bboxes2)
    # if ious[0] > 0.3:
    #     return True


if __name__ == "__main__":
    # Example usage
    result_path = "/root/exp"
    result_path = os.path.expanduser(result_path)
    exp_names = [
        # "20250925-173730-ped_field_siam_1",
        ]
    success_in_regions = [0] * 4
    dos_count_in_regions = [0] * 4
    total_trials_regions = [0] * 4

    success_appearances = [0] * 2 # same, different appearance
    dos_count_appearances = [0] * 2
    total_trials_appearances = [0] * 2

    loss_count_in_regions = [0] * 4
    loss_count_appearances = [0] * 2

    for exp_name in exp_names:
        print(f"Processing experiment: {exp_name}")
        exp_path = os.path.join(result_path, exp_name)
        dirs = list_dirs(exp_path)
        success = 0
        one_success = 0
        success_at_halfway = 0
        one_success_at_halfway = 0
        suc_trials = []
        surrogate_success = 0
        skip_count = 0
        surrogate_error = []
        widths, heights = [], []
        mal_cam_offsets = []
        dos_count = 0
        dos_count_at_halfway = 0
        loss_count = 0
        loss_count_at_halfway = 0
        loss_flags = []  # whether the trial is a loss (tracker lost the target) or not
        longest_streaks = []  # longest consecutive success frames per trial

        for trial_num in tqdm(range(len(dirs) - 1)):
            # print(f"Processing trial {trial_num + 1}/{len(dirs) - 1}")
            trial_path = os.path.join(exp_path, f"trial_{trial_num+1:03d}")
            try:
                sim_state = SimulationState(trial_path)
            except Exception as e:
                # raise ValueError(f"Failed to load simulation state for {trial_path}: {e}")
                print(f"Failed to load simulation state for {trial_path}: {e}")
                skip_count += 1
                continue
            

            # draw_bounding_boxes(sim_state)
            # raise ValueError("Debugging: check the generated video and bounding boxes")

            # print(f"Trial {trial_num + 1}: len={len(sim_state)}, ")
            success_flag = False
            current_streak = 0
            max_streak = 0
            for i in range(len(sim_state)):
                if check_success(sim_state, i):
                    current_streak += 1
                    if current_streak > max_streak:
                        max_streak = current_streak
                    if not success_flag:
                        if check_success(sim_state, i + 1):
                            success_flag = True
                            success += 1
                            suc_trials.append((trial_num + 1, i, 0.0))  # IoU is not calculated here
                        elif i == len(sim_state) - 1: # switched at the last frame
                            success_flag = True
                            success += 1
                            suc_trials.append((trial_num + 1, i, 0.0))
                else:
                    current_streak = 0
            longest_streaks.append(max_streak)
                    
            # for i in range(len(sim_state)):
            #     if check_success(sim_state, i):
            #         one_success += 1
            #         break
                
            if trial_num == (len(dirs) - 1) // 2:
                success_at_halfway = success
                one_success_at_halfway = one_success
                dos_count_at_halfway = dos_count
                loss_count_at_halfway = loss_count
            
            trial_settings = sim_state.get_settings_for_trial(trial_num + 1)
            if trial_settings is not None:
                victim_velocity = trial_settings["victim_velocity"]
            region = get_direction_region(*victim_velocity)
            total_trials_regions[region] += 1
            if success_flag:
                success_in_regions[region] += 1

            target_objects = trial_settings['gz_models']
            diff_appearance = target_objects[0] != target_objects[1]
            total_trials_appearances[int(diff_appearance)] += 1 # 0(False): same, 1(True): different
            if success_flag:
                success_appearances[int(diff_appearance)] += 1

            try:
                last_frame = sim_state.get_frame(len(sim_state) - 1)
            except Exception as e:
                print(f"Failed to get last frame for trial {trial_num + 1}: {e}")
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
                    dos_count_in_regions[region] += 1
                    dos_count_appearances[int(diff_appearance)] += 1
            else:
                bboxes1 = vic_pred_uav.reshape(-1, 4)
                bboxes2 = vic_det_uav.reshape(-1, 4)
                ious = get_iou(bboxes1, bboxes2)
                if ious[0] < 0.01 or vic_pred_w_conf_uav[-1] < 0.15:
                    loss_flag = True
                    if not success_flag:
                        dos_count += 1
                        dos_count_in_regions[region] += 1
                        dos_count_appearances[int(diff_appearance)] += 1
            if loss_flag:
                loss_flags.append(1)
                loss_count += 1
                loss_count_in_regions[region] += 1
                loss_count_appearances[int(diff_appearance)] += 1

            mal_cam_offsets_ = []
            for idx in range(len(sim_state)):
                frame = sim_state.get_frame(idx)
                # vic_det_uav = frame[f"{sim_state.vic_id}_det_uav"]
                # atk_det_uav = frame[f"{sim_state.atk_id}_det_uav"]
                # vic_pred_uav = frame[f"{sim_state.vic_id}_pred_uav"]
                # atk_pred_uav = frame[f"{sim_state.atk_id}_pred_uav"]
                
                # vic_pred_xywh_uav = convert_bbox(vic_pred_uav, "x1y1x2y2", "xywh")
                # widths.append(vic_pred_xywh_uav[2])
                # heights.append(vic_pred_xywh_uav[3])
                # surr_pred_uav = frame["surrogate_pred"]
                # surr_pred_xywh_uav = convert_bbox(surr_pred_uav[:4], "x1y1x2y2", "xywh")
                # error = np.linalg.norm(np.array(vic_pred_xywh_uav[:2]) - np.array(surr_pred_xywh_uav[:2]))
                # normalized_error = error / np.linalg.norm(np.array(vic_pred_xywh_uav[2:4]))
                # surrogate_error.append(error)
                
                mal_cam_offset = frame["mal_cam_offset"]
                if mal_cam_offset is not None and mal_cam_offset[0] is not None:
                    mal_cam_offsets_.append(mal_cam_offset)
            if len(mal_cam_offsets_) > 0:
                mal_cam_offsets.append(np.sum(mal_cam_offsets_, axis=0))

        print(
            f"Total trials: {len(dirs) - 1 - skip_count}, Success count: {success}, Success rate: {success / (len(dirs) - 1 - skip_count):.2f}, \n",
            f"Success at halfway count: {success_at_halfway}, Success at halfway rate: {success_at_halfway / ((len(dirs) - 1)/2 - skip_count):.2f}\n",
            f"Success increase after halfway: {success - success_at_halfway}, Success increase rate after halfway: {(success - success_at_halfway) / ((len(dirs) - 1)/2 - skip_count):.2f}\n",
            
            f"\n",
            f"One Success count: {one_success}, One Success rate: {one_success / (len(dirs) - 1 - skip_count):.2f}\n",
            f"One Success at halfway count: {one_success_at_halfway}, One Success at halfway rate: {one_success_at_halfway / (len(dirs) - 1 - skip_count):.2f}\n",
            f"One Success increase after halfway: {one_success - one_success_at_halfway}, One Success increase rate after halfway: {(one_success - one_success_at_halfway) / ((len(dirs) - 1)/2 - skip_count):.2f}\n",
            
            f"\n",
            f"Total trials: {len(dirs) - 1 - skip_count}, Surrogate Success count: {surrogate_success}, Surrogate Success rate: {surrogate_success / (len(dirs) - 1 - skip_count):.2f}\n",
            
            f"\n",
            f"DoS count: {dos_count}",
            f"DoS rate: {dos_count / (len(dirs) - 1 - skip_count):.2f}",
            f"DoS at halfway count: {dos_count_at_halfway}, DoS at halfway rate: {dos_count_at_halfway / ((len(dirs) - 1)/2 - skip_count):.2f}",
            f"\n",
            f"Loss count: {loss_count}",
            f"Loss flags: {loss_flags}"
            # f"Average surrogate error: {np.mean(surrogate_error) if len(surrogate_error) > 0 else 0:.2f}",
            # f"Average bbox width: {np.mean(widths) if len(widths) > 0 else 0:.2f}, height: {np.mean(heights) if len(heights) > 0 else 0:.2f}",
            # f"Average mal cam offset: {np.mean(mal_cam_offsets, axis=0) if len(mal_cam_offsets) > 0 else [0,0,0]}",
        )
        print(f"Longest consecutive success streaks per trial: {longest_streaks}")
        print(f"Max streak: {max(longest_streaks) if longest_streaks else 0}, "
              f"Mean streak: {np.mean(longest_streaks) if longest_streaks else 0:.2f}")
        if success > 0:
            for trial_num, frame_id, iou in suc_trials:
                print(f"Trial {trial_num}, Frame {frame_id}, IoU: {iou:.2f}")

    # region success rates
    print("Region success rates:")
    for i in range(4):
        if total_trials_regions[i] > 0:
            success_rate = success_in_regions[i] / total_trials_regions[i]
            print(f"Region {i}: Success rate: {success_rate:.2f} ({success_in_regions[i]}/{total_trials_regions[i]})")
        else:
            print(f"Region {i}: No trials")
    print("DoS counts:")
    for i in range(4):
        print(f"Region {i}: DoS rate: {dos_count_in_regions[i] / total_trials_regions[i] if total_trials_regions[i] > 0 else 0:.2f}")
    
    # appearance success rates
    print("Appearance success rates:")
    for i in range(2):
        if total_trials_appearances[i] > 0:
            success_rate = success_appearances[i] / total_trials_appearances[i]
            print(f"Appearance {i} ({'different' if i==1 else 'same'}): Success rate: {success_rate:.2f} ({success_appearances[i]}/{total_trials_appearances[i]})")
        else:
            print(f"Appearance {i} ({'different' if i==1 else 'same'}): No trials")
    print("DoS counts by appearance:")
    for i in range(2):
        print(f"Appearance {i} ({'different' if i==1 else 'same'}): DoS rate: {dos_count_appearances[i] / total_trials_appearances[i] if total_trials_appearances[i] > 0 else 0:.2f}")