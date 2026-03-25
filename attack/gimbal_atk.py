import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
import cv2
import sys
import time
import rclpy
import pickle
import argparse
import subprocess
import copy as cp
import numpy as np
import tensorflow as tf
import pysocialforce as psf
tf.random.set_seed(42)  # Set random seed for reproducibility
from transforms3d.euler import quat2euler
from scipy.optimize import linear_sum_assignment

from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Pose, Point, TwistStamped, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from vision_msgs.msg import Detection2DArray, Detection3DArray
from image_geometry import PinholeCameraModel

from onboard.tracker import TrackerConfig, OnboardTracker
from utils.topic_bridge_manager import TopicBridgeManager
from attack.motion_atk import (
    project_box_camera_to_world,
    project_box_3d_to_2d,
    parse_detection2d_array,
    parse_detection3d_array,
)
from utils.eval.helper import convert_bbox, get_color, visualize_bbox_trace_video, get_iou
from attack.opt_gim_motion import AttackConfig, TFGimbalAttackOptimizer
from attack.mock_tracker import MockTracker, MockTrackerSiamRPN
from utils.gz_helper import GzTransportController, GzClockSubscriber, GzGimbalController
from utils.sort import Sort


class TrackerBase:
    def __init__(self, id2name, tracker_config: TrackerConfig, **kwargs):
        self.id2name = id2name
        self.name2id = {v: k for k, v in id2name.items()}
        self.object_ids = list(id2name.keys())
        self.followme_trackers = OnboardTracker(tracker_config)
        self.history = {obj_id: [] for obj_id in self.object_ids}
        self.pred_history = {obj_id: [] for obj_id in self.object_ids}
        self.pred_w_conf_history = {obj_id: [] for obj_id in self.object_ids}
        self.all_pred_trks = []
        self.all_matched = []
        self.world_history = {obj_id: [] for obj_id in self.object_ids}
        self.current_det = {}
        self.world_corners = {}
        self.world_center = {}
        
        self.model_scales = {
            "pedestrian1": (1.0, 1.0),  # gz casual_female
            "pedestrian2": (1.0, 1.0),  # gz casual_female
            "pedestrian3": (1.0, 1.0),  # gz casual_female
            # "pedestrian2": (0.4, 0.7),  # gz stand
            # "pedestrian3": (0.4, 0.7),  # gz male visitor
            "car1": (1.0, 1.0),  # gz hatchback
            "car2": (1.0, 1.0),  # gz pickup
            "car3": (1.0, 1.0),  # gz hatchback_blue
            "default": (1.0, 1.0),  # default scale
        }
        
        self.object_scales = {
            "10": self.model_scales[kwargs.get("victim_model", "default")],
            "20": self.model_scales[kwargs.get("atker_model", "default")],
        }
        self.uav_height = kwargs.get("uav_height", 8.0)

    def set_current_det(self, det2d, det3d, attack_started=False):
        """Set current detection"""
        for obj_id in self.object_ids:
            self.current_det[obj_id] = {
                "2d": det2d[obj_id],
                "3d": det3d[obj_id],
            }
        if not attack_started:
            return

        noise_3d = np.random.normal(0, 0.1, 2)
        for obj_id in self.object_ids:
            # Apply noise to x,y position
            self.current_det[obj_id]["3d"]["position"][0] += noise_3d[0]
            self.current_det[obj_id]["3d"]["position"][1] += noise_3d[1]

    def get_bbox_from_detection(self, obj_name, source="2d", box_format="x1y1x2y2", wscale=1.0, hscale=1.0):
        """Get bounding box from Gazebo detections"""
        if obj_name not in self.name2id:
            raise ValueError(f"Object {obj_name} not found in current detection")

        obj_id = self.name2id[obj_name]
        if source not in self.current_det[obj_id]:
            raise ValueError(f"Source {source} not found in current detection")
        if source == "2d":
            raise NotImplementedError("Use 3d instead. ")
            x, y = self.current_det[obj_id][source]["position"]
            w, h = self.current_det[obj_id][source]["dimensions"]
            return convert_bbox([x, y, w, h], "xywh", box_format, wscale=wscale, hscale=hscale)
        elif source == "3d":
            x1, y1, x2, y2 = self.history[obj_id][-1]
            return convert_bbox([x1, y1, x2, y2], "x1y1x2y2", box_format, wscale=wscale, hscale=hscale)
        else:
            raise ValueError(f"Invalid source: {source}")

    def update_coordinates_position(self, detections, camera_model):
        """Update coordinates position"""
        for obj_id in self.object_ids:
            det = detections[obj_id]
            _, bbox = project_box_3d_to_2d(det, camera_model)
            bbox = convert_bbox(bbox, "x1y1x2y2", "x1y1x2y2", wscale=self.object_scales[obj_id][0], hscale=self.object_scales[obj_id][1])
            self.history[obj_id].append(bbox)
    
    def update_pred_position(self, img, attack_started=False, camera=None):
        if not self.followme_trackers.has_init:
            vic_box = self.history[self.name2id["victim"]][-1]
            atk_box = self.history[self.name2id["attacker"]][-1]
            vic_box = np.array(vic_box)
            atk_box = np.array(atk_box)
            if camera is not None:
                cam_params = {
                    "Ki": camera.camera_model.projection_matrix(),
                    "Ko": camera.inv_mat,
                    "uav_height": self.uav_height,
                }
            self.followme_trackers.init_tracker(img, vic_box, atk_box, cam_params=cam_params)
            vic_w_conf = np.append(vic_box, 1.0)
            atk_w_conf = np.append(atk_box, 1.0)
            pred_trks, res, matched = [], [], [self.followme_trackers.init_vic_id, self.followme_trackers.init_atk_id]
        else:
            # pred_trks: all tracklets no matter associated to detection or not [N, 5]
            # res: only associated tracklets to detections [M, 5]
            gz_det = (
                self.history[self.name2id["victim"]][-1],
                self.history[self.name2id["attacker"]][-1],
            )
            dets, pred_trks, res, matched = self.followme_trackers.track(img, vic_gt_box=gz_det[0], atk_gt_box=gz_det[1])
            vic_box, atk_box = self.followme_trackers.get_vic_atk_boxes(res)
            vic_w_conf, atk_w_conf = self.followme_trackers.get_vic_atk_det_boxes(dets, gz_det, res)
            if (vic_box[0] is None or atk_box[0] is None) and self.followme_trackers.tracker_type == "SORT":
                if not attack_started: # stable tracking when initializing
                    vic_box = vic_w_conf[:4]
                    atk_box = atk_w_conf[:4]

        self.pred_history[self.name2id["victim"]].append(vic_box)
        self.pred_history[self.name2id["attacker"]].append(atk_box)
        self.pred_w_conf_history[self.name2id["victim"]].append(vic_w_conf)
        self.pred_w_conf_history[self.name2id["attacker"]].append(atk_w_conf)
        self.all_pred_trks.append(pred_trks) # [frame_num, N, 5]. x1y1x2y2id
        self.all_matched.append(matched)

    def update_world_position(self, detections, matrix, static_camera=False):
        """Update world position using transformation matrix"""
        for obj_id in self.object_ids:
            corners, center = project_box_camera_to_world(
                detections[obj_id], matrix, static_camera=static_camera
            )
            self.world_corners[obj_id] = corners
            self.world_center[obj_id] = center
            self.world_history[obj_id].append(center)

        return self.world_corners, self.world_center

    def save_tracking_data(self, save_dir):
        """Save tracking data to file"""
        for obj_id in self.object_ids:
            history = np.array(self.history[obj_id])
            pred_history = np.array(self.pred_history[obj_id])
            pred_w_conf_history = np.array(self.pred_w_conf_history[obj_id])
            world_history = np.array(self.world_history[obj_id])
            for name in [
                "history",
                "pred_history",
                "pred_w_conf_history",
                "world_history",
            ]:
                np.save(
                    os.path.join(save_dir, f"{obj_id}_{name}.npy"),
                    locals()[name],
                )
        # pickle dump all_pred_trks
        with open(os.path.join(save_dir, "all_pred_trks.pkl"), "wb") as f:
            pickle.dump(self.all_pred_trks, f)
        with open(os.path.join(save_dir, "all_matched.pkl"), "wb") as f:
            pickle.dump(self.all_matched, f)

class SimStates(TrackerBase):
    """Trace tracker for the attack simulation"""

    def __init__(self, id2name, tracker_type="SORT", surrogate_model="SORT", atk_cycle=2.03, **kwargs):
        self.tracker_type = tracker_type
        self.surrogate_model = surrogate_model
        self.atk_cycle = atk_cycle
        self.gim_max_speed = kwargs.get("gim_max_speed", [1.0, 1.0, 2.0])
        self.tracker_config = TrackerConfig(
            tracker_type=tracker_type,
            track_conf_threshold=0.2,
            det_conf_threshold=0.4,
        )
        super().__init__(
            id2name, self.tracker_config, 
            victim_model=kwargs.get("victim_model", None), atker_model=kwargs.get("atker_model", None),
            uav_height=kwargs.get("uav_height", 8.0)
        )

        self.init_height = -1
        self.uav_error_history = []
        self.gimbal_error_history = []
        self.opt_time_history = []

        self.bridge = CvBridge()
        
        self.config = AttackConfig(
            mock_tracker_type=surrogate_model,
            frequency= atk_cycle,  # attack cycle in Hz
            T_fps=1/kwargs.get("fps", 30),  # frame interval in seconds (1/fps)
        )
        self.config.omega_max = np.array(kwargs.get("gim_max_speed", [1.0, 1.0, 2.0]), dtype=np.float32)  # gimbal max speed in pitch, roll, yaw (rad/s)
        self.config.omega_max_norm = np.linalg.norm(self.config.omega_max)
        if self.surrogate_model == "UCMCTrack":
            self.config.w5 = 1.0
            
        if surrogate_model == "SORT":

            if tracker_type == "SORT":
                self.trackers = MockTracker()
            elif tracker_type == "SiamRPN":
                self.trackers = MockTracker()
                self.sort = Sort(max_age=100, min_hits=1, iou_threshold=0.2)
                self.victim_tracklet_id = 1
                self.attacker_tracklet_id = 2
                
        elif surrogate_model == "SiamRPN":
            if tracker_type == "SiamRPN":
                self.trackers = MockTrackerSiamRPN(
                    self.followme_trackers._tracker, self.followme_trackers.cfg
                )
            elif tracker_type == "SORT":
                init_config = TrackerConfig(
                    tracker_type="SiamRPN",
                    track_conf_threshold=0.2,
                    det_conf_threshold=0.4,
                )
                init_model = OnboardTracker(init_config)
                self.trackers = MockTrackerSiamRPN(init_model._tracker, init_model.cfg, env_conf=0.4)
        elif surrogate_model == "DaSiamRPN":
            init_config = TrackerConfig(
                tracker_type="SiamRPN",
                track_conf_threshold=0.2,
                det_conf_threshold=0.4,
            )
            init_model = OnboardTracker(init_config)
            self.trackers = MockTrackerSiamRPN(init_model._tracker, init_model.cfg)
        elif surrogate_model == "KCF":
            init_config = TrackerConfig(
                tracker_type="SiamRPN",
                track_conf_threshold=0.2,
                det_conf_threshold=0.4,
            )
            init_model = OnboardTracker(init_config)
            self.trackers = MockTrackerSiamRPN(init_model._tracker, init_model.cfg, env_conf=0.7)
        elif surrogate_model == "UCMCTrack":
            self.trackers = MockTracker()
            self.sort = Sort(max_age=3, min_hits=1, iou_threshold=0.2)
            self.victim_tracklet_id = 1 # since we passed victim as the first object in update_surrogate_tracker
            self.attacker_tracklet_id = 2 # and the id starts from 1
            self.sort.kept_ids = [self.victim_tracklet_id, self.attacker_tracklet_id]

        self.optimizer = TFGimbalAttackOptimizer(self.config)
        self.opt_init_states = {
            "trackers": self.trackers,
            "victim_3d_center": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "attacker_3d_center": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "victim_velocity": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "attacker_velocity": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "uav_linear_velocity": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "uav_angular_velocity": tf.Variable([0.0, 0.0, 0.0], dtype=tf.float32),
            "cam_pose": tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
            "frame_num": None,
            "world_file": kwargs.get("world_file", "default"),
            "victim_2d_center": tf.Variable([0.0, 0.0], dtype=tf.float32),
            "attacker_2d_center": tf.Variable([0.0, 0.0], dtype=tf.float32),
        }

        self.attack_object_relative_direction = None

    def optimize_attack(
        self, 
        camera_pose, 
        world_centers, 
        pixel_centers,
        estimated_vic_atk_velocity, 
        estimated_uav_velocity, 
        frame_num, 
        save_dir, 
        log_func,
    ):
        """Calculate error for victim's motion control"""
        # Calculate error for gimbal control
        victim_center = world_centers[self.name2id["victim"]]
        attacker_center = world_centers[self.name2id["attacker"]]
        victim_2d_center = pixel_centers[0]
        attacker_2d_center = pixel_centers[1]

        if self.surrogate_model == "SORT":
            if self.tracker_type == "SORT":
                victim_kf, attacker_kf = self.followme_trackers.find_vic_atk_trackers()
                self.trackers.add(victim_kf, 10)
                self.trackers.add(attacker_kf, 20)
            elif self.tracker_type == "SiamRPN":
                victim_kf = self.sort.find_tracker_by_id(self.victim_tracklet_id)
                attacker_kf = self.sort.find_tracker_by_id(self.attacker_tracklet_id)
                self.trackers.add(victim_kf, 10)
                self.trackers.add(attacker_kf, 20)
        elif self.surrogate_model == "SiamRPN":
            if self.tracker_type == "SiamRPN":
                if self.trackers.initial_center_pos is None:
                    self.trackers.reset_state(center_pos=self.followme_trackers._tracker.center_pos,
                                            size=self.followme_trackers._tracker.size)
                else:
                    # update the mock tracker with the gt state
                    # self.trackers.reset_state(
                    #     center_pos=self.followme_trackers._tracker.center_pos,
                    #     size=self.followme_trackers._tracker.size
                    # )

                    self.trackers.reset_state(
                        center_pos=self.trackers.center_pos,
                        size=self.trackers.size,
                    )
            elif self.tracker_type == "SORT":
                if self.trackers.initial_center_pos is None:
                    victim_kf, attacker_kf = self.followme_trackers.find_vic_atk_trackers()
                    bbox = victim_kf.get_state()[0]
                    xywh = convert_bbox(bbox, "x1y1x2y2", "xywh")
                    self.trackers.reset_state(center_pos=xywh[:2],
                                            size=xywh[2:])
                else:
                    self.trackers.reset_state(
                        center_pos=self.trackers.center_pos,
                        size=self.trackers.size,
                    )
        elif self.surrogate_model == "DaSiamRPN":
            if self.trackers.initial_center_pos is None:
                self.trackers.reset_state(center_pos=self.followme_trackers._tracker["target_pos"],
                                         size=self.followme_trackers._tracker["target_sz"])
            else:
                self.trackers.reset_state(
                    center_pos=self.trackers.center_pos,
                    size=self.trackers.size,
                )
        elif self.surrogate_model == "KCF":
            if self.trackers.initial_center_pos is None:
                self.trackers.reset_state(center_pos=self.followme_trackers._tracker.center,
                                         size=self.followme_trackers._tracker.size)
            else:
                self.trackers.reset_state(
                    center_pos=self.trackers.center_pos,
                    size=self.trackers.size,
                )
        elif self.surrogate_model == "UCMCTrack":
            victim_kf = self.sort.find_tracker_by_id(self.victim_tracklet_id)
            attacker_kf = self.sort.find_tracker_by_id(self.attacker_tracklet_id)
            self.trackers.add(victim_kf, 10)
            self.trackers.add(attacker_kf, 20)

        self.opt_init_states["victim_3d_center"].assign(victim_center)
        self.opt_init_states["attacker_3d_center"].assign(attacker_center)
        self.opt_init_states["victim_velocity"].assign(estimated_vic_atk_velocity["victim_velocity"])
        self.opt_init_states["attacker_velocity"].assign(estimated_vic_atk_velocity["attacker_velocity"])
        self.opt_init_states["uav_linear_velocity"].assign(estimated_uav_velocity["linear_velocity"])
        self.opt_init_states["uav_angular_velocity"].assign(estimated_uav_velocity["angular_velocity"])
        self.opt_init_states["cam_pose"].assign(camera_pose[:7])
        self.opt_init_states["frame_num"] = frame_num
        self.opt_init_states["victim_2d_center"].assign(victim_2d_center[0:2])
        self.opt_init_states["attacker_2d_center"].assign(attacker_2d_center[0:2])

        opt_start = time.time()
        optimal_omega_seq, opt_iteration_trace, _ = self.optimizer.optimize_attack(self.opt_init_states, save_path=save_dir)
        opt_end = time.time()
        self.opt_time_history.append(opt_end - opt_start)
        log_func(f"Optimization time: {opt_end - opt_start:.2f} seconds")

        # negate pitch and yaw to match the gimbal control set points
        optimal_omega_seq = optimal_omega_seq * np.array([1.0, -1.0, -1.0], dtype=np.float32)
        print(f"Optimal omega sequence: {optimal_omega_seq}")
        optimal_offset_seq = optimal_omega_seq * self.config.delta_t
                
        return optimal_omega_seq, opt_iteration_trace, optimal_offset_seq

    def gimbal_control_msg(self, optimal_omega):
        w_roll, w_pitch, w_yaw = optimal_omega
        time_interval = self.config.delta_t
        frame_interval = self.config.T_fps   
        delta_roll = w_roll * frame_interval
        delta_pitch = - w_pitch * frame_interval
        delta_yaw = - w_yaw * frame_interval
        
        gimbal_error = Point()
        gimbal_error.x = float(delta_yaw)  # yaw
        gimbal_error.y = float(delta_pitch)  # pitch
        gimbal_error.z = float(delta_roll)  # roll

        self.gimbal_error_history.append([gimbal_error.x, gimbal_error.y, gimbal_error.z])

        return gimbal_error
    
    def followme_msg(self, detection, camera_info, noise=0.0):
        """Calculate error for Follow-me target's motion control"""
        error = Point()
        # Center the Follow-me target in the camera frame and maintain initial height
        w, h = camera_info.width, camera_info.height
        if isinstance(detection, list) or isinstance(detection, np.ndarray):
            if detection[0] is None:
                error.x, error.y, error.z = 0.0, 0.0, 0.0
                return error
            xywh = convert_bbox(detection, "x1y1x2y2", "xywh")
            detection = {"position": [xywh[0], xywh[1]], "dimensions": [xywh[2], xywh[3]]}
        if self.init_height < 0:
            self.init_height = detection["dimensions"][1]

        # random noises
        center_noise = np.random.uniform(-noise, noise, size=2)

        # Calculate error for UAV and gimbal motion control
        error.x = detection["position"][0] - w / 2 + center_noise[0]
        error.y = detection["position"][1] - h / 2 + center_noise[1]
        error.z = detection["dimensions"][1] - self.init_height

        return error

    def attacker_pose_msg(self, camera_pose, victim_pose, attacker_pose, victim_velocity, dist=5.0, strategy="angle", angle=90.0, motion_model=False, sim_time_to_move=0.0):
        """
        Calculate attacker position based on strategy.
        
        Args:
            camera_pose: Camera position
            victim_pose: Victim position
            attacker_pose: Attacker position
            victim_velocity: Victim velocity
            dist: Distance from victim to attacker
            strategy: "perpendicular", "line", "follow", or "angle"
            angle: Angle in degrees (0-90) between attacker-victim line and victim-camera line
                0° = attacker on the line (same as "line")
                90° = attacker perpendicular (same as "perpendicular")
        """
        vic_xy = np.array(victim_pose[:2])
        
        if strategy == "perpendicular":
            cam_xy = np.array(camera_pose[:2])
            direction = cam_xy - vic_xy
            if np.linalg.norm(direction) < 1e-3:
                raise ValueError("Camera and victim are at the same position")
            perp_direction = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            offset = perp_direction * dist
            # move to the opposite side of the victim's velocity
            
            # victim_direction = np.array(victim_velocity) / np.linalg.norm(victim_velocity)
            # if np.dot(perp_direction, victim_direction) > 0:
            #     offset = - offset

        elif strategy == "line":
            cam_xy = np.array(camera_pose[:2])
            direction = vic_xy - cam_xy
            if np.linalg.norm(direction) < 1e-3:
                raise ValueError("Camera and victim are at the same position")
            unit_direction = direction / np.linalg.norm(direction)
            offset = - unit_direction * dist
            
        elif strategy == "angle":
            assert 0.0 <= angle <= 90.0, "Angle must be between 0 and 90 degrees"
            # New strategy: arbitrary angle between attacker-victim and victim-camera lines
            cam_xy = np.array(camera_pose[:2])
            cam_to_vic = vic_xy - cam_xy
            if np.linalg.norm(cam_to_vic) < 1e-3:
                raise ValueError("Camera and victim are at the same position")
            
            # Normalize the camera-to-victim direction
            cam_to_vic_norm = cam_to_vic / np.linalg.norm(cam_to_vic)
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Calculate perpendicular direction (90° rotated)
            perp_direction = np.array([-cam_to_vic_norm[1], cam_to_vic_norm[0]])
            
            # Blend between the line direction (0°) and perpendicular direction (90°)
            # At 0°: attacker is on the opposite side of the camera
            # At 90°: attacker is perpendicular to the camera-victim line
            line_component = -cam_to_vic_norm * np.cos(angle_rad)
            perp_component = perp_direction * np.sin(angle_rad)
            
            offset_direction = line_component + perp_component
            offset = offset_direction * dist
            
        elif strategy == "follow":
            victim_direction = np.array(victim_velocity) / np.linalg.norm(victim_velocity)
            offset = - victim_direction * dist
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        atk_xy = vic_xy + offset

        if motion_model:
            # cap the attacker's velocity
            atk_dist_to_move = np.linalg.norm(atk_xy - attacker_pose[:2])
            max_atk_move = 2.5 * sim_time_to_move  # assume max speed of attacker is 2.5 m/s
            if atk_dist_to_move > max_atk_move:
                atk_xy = attacker_pose[:2] + (atk_xy - attacker_pose[:2]) / atk_dist_to_move * max_atk_move
            # add some noise to the attacker's position
            noise = np.random.uniform(-0.1, 0.1, size=2)
            atk_xy += noise
        
        atk_pose = PoseStamped()
        atk_pose.header.frame_id = "world"
        atk_pose.pose.position.x = float(atk_xy[0])
        atk_pose.pose.position.y = float(atk_xy[1])
        atk_pose.pose.position.z = float(victim_pose[2])
        atk_pose.pose.orientation.x = 0.0
        atk_pose.pose.orientation.y = 0.0
        atk_pose.pose.orientation.z = 0.0
        atk_pose.pose.orientation.w = 1.0
        return atk_pose
    
    def update_surrogate_tracker(self, victim_box, attacker_box):
        """Update the surrogate tracker with the current image and bounding boxes"""
        
        if self.surrogate_model == "SORT":
            if self.tracker_type == "SORT":
                victim_kf, attacker_kf = self.followme_trackers.find_vic_atk_trackers()
                if victim_kf is None or attacker_kf is None:
                    return None
                bbox = victim_kf.get_state()[0]
                return {"bbox": bbox, "best_score": 1.0, "best_pscore": 1.0}
            elif self.tracker_type == "SiamRPN":
                tracked_results, pred_trks, matched = self.sort.update(np.array(
                    [victim_box[:4] + [1], attacker_box[:4] + [1]]
                ))
                victim_kf = self.sort.find_tracker_by_id(self.victim_tracklet_id)
                bbox = victim_kf.get_state()[0]
                return {"bbox": bbox, "best_score": 1.0, "best_pscore": 1.0}
            
        elif self.surrogate_model == "SiamRPN":
            if self.trackers.initial_center_pos is None:
                self.trackers.reset_state(center_pos=self.followme_trackers._tracker.center_pos,
                                        size=self.followme_trackers._tracker.size)
            return self.trackers.predict(victim_box, attacker_box)
        elif self.surrogate_model == "DaSiamRPN":
            if self.trackers.initial_center_pos is None:
                self.trackers.reset_state(center_pos=self.followme_trackers._tracker["target_pos"],
                                         size=self.followme_trackers._tracker["target_sz"])
            return self.trackers.predict(victim_box, attacker_box)
        elif self.surrogate_model == "KCF":
            if self.trackers.initial_center_pos is None:
                self.trackers.reset_state(center_pos=self.followme_trackers._tracker.center,
                                         size=self.followme_trackers._tracker.size)
            return self.trackers.predict(victim_box, attacker_box)
        elif self.surrogate_model == "UCMCTrack":
            tracked_results, pred_trks, matched = self.sort.update(np.array(
                [victim_box[:4] + [1], attacker_box[:4] + [1]]
            ))
            victim_kf = self.sort.find_tracker_by_id(self.victim_tracklet_id)
            bbox = victim_kf.get_state()[0]
            return {"bbox": bbox, "best_score": 1.0, "best_pscore": 1.0}
        else:
            raise ValueError(f"Invalid surrogate model: {self.surrogate_model}")

    def save_tracking_data(self, save_dir):
        """Save tracking data to file"""
        super().save_tracking_data(save_dir)
        for name in ["uav_error_history", "gimbal_error_history"]:
            np.save(
                os.path.join(save_dir, f"{name}.npy"),
                getattr(self, name),
            )

class CameraSetup:
    """Handles camera information and transformations"""

    def __init__(self, camera_frame_id):
        self.frame_id = camera_frame_id
        self.camera_model = PinholeCameraModel()
        self.info = None
        self.pose = None
        self.pose_history = []
        self.pose_wo_noise_history = []
        self.mat = None
        self.inv_mat = None

    def update_camera_info(self, msg):
        """Update camera info from CameraInfo message"""
        if msg.header.frame_id == self.frame_id:
            self.camera_model.from_camera_info(msg)
            self.info = msg
            return True
        return False

    def update_camera_pose(self, msg, ros_time, sim_time):
        """Update camera pose from Pose message"""
        self.pose_wo_noise_history.append([msg.position.x, msg.position.y, msg.position.z])
        noisy_pos, _ = self._uav_det_error(
            (msg.position.x, msg.position.y, msg.position.z),
            target_mse=2.21,
            noise_type='gaussian'
        )

        pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]

        # Calculate Euler angles
        roll, pitch, yaw = quat2euler(
            [pose[6], pose[3], pose[4], pose[5]],  # wxyz
            axes="sxyz",
        )

        # Store angles and timestamp
        pose.extend([roll, pitch, yaw, ros_time.nanoseconds, sim_time])
        self.pose = pose
        self.pose_history.append(pose)

        return True

    def update_transformation_matrix(self, mat_msg):
        """Update transformation matrix from Float64MultiArray message"""
        self.mat = np.array(mat_msg.data[:16]).reshape(4, 4)
        self.inv_mat = np.array(mat_msg.data[16:32]).reshape(4, 4)

    def save_camera_data(self, save_dir, camera_id):
        """Save camera data to file"""
        for name in ["pose_history", "pose_wo_noise_history"]:
            np.save(
                os.path.join(save_dir, f"camera{camera_id}_{name}.npy"),
                getattr(self, name),
            )
            
    def _uav_det_error(self, original_point, target_mse=2.21, noise_type='gaussian'):
        """
        Inject noise (error) into 3D points to achieve a target MSE.
        Using 2.21 MSE from https://arxiv.org/pdf/2405.16464
        
        Parameters:
        - original_point: tuple or array (x, y, z)
        - target_mse: desired MSE (for single point, this equals squared distance)
        - noise_type: 'gaussian', 'uniform', 'spherical', or 'custom'
        
        Returns:
        - noisy_point: new 3D point
        - actual_mse: actual MSE achieved
        """
        original_point = np.array(original_point)
        
        if noise_type == 'gaussian':
            # For 3D Gaussian noise, if we want MSE = target_mse:
            # MSE = E[(x_noise - x_orig)²] + E[(y_noise - y_orig)²] + E[(z_noise - z_orig)²]
            # For isotropic Gaussian noise with std σ: MSE = 3σ²
            # Therefore: σ = sqrt(target_mse / 3)
            sigma = np.sqrt(target_mse / 3)
            noise = np.random.normal(0, sigma, 3)
            
        # Add noise to original points
        noisy_point = original_point + noise
        # Calculate actual MSE
        actual_mse = np.sum((noisy_point - original_point) ** 2)
        return noisy_point, actual_mse


class ImageProcessor:
    """Handles image processing and visualization"""

    def __init__(self, save_dir):
        self.bridge = CvBridge()
        self.setup_save_directories(save_dir)
        self.raw_imgs = []
        self.box_imgs = []

    def setup_save_directories(self, save_dir):
        """Set up directories for saving images"""
        self.img_dir = os.path.join(save_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.save_dir = save_dir

    def save_track_image(
        self, 
        img, 
        frame_num, 
        boxes=None, 
        filename="test", 
        show_window=False, 
        labels=None, 
        blurred_img=None, 
        dist_suc=None,
        angular_velocity=None,
        optimal_omega=None,
        gimbal_setpoint=None,
        gimbal_actual=None,
    ):
        """Save tracking visualization image"""
        img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.raw_imgs.append(img.copy())
        if blurred_img is not None:
            img = blurred_img.copy()

        if boxes is not None:
            cv2.putText(
                img,
                f"Frame: {frame_num}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if dist_suc is not None:
                cv2.putText(
                    img,
                    f"Distance to success: {dist_suc}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            if optimal_omega is not None:
                cv2.putText(
                    img,
                    f"Optimal Omega: [{optimal_omega[0]:.2f}, {optimal_omega[1]:.2f}, {optimal_omega[2]:.2f}]",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            if angular_velocity is not None and angular_velocity[0] is not None:
                cv2.putText(
                    img,
                    f"Angular Velocity: [{angular_velocity[0]:.2f}, {angular_velocity[1]:.2f}, {angular_velocity[2]:.2f}]",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            if gimbal_setpoint is not None and gimbal_setpoint[0] is not None:
                cv2.putText(
                    img,
                    f"Gimbal Setpoint: [{gimbal_setpoint[0]:.2f}, {gimbal_setpoint[1]:.2f}, {gimbal_setpoint[2]:.2f}]",
                    (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            if gimbal_actual is not None and gimbal_actual[0] is not None:
                cv2.putText(
                    img,
                    f"Gimbal Actual: [{gimbal_actual[0]:.2f}, {gimbal_actual[1]:.2f}, {gimbal_actual[2]:.2f}]",
                    (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            # Draw bounding boxes on the image
            for i, box in enumerate(boxes):
                # box = center2bbox(box)
                if box[0] is None:
                    cv2.putText(
                        img,
                        f"{labels[i]} lost",
                        (50, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    continue
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    get_color(i),
                    2,
                )
                if labels is not None and i < len(labels):
                    cv2.putText(
                        img,
                        labels[i],
                        (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        get_color(i),
                        2,
                    )
                    cv2.putText(
                        img,
                        labels[i],
                        (10, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        get_color(i),
                        2,
                    )
            # Save the image with bounding boxes
            # cv2.imwrite(
            #     os.path.join(self.img_dir, f"{filename}_{frame_num}.jpg"),
            #     img,
            # )
            self.box_imgs.append(img.copy())

        if show_window:
            window_name = f"{filename} - Frame {frame_num}"
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
    
    def save_image_data(self):
        """Save raw images as a video"""
        if len(self.raw_imgs) == 0:
            return
        
        # use mp4 instead of avi
        video_path = os.path.join(self.save_dir, "raw_video.mp4")
        height, width, _ = self.raw_imgs[0].shape
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height)
        )
        for img in self.raw_imgs:
            out.write(img)
        out.release()
        print(f"Saved raw images to {video_path}")

        # Save box images as a separate video
        if len(self.box_imgs) == 0:
            return
        video_path = os.path.join(self.save_dir, "box_video.mp4")
        height, width, _ = self.box_imgs[0].shape
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height)
        )
        for img in self.box_imgs:
            out.write(img)
        out.release()
        print(f"Saved box images to {video_path}")

        # video_path = os.path.join(self.save_dir, "raw_video.avi")
        # height, width, _ = self.raw_imgs[0].shape
        # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        # for img in self.raw_imgs:
        #     out.write(img)
        # out.release()

        # video_path = os.path.join(self.save_dir, "box_video.avi")
        # height, width, _ = self.box_imgs[0].shape
        # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        # for img in self.box_imgs:
        #     out.write(img)
        # out.release()
        # print(f"Saved raw video to {video_path}")

    def add_blur(frame, omega):
        import math
        image_width = 720
        fov = 2
        focal_len = image_width / (2 * math.tan(fov/2))
        shutter_speed = 1/60

        shift = focal_len * np.linalg.norm(np.array(omega[:2]) * shutter_speed)

        magn = max(3, int(shift * 2) | 1)
        angle = np.degrees(math.atan2(omega[1], omega[0]))
        kernel = np.zeros((magn, magn), dtype=np.float32)

        c = magn // 2
        dx = int(c * math.cos(math.radians(angle)))
        dy = int(c * math.sin(math.radians(angle)))
        cv2.line(kernel, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
        kernel /= kernel.sum()

        blurred = cv2.filter2D(frame, -1, kernel)

        # Rotational motion blur
        shift = abs(focal_len * omega[2] * shutter_speed)
        steps = max(3, int(shift * 2) | 1)

        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        angles = np.linspace(-omega[2] * shutter_speed * 90 / np.pi, omega[2] * shutter_speed * 90 / np.pi, steps)
        acc = np.zeros_like(blurred, dtype=np.float32)

        for theta in angles:
            M = cv2.getRotationMatrix2D(center, theta, 1.0)
            acc += cv2.warpAffine(blurred, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)

        blurred = np.clip(acc / steps, 0, 255).astype(np.uint8)

        return blurred

class SocialForceModel:
    """Handles social force modeling between agents"""

    def __init__(self, config_file="config/socialForce/social_force.toml"):
        self.model = None
        self.config_file = config_file
        self.enabled = True
        self.max_delta_speed = 0.0
        self.state_history = []
        self.distance_history = []
        self.timestamp_history = []

    def initialize(self, ped_states):
        """Initialize social force model with initial states"""
        self.model = psf.Simulator(
            state=ped_states,
            config_file=self.config_file,
        )

        # Check if social forces are enabled
        config = self.model.forces[1].config
        if config("factor", 0) == 0:
            self.enabled = False

        return self.enabled

    def get_state(self, state: TwistStamped):
        """Update victim state from TwistStamped message"""
        # Extract pose and velocity from message
        pose = [
            state.twist.angular.x,
            state.twist.angular.y,
            state.twist.angular.z,
        ]
        speed = [
            state.twist.linear.x,
            state.twist.linear.y,
            state.twist.linear.z,
        ]
        goal = np.array(pose[:2]) + np.array(speed[:2]) * 10

        ped_state = np.array([pose[0], pose[1], speed[0], speed[1], goal[0], goal[1]])
        return ped_state

    def update(self, ped_states):
        """Update the social force model and get new states"""
        if not self.enabled or self.model is None:
            return None

        vic_pose = ped_states[0][:2]
        vic_speed = ped_states[0][2:4]
        atk_pose = ped_states[1][:2]

        # Update pedestrian states
        self.model.peds.update(state=ped_states, groups=None)
        self.model.peds.update_initial_speeds(1)  # Update attacker's initial speed

        # Step the simulation
        self.model.step(1)

        # Get updated states
        psf_states, _ = self.model.peds.get_states()
        curr_state = psf_states[-1]
        vic_state, atk_state = curr_state[0], curr_state[1]
        self.state_history.append(vic_state.tolist())

        # Track maximum speed change
        delta = np.array(vic_speed[:2]) - np.array(vic_state[2:4])
        self.max_delta_speed = max(self.max_delta_speed, np.linalg.norm(delta))

        # Track distance between victim and attacker
        distance = np.linalg.norm(
            np.array(vic_pose[:2]) - np.array(atk_pose[:2])
        ).item()
        self.distance_history.append(distance)

        return vic_state

    def create_velocity_message(self, vic_state):
        """Create velocity message from social force state"""
        vic_vel = TwistStamped()
        vic_vel.header.stamp = None  # Will be set by caller
        vic_vel.header.frame_id = "world"
        vic_vel.twist.linear.x = vic_state[2].item()
        vic_vel.twist.linear.y = vic_state[3].item()
        vic_vel.twist.linear.z = 0.0
        vic_vel.twist.angular.x = vic_state[0].item()
        vic_vel.twist.angular.y = vic_state[1].item()
        vic_vel.twist.angular.z = 0.0

        return vic_vel

    def save_social_force_data(self, save_dir):
        """Save social force data to file"""
        for name in ["state_history", "distance_history", "timestamp_history"]:
            np.save(
                os.path.join(save_dir, f"{name}.npy"),
                getattr(self, name),
            )

class AttackManager(Node):
    """Main node class for managing the attack simulation"""

    def __init__(
        self,
        save_dir: str,
        atk_duration: int,
        world_file: str,
        atker_dist: float,
        tracker_type: str,
        surrogate_model: str,
        victim_velocity: list,
        **kwargs,
    ):
        super().__init__("attack_manager")

        self.id2name = {
            "10": "victim",
            "20": "attacker",
        }  # the vic and atk here are the pedestrians
        self.name2id = {v: k for k, v in self.id2name.items()}

        # Setup trackers
        self.victim_vehicle = SimStates(
            self.id2name, tracker_type=tracker_type, surrogate_model=surrogate_model, 
            atk_cycle=kwargs.get("atk_cycle", 2.03), gim_max_speed=kwargs.get("gim_max_speed", [1.0, 1.0, 2.0]),
            victim_model=kwargs.get("victim_model", None), atker_model=kwargs.get("atker_model", None),
            world_file=world_file, uav_height=kwargs.get("uav_height", 8.0), fps=kwargs.get("fps", 30),
        )  # vic uav
        self.attacker_machine = self.victim_vehicle  # white box attack for now

        # Setup cameras
        self.main_camera = CameraSetup("x500_gimbal_0/camera_link/camera")

        # Setup attack state
        self.lose_target_count = 0
        self.reset_count = 0
        self.surrogate_fail_count = 0
        self.frame_num = 0
        self.attack_started = False
        self.attacker_inplace_frame = -1
        self.start_frame = -1
        self.start_buffer = 0 # number of frames to wait for followme to stabilize
        self.start_sim_time = None
        self.optimal_omega_seq = None
        self.opt_iteration_trace = None
        self.optimal_offset_seq = None
        self.execution_frame_left = 0
        self.current_time_cam = None
        self.current_time_img = None
        self.switched = False  # whether the attack has switched to the attacker
        self.switch_frame = -1  # frame when the attack switched to the attacker
        self.dist_to_success = -1.0  # distance to success
        self.sort_init_buffer = 20 # number of frames to wait for SORT to initialize
        self.check_suc = 0 # number of frames to check for success after attack execution
        
        self.atk_duration = atk_duration
        self.fps = kwargs.get("fps", 30)
        self.world_file = world_file
        self.atker_dist = atker_dist
        self.atker_angle = kwargs.get("atker_angle", 90.0)
        self.tracker_type = tracker_type
        self.surrogate_model = surrogate_model
        self.victim_velocity = victim_velocity

        # Setup tracking data
        self.main_timestamp_history = []
        self.vic_pose_history = []
        self.atk_pose_history = []
        self.malicious_camera_offset_history = []
        self.omega_history = []
        self.gimbal_setpoint_history = []
        self.iteration_trace_history = []
        self.surrogate_pred_history = []
        self.main_camera_pose_history = []
        self.main_uav_pose_history = []
        self.main_vic_pose_history = []
        self.main_atk_pose_history = []
        self.estimated_vic_atk_velocity_history = []
        self.estimated_uav_velocity_history = []
        self.coord_transform_time_history = []
        self.surr_time_history = []
        self.opt_time_history = []

        # Setup image processor
        self.image_processor = ImageProcessor(save_dir)

        self.sf_model = SocialForceModel()
        
        # Setup Gazebo transport controller
        self.init_pitch = -0.5 if kwargs["uav_height"] < 8.0 else -1.0
        self.gz_controller = GzTransportController(world_file)
        self.gz_controller.resume_simulation()
        self.gz_clock = GzClockSubscriber()
        self.real_time_factor = kwargs.get("real_time_factor", 0.5)  # simulationTime/realTime
        self.attack_interval_method = kwargs.get("attack_interval_method", "simdt")
        self.gz_gimbal_controller = GzGimbalController(
            self.gz_clock, 
            self.init_pitch, 
            real_time_factor=self.real_time_factor, 
            attack_interval_method=self.attack_interval_method
        )
        self.skip_initial_frames = 0  # skip initial frames to allow system to stabilize
        self.last_optimize_frame = -1

        # Setup logfile
        self.save_dir = save_dir
        self.log_path = os.path.join(save_dir, "log.txt")
        self.log_file = open(self.log_path, "w")
        self._setup_dirs()

        # Setup ROS subscriptions
        self._setup_subscriptions()

        # Setup ROS publishers
        self._setup_publishers()

        # Initialize node
        self.log("Initialized")
        time.sleep(5)  # Wait for everything to be ready

        # Send start signal
        # self.start_pub.publish(Bool(data=True))

    def _setup_subscriptions(self):
        """Setup ROS subscriptions"""
        # Detection and camera image subscriptions
        boxes2d_sub = Subscriber(self, Detection2DArray, "/boxes_visible_2d")
        boxes3d_sub = Subscriber(self, Detection3DArray, "/boxes_3d")
        image_sub = Subscriber(self, Image, "/camera")
        ts = TimeSynchronizer([boxes2d_sub, boxes3d_sub, image_sub], 1)
        ts.registerCallback(self.synced_callback)

        # Camera info subscriptions
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera_info", self.camera_info_callback, 1
        )
        self.camera_pose_sub = self.create_subscription(
            Pose, "/camera/pose", self.camera_pose_callback, 1
        )
        self.mat_sub = self.create_subscription(
            Float64MultiArray, "/camera/transform_matrix", self.mat_callback, 1
        )
        self.uav_pose_sub = self.create_subscription(
            Pose, "/mount/pose", self.uav_pose_callback, 1
        )

        # Agent state subscriptions
        vic_state_sub = Subscriber(self, TwistStamped, "/actor_walking/pose_vel")
        atk_state_sub = Subscriber(self, TwistStamped, "/actor_following/pose_vel")
        self.synced_pose = ApproximateTimeSynchronizer(
            [vic_state_sub, atk_state_sub], 10, 0.05
        )
        self.synced_pose.registerCallback(self.synced_pose_callback)

    def _setup_publishers(self):
        """Setup ROS publishers"""
        # Victim's bbox error publisher
        self.vic_gimbal_pub = self.create_publisher(Point, "object/gimbal", 1)
        self.vic_followme_pub = self.create_publisher(Point, "object/follow", 1)
        self.gimbal_perturb_pub = self.create_publisher(Point, "/stability_atk", 1)
        self.stop_pub = self.create_publisher(Bool, "/stop_movement/actor_following", 1)
        self.start_pub = self.create_publisher(Bool, "/start_movement", 1)
        self.start_pub_atk = self.create_publisher(Bool, "/start_movement/atker", 1)
        self.atk_pose_pub = self.create_publisher(
            PoseStamped, "/actor_following/atk_pose", 1
        )  # Publish attacker's pose for visualization
        self.vic_sf_pub = self.create_publisher(
            TwistStamped, "/actor_walking/social_force", 1
        )

    def log(self, message):
        """Log message to console and file"""
        print(message)
        self.log_file.write(message + "\n")

    def camera_info_callback(self, msg):
        """Handle camera info messages"""
        self.main_camera.update_camera_info(msg)

    def camera_pose_callback(self, msg):
        """Handle camera pose messages"""
        self.main_camera.update_camera_pose(msg, self.get_clock().now(), self.gz_clock.get_sim_time())

    def mat_callback(self, mat_msg):
        """Handle transformation matrix messages"""
        self.main_camera.update_transformation_matrix(mat_msg)
        
    def uav_pose_callback(self, msg):
        """Handle UAV pose messages"""
        # basically called at every simulation step - 0.004s.
        # if self.current_time_cam is not None:
        #     print(f"UAV | time passed from last call: {(self.get_clock().now().nanoseconds - self.current_time_cam[0].nanoseconds) / 1e9:.3f} s")
        #     print(f"UAV | sim time: {self.gz_clock.get_sim_time() - self.current_time_cam[1]:.3f} s")
        self.current_time_cam = [self.get_clock().now(), self.gz_clock.get_sim_time()]
        self.uav_pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        # Calculate Euler angles
        roll, pitch, yaw = quat2euler(
            [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z],  # wxyz
            axes="sxyz",
        )
        self.uav_pose.extend([roll, pitch, yaw])

    def synced_pose_callback(self, vic: TwistStamped, atk: TwistStamped):
        """Handle synchronized victim and attacker state messages"""
        ros_time = self.get_clock().now()
        sim_time = self.gz_clock.get_sim_time()
        # .twist.linear/angular -> linear velocity/current position
        self.vic_pose_history.append(
            [vic.twist.angular.x, vic.twist.angular.y, vic.twist.angular.z, ros_time.nanoseconds, sim_time]
        )
        self.atk_pose_history.append(
            [atk.twist.angular.x, atk.twist.angular.y, atk.twist.angular.z, ros_time.nanoseconds, sim_time]
        )

        # Initialize social force model if needed
        vic_data = self.sf_model.get_state(vic)
        atk_data = self.sf_model.get_state(atk)
        ped_states = np.array([vic_data, atk_data])
        self.vic_pose = vic_data[:2]

        if self.sf_model.model is None:
            self.sf_model.initialize(ped_states)
            if not self.sf_model.enabled:
                return

        # Update social force model
        vic_sf_state = self.sf_model.update(ped_states)
        if vic_sf_state is None:
            return

        # Create and publish velocity message
        vic_vel = self.sf_model.create_velocity_message(vic_sf_state)
        vic_vel.header.stamp = self.get_clock().now().to_msg()
        self.vic_sf_pub.publish(vic_vel)

        # Record timestamp
        self.sf_model.timestamp_history.append((self.get_clock().now(), self.gz_clock.get_sim_time()))

    def vic_atk_velocity(self):
        """Calculate the velocity of the victim and attacker"""
        if len(self.vic_pose_history) < 10 or len(self.atk_pose_history) < 10 or self.optimal_omega_seq is None:
            return {
                "victim_velocity": [0.0, 0.0, 0.0],
                "attacker_velocity": [0.0, 0.0, 0.0],
            }
        
        # Calculate velocity as the difference between the last two positions
        vic_pos = np.array(self.vic_pose_history[-1][:3])
        atk_pos = np.array(self.atk_pose_history[-1][:3])
        prev_vic_pos = np.array(self.vic_pose_history[-10][:3])
        prev_atk_pos = np.array(self.atk_pose_history[-10][:3])

        vic_velocity = (vic_pos - prev_vic_pos) / (self.vic_pose_history[-1][-1] - self.vic_pose_history[-10][-1])
        atk_velocity = (atk_pos - prev_atk_pos) / (self.atk_pose_history[-1][-1] - self.atk_pose_history[-10][-1])
        vic_velocity[2] = 0.0  # Ignore vertical velocity
        atk_velocity[2] = 0.0  # Ignore vertical velocity

        return {
            "victim_velocity": vic_velocity.tolist(),
            "attacker_velocity": atk_velocity.tolist(),
        }

    
    def get_uav_velocity(self):
        """Calculate the velocity of the UAV"""
        if len(self.main_timestamp_history) < 5:
            return {
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0]
            }

        curr_idx = len(self.main_timestamp_history) - 5
        time_diff = self.main_timestamp_history[-1][1] - self.main_timestamp_history[curr_idx][1]

        uav_velocity = np.array(self.main_uav_pose_history[-1]) - np.array(self.main_uav_pose_history[curr_idx])
        uav_velocity /= time_diff
        
        if np.linalg.norm(uav_velocity) > 5: # filter out incorrect gazebo measurements 
            uav_velocity = np.zeros(6)
        
        print(f"UAV velocity: {uav_velocity[:3]}, Angular velocity: {uav_velocity[-3:]} | {time_diff} s | selected idx: {curr_idx}")
        return {
            "linear_velocity": uav_velocity[:3].tolist(),
            "angular_velocity": uav_velocity[-3:].tolist()
        }

    def check_detection(self, msg2d, msg3d):
        """Check if both detections are present"""
        det2d = parse_detection2d_array(msg2d)
        det3d = parse_detection3d_array(msg3d)
        if (
            self.name2id["victim"] not in det2d
            or self.name2id["attacker"] not in det2d
            or self.name2id["victim"] not in det3d
            or self.name2id["attacker"] not in det3d
        ):
            self.log("No victim or attacker detection found")
            return False, det2d, None
        return True, det2d, det3d

    def check_switch(self, pred_box, atk_box, vic_box, vic_w_conf):
        if pred_box[0] is None or atk_box[0] is None: # pred_box[0] can be None when using SORT
            self.log(f"Predicted box or attack box is None at frame {self.frame_num}")
            if self.switched:
                self.log(f"Hijacking lost at frame {self.frame_num}")
                if self.frame_num - self.switch_frame > 15:
                    self.stop_ros_node()
                self.switched = False
                self.switch_frame = -1
            return False
        
        iou = get_iou(np.array([pred_box]), np.array([atk_box]))
        iou_vic = get_iou(np.array([pred_box]), np.array([vic_box]))
        
        pred_center = np.array([(pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2])
        atk_center = np.array([(atk_box[0] + atk_box[2]) / 2, (atk_box[1] + atk_box[3]) / 2])
        self.dist_to_success = np.linalg.norm(pred_center - atk_center)
        # if self.dist_to_success < 40:  # 40 pixels
        self.log(f"Frame {self.frame_num} | IoU: {iou[0]:.2f} | IoU vic: {iou_vic[0]:.2f} | Distance to success: {self.dist_to_success:.2f} pixels | Victim confidence: {vic_w_conf}")
        if iou[0] > 0.3 and iou[0] > iou_vic[0]:  # attack box overlaps with predicted box and has higher IoU than victim box
            if not self.switched:
                self.log(f"Hijacking successful at frame {self.frame_num}")
                self.switched = True
                self.switch_frame = self.frame_num
                self.start_pub_atk.publish(Bool(data=False))
            else:
                if self.frame_num - self.switch_frame > 10:
                    print(f"Hijacking maintained for {self.frame_num - self.switch_frame} frames, stopping simulation")
                    self.stop_ros_node()
            return True
        if self.switched:
            self.log(f"Hijacking lost at frame {self.frame_num}")
            if self.frame_num - self.switch_frame > 10: 
                self.stop_ros_node()
            self.switched = False
            self.switch_frame = -1
        return False
        
    
    def has_init_followme(self):
        # follow-me system #
        # self.gz_gimbal_controller.followme_gimbal_angles(followme_msg) # follow-me gimbal control
        # start attack simulation when the victim is centered
        if not self.attack_started:
            # followme_msg = self.victim_vehicle.followme_msg(det2d[self.name2id["victim"]], self.main_camera.info)
            followme_msg = self.victim_vehicle.followme_msg(self.victim_vehicle.pred_history[self.name2id["victim"]][-1], self.main_camera.info)
            print(f"Followme error: {followme_msg.x:.1f}, {followme_msg.y:.1f}, {followme_msg.z:.1f}")
            # if abs(followme_msg.x) < 20 and abs(followme_msg.y) < 20: 
            followme_msg.x = 0.0
            followme_msg.y = 0.0
            followme_msg.z = 0.0
            self.attack_started = True
            self.start_frame = self.frame_num
            self.start_sim_time = self.gz_clock.get_sim_time()
            self.log(f"Followme stabilized at frame {self.frame_num}.")
            self.vic_followme_pub.publish(followme_msg)
            return False
        if self.frame_num < self.start_frame + self.start_buffer:
            error = Point()
            error.x, error.y, error.z = 0.0, 0.0, 0.0
            self.vic_followme_pub.publish(error)
            return False
        return True
        
    def synced_callback(
        self, msg2d: Detection2DArray, msg3d: Detection3DArray, img: Image
    ):
        """Process synchronized detection and camera info messages"""
        print(f"\n--- Frame {self.frame_num} ---")
        sim_time = self.gz_clock.get_sim_time()
        ros_time = self.get_clock().now()

        if self.current_time_img is not None:
            sim_time_passed = sim_time - self.current_time_img[1]
            ros_time_passed = (ros_time.nanoseconds - self.current_time_img[0].nanoseconds) / 1e9
            self.log(
                f"Processing frame {self.frame_num} at {ros_time_passed:.3f} s. "
                f"Sim time passed: {sim_time_passed:.3f} s"
            )
            if sim_time_passed < 0.010:
                self.log(f"Skipping frame {self.frame_num} due to small time difference")
                return
            sim_time_to_move = sim_time - self.current_time_img[1]
        else:
            sim_time_to_move = 0.0
        self.current_time_img = [ros_time, sim_time]
        
        if len(self.vic_pose_history) == 0:
            self.log("No victim pose history yet")
            return
        # move the attacker in parallel with the victim
        if not self.switched:
            self.start_pub_atk.publish(Bool(data=True))
            atker_pose = self.victim_vehicle.attacker_pose_msg(
                self.main_camera.pose_wo_noise_history[-1], self.vic_pose_history[-1], self.atk_pose_history[-1], dist=self.atker_dist, victim_velocity=self.victim_velocity,
                angle=self.atker_angle, motion_model=self.attack_started, sim_time_to_move=sim_time_to_move
            )
            # print(f"Moving attacker to position: {atker_pose.pose.position.x:.2f}, {atker_pose.pose.position.y:.2f}, {atker_pose.pose.position.z:.2f}")
            self.atk_pose_pub.publish(atker_pose) # attacker move to vulnerable position
            if self.attacker_inplace_frame < 0:
                self.attacker_inplace_frame = self.frame_num
        
        if self.skip_initial_frames < 30:
            self.skip_initial_frames += 1
            return
        
        # Ensure camera info is available
        if (
            self.main_camera.info is None
            or self.main_camera.mat is None
            or self.main_camera.pose is None
        ):
            self.log("No camera info or trans matrix yet")
            return
        
        # ensure objects are within the camera view
        valid, det2d, det3d = self.check_detection(msg2d, msg3d)
        if not valid:
            self.lose_target_count += 1
            if self.lose_target_count > 5:
                self.gz_gimbal_controller.reinitialize_gimbal() # fix simulation spikes
                self.lose_target_count = 0
                self.reset_count += 1
            if self.reset_count > 5:
                self.log("Too many resets, stopping simulation")
                self.stop_ros_node()
            # if self.lose_target_count > self.fps: # lose either of the target for 1 second
            #     self.log(f"Vic: Lost target for {self.lose_target_count} frames")
            #     self.stop_ros_node()
            return
        
        self.victim_vehicle.set_current_det(det2d, det3d, self.attack_started)
        coord_trans_start = time.time()
        self.victim_vehicle.update_coordinates_position(
            det3d, self.main_camera.camera_model
        )  # bbox from 3d detection
        coord_trans_end = time.time()
        self.coord_transform_time_history.append(coord_trans_end - coord_trans_start)
        # self.log(f"Coord transform time: {coord_trans_end - coord_trans_start:.3f} s")

        blurred_img = None
        update_pred_start = time.time()
        if self.optimal_omega_seq is not None and len(self.optimal_omega_seq) > 0 and not self.switched:
            blurred_img = ImageProcessor.add_blur(
                self.image_processor.bridge.imgmsg_to_cv2(img, "bgr8"),
                self.optimal_omega_seq[0],
            )
            self.victim_vehicle.update_pred_position(blurred_img, attack_started=self.attack_started,
                                                     camera=self.main_camera)
        else:
            self.victim_vehicle.update_pred_position(
                self.image_processor.bridge.imgmsg_to_cv2(img, "bgr8"), attack_started=self.attack_started,
                camera=self.main_camera,
            )
        update_pred_end = time.time()
        # self.log(f"Update pred time: {update_pred_end - update_pred_start:.3f} s")
        world_corners, world_center = self.victim_vehicle.update_world_position(
            det3d, self.main_camera.mat, static_camera=False
        )
        
        vic_box = self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="x1y1x2y2")
        atk_box = self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="x1y1x2y2")
        # vic_box = self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="x1y1x2y2", wscale=1.5, hscale=1.5)
        # atk_box = self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="x1y1x2y2", wscale=0.7, hscale=1.0)
        vic_pred_box = self.victim_vehicle.pred_history[self.name2id["victim"]][-1]
        atk_pred_box = self.victim_vehicle.pred_history[self.name2id["attacker"]][-1]
        vic_w_conf = self.victim_vehicle.pred_w_conf_history[self.name2id["victim"]][-1]
        atk_w_conf = self.victim_vehicle.pred_w_conf_history[self.name2id["attacker"]][-1]
        
        boxes = [vic_w_conf[:4], atk_w_conf[:4]]
        labels = [f"vic conf {vic_w_conf[4]:.2f}", f"atk conf {atk_w_conf[4]:.2f}"]
        if self.tracker_type == "SORT" or self.tracker_type == "UCMCTrack":
            all_pred_trks = self.victim_vehicle.all_pred_trks[-1] # [N, 5]. x1y1x2y2id
            all_pred_trks_boxes = [] # [box[:4] for box in all_pred_trks]
            all_pred_trks_labels = [] # [f"id {int(box[4])}" for box in all_pred_trks]
            boxes.extend([vic_pred_box, atk_pred_box])
            labels.extend(["vic pred", "atk pred"])
            for i, box in enumerate(all_pred_trks):
                if box[4] in [0, 1, 2, 3]:  # only keep victim and attacker (sometimes the detector detect an extra person with id 2 or 3)
                    all_pred_trks_boxes.append(box[:4])
                    all_pred_trks_labels.append(f"id {int(box[4])}")
            boxes = boxes + all_pred_trks_boxes
            labels = labels + all_pred_trks_labels
        
        atk_box = self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="x1y1x2y2")
        vic_box = self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="x1y1x2y2")
        boxes.extend([vic_box, atk_box])
        labels.extend(["vic gt", "atk gt"])

        self.image_processor.save_track_image(
            img,
            self.frame_num,
            boxes=boxes,
            filename="image",
            show_window=False,
            labels=labels,
            blurred_img=blurred_img,
            # dist_suc=self.dist_to_success,
            # angular_velocity=omega,
            # optimal_omega=self.optimal_omega_seq[0] if self.optimal_omega_seq is not None and len(self.optimal_omega_seq) > 0 else None,
            # gimbal_setpoint=gimbal_setpoint,
            # gimbal_actual=self.main_camera.pose[7:10] if self.main_camera.pose is not None else None,
        )
        
        surr_start = time.time()
        try:
            res = self.victim_vehicle.update_surrogate_tracker(
                self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="x1y1x2y2"),
                self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="x1y1x2y2"),
            )
        except Exception as e:
            self.log(f"Surrogate tracker update failed at frame {self.frame_num}: {e}")
            res = None
            self.stop_ros_node()
            
        surr_end = time.time()
        self.surr_time_history.append(surr_end - surr_start)
        # self.log(f"Surrogate update time: {surr_end - surr_start:.3f} s")
        surrogate_box = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x1y1x2y2 + best_score + best_pscore
        if res is not None:
            if isinstance(res['bbox'], np.ndarray):
                surrogate_box = res['bbox'].tolist()
            elif isinstance(res['bbox'], list):
                surrogate_box = res['bbox']
            elif isinstance(res['bbox'], tf.Tensor):
                surrogate_box = res['bbox'].numpy().tolist()
            surrogate_box.append(res['best_score'])
            surrogate_box.append(res['best_pscore'])
            self.surrogate_pred_history.append(surrogate_box)
        else:
            self.surrogate_pred_history.append([-1, -1, -1, -1, -1, -1])
            self.surrogate_fail_count += 1
            if self.surrogate_fail_count > 20:
                self.log("Surrogate tracker failed too many times, stopping simulation")
                self.stop_ros_node()
            return # wait until surrogate tracker is ready

        if (self.tracker_type == "SORT" or self.tracker_type == "UCMCTrack") and self.frame_num < self.sort_init_buffer:
            self.log(f"Warming up SORT tracker at frame {self.frame_num}")
            self.frame_num += 1
            return
        
        omega = [None, None, None]
        malicious_camera_offset = [None, None, None]
        gimbal_setpoint = [None, None, None]
        estimated_vic_atk_velocity = {
            "victim_velocity": [None, None, None],
            "attacker_velocity": [None, None, None],
        }
        estimated_uav_velocity = {
            "linear_velocity": [None, None, None],
            "angular_velocity": [None, None, None],
        }
        
        if self.has_init_followme():
            # simulate acoustic attack #
            print(f"Frame {self.frame_num}.")

            pred_box = self.victim_vehicle.pred_history[self.name2id["victim"]][-1]
            vic_w_conf = self.victim_vehicle.pred_w_conf_history[self.name2id["victim"]][-1]
            atk_box = self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="x1y1x2y2")
            vic_box = self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="x1y1x2y2")
            self.check_switch(pred_box, atk_box, vic_box, vic_w_conf)
                
            # continue followme during the attack
            if self.victim_vehicle.pred_w_conf_history[self.name2id["victim"]][-1][4] > 0.5:
                followme_msg = self.victim_vehicle.followme_msg(self.victim_vehicle.pred_history[self.name2id["victim"]][-1], self.main_camera.info)
                # self.vic_followme_pub.publish(followme_msg)
            else:
                followme_msg = Point()
                followme_msg.x, followme_msg.y, followme_msg.z = 0.0, 0.0, 0.0
                self.vic_followme_pub.publish(followme_msg)
                self.log("Victim lost confidence in tracking, stopping follow-me")
                    
            if self.execution_frame_left == 0 and not self.switched and self.check_suc == 0:
                self.log("Pausing simulation to optimize attack...")
                self.start_pub.publish(Bool(data=False))
                self.start_pub_atk.publish(Bool(data=False))

                self.gz_gimbal_controller.stop_motion()
                self.gz_controller.pause_simulation()
                estimated_vic_atk_velocity = self.vic_atk_velocity()
                estimated_uav_velocity = self.get_uav_velocity()
                try:
                    self.optimal_omega_seq, self.opt_iteration_trace, self.optimal_offset_seq = self.victim_vehicle.optimize_attack(
                        self.main_camera.pose, 
                        self.victim_vehicle.world_center, 
                        [
                            self.victim_vehicle.get_bbox_from_detection("victim", source="3d", box_format="xywh"),
                            self.victim_vehicle.get_bbox_from_detection("attacker", source="3d", box_format="xywh"),
                        ],
                        estimated_vic_atk_velocity, 
                        estimated_uav_velocity, 
                        self.frame_num, 
                        self.save_dir, 
                        self.log,
                    )
                    self.gz_gimbal_controller.set_wg(self.optimal_offset_seq[0])
                    self.gz_gimbal_controller.start_motion()
                    self.last_optimize_frame = self.frame_num
                    self.execution_frame_left = self.victim_vehicle.config.frames_per_interval
                    self.check_suc = 1

                except Exception as e:
                    # raise e
                    self.log(f"Optimization failed: {e}")
                    self.stop_ros_node()
                    return
                self.iteration_trace_history.append(self.opt_iteration_trace)
                self.gz_controller.resume_simulation()
                self.start_pub.publish(Bool(data=True))
                self.start_pub_atk.publish(Bool(data=True))
                self.log("Resuming simulation after optimization...")
                
            if self.execution_frame_left > 0 and not self.switched:
                # omega, malicious_camera_offset, gimbal_setpoint = self.gz_gimbal_controller.control_callback(optimal_velocity=self.optimal_omega_seq[0])
                omega, malicious_camera_offset, gimbal_setpoint = self.gz_gimbal_controller.control_callback() # use realistic acoustic gimbal attack simulation
                self.execution_frame_left -= 1
            elif self.check_suc > 0:
                self.check_suc -= 1

        self.main_timestamp_history.append([ros_time.nanoseconds, sim_time])
        self.malicious_camera_offset_history.append(malicious_camera_offset)
        self.omega_history.append(omega)
        self.gimbal_setpoint_history.append(gimbal_setpoint)
        self.main_camera_pose_history.append(self.main_camera.pose)
        self.main_uav_pose_history.append(self.uav_pose)
        self.main_vic_pose_history.append(self.vic_pose_history[-1][:3])
        self.main_atk_pose_history.append(self.atk_pose_history[-1][:3])
        self.estimated_vic_atk_velocity_history.append(
            estimated_vic_atk_velocity["victim_velocity"] + estimated_vic_atk_velocity["attacker_velocity"]
        )
        self.estimated_uav_velocity_history.append(
            estimated_uav_velocity["linear_velocity"] + estimated_uav_velocity["angular_velocity"]
        )

        # Publish gimbal perturbation

        
        
        if self.attack_started:
            atk_timeout_frame = (self.frame_num > self.start_frame + self.atk_duration + self.start_buffer)
            atk_timeout_time = (sim_time - self.start_sim_time) > (self.atk_duration * self.fps)
            if atk_timeout_frame or atk_timeout_time:
                self.log("Time out. Stopping simulation")
                self.stop_ros_node()

        # Increment frame counter
        self.frame_num += 1
        if self.frame_num % 10 == 0:
            self.log(f"Simulation running at frame {self.frame_num}")
            self.log(f"Lost target for {self.lose_target_count} frames")
            np_ts = np.array(self.main_timestamp_history)
            self.log(
                f"Average time interval: {np.mean(np.diff(np_ts[:, 1]))} s"
            )
            self.log(
                f"Attack duration: {(np_ts[-1][1] - np_ts[0][1])} s"
            )
            self.log("")

    def _setup_dirs(self):
        """Set up directories for saving images and logs"""
        self.vic_vehicle_dir = os.path.join(self.save_dir, "vic_vehicle")
        # self.atk_machine_dir = os.path.join(self.save_dir, "atker_machine")
        os.makedirs(self.vic_vehicle_dir, exist_ok=True)
        # os.makedirs(self.atk_machine_dir, exist_ok=True)

    def stop_ros_node(self):
        """Save log file and stop the ROS node"""
        # save tracking data
        self.victim_vehicle.save_tracking_data(self.vic_vehicle_dir)
        # self.attacker_machine.save_tracking_data(self.atk_machine_dir)
        self.main_camera.save_camera_data(self.save_dir, "0")
        self.image_processor.save_image_data()
        # self.sf_model.save_social_force_data(self.save_dir)
        self.gz_gimbal_controller.profiled_motion_models.save_traces(self.save_dir)
        for name in [
            "main_timestamp_history", 
            "vic_pose_history", 
            "atk_pose_history", 
            "malicious_camera_offset_history", 
            "omega_history", 
            "gimbal_setpoint_history",
            "iteration_trace_history", 
            "surrogate_pred_history",
            "main_camera_pose_history",
            "main_uav_pose_history",
            "main_vic_pose_history",
            "main_atk_pose_history",
            "estimated_vic_atk_velocity_history",
            "estimated_uav_velocity_history",
        ]:
            np.save(
                os.path.join(self.save_dir, f"{name}.npy"),
                getattr(self, name),
            )

        self.log("Motion attack simulation result:")
        self.log(f"Simulation stop at frame {self.frame_num}")
        self.log(f"Coordinate transformation time per frame: {np.mean(self.coord_transform_time_history):.10f} s")
        self.log(f"Surrogate time per frame: {np.mean(self.surr_time_history):.10f} s")
        np_ts = np.array(self.main_timestamp_history)
        # self.log(f"Average attack interval: {np.mean(np.diff(np_ts[:, 1]))} s")
        # self.log(
        #     f"Attack duration: {(np_ts[-1][1] - np_ts[0][1])} s"
        # )
        self.log("ATTACK STOPPED")
        self.log_file.close()

        time.sleep(1)
        self.destroy_node()
        rclpy.shutdown()
        exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Attack Manager")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/Documents/exp/uavAtk",
        help="Directory to save the experiment results",
    )
    parser.add_argument(
        "--atk_duration",
        type=int,
        default=15,
        help="Duration of the attack in frames",
    )
    parser.add_argument(
        "--world_file",
        type=str,
        default="default",
        help="World file name",
    )
    parser.add_argument(
        "--atker_dist",
        type=float,
        default=5.0,
        help="Distance between attacker and victim",
    )
    parser.add_argument(
        "--atker_angle",
        type=float,
        default=90.0,
        help="Angle between attacker and victim in degrees",
    )
    # select only among "SORT", "SiamRPN", "KCF"
    parser.add_argument(
        "--tracker_type",
        choices=["SORT", "SiamRPN", "KCF", "DaSiamRPN", "UCMCTrack"],
        default="SiamRPN",
        help="Type of tracker to use",
    )
    parser.add_argument(
        "--surrogate_model",
        choices=["SORT", "SiamRPN", "KCF", "DaSiamRPN", "UCMCTrack"],
        default="SiamRPN",
        help="Type of surrogate model to use for the attack",
    )
    # victim velocity in x and y direction
    parser.add_argument(
        "--victim_velocity",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Victim velocity in x and y direction",
    )
    parser.add_argument(
        "--atk_cycle",
        type=float,
        default=2.03,
        help="Attack cycle time in frequency (Hz)"
    )
    parser.add_argument(
        "--gim_max_speed",
        type=float,
        nargs=3,
        default=[0.1, 2.0, 2.0],
        help="Gimbal maximum speed in pitch, roll, yaw (rad/s)",
    )
    parser.add_argument(
        "--victim_model",
        type=str,
        default="pedestrian1",
        help="Model name of the victim used in the simulation",
    )
    parser.add_argument(
        "--atker_model",
        type=str,
        default="pedestrian2",
        help="Model name of the attacker used in the simulation",
    )
    parser.add_argument(
        "--uav_height",
        type=float,
        default=8.0,
        help="Height of the UAV",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS",
    )
    parser.add_argument(
        "--real_time_factor",
        type=float,
        default=0.5,
        help="Real time factor for the simulation",
    )
    parser.add_argument(
        "--attack_interval_method",
        choices=["simdt", "1overfps"],
        default="simdt",
        help="Method to determine the time interval used in the gimbal injection attack simulation",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.expanduser(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    topic_manager = TopicBridgeManager()
    topic_manager.start_common_bridges()

    rclpy.init()
    attack_manager = AttackManager(
        save_dir=save_dir,
        atk_duration=args.atk_duration,
        world_file=args.world_file,
        atker_dist=args.atker_dist,
        tracker_type=args.tracker_type,
        surrogate_model=args.surrogate_model,
        victim_velocity=args.victim_velocity,
        atk_cycle=args.atk_cycle,
        gim_max_speed=args.gim_max_speed,
        victim_model=args.victim_model,
        atker_model=args.atker_model,
        uav_height=args.uav_height,
        fps=args.fps,
        real_time_factor=args.real_time_factor,
        attack_interval_method=args.attack_interval_method,
        atker_angle=args.atker_angle,
    )
    rclpy.spin(attack_manager)


if __name__ == "__main__":
    main()
