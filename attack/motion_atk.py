#!/usr/bin/env python3

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

import cv2
import time
import copy
import rclpy
import argparse
import subprocess
import numpy as np
import tensorflow as tf
import pysocialforce as psf
import gz.transport13 as gzt
import gz.msgs10.pose_v_pb2 as pose_v_pb2

# tf.config.set_visible_devices([], "GPU")

from typing import List, Dict
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler, euler2quat

from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Pose, PoseArray, Point, TwistStamped, PoseStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from vision_msgs.msg import Detection2DArray, Detection3DArray

from utils.sort_TF import KalmanBoxTracker
from utils.topic_bridge_manager import TopicBridgeManager
from attack.ID_Transfer import (
    optimize_3d_coord,
    iou_batch,
    project_box_3d_to_2d,
    project_box_camera_to_world,
    camera_stability_analysis,
    world_to_camera,
)


class TrackerObject:
    """Base class for tracking objects (victim or attacker)"""

    def __init__(self, id2name: dict, noise_3d=0.0, det_noise=0.0):
        self.id2name = id2name
        self.name2id = {v: k for k, v in id2name.items()}
        self.object_ids = list(id2name.keys())
        self.trackers = {obj_id: None for obj_id in self.object_ids}
        self.history = {obj_id: [] for obj_id in self.object_ids}
        self.pred_history = {obj_id: [] for obj_id in self.object_ids}
        self.world_history = {obj_id: [] for obj_id in self.object_ids}
        self.id_switch_history = []
        self.current_det = {}
        self.world_corners = {}
        self.world_center = {}

        # Apply random noise for 3D position
        self.noise_3d = np.random.normal(0, noise_3d, 2)
        if noise_3d > 0:
            self.noise_3d = self.noise_3d / np.linalg.norm(self.noise_3d) * noise_3d

        self.det_noise = det_noise

    def update_tracking(self, camera_model, occluded=False):
        """Update Kalman filter tracking with new bbox"""
        history = {}
        pred_history = {}
        for obj_id in self.object_ids:
            det = self.current_det[obj_id]
            _, bbox = project_box_3d_to_2d(det, camera_model)

            # Update actual tracker with potentially noisy input
            if self.det_noise > 0:
                bbox += tf.random.normal(bbox.shape, 0, self.det_noise)

            # Run Kalman filter
            self.trackers[obj_id], pred_bbox = self._kalman_filter(
                bbox, self.trackers[obj_id], occluded
            )

            if pred_bbox is not None:
                history[obj_id] = bbox
                pred_history[obj_id] = pred_bbox
            else:
                return False

        for obj_id in self.object_ids:
            self.history[obj_id].append(history[obj_id])
            self.pred_history[obj_id].append(pred_history[obj_id])
        return True

    def _kalman_filter(self, bbox, tracker: KalmanBoxTracker, occluded=False):
        """Run Kalman filter prediction and update"""
        if tracker is None:
            tracker = KalmanBoxTracker(bbox)
            pred_bbox = None
        elif occluded:
            pred_bbox = tracker.predict_no_trace()[0]
            tracker.predict()
            tracker.update(pred_bbox)
        else:
            pred_bbox = tracker.predict_no_trace()[0]
            tracker.predict()
            tracker.update(bbox)
        return tracker, pred_bbox

    def set_detection_3d(self, detections):
        """Set the current 3D detection data"""
        self.current_det = detections
        if self.noise_3d is not None:
            for obj_id in self.object_ids:
                # Apply noise to x,y position
                self.current_det[obj_id]["position"][0] += self.noise_3d[0]
                self.current_det[obj_id]["position"][1] += self.noise_3d[1]

    def update_world_position(self, matrix, static_camera=False):
        """Update world position using transformation matrix"""
        for obj_id in self.object_ids:
            corners, center = project_box_camera_to_world(
                self.current_det[obj_id], matrix, static_camera=static_camera
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
            world_history = np.array(self.world_history[obj_id])
            for name in [
                "history",
                "pred_history",
                "world_history",
            ]:
                np.save(
                    os.path.join(save_dir, f"{obj_id}_{name}.npy"),
                    locals()[name],
                )
        for name in ["id_switch_history"]:
            np.save(
                os.path.join(save_dir, f"{name}.npy"),
                getattr(self, name),
            )


class VictimTracker(TrackerObject):
    """Handles victim UAV's tracking and related functionality"""

    def __init__(self, id2name: dict, noise_3d=0.0, det_noise=0.0):
        super().__init__(id2name, noise_3d, det_noise)
        self.init_height = -1
        self.error_history = []
        self.curr_target_id = self.name2id[
            "victim"
        ]  # start with following the victim pedestrian

    def calculate_error(self, detection, camera_info):
        """Calculate error for victim's motion control"""
        # Center the Follow-me target in the camera frame and maintain initial height
        w, h = camera_info.width, camera_info.height
        if self.init_height < 0:
            self.init_height = detection["dimensions"][1]

        # Calculate error for UAV and gimbal motion control
        error = Point()
        error.x = detection["position"][0] - w / 2
        error.y = detection["position"][1] - h / 2
        error.z = detection["dimensions"][1] - self.init_height

        self.error_history.append([error.x, error.y, error.z])
        return error

    def save_tracking_data(self, save_dir):
        """Save tracking data to file"""
        super().save_tracking_data(save_dir)
        for name in ["error_history"]:
            np.save(
                os.path.join(save_dir, f"{name}.npy"),
                getattr(self, name),
            )


class AttackerTracker(TrackerObject):
    """Handles attacker's camera tracking and attack optimization"""

    def __init__(self, id2name: dict, noise_3d=0.0, det_noise=0.0, **kwargs):
        super().__init__(id2name, noise_3d, det_noise)
        # self.id2name = {"10": "victim", "20": "attacker"} # the vic and atk here are the pedestrians
        # self.name2id = {v: k for k, v in self.id2name.items()}
        self.center_displacement_history = []
        self.displacement_history = []
        self.displacement_coord_history = []
        self.target_vic_box = []
        self.curr_atker = self.name2id["attacker"]
        self.atk_speed = kwargs.get("atk_speed", 3.3)
        self.compensate_latency = kwargs.get("compensate_latency", 0)
        self.lr = 0.1
        self.comp_latency = 0.9  # TODO: adaptive computation latency

        # optmization parameters
        self.world_corners_history = []
        self.mat_history = []
        self.inv_mat_history = []
        self.proj_matrix_history = []

    def create_pose_message(self, z_height=None):
        """Create a pose message for publishing"""
        pose = PoseStamped()
        pose.header.stamp = None  # Will be set by caller
        pose.header.frame_id = "world"
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        # Use either provided z_height or current world position
        if z_height is not None:
            pose.pose.position.z = z_height
        else:
            pose.pose.position.z = self.world_center[self.curr_atker][2].item()
            # pose.position.z = self.world_center[2].item()

        # Set orientation from detection quaternion
        q = self.current_det[self.curr_atker]["quaternion"]
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        return pose

    def tracker_after_latency(self, name, frame_latency):
        """Update tracker position after latency"""
        obj_id = self.name2id[name]
        # make a copy of the tracker
        tracker = copy.deepcopy(self.trackers[obj_id])

        for _ in range(frame_latency):
            pred_box = tracker.predict_no_trace()[0]
            tracker.predict()
            tracker.update(pred_box)
        return tracker, tracker.predict_no_trace()[0]

    def optimize_attack(self, mat, inv_mat, camera_model, static_camera=False, eot=False, optimize=True):
        """Optimize attack position to cause ID switch"""
        vic_id = self.name2id["victim"]
        atk_id = self.name2id["attacker"]

        # Calculate displacement distance enough for the attacker to keep moving during next computation
        delta_location = self.atk_speed * self.comp_latency
        # this ensures the computation latency does not further go up due to increased iteration caused by larger delta_location
        lr = delta_location / 0.3 * self.lr  # 0.3 was the original delta_location.

        vic_tracker = None
        vic_pred_box = None
        if self.compensate_latency > 0:
            vic_tracker, vic_pred_box = self.tracker_after_latency(
                "victim", self.compensate_latency
            )
        else:
            vic_tracker = self.trackers[vic_id]
            vic_pred_box = self.pred_history[vic_id][-1]

        self.target_vic_box.append(vic_pred_box)
        self.world_corners_history.append(copy.deepcopy(self.world_corners[atk_id]))
        self.mat_history.append(mat)
        self.inv_mat_history.append(inv_mat)
        self.proj_matrix_history.append(camera_model.projection_matrix())
        
        # Create pose message with optimized position
        pose = self.create_pose_message()
        if not optimize:
            return pose, [0, 0]
        
        center_displacement = optimize_3d_coord(
            self.trackers[atk_id],
            vic_tracker,
            self.world_corners[atk_id],
            vic_pred_box,
            inv_mat,
            camera_model,
            lr=lr,
            iteration=20,
            delta_location=delta_location,
            static_camera=static_camera,
        )

        # Update position if optimization succeeded
        if not np.isnan(
            center_displacement
        ).any():  # TODO: sometimes the optimization fails
            pose.pose.position.x = (
                center_displacement[0].item() + self.world_center[atk_id][0]
            )
            pose.pose.position.y = (
                center_displacement[1].item() + self.world_center[atk_id][1]
            )
            self.displacement_history.append(
                center_displacement + self.world_center[atk_id][:2]
            )
        else:
            pose.pose.position.x = self.world_center[atk_id][0].item()
            pose.pose.position.y = self.world_center[atk_id][1].item()
            self.displacement_history.append(self.world_center[atk_id][:2])

        return pose, center_displacement

    def save_tracking_data(self, save_dir):
        """Save tracking data to file"""
        super().save_tracking_data(save_dir)
        for name in [
            "center_displacement_history",
            "displacement_history",
            "displacement_coord_history",
            "world_corners_history",
            "mat_history",
            "inv_mat_history",
            "proj_matrix_history",
            "target_vic_box",
        ]:
            np.save(
                os.path.join(save_dir, f"{name}.npy"),
                getattr(self, name),
            )


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
        self.model.step(5)

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


class CameraSetup:
    """Handles camera information and transformations"""

    def __init__(self, camera_frame_id):
        self.frame_id = camera_frame_id
        self.camera_model = PinholeCameraModel()
        self.info = None
        self.pose = None
        self.pose_history = []
        self.mainloop_pose_history = []
        self.mat = None
        self.inv_mat = None
        self.last_update_timestamp = None
        self.measure_period = 0.01  # 10ms

    def update_camera_info(self, msg):
        """Update camera info from CameraInfo message"""
        if msg.header.frame_id == self.frame_id:
            self.camera_model.from_camera_info(msg)
            self.info = msg
            return True
        return False

    def update_camera_pose(self, msg, current_time):
        """Update camera pose from Pose message"""
        if self.last_update_timestamp is None:
            self.last_update_timestamp = current_time
            return False

        if (
            current_time - self.last_update_timestamp
        ).nanoseconds < self.measure_period * 1e9:
            return False

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
        pose.extend([roll, pitch, yaw, current_time.nanoseconds])
        self.pose = pose
        self.pose_history.append(pose)
        self.last_update_timestamp = current_time

        return True

    def update_transformation_matrix(self, mat_msg):
        """Update transformation matrix from Float64MultiArray message"""
        self.mat = np.array(mat_msg.data[:16]).reshape(4, 4)
        self.inv_mat = np.array(mat_msg.data[16:32]).reshape(4, 4)

    def get_stability_analysis(self):
        """Analyze camera stability"""
        return camera_stability_analysis(self.pose_history)

    def save_camera_data(self, save_dir, camera_id):
        """Save camera data to file"""
        for name in ["pose_history", "mainloop_pose_history"]:
            np.save(
                os.path.join(save_dir, f"camera{camera_id}_{name}.npy"),
                getattr(self, name),
            )


class ImageProcessor:
    """Handles image processing and visualization"""

    def __init__(self, save_dir, save_interval):
        self.bridge = CvBridge()
        self.setup_save_directories(save_dir)
        self.save_interval = save_interval

    def setup_save_directories(self, save_dir):
        """Set up directories for saving images"""
        self.img_dir = os.path.join(save_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.raw_img_dir = os.path.join(save_dir, "raw_images")
        os.makedirs(self.raw_img_dir, exist_ok=True)
        self.save_dir = save_dir

    def save_track_image(
        self, img, bboxes, frame_num, filename="test", extra_bbox=None
    ):
        """Save tracking visualization image"""
        # TODO: will save in the end of the simulation instead of each step be better?
        victim_bbox, attacker_bbox, victim_pred_bbox, attacker_pred_bbox = bboxes

        # Save raw image
        if frame_num % self.save_interval == 0:
            cv2.imwrite(
                os.path.join(self.raw_img_dir, f"{filename}_{frame_num}.jpg"),
                self.bridge.imgmsg_to_cv2(img, "bgr8"),
            )

        # Draw bounding boxes
        cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")

        # Draw ground truth boxes
        cv2.rectangle(
            cv_img,
            (int(victim_bbox[0]), int(victim_bbox[1])),
            (int(victim_bbox[2]), int(victim_bbox[3])),
            (0, 255, 0),  # green
            2,
        )
        cv2.rectangle(
            cv_img,
            (int(attacker_bbox[0]), int(attacker_bbox[1])),
            (int(attacker_bbox[2]), int(attacker_bbox[3])),
            (255, 0, 0),  # blue
            2,
        )

        # Draw predicted boxes if available
        if victim_pred_bbox is not None and attacker_pred_bbox is not None:
            cv2.rectangle(
                cv_img,
                (int(victim_pred_bbox[0]), int(victim_pred_bbox[1])),
                (int(victim_pred_bbox[2]), int(victim_pred_bbox[3])),
                (0, 255, 255),  # yellow
                2,
            )
            cv2.rectangle(
                cv_img,
                (int(attacker_pred_bbox[0]), int(attacker_pred_bbox[1])),
                (int(attacker_pred_bbox[2]), int(attacker_pred_bbox[3])),
                (255, 255, 0),  # cyan
                2,
            )

        # Draw extra bounding boxes if available
        if extra_bbox is not None:
            for i, bbox in enumerate(extra_bbox):
                cv2.rectangle(
                    cv_img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 0, 255),  # red
                    2,
                )
                label = f"Extra {i}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 0, 255)  # red, same as box
                font_thickness = 2
                text_position = (
                    int(bbox[0]),
                    int(bbox[1] - 5),
                )  # Slightly above the box

                cv2.putText(
                    cv_img,
                    label,
                    text_position,
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )

        # Save annotated image
        cv2.imwrite(
            os.path.join(self.img_dir, f"{filename}_{frame_num}.jpg"),
            cv_img,
        )


def parse_detection2d_array(msg2d):
    """
    Convert a Detection2DArray message to a dict of box dictionaries.

    Args:
        msg2d (vision_msgs.msg.Detection2DArray): Detection2D array message

    Returns:
        list: List of dictionaries, each containing:
            - position: [x, y]
            - dimensions: [width, height]
            - class_id: string
    """
    boxes = {}
    for detection in msg2d.detections:
        box = {
            "position": [
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
            ],
            "dimensions": [
                detection.bbox.size_x,
                detection.bbox.size_y,
            ],
            "class_id": detection.results[0].hypothesis.class_id,
        }
        boxes[box["class_id"]] = box

    return boxes


def parse_detection3d_array(msg3d):
    """
    Convert a Detection3DArray message to a dict of box dictionaries.

    Args:
        msg3d (vision_msgs.msg.Detection3DArray): Detection3D array message

    Returns:
        list: List of dictionaries, each containing:
            - position: [x, y, z]
            - dimensions: [width, height, length]
            - quaternion: [x, y, z, w]
            - class_id: string
    """
    boxes = {}
    for detection in msg3d.detections:
        box = {
            "position": [
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                detection.bbox.center.position.z,
            ],
            "dimensions": [
                detection.bbox.size.x,
                detection.bbox.size.y,
                detection.bbox.size.z,
            ],
            "quaternion": [
                detection.bbox.center.orientation.x,
                detection.bbox.center.orientation.y,
                detection.bbox.center.orientation.z,
                detection.bbox.center.orientation.w,
            ],
            "class_id": detection.results[0].hypothesis.class_id,
        }
        boxes[box["class_id"]] = box

    return boxes


def get_pose_index(target_name: List[str]) -> Dict[str, int]:
    """
    Get the index of a list of vehicles by name.
    """
    gz_node = (
        gzt.Node()
    )  # Gazebo transport node destroyed automatically when out of scope
    received_messages = []

    def callback(msg):
        received_messages.append(msg)

    sub = gz_node.subscribe(pose_v_pb2.Pose_V, "/world/default/pose/info", callback)
    while len(received_messages) == 0:
        time.sleep(0.1)

    found_indices = {}
    for i, pose in enumerate(received_messages[0].pose):
        if pose.name in target_name:
            print(f"Found {pose.name} at index {i}")
            found_indices[pose.name] = i

    if len(found_indices) != len(target_name):
        print("Some vehicles not found")
    return found_indices


class AttackManager(Node):
    """Main node class for managing the attack simulation"""

    def __init__(
        self,
        save_dir: str,
        save_interval: int,
        atk_duration: int,
        det_noise=0.0,
        noise_3d=0.0,
        atk_speed=3.3,
        occlusion=False,
        static_camera=False,
        eot=False,
        compensate_latency=0,
    ):
        super().__init__("attack_manager")

        self.id2name = {
            "10": "victim",
            "20": "attacker",
        }  # the vic and atk here are the pedestrians
        self.name2id = {v: k for k, v in self.id2name.items()}

        # Setup trackers
        self.victim_vehicle = VictimTracker(
            self.id2name, noise_3d, det_noise
        )  # vic uav
        self.attacker_machine = AttackerTracker(
            self.id2name,
            noise_3d,
            det_noise,
            atk_speed=atk_speed,
            compensate_latency=compensate_latency,
        )  # atk camera

        # Setup cameras
        self.main_camera = CameraSetup("x500_gimbal_0/camera_link/camera")

        # Setup attack state
        self.first_switch = False
        self.first_switch_frame = 0
        self.switched = False
        self.switch_frame = 0
        self.lose_target_count = 0
        self.frame_num = 0
        self.atk_duration = atk_duration
        self.enable_occlusion = occlusion
        self.occluded = False
        self.occluded_frame = 0
        self.static_camera = static_camera
        self.eot = eot
        self.vic_pose = None

        # Setup tracking data
        self.main_timestamp_history = []

        # Setup image processor
        self.image_processor = ImageProcessor(save_dir, save_interval)
        self.save_interval = save_interval

        # Setup social force model
        self.sf_model = SocialForceModel()

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
        self._send_start_signal()

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

        # Agent state subscriptions
        vic_state_sub = Subscriber(self, TwistStamped, "/actor_walking/pose_vel")
        atk_state_sub = Subscriber(self, TwistStamped, "/actor_following/pose_vel")
        self.synced_pose = ApproximateTimeSynchronizer(
            [vic_state_sub, atk_state_sub], 10, 0.05
        )
        self.synced_pose.registerCallback(self.synced_pose_callback)

    def _setup_publishers(self):
        """Setup ROS publishers"""
        # Attacker's crafted pose publisher
        self.atk_pub = self.create_publisher(PoseStamped, "/actor_following/atk_pose", 1)

        # Victim's bbox error publisher
        self.vic_pub = self.create_publisher(Point, "object", 1)

        # Victim's social force publisher
        self.vic_sf_pub = self.create_publisher(
            TwistStamped, "/actor_walking/social_force", 1
        )

    def _send_start_signal(self):
        """Send start signal to begin movement"""
        self.start_signal = subprocess.Popen(
            [
                "ros2",
                "topic",
                "pub",
                "/start_movement",
                "std_msgs/msg/Bool",
                "data: true",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        current_time = self.get_clock().now()
        self.main_camera.update_camera_pose(msg, current_time)

    def mat_callback(self, mat_msg):
        """Handle transformation matrix messages"""
        self.main_camera.update_transformation_matrix(mat_msg)

    def synced_pose_callback(self, vic_state: TwistStamped, atk_state: TwistStamped):
        """Handle synchronized victim and attacker state messages"""
        # Initialize social force model if needed
        vic_data = self.sf_model.get_state(vic_state)
        atk_data = self.sf_model.get_state(atk_state)
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
        self.sf_model.timestamp_history.append(self.get_clock().now().nanoseconds)

    def convert_to_bbx(self, victim_2d, attacker_2d):
        """Extract bounding boxes from detection messages"""
        # Convert to format needed for tracking
        victim_bbox = [
            victim_2d["position"][0] - victim_2d["dimensions"][0] / 2,
            victim_2d["position"][1] - victim_2d["dimensions"][1] / 2,
            victim_2d["position"][0] + victim_2d["dimensions"][0] / 2,
            victim_2d["position"][1] + victim_2d["dimensions"][1] / 2,
        ]
        attacker_bbox = [
            attacker_2d["position"][0] - attacker_2d["dimensions"][0] / 2,
            attacker_2d["position"][1] - attacker_2d["dimensions"][1] / 2,
            attacker_2d["position"][0] + attacker_2d["dimensions"][0] / 2,
            attacker_2d["position"][1] + attacker_2d["dimensions"][1] / 2,
        ]

        return np.array(victim_bbox, dtype=np.float32), np.array(
            attacker_bbox, dtype=np.float32
        )

    def check_for_id_switch(
        self, victim_bbox, attacker_bbox, victim_pred_bbox, attacker_pred_bbox
    ):
        """Check if ID switch has occurred based on IoU"""
        ious = iou_batch(
            [victim_bbox, attacker_bbox], [victim_pred_bbox, attacker_pred_bbox]
        )

        # Check if cross-association IoUs are higher than self-association
        if ious[0, 0] + ious[1, 1] < ious[0, 1] + ious[1, 0]:
            if self.frame_num > 20:  # Ignore initialization phase
                return True
        return False

    def check_for_occlusion(
        self, attacker_world_center, victim_world_center, victim_bbox, victim_pred_bbox
    ):
        """Check for occlusion between victim and attacker"""
        if not self.enable_occlusion:
            return

        # Check distance between attacker and victim
        distance = np.linalg.norm(
            np.array(attacker_world_center) - np.array(victim_world_center)
        )

        if distance < 1.5:
            self.occluded = True
            self.occluded_frame = self.frame_num
        elif self.occluded:
            # Check if victim has moved significantly from predicted position
            vic_bbx_center = [
                victim_bbox[0] + (victim_bbox[2] - victim_bbox[0]) / 2,
                victim_bbox[1] + (victim_bbox[3] - victim_bbox[1]) / 2,
            ]
            vic_bbx_center_pred = [
                victim_pred_bbox[0] + (victim_pred_bbox[2] - victim_pred_bbox[0]) / 2,
                victim_pred_bbox[1] + (victim_pred_bbox[3] - victim_pred_bbox[1]) / 2,
            ]
            vic_region = (
                victim_bbox[2] - victim_bbox[0] + victim_bbox[3] - victim_bbox[1]
            ) / 2

            if (
                np.linalg.norm(np.array(vic_bbx_center) - np.array(vic_bbx_center_pred))
                > vic_region
            ):
                self.occluded = False

        # Check if occlusion duration threshold exceeded
        if self.occluded and self.frame_num == self.occluded_frame + 200:
            self.log(f"Occluded. Stop the attack at frame {self.frame_num}")
            self.stop_ros_node()

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
            return False, None, None
        return True, det2d, det3d

    def synced_callback(
        self, msg2d: Detection2DArray, msg3d: Detection3DArray, img: Image
    ):
        """Process synchronized detection and camera info messages"""
        start_time = self.get_clock().now()

        # Ensure camera info is available
        if self.main_camera.info is None or self.main_camera.mat is None:
            self.log("No camera info or trans matrix yet")
            return

        valid, det2d, det3d = self.check_detection(msg2d, msg3d)
        if not valid:
            self.lose_target_count += 1
            if self.lose_target_count > 100:
                self.log(f"Vic: Lost target for {self.lose_target_count} frames")
                self.stop_ros_node()
            return
        victim_2d = det2d[self.name2id["victim"]]

        # Set current 3D detections for trackers
        self.victim_vehicle.set_detection_3d(det3d)
        self.attacker_machine.set_detection_3d(det3d)

        # Publish victim error for motion control
        motion_error = self.victim_vehicle.calculate_error(
            victim_2d, self.main_camera.info
        )
        self.vic_pub.publish(
            motion_error
        )  # currently using GT 2D bbox. TODO: use predicted/matched 2D bbox?
        if motion_error:
            self.victim_vehicle.error_history.append(
                [motion_error.x, motion_error.y, motion_error.z]
            )

        # Update trackers and get predicted bounding boxes
        victim_tracking_updated = self.victim_vehicle.update_tracking(
            self.main_camera.camera_model,  # if camera info available, camera model is also available
            self.occluded,
        )
        attacker_tracking_updated = self.attacker_machine.update_tracking(
            self.main_camera.camera_model,
            False,  # Attacker is never occluded in this scenario
        )
        if not (victim_tracking_updated and attacker_tracking_updated):
            return

        # Update world positions, if 2D tracking is successful
        _, _ = self.attacker_machine.update_world_position(self.main_camera.mat)
        _, _ = self.victim_vehicle.update_world_position(self.main_camera.mat)

        # Check for occlusion
        # if self.enable_occlusion:
        #     self.check_for_occlusion(
        #         attacker_world_center,
        #         victim_world_center,
        #         victim_bbox,
        #         self.victim_vehicle.pred_history[-1]
        #     )

        # Record timestamp
        self.main_camera.mainloop_pose_history.append(self.main_camera.pose)
        self.main_timestamp_history.append(self.get_clock().now().nanoseconds)
        self.attacker_machine.id_switch_history.append(
            self.check_for_id_switch(
                self.attacker_machine.history[self.name2id["victim"]][-1],
                self.attacker_machine.history[self.name2id["attacker"]][-1],
                self.attacker_machine.pred_history[self.name2id["victim"]][-1],
                self.attacker_machine.pred_history[self.name2id["attacker"]][-1],
            )
        )
        self.victim_vehicle.id_switch_history.append(
            self.check_for_id_switch(
                self.victim_vehicle.history[self.name2id["victim"]][-1],
                self.victim_vehicle.history[self.name2id["attacker"]][-1],
                self.victim_vehicle.pred_history[self.name2id["victim"]][-1],
                self.victim_vehicle.pred_history[self.name2id["attacker"]][-1],
            )
        )

        # check switch until the end of the attack duration
        if self.victim_vehicle.id_switch_history[-1]:
            if not self.first_switch:
                self.first_switch = True
                self.first_switch_frame = self.frame_num
            # flip the flag
            self.switched = not self.switched
            self.switch_frame = self.frame_num
            self.log(f"ID switch detected at frame {self.frame_num} | {self.switched}")

        # optimize attack position
        attack_pose, center_dis = self.attacker_machine.optimize_attack(
            self.main_camera.mat,
            self.main_camera.inv_mat,
            self.main_camera.camera_model,
            eot=self.eot,
            optimize=((self.frame_num % self.save_interval) == 0),
            # optimize=self.frame_num == 0, # only optimize at the beginning
        )
        # Only process at specified intervals
        if attack_pose is not None:
            # Publish attack pose
            opt_finish_time = self.get_clock().now()
            print(
                f"time: {(opt_finish_time.nanoseconds - start_time.nanoseconds) / 1e9} s"
            )
            attack_pose.header.stamp = opt_finish_time.to_msg() # time sent

            attack_pose.pose.position.x = self.vic_pose[0]
            attack_pose.pose.position.y = self.vic_pose[1]

            self.atk_pub.publish(attack_pose)
            self.attacker_machine.center_displacement_history.append(center_dis)

        atk_pose_world = self.attacker_machine.displacement_history[-1]
        atk_pose_world = np.append(atk_pose_world, 2.3)
        atk_pose_world = atk_pose_world.astype(np.float32)
        atk_pose_camera = world_to_camera(atk_pose_world, self.main_camera.inv_mat)
        atk_coord, _ = project_box_3d_to_2d(
            atk_pose_camera, self.main_camera.camera_model
        )
        atk_coord = atk_coord[0]
        self.attacker_machine.displacement_coord_history.append(atk_coord)

        # Save atk visualization
        # atk_coord = self.attacker_machine.displacement_coord_history[-1]
        self.image_processor.save_track_image(
            img,
            [
                self.attacker_machine.history[self.name2id["victim"]][-1],
                self.attacker_machine.history[self.name2id["attacker"]][-1],
                self.attacker_machine.pred_history[self.name2id["victim"]][-1],
                self.attacker_machine.pred_history[self.name2id["attacker"]][-1],
            ],
            self.frame_num,
            "atk_machine",
            extra_bbox=[
                [
                    atk_coord[0] - 10,
                    atk_coord[1] - 10,
                    atk_coord[0] + 10,
                    atk_coord[1] + 10,
                ],
                self.attacker_machine.target_vic_box[-1],
            ],
        )

        # Increment frame counter
        self.frame_num += 1

        # Log progress periodically
        if self.frame_num % 100 == 0:
            self.log(f"Simulation running at frame {self.frame_num}")
            self.log(f"Lost target for {self.lose_target_count} frames")
            self.log(f"Attacker switched: {self.switched}")
            self.log(
                f"Average time interval: {np.mean(np.diff(self.main_timestamp_history)) / 1e9} s"
            )
            # sample every 10 timestamps
            self.log(
                f"Average attack interval: {np.mean(np.diff(self.main_timestamp_history[::10])) / 1e9} s"
            )
            self.log(
                f"Attack duration: {(self.main_timestamp_history[-1] - self.main_timestamp_history[0]) / 1e9} s"
            )
            self.log("")

        # Check if attack duration reached
        if self.frame_num > self.atk_duration:
            self.log("Attack duration reached")
            self.stop_ros_node()

    def _setup_dirs(self):
        """Set up directories for saving images and logs"""
        self.vic_vehicle_dir = os.path.join(self.save_dir, "vic_vehicle")
        self.atk_machine_dir = os.path.join(self.save_dir, "atker_machine")
        os.makedirs(self.vic_vehicle_dir, exist_ok=True)
        os.makedirs(self.atk_machine_dir, exist_ok=True)

    def stop_ros_node(self):
        """Save log file and stop the ROS node"""
        # save tracking data
        self.victim_vehicle.save_tracking_data(self.vic_vehicle_dir)
        self.attacker_machine.save_tracking_data(self.atk_machine_dir)
        self.sf_model.save_social_force_data(self.save_dir)
        self.main_camera.save_camera_data(self.save_dir, "0")
        for name in ["main_timestamp_history"]:
            np.save(
                os.path.join(self.save_dir, f"{name}.npy"),
                getattr(self, name),
            )

        self.log("Motion attack simulation result:")
        if self.first_switch:
            self.log(f"Attacker switched at frame {self.first_switch_frame}")
        self.log(f"Simulation stop at frame {self.frame_num}")
        if self.sf_model.enabled:
            self.log(f"Maximum victim delta speed: {self.sf_model.max_delta_speed}")
        self.log(
            f"Attack interval: {np.mean(np.diff(self.main_timestamp_history[::10])) / 1e9} s"
        )
        indices = np.arange(0, len(self.main_timestamp_history) - 1, 10)
        np_timestamps = np.array(self.main_timestamp_history)
        self.log(
            f"Latency one: {np.mean(np_timestamps[indices + 1] - np_timestamps[indices]) / 1e9} s"
        )
        self.log(
            f"Attack duration: {(self.main_timestamp_history[-1] - self.main_timestamp_history[0]) / 1e9} s"
        )
        self.log(
            f"Camera stability analysis: {camera_stability_analysis(self.main_camera.pose_history)}"
        )
        self.log("ATTACK STOPPED")
        self.log_file.close()

        time.sleep(1)
        self.destroy_node()
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Attack Manager")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="~/Documents/exp/uavAtk",
        help="Directory to save the experiment results",
    )
    parser.add_argument(
        "--save_img_interval",
        type=int,
        default=10,
        help="Interval to save images. "
        "Also interval to update the tracking and perform the attack",
    )
    parser.add_argument(
        "--atk_duration",
        type=int,
        default=400,
        help="Duration of the attack",
    )
    parser.add_argument(
        "--det_noise",
        type=float,
        default=5.0,
        help=(
            "Std for object detection gaussian noise. "
            "Add noise to the ground truth 2D bounding box. "
            "Bbx typically are around 100x100 pixels."
            "Low: 1, Medium: 5, High: 10"
        ),
    )
    parser.add_argument(
        "--noise_3d",
        type=float,
        default=0.0,
        help="Std for object detection 3D gaussian noise",
    )
    parser.add_argument(
        "--atk_speed",
        type=float,
        default=3.3,
        help="Attacker speed in m/s. Default: 3.3",
    )
    parser.add_argument(
        "--compensate_latency",
        type=int,
        default=0,
        help=("Compensate latency in frames. "),
    )
    parser.add_argument(
        "--enable_occlusion",
        action="store_true",
        help="Enable occlusion attack",
        default=False,
    )
    parser.add_argument(
        "--static_camera",
        action="store_true",
        help="Use static camera for attacker",
        default=False,
    )
    parser.add_argument(
        "--eot",
        action="store_true",
        help="Use expectation of transformation for attacker optimization",
        default=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    args.save_dir = os.path.expanduser(args.save_dir)

    with open(os.path.join(args.save_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    topic_manager = TopicBridgeManager()
    topic_manager.start_common_bridges()

    rclpy.init()
    node = AttackManager(
        args.save_dir,
        args.save_img_interval,
        args.atk_duration,
        args.det_noise,
        args.noise_3d,
        args.atk_speed,
        args.enable_occlusion,
        args.static_camera,
        args.eot,
        args.compensate_latency,
    )

    rclpy.spin(node)
    # try:  # stable experiment running
    #     rclpy.spin(node)
    # except:
    #     print("Error in the attack manager")


if __name__ == "__main__":
    main()
