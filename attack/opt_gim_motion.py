import os
import cv2
import time
import copy as cp
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)  # Set random seed for reproducibility

from tqdm import tqdm
from typing import Tuple, List, Dict
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from attack.ID_Transfer import get_2d_bbox_tf
from utils.trans_mat import TransformMatrix
from utils.sort_TF import KalmanBoxTracker, convert_bbox_to_z, convert_x_to_bbox
from utils.eval.helper import get_iou_tf, convert_bbox, convert_bbox_tf, visualize_bbox_trace_video, visualize_boxes
from attack.coord_trans import CoordinateTransformation

@dataclass
class AttackConfig:
    """Configuration parameters for the attack optimization"""
    
    mock_tracker_type: str = "SORT"  # Options: "SORT", "SiamRPN"

    w1: float = 0.0  # Victim tracking error weight
    w2: float = 0.0  # Confidence reduction weight
    w3: float = 1.0  # Attacker to victim prediction tracking error weight
    w4: float = 0.0  # Attacker to attacker prediction tracking error weight
    w5: float = 0.0  # Angular velocity magnitude penalty
    w6: float = 0.0  # Angular velocity smoothness penalty
    
    # gimbal properties
    # This should be the maximum injectable angular velocity for each gimbal axis across different resonant frequencies
    # It will be clipped during execution if it is not achievable on all axes at the same time
    omega_max: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 2.0, 2.0], dtype=np.float32)
    )  # Max angular velocity [rad/s], roll, pitch, yaw. 
    omega_max_norm: float = field(
        default_factory=lambda: np.linalg.norm(np.array([0.1, 2.0, 2.0], dtype=np.float32))
    )
    frequency: float = 4.0 # 134.99 deg/s at 2.0349 Hz
    fps: float = 30.0  # Camera frame rate

    learning_rate: float = 0.5
    max_iterations: int = 1
    convergence_threshold: float = 1e-6
    det_failure_omega_threshold: float = 3.0 # not used anymore

    N: int = 1  # Number of attack intervals to optimize (future two intervals)
    T_fps: float = 1.0 / 30.0  # Time per frame (1/fps)
    
    det_std: float = 3.0 # Standard deviation for detection noise (meters)

    motion_models: List[np.ndarray] = field(default_factory=lambda: [
        np.array([0.1, 0.1, 2.0], dtype=np.float32),
        # np.array([0.1, 4.0, 0.1], dtype=np.float32),
        # np.array([1.0, 0.1, 0.1], dtype=np.float32),
    ])

    @property
    def delta_t(self) -> float:
        """Time interval for each optimization step"""
        return 1.0 / self.frequency * 0.5
    
    @property
    def frames_per_interval(self) -> int:
        """Number of frames per optimization interval"""
        return max(1, int(self.delta_t / self.T_fps))
    
    @property
    def total_frames(self) -> int:
        """Total number of frames in the attack sequence"""
        return self.N * self.frames_per_interval


class DummyCamera:
    def __init__(self, projection_matrix):
        self._projection_matrix = projection_matrix

    def projection_matrix(self):
        return self._projection_matrix
    

class UAVRandomWalkModel:
    """
    Random walk model for UAV, victim, and attacker linear and angular velocities with proper uncertainty propagation
    """
    
    def __init__(self, config):
        self.config = config
        
        # Random walk parameters for UAV linear velocity [vx, vy, vz]
        self.uav_linear_vel_drift = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)  # no systematic drift
        self.uav_linear_vel_noise_std = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)  # m/s per sqrt(dt)
        
        # Random walk parameters for UAV angular velocity [wx, wy, wz] 
        self.uav_angular_vel_drift = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)  # no systematic drift
        self.uav_angular_vel_noise_std = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)  # rad/s per sqrt(dt)
        
        # Random walk parameters for victim velocity [vx, vy, vz]
        self.victim_vel_drift = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)  # no systematic drift
        self.victim_vel_noise_std = tf.constant([0.01, 0.01, 0.01], dtype=tf.float32)  # m/s per sqrt(dt) - more uncertainty in xy
        
        # Random walk parameters for attacker velocity [vx, vy, vz]
        self.attacker_vel_drift = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)  # no systematic drift
        self.attacker_vel_noise_std = tf.constant([0.01, 0.01, 0.01], dtype=tf.float32)  # m/s per sqrt(dt)

        # Position noise (GPS/estimation uncertainty)
        self.position_noise_std = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)  # meters. refer to MMAUD

        # Gimbal control noise (uncertainty in injected gimbal angular velocities)
        self.noise_to_signal_ratio = 0.082  # 11.1734/134.9868

    def evolve_uav_linear_velocity(self, current_vel: tf.Tensor, dt: float) -> tf.Tensor:
        """
        Evolve UAV linear velocity using random walk model
        v(t+dt) = v(t) + drift*dt + noise*sqrt(dt)
        """
        dt_tensor = tf.constant(dt, dtype=tf.float32)
        
        # Drift component (deterministic)
        drift_component = self.uav_linear_vel_drift * dt_tensor
        
        # Random walk component (stochastic)
        noise_component = tf.random.normal(
            shape=(3,), 
            mean=0.0, 
            stddev=self.uav_linear_vel_noise_std * tf.sqrt(dt_tensor)
        )
        
        new_velocity = current_vel + drift_component + noise_component
        return new_velocity
    
    def evolve_uav_angular_velocity(self, current_angular_vel: tf.Tensor, dt: float) -> tf.Tensor:
        """
        Evolve UAV angular velocity using random walk model
        ω(t+dt) = ω(t) + drift*dt + noise*sqrt(dt)
        """
        dt_tensor = tf.constant(dt, dtype=tf.float32)
        
        # Drift component 
        drift_component = self.uav_angular_vel_drift * dt_tensor
        
        # Random walk component
        noise_component = tf.random.normal(
            shape=(3,), 
            mean=0.0, 
            stddev=self.uav_angular_vel_noise_std * tf.sqrt(dt_tensor)
        )
        
        new_angular_velocity = current_angular_vel + drift_component + noise_component
        return new_angular_velocity
    
    def evolve_victim_velocity(self, current_vel: tf.Tensor, dt: float) -> tf.Tensor:
        """
        Evolve victim velocity using random walk model
        v(t+dt) = v(t) + drift*dt + noise*sqrt(dt)
        """
        dt_tensor = tf.constant(dt, dtype=tf.float32)
        
        # Drift component (deterministic)
        drift_component = self.victim_vel_drift * dt_tensor
        
        # Random walk component (stochastic)
        noise_component = tf.random.normal(
            shape=(3,), 
            mean=0.0, 
            stddev=self.victim_vel_noise_std * tf.sqrt(dt_tensor)
        )
        
        new_velocity = current_vel + drift_component + noise_component
        return new_velocity
    
    def evolve_attacker_velocity(self, current_vel: tf.Tensor, dt: float) -> tf.Tensor:
        """
        Evolve attacker velocity using random walk model
        v(t+dt) = v(t) + drift*dt + noise*sqrt(dt)
        """
        dt_tensor = tf.constant(dt, dtype=tf.float32)
        
        # Drift component (deterministic)
        drift_component = self.attacker_vel_drift * dt_tensor
        
        # Random walk component (stochastic)
        noise_component = tf.random.normal(
            shape=(3,), 
            mean=0.0, 
            stddev=self.attacker_vel_noise_std * tf.sqrt(dt_tensor)
        )
        
        new_velocity = current_vel + drift_component + noise_component
        return new_velocity
    
    def add_position_noise(self, position: tf.Tensor) -> tf.Tensor:
        """Add GPS/estimation noise to position"""
        noise = tf.random.normal(
            shape=(3,), 
            mean=0.0, 
            stddev=self.position_noise_std
        )
        return position + noise
    
    def add_gimbal_ctrl_noise(self, omega: tf.Tensor) -> tf.Tensor:
        """
        Add noise to gimbal control signal
        This simulates uncertainty in the gimbal control system
        """
        # Scale noise by signal-to-noise ratio
        gimbal_ctrl_noise_std = omega * self.noise_to_signal_ratio

        noise = tf.random.normal(
            shape=omega.shape, 
            mean=0.0, 
            stddev=gimbal_ctrl_noise_std
        )
        return noise


class ExpectationOverTransformation:
    """
    Implement Expectation over Transformation for robust optimization under UAV, victim, and attacker uncertainties
    """
    
    def __init__(self, config, random_walk_model: UAVRandomWalkModel):
        self.config = config
        self.random_walk_model = random_walk_model
        self.num_samples = 3  # Number of Monte Carlo samples for EoT
        
    def sample_full_trajectory(self, 
                              initial_uav_position: tf.Tensor,
                              initial_uav_linear_vel: tf.Tensor, 
                              initial_uav_angular_vel: tf.Tensor,
                              initial_victim_velocity: tf.Tensor,
                              initial_attacker_velocity: tf.Tensor,
                              init_omega: tf.Tensor):
        """
        Sample trajectories for UAV, victim, and attacker using random walk models
        
        Args:
            initial_uav_position: Initial UAV position [3]
            initial_uav_linear_vel: Initial UAV linear velocity [3]
            initial_uav_angular_vel: Initial UAV angular velocity [3]
            initial_victim_velocity: Initial victim velocity [3]
            initial_attacker_velocity: Initial attacker velocity [3]
            init_omega: Gimbal control sequence [N, 3]
        
        Returns:
            Dictionary containing sampled trajectories with keys:
            - 'uav_positions': [num_steps, 3]
            - 'uav_linear_velocities': [num_steps, 3] 
            - 'uav_angular_velocities': [num_steps, 3]
            - 'victim_velocities': [num_steps, 3]
            - 'attacker_velocities': [num_steps, 3]
            - 'gimbal_controls': [num_steps, 3]
        """
        num_steps = self.config.total_frames
        dt = self.config.T_fps
        
        # Initialize trajectory arrays
        uav_positions = tf.TensorArray(tf.float32, size=num_steps)
        uav_linear_vels = tf.TensorArray(tf.float32, size=num_steps)
        uav_angular_vels = tf.TensorArray(tf.float32, size=num_steps)
        victim_vels = tf.TensorArray(tf.float32, size=num_steps)
        attacker_vels = tf.TensorArray(tf.float32, size=num_steps)
        gimbal_ctrls = tf.TensorArray(tf.float32, size=num_steps)

        # Set initial conditions
        current_uav_pos = initial_uav_position
        current_uav_linear_vel = initial_uav_linear_vel
        current_uav_angular_vel = initial_uav_angular_vel
        current_victim_vel = initial_victim_velocity
        current_attacker_vel = initial_attacker_velocity
        current_gimbal_ctrl = tf.zeros((3,), dtype=tf.float32)

        # Store initial states
        uav_positions = uav_positions.write(0, current_uav_pos)
        uav_linear_vels = uav_linear_vels.write(0, current_uav_linear_vel)
        uav_angular_vels = uav_angular_vels.write(0, current_uav_angular_vel)
        victim_vels = victim_vels.write(0, current_victim_vel)
        attacker_vels = attacker_vels.write(0, current_attacker_vel)
        gimbal_ctrls = gimbal_ctrls.write(0, current_gimbal_ctrl)

        # Evolve trajectories
        for i in range(num_steps):
            # Evolve UAV velocities using random walk
            current_uav_linear_vel = self.random_walk_model.evolve_uav_linear_velocity(current_uav_linear_vel, dt)
            current_uav_angular_vel = self.random_walk_model.evolve_uav_angular_velocity(current_uav_angular_vel, dt)
            
            # Evolve victim and attacker velocities using random walk
            current_victim_vel = self.random_walk_model.evolve_victim_velocity(current_victim_vel, dt)
            current_attacker_vel = self.random_walk_model.evolve_attacker_velocity(current_attacker_vel, dt)
            
            # Update UAV position using current velocity
            current_uav_pos = current_uav_pos + current_uav_linear_vel * dt
            
            # Store current state
            uav_positions = uav_positions.write(i, current_uav_pos)
            uav_linear_vels = uav_linear_vels.write(i, current_uav_linear_vel)
            uav_angular_vels = uav_angular_vels.write(i, current_uav_angular_vel)
            victim_vels = victim_vels.write(i, current_victim_vel)
            attacker_vels = attacker_vels.write(i, current_attacker_vel)

        # Add gimbal control noise for each frame
        for i in range(self.config.N):
            for j in range(self.config.frames_per_interval):
                # Add gimbal control noise
                current_gimbal_ctrl = self.random_walk_model.add_gimbal_ctrl_noise(init_omega[i])
                gimbal_ctrls = gimbal_ctrls.write(i * self.config.frames_per_interval + j, current_gimbal_ctrl)

        # Stack results into tensors
        uav_positions = uav_positions.stack()
        uav_linear_vels = uav_linear_vels.stack()
        uav_angular_vels = uav_angular_vels.stack()
        victim_vels = victim_vels.stack()
        attacker_vels = attacker_vels.stack()
        gimbal_ctrls = gimbal_ctrls.stack()

        return {
            'uav_positions': uav_positions,
            'uav_linear_velocities': uav_linear_vels,
            'uav_angular_velocities': uav_angular_vels,
            'victim_velocities': victim_vels,
            'attacker_velocities': attacker_vels,
            'gimbal_controls': gimbal_ctrls
        }

    # Backward compatibility - keep the old method name
    def sample_uav_trajectory(self, 
                            initial_position: tf.Tensor,
                            initial_linear_vel: tf.Tensor, 
                            initial_angular_vel: tf.Tensor,
                            init_omega: tf.Tensor):
        """Legacy method for backward compatibility - only samples UAV trajectory"""
        num_steps = self.config.total_frames
        dt = self.config.T_fps
        
        # Initialize trajectory arrays
        positions = tf.TensorArray(tf.float32, size=num_steps)
        linear_vels = tf.TensorArray(tf.float32, size=num_steps)
        angular_vels = tf.TensorArray(tf.float32, size=num_steps)
        gimbal_ctrls = tf.TensorArray(tf.float32, size=num_steps)

        # Set initial conditions
        current_pos = initial_position
        current_linear_vel = initial_linear_vel
        current_angular_vel = initial_angular_vel
        current_gimbal_ctrl = tf.zeros((3,), dtype=tf.float32)

        positions = positions.write(0, current_pos)
        linear_vels = linear_vels.write(0, current_linear_vel)
        angular_vels = angular_vels.write(0, current_angular_vel)
        gimbal_ctrls = gimbal_ctrls.write(0, current_gimbal_ctrl)

        # Evolve trajectory
        for i in range(num_steps):
            # Evolve velocities using random walk
            current_linear_vel = self.random_walk_model.evolve_uav_linear_velocity(current_linear_vel, dt)
            current_angular_vel = self.random_walk_model.evolve_uav_angular_velocity(current_angular_vel, dt)
            
            # Update position using current velocity
            current_pos = current_pos + current_linear_vel * dt
            
            # Store current state
            positions = positions.write(i, current_pos)
            linear_vels = linear_vels.write(i, current_linear_vel)
            angular_vels = angular_vels.write(i, current_angular_vel)

        for i in range(self.config.N):
            for j in range(self.config.frames_per_interval):
                # Add gimbal control noise
                current_gimbal_ctrl = self.random_walk_model.add_gimbal_ctrl_noise(init_omega[i])
                gimbal_ctrls = gimbal_ctrls.write(i * self.config.frames_per_interval + j, current_gimbal_ctrl)

        # Stack results into tensors
        positions = positions.stack()
        linear_vels = linear_vels.stack()
        angular_vels = angular_vels.stack()
        gimbal_ctrls = gimbal_ctrls.stack()

        return {
            'positions': positions,
            'linear_velocities': linear_vels,
            'angular_velocities': angular_vels,
            'gimbal_controls': gimbal_ctrls
        }


class TFGimbalAttackOptimizer:
    """TensorFlow-based gimbal attack optimization with automatic gradients"""

    def __init__(self, config):
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.transform_tool = TransformMatrix()
        
        # self.omega_sequence = tf.Variable(
        #     tf.random.normal((self.config.N, 3), stddev=1.00), # prior knowledge: need large motion
        #     trainable=True,
        #     name="omega_sequence",
        # )
        self.omega_sequence = tf.Variable(
            tf.ones((self.config.N, 3), dtype=tf.float32) * self.config.omega_max,
            trainable=True,
            name="omega_sequence",
        )
        # zero initialization
        self.current_orientation = tf.Variable(
            tf.zeros((3,), dtype=tf.float32), trainable=False, name="current_orientation"
        )

        self.random_walk_model = UAVRandomWalkModel(config)
        self.eot_uav_det = ExpectationOverTransformation(config, self.random_walk_model)

        self.coord_transform = CoordinateTransformation()

    def shift_and_reinitialize_omega(self, shift_steps=1):
        """
        Shift omega sequence forward by shift_steps and pad with zeros
        
        Args:
            shift_steps: Number of steps to shift forward (default=1)
        """
        current_omega = self.omega_sequence.value()
        
        if shift_steps >= self.config.N:
            # If shifting more than sequence length, reinitialize completely
            new_omega = tf.random.normal((self.config.N, 3), stddev=1.0)
            # initialize with the maximum angular velocity value
            # new_omega = tf.ones((self.config.N, 3), dtype=tf.float32) * self.config.omega_max
        else:
            # Shift existing values forward and pad with zeros
            shifted_part = current_omega[shift_steps:]  # Take from shift_steps to end
            # zero_padding = tf.zeros((shift_steps, 3), dtype=tf.float32)  # Pad with zeros
            padding = tf.random.normal((shift_steps, 3), stddev=1.0)
            # padding = tf.ones((shift_steps, 3), dtype=tf.float32) * self.config.omega_max  # Pad with max omega value
            new_omega = tf.concat([shifted_part, padding], axis=0)

        self.omega_sequence.assign(new_omega)

    def apply_gimbal_rotation_tf(self, ex_mat, in_mat, atk_states, dt, frame_idx=None):
        """Apply gimbal rotation effect to object boxes with sampled velocities"""
        dt = tf.convert_to_tensor(dt, dtype=tf.float32)

        results = self.coord_transform.apply_gimbal_rotation_optimized(
            ex_mat, in_mat, atk_states, dt
        )

        return results
    
    def huber_loss_tf(self, x, delta, scale=1.0):
        """
        Huber loss: quadratic for small errors, linear for large errors
        Maintains constant gradient for large errors (good for optimization)
        """
        return tf.cond(
            x <= delta,
            lambda: scale * 0.5 * tf.square(x),  # Quadratic for small errors
            lambda: scale * delta * (x - 0.5 * delta)  # Linear for large errors (constant gradient!)
        )

    def exponential_reward(self, omega, decay_factor=0.1):
        """
        Exponential reward that grows more aggressively initially
        
        Characteristics:
        - Strong encouragement for small->medium omega
        - Natural saturation due to exponential decay
        - Good for bootstrap learning
        """
        omega_magnitude = tf.norm(omega)
        omega_ratio = omega_magnitude / self.config.omega_max_norm
        
        # Exponential growth with decay
        reward = omega_magnitude * (1.0 - tf.exp(-omega_ratio / decay_factor))
        
        return reward

    #@tf.function
    def compute_loss_tf(self, omega_sequence, atk_states):
        """Compute loss function with decoupled time intervals and realistic IoU-based association"""
        tracker_state = atk_states["trackers"]
        tracker_state.reset_all_states()
        position, quaternion = (
            atk_states["cam_pose"][:3],
            atk_states["cam_pose"][3:7],
        )  # qx, qy, qz, qw
        qx, qy, qz, qw = tf.unstack(quaternion, num=4)
        quaternion = tf.stack([qw, qx, qy, qz], axis=0)

        # Get sampled trajectories from atk_states
        sampled_trajectory = atk_states["sampled_trajectory"]
        uav_linear_velocity_noise = sampled_trajectory["uav_linear_velocities"]
        uav_angular_velocity_noise = -sampled_trajectory["uav_angular_velocities"]
        omega_sequence_noise = sampled_trajectory["gimbal_controls"]

        N = self.config.N
        delta_t = self.config.delta_t
        T_fps = self.config.T_fps
        frames_per_interval = self.config.frames_per_interval
        iou_threshold = 0.2  # Minimum IoU for association

        total_loss = 0.0
        frame_traces = []

        # Track IDs
        victim_track_id = 10
        attacker_track_id = 20

        # Iterate over optimization intervals
        for i in range(N):
            current_omega = omega_sequence[i]
            # variable to hold the minimum attacker_error
            min_attacker_error = tf.constant(float("inf"))

            # Process each frame within this optimization interval
            for j in range(frames_per_interval):
                frame_idx = i * frames_per_interval + j
                # Calculate time elapsed from start of interval
                frame_time_in_interval = j * T_fps
                
                uav_linear_velocity = uav_linear_velocity_noise[frame_idx]
                uav_angular_velocity = uav_angular_velocity_noise[frame_idx] * -1.0 # opposite coordinate system (uav & gimbal)
                position = position + uav_linear_velocity * T_fps
                current_omega_noised = current_omega + omega_sequence_noise[frame_idx]
                
                # Update orientation: Ω_{ij} = Ω_{i-1} + ω_i * j * T_fps
                quaternion = self.transform_tool.integrate_quaternion(
                    quaternion, current_omega_noised, T_fps
                )
                quaternion = self.transform_tool.integrate_quaternion(
                    quaternion, uav_angular_velocity, T_fps
                )

                extri_mat = self.transform_tool.get_trans_matrix(position, quaternion)
                intri_mat = self.transform_tool.get_intrinsic_matrix()

                # Apply gimbal rotation to frame with frame index for velocity sampling
                rotated_bboxes = self.apply_gimbal_rotation_tf(
                    extri_mat, intri_mat, atk_states, frame_time_in_interval + i * delta_t, frame_idx
                )

                # Step 1: Predict both trackers #
                _, victim_2d_pred = tracker_state.predict_step(victim_track_id)
                _, attacker_2d_pred = tracker_state.predict_step(attacker_track_id)
                
                # Step 2: Compute IoU matrix and perform matching #
                omega_magnitude = tf.norm(current_omega)
                
                victim_2d_pred_squeezed = tf.squeeze(victim_2d_pred)
                attacker_2d_pred_squeezed = tf.squeeze(attacker_2d_pred)
                detections = [
                    rotated_bboxes["victim_2d_bbox"],
                    rotated_bboxes["victim_2d_bbox"],
                    rotated_bboxes["attacker_2d_bbox"],
                    rotated_bboxes["attacker_2d_bbox"],
                ]
                predictions = [
                    victim_2d_pred_squeezed,
                    attacker_2d_pred_squeezed,
                    victim_2d_pred_squeezed,
                    attacker_2d_pred_squeezed,
                ]
                iou_matrix = get_iou_tf(
                    tf.stack(detections, axis=0),
                    tf.stack(predictions, axis=0),
                )
                iou_matrix = tf.reshape(iou_matrix, (2, 2))  # Reshape to [2 detections x 2 predictions]
                
                # Step 3: Perform IoU-based association and tracker updates #
                (
                    victim_tracker_gets_victim_det,
                    victim_tracker_gets_attacker_det,
                    attacker_tracker_gets_attacker_det,
                    attacker_tracker_gets_victim_det
                ) = tracker_state.iou_matching(iou_matrix, iou_threshold)
                
                if victim_tracker_gets_victim_det:
                    tracker_state.update_step(victim_track_id, rotated_bboxes["victim_2d_bbox"])
                elif victim_tracker_gets_attacker_det:
                    tracker_state.update_step(victim_track_id, rotated_bboxes["attacker_2d_bbox"])
                if attacker_tracker_gets_attacker_det:
                    tracker_state.update_step(attacker_track_id, rotated_bboxes["attacker_2d_bbox"])
                elif attacker_tracker_gets_victim_det:
                    tracker_state.update_step(attacker_track_id, rotated_bboxes["victim_2d_bbox"])
                    
                
                # Record frame-level data for visualization
                frame_bbox_data = tf.stack([
                    tf.squeeze(victim_2d_pred),
                    tf.squeeze(attacker_2d_pred), 
                    rotated_bboxes["victim_2d_bbox"],
                    rotated_bboxes["attacker_2d_bbox"]
                ], axis=0)  # Shape: [4, 4] - 4 boxes, 4 coordinates each
                frame_traces.append(frame_bbox_data)

                # Step 4: Compute loss based on actual associations #
                attacker_xy_actual = convert_bbox_tf(rotated_bboxes["attacker_2d_bbox"], "x1y1x2y2", "xywh")
                victim_xy_pred = convert_bbox_tf(victim_2d_pred, "x1y1x2y2", "xywh")
                attacker_error = tf.norm(attacker_xy_actual[:2] - victim_xy_pred[:2])
                # use sqrt(attacker.w* attacker.h) as the threshold
                attacker_error_threshold = tf.sqrt(
                    attacker_xy_actual[2] * attacker_xy_actual[3]
                ) / 2.0  # Half the scale of the attacker's bbox
                attacker_error = self.huber_loss_tf(attacker_error, attacker_error_threshold)
                min_attacker_error = tf.minimum(min_attacker_error, attacker_error)
            
            omega_reward = self.exponential_reward(current_omega, decay_factor=0.1)

            # Add interval-level penalties
            interval_loss = (
                self.config.w3 * min_attacker_error  # Want to lose the victim target
                + self.config.w5 * omega_reward  # Angular velocity penalty
            )
            total_loss += interval_loss

        return total_loss, tf.stack(frame_traces, axis=0)
    
    #@tf.function
    def compute_loss_tf_siam(self, omega_sequence, atk_states):
        """Compute loss function for SiamRPN tracker optimization"""
        tracker_state = atk_states["trackers"]  # Now expecting a SiamRPN tracker
        
        position, quaternion = (
            atk_states["cam_pose"][:3],
            atk_states["cam_pose"][3:7],
        )  # qx, qy, qz, qw
        qx, qy, qz, qw = tf.unstack(quaternion, num=4)
        quaternion = tf.stack([qw, qx, qy, qz], axis=0)

        sampled_trajectory = atk_states["sampled_trajectory"]
        uav_linear_velocity_noise = sampled_trajectory["uav_linear_velocities"]
        uav_angular_velocity_noise = -sampled_trajectory["uav_angular_velocities"]
        omega_sequence_noise = sampled_trajectory["gimbal_controls"]

        N = self.config.N
        delta_t = self.config.delta_t
        T_fps = self.config.T_fps
        frames_per_interval = self.config.frames_per_interval

        total_loss = 0.0
        frame_traces = []
        
        tracker_state.reset_state()

        # Iterate over optimization intervals
        for i in range(N):
            current_omega = omega_sequence[i]
            min_attacker_error = tf.constant(float("inf"))
            it_start = time.time()
            
            # Process each frame within this optimization interval
            for j in range(frames_per_interval):
                frame_start = time.time()
                
                frame_idx = i * frames_per_interval + j
                # Calculate time elapsed from start of interval
                frame_time_in_interval = j * T_fps

                uav_linear_velocity = uav_linear_velocity_noise[frame_idx]
                uav_angular_velocity = uav_angular_velocity_noise[frame_idx] * -1
                position = position + uav_linear_velocity * T_fps
                current_omega_noised = current_omega + omega_sequence_noise[frame_idx]
                
                # Update orientation: Ω_{ij} = Ω_{i-1} + ω_i * j * T_fps
                quaternion = self.transform_tool.integrate_quaternion(
                    quaternion, current_omega_noised, T_fps
                )
                quaternion = self.transform_tool.integrate_quaternion(
                    quaternion, uav_angular_velocity, T_fps
                )
                integration_time = time.time()
                # seems not a very good idea to jump over steps as the tracking algorithm relies on temporal consistency
                # if j != 0 and j != frames_per_interval - 1: 
                #     continue

                extri_mat = self.transform_tool.get_trans_matrix(position, quaternion)
                intri_mat = self.transform_tool.get_intrinsic_matrix()
                matrix_time = time.time()

                # Apply gimbal rotation to frame
                rotated_bboxes = self.apply_gimbal_rotation_tf(
                    extri_mat, intri_mat, atk_states, frame_time_in_interval + i * delta_t, frame_idx
                )
                rotation_time = time.time()

                # Get SiamRPN prediction
                prediction_result = tracker_state.predict(rotated_bboxes["victim_2d_bbox"], rotated_bboxes["attacker_2d_bbox"])
                victim_2d_pred = prediction_result['bbox']
                confidence_score = prediction_result['best_pscore']
                prediction_time = time.time()
                
                # Create placeholder for attacker prediction (SiamRPN only tracks one object)
                # We'll use zeros as placeholder to maintain consistent shape for visualization
                placeholder_attacker_pred = tf.zeros_like(rotated_bboxes["attacker_2d_bbox"])
                
                # Record frame-level data for visualization
                frame_bbox_data = tf.stack([
                    victim_2d_pred,
                    placeholder_attacker_pred,  # Placeholder for attacker prediction
                    rotated_bboxes["victim_2d_bbox"],
                    rotated_bboxes["attacker_2d_bbox"]
                ], axis=0)  # Shape: [4, 4] - 4 boxes, 4 coordinates each
                frame_traces.append(frame_bbox_data)
                frame_record_time = time.time()
                
                # Compute tracking errors
                attacker_xy_actual = convert_bbox_tf(rotated_bboxes["attacker_2d_bbox"], "x1y1x2y2", "xywh")
                victim_xy_pred = convert_bbox_tf(victim_2d_pred, "x1y1x2y2", "xywh")

                # Error metrics
                attacker_error = tf.norm(attacker_xy_actual[:2] - victim_xy_pred[:2])
                attacker_error_threshold = tf.sqrt(
                    attacker_xy_actual[2] * attacker_xy_actual[3]
                ) / 2.0
                attacker_error_threshold = tf.maximum(attacker_error_threshold, 1e-6)
                attacker_error = self.huber_loss_tf(attacker_error, attacker_error_threshold)
                min_attacker_error = tf.minimum(min_attacker_error, attacker_error)
                error_time = time.time()
                # Debug timing information
            #     print(f"Frame {frame_idx} timing (s): "
            #           f"Total: {error_time - frame_start:.4f}, "
            #           f"Integration: {integration_time - frame_start:.4f}, "
            #           f"Matrix: {matrix_time - integration_time:.4f}, "
            #           f"Rotation: {rotation_time - matrix_time:.4f}, "
            #           f"Prediction: {prediction_time - rotation_time:.4f}, "
            #           f"Record: {frame_record_time - prediction_time:.4f}, "
            #           f"Error: {error_time - frame_record_time:.4f}") 
            # print(f"Interval {i} processed in {time.time() - it_start:.4f} seconds.\n\n")

            omega_reward = self.exponential_reward(current_omega, decay_factor=0.1)

            # Add interval-level penalties
            interval_loss = (
                self.config.w3 * min_attacker_error  # Want to lose the victim target
                + self.config.w5 * omega_reward  # Angular velocity penalty
            )
            total_loss += interval_loss
            
        return total_loss, tf.stack(frame_traces, axis=0)

    def apply_constraints_tf(self, omega_sequence):
        """Apply constraints using TensorFlow operations"""
        # Clip to velocity limits
        omega_clipped = tf.clip_by_value(
            omega_sequence,  # TODO: further improve the realistic constraints
            -self.config.omega_max,
            self.config.omega_max,
        )
        return omega_clipped
    
    def frequency_selection(self, atk_states, save_path):
        """
        Frequency selection strategy to choose the best resonance frequency.

        Essentially selecting the best omega constraint range or the 
        direction of the injected 2D image movement.

        The selection is done by selecting the injected motion most aligned with
        the relative position vector from victim to attacker in image plane.
        """
        victim_2d_center = atk_states["victim_2d_center"]
        attacker_2d_center = atk_states["attacker_2d_center"]
        cam_pose = atk_states["cam_pose"]
        position, quaternion = (
            cam_pose[:3],
            cam_pose[3:7],
        )  # qx, qy, qz, qw
        qx, qy, qz, qw = tf.unstack(quaternion, num=4)
        quaternion = tf.stack([qw, qx, qy, qz], axis=0)
        if self.config.mock_tracker_type == "SiamRPN" or self.config.mock_tracker_type == "DaSiamRPN" or self.config.mock_tracker_type == "KCF":
            tracker_state = atk_states["trackers"]
            tracking_box_center = tracker_state.center_pos.numpy()
        elif self.config.mock_tracker_type == "SORT" or self.config.mock_tracker_type == "UCMCTrack":
            tracker_state = atk_states["trackers"]
            victim_track_id = 10
            _, tracking_box = tracker_state.get_current_state(victim_track_id)
            tracking_box_center = convert_bbox_tf(tracking_box, "x1y1x2y2", "xywh")[:2]
        else:
            raise ValueError(f"Unsupported tracker type: {self.config.mock_tracker_type}")
        
        boxes_to_visualize = []
        boxes_labels = []
        victim_box = victim_2d_center.numpy().tolist() + [50, 50]
        attacker_box = attacker_2d_center.numpy().tolist() + [50, 50]
        boxes_to_visualize.append(
            convert_bbox(victim_box, "xywh", "x1y1x2y2")
        )
        boxes_to_visualize.append(
            convert_bbox(attacker_box, "xywh", "x1y1x2y2")
        )
        boxes_labels.append("victim_center")
        boxes_labels.append("attacker_center")
        
        injected_motion_directions = []
        for model_idx, model in enumerate(self.config.motion_models):
            quaternion_injected = self.transform_tool.integrate_quaternion(
                quaternion, model, self.config.delta_t
            )
            extri_mat = self.transform_tool.get_trans_matrix(position, quaternion_injected)
            intri_mat = self.transform_tool.get_intrinsic_matrix()
            results = self.coord_transform.apply_gimbal_rotation_optimized(
                extri_mat, intri_mat, atk_states, dt=0.0
            )
            victim_2d_center_injected = convert_bbox_tf(
                results["victim_2d_bbox"], "x1y1x2y2", "xywh"
            )[:2]
            attacker_2d_center_injected = convert_bbox_tf(
                results["attacker_2d_bbox"], "x1y1x2y2", "xywh"
            )[:2]
            injected_motion_direction = attacker_2d_center_injected - attacker_2d_center
            injected_motion_directions.append(injected_motion_direction)
            
            boxes_to_visualize.append(results["victim_2d_bbox"].numpy().tolist())
            boxes_to_visualize.append(results["attacker_2d_bbox"].numpy().tolist())
            boxes_labels.append(f"victim_model{model_idx}")
            boxes_labels.append(f"attacker_model{model_idx}")

        visualize_boxes(
            boxes_to_visualize,
            boxes_labels,
            os.path.join(
                save_path,
                "images",
                f"frequency_selection_frame{atk_states['frame_num']}.jpg",
            ),
        )

        desired_direction = attacker_2d_center - tracking_box_center
        desired_direction = desired_direction / (tf.norm(desired_direction) + 1e-6)
        injected_motion_directions = tf.stack(injected_motion_directions, axis=0)
        injected_motion_directions = injected_motion_directions / (tf.norm(injected_motion_directions, axis=1, keepdims=True) + 1e-6)
        alignment_scores = tf.matmul(injected_motion_directions, tf.expand_dims(desired_direction, axis=-1))
        alignment_scores = tf.abs(alignment_scores)  # Use absolute value to consider both directions. As long as on the same line.
        best_model_idx = tf.argmax(alignment_scores, axis=0).numpy()[0]
        best_model = self.config.motion_models[best_model_idx]
        print(f"Selected best motion model: {best_model} with alignment score {alignment_scores[best_model_idx].numpy()[0]:.4f} at index {best_model_idx}")
        return best_model


    def optimize_attack(
        self, init_atk_states, iterations=10, early_stop=None, save_path="/home/jiaruili/Documents/exp/uavAtk"
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Main optimization loop using TensorFlow automatic differentiation with enhanced EoT"""

        self.config.omega_max = self.frequency_selection(init_atk_states, save_path)
        self.config.omega_max_norm = np.linalg.norm(self.config.omega_max)
        print(f"Updated omega_max to {self.config.omega_max}, norm: {self.config.omega_max_norm:.4f}")

        omega_sequence = self.omega_sequence
        opt_omega_sequence = None
        opt_iteration = 0

        loss_history = []
        best_loss = float("inf")
        iterations = self.config.max_iterations
        pbar = tqdm(range(iterations), desc="Optimizing", unit="iter")

        for iteration in range(iterations):
            it_start = time.time()
            with tf.GradientTape() as tape:
                # Apply constraints (differentiable)
                omega_constrained = self.apply_constraints_tf(omega_sequence)
                loss = 0.0

                for sample in range(self.eot_uav_det.num_samples):
                    # Sample full trajectory including victim and attacker velocities
                    sampled_trajectory = self.eot_uav_det.sample_full_trajectory(
                        initial_uav_position=init_atk_states["cam_pose"][:3],
                        initial_uav_linear_vel=init_atk_states["uav_linear_velocity"],
                        initial_uav_angular_vel=init_atk_states["uav_angular_velocity"],
                        initial_victim_velocity=init_atk_states["victim_velocity"],
                        initial_attacker_velocity=init_atk_states["attacker_velocity"],
                        init_omega=omega_constrained,
                    )
                    
                    # Store sampled trajectory in attack states
                    init_atk_states["sampled_trajectory"] = sampled_trajectory

                    # Compute loss
                    if self.config.mock_tracker_type == "SORT" or self.config.mock_tracker_type == "UCMCTrack":
                        sample_loss, iteration_trace = self.compute_loss_tf(
                            omega_constrained, init_atk_states
                        )
                        # visualize_bbox_trace_video(
                        #     iteration_trace,
                        #     os.path.join(
                        #         save_path,
                        #         f"/home/jiaruili/Documents/exp/uavAtk/saved/set4/gimbal_attack_frame{init_atk_states['frame_num']}_iter{iteration:03d}_sample{sample}_trace.mp4",
                        #     ),
                        # )
                    elif self.config.mock_tracker_type == "SiamRPN" or self.config.mock_tracker_type == "DaSiamRPN" or self.config.mock_tracker_type == "KCF":
                        sample_loss, iteration_trace = self.compute_loss_tf_siam(
                            omega_constrained, init_atk_states
                        )
                    
                    loss += sample_loss

                # Average loss over samples
                loss /= self.eot_uav_det.num_samples
            loss_end = time.time()

            # Compute gradients
            gradients = tape.gradient(loss, [omega_sequence])
            grad_end = time.time()

            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, [omega_sequence]))
            apply_end = time.time()

            # Manual constraint enforcement (non-differentiable projection)
            omega_sequence.assign(self.apply_constraints_tf(omega_sequence))
            constraint_end = time.time()

            current_loss = loss.numpy().item()
            loss_history.append(current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                opt_omega_sequence = omega_sequence.numpy()
                opt_iteration = iteration
                opt_iteration_trace = iteration_trace.numpy()

            pbar.set_postfix(
                {
                    "Loss": f"{current_loss:.6f}",
                    "Best": f"{best_loss:.6f}",
                }
            )
            pbar.update(1)
            it_end = time.time()
            
            # print("------------------------------------",
            #       f"Iteration {iteration} timing (s): "
            #       f"Total: {it_end - it_start:.4f}, "
            #       f"Loss: {loss_end - it_start:.4f}, "
            #       f"Gradients: {grad_end - loss_end:.4f}, "
            #       f"Apply: {apply_end - grad_end:.4f}, "
            #       f"Constraint: {constraint_end - apply_end:.4f}",
            #       "------------------------------------",)

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Loss = {current_loss:.6f}")
                print(f"Total frames processed: {self.config.total_frames}")

        pbar.close()
        print(f"Final loss: {current_loss:.6f}, Best: {best_loss:.6f}, Best iteration: {opt_iteration}")
        
        if opt_omega_sequence is None:
            opt_omega_sequence = omega_sequence.numpy()
            opt_iteration = iterations - 1
        
        # assign the best omega sequence to the variable
        self.omega_sequence.assign(opt_omega_sequence)
        self.shift_and_reinitialize_omega(shift_steps=1) # delta_t is the execution time interval
        
        if self.config.mock_tracker_type == "SORT":
            init_atk_states["trackers"].reset_all_states() # reset the tracker state after optimization
        elif self.config.mock_tracker_type == "SiamRPN" or self.config.mock_tracker_type == "DaSiamRPN" or self.config.mock_tracker_type == "KCF":
            init_atk_states["trackers"].reset_state() # reset the tracker state after optimization

        visualize_bbox_trace_video(
            opt_iteration_trace,
            os.path.join(
                save_path,
                f"gimbal_attack_frame{init_atk_states['frame_num']}_trace_{opt_iteration:03d}.mp4",
            ),
        )

        return opt_omega_sequence, opt_iteration_trace, loss_history

