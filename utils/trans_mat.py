import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import quat2mat, mat2quat

from attack.ID_Transfer import get_2d_bbox_tf
from utils.eval.helper import list_dirs, get_bbx_center


class TransformMatrix:
    """
    A class to handle transformation matrices for the UAV gimbal camera using TensorFlow operations.
    """

    def __init__(self):
        self.intrinsic_proj_mat = tf.constant(
            [[616.40888214, 0.0, 960.0, 0.0],
             [0.0, 616.4089036, 540.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]],
            dtype=tf.float32
        ) # project camera 3D coordinates to 2D pixel coordinates

    def _set(self, p, q):
        """Set position and quaternion as TensorFlow tensors."""
        self._p = tf.convert_to_tensor(p, dtype=tf.float32)
        self._q = tf.convert_to_tensor(q, dtype=tf.float32)

    def get_intrinsic_matrix(self):
        return self.intrinsic_proj_mat

    def quaternion_to_matrix(self, q):
        """
        Construct a 4x4 transformation matrix from a quaternion using TensorFlow operations.

        Args:
            q (list, np.ndarray, or tf.Tensor): Quaternion [qw, qx, qy, qz]

        Returns:
            tf.Tensor: 4x4 transformation matrix
        """
        # Convert to tensor and ensure float32 dtype
        q = tf.convert_to_tensor(q, dtype=tf.float32)
        
        # Normalize the quaternion
        q = q / tf.norm(q)

        qw, qx, qy, qz = tf.unstack(q)

        # Create the rotation matrix from quaternion
        # Row 0
        r00 = 1 - 2 * qy * qy - 2 * qz * qz
        r01 = 2 * qx * qy - 2 * qz * qw
        r02 = 2 * qx * qz + 2 * qy * qw
        r03 = tf.constant(0.0)

        # Row 1
        r10 = 2 * qx * qy + 2 * qz * qw
        r11 = 1 - 2 * qx * qx - 2 * qz * qz
        r12 = 2 * qy * qz - 2 * qx * qw
        r13 = tf.constant(0.0)

        # Row 2
        r20 = 2 * qx * qz - 2 * qy * qw
        r21 = 2 * qy * qz + 2 * qx * qw
        r22 = 1 - 2 * qx * qx - 2 * qy * qy
        r23 = tf.constant(0.0)

        # Row 3
        r30 = tf.constant(0.0)
        r31 = tf.constant(0.0)
        r32 = tf.constant(0.0)
        r33 = tf.constant(1.0)

        # Stack rows to create the matrix
        row0 = tf.stack([r00, r01, r02, r03])
        row1 = tf.stack([r10, r11, r12, r13])
        row2 = tf.stack([r20, r21, r22, r23])
        row3 = tf.stack([r30, r31, r32, r33])

        matrix = tf.stack([row0, row1, row2, row3])

        return matrix

    def get_trans_matrix(self, position, quaternion=None, rot_mat=None):
        """
        Construct a 4x4 transformation matrix from a position and quaternion using TensorFlow operations.
        Returns the extrinsic transformation matrix that transform the camera coordinates to the world coordinates.
        Inverse of this matrix transforms the world coordinates to the camera coordinates.

        Args:
            position (list, np.ndarray, or tf.Tensor): Position vector [x, y, z]
            quaternion (list, np.ndarray, or tf.Tensor): Quaternion [qw, qx, qy, qz]
            rot_mat (tf.Tensor): Rotation matrix (optional)

        Returns:
            tf.Tensor: 4x4 transformation matrix
        """
        # Convert position to tensor
        position = tf.convert_to_tensor(position, dtype=tf.float32)

        # Get rotation matrix from quaternion or use provided rotation matrix
        if quaternion is not None:
            matrix = self.quaternion_to_matrix(quaternion)
        elif rot_mat is not None:
            rot_mat = tf.convert_to_tensor(rot_mat, dtype=tf.float32)
            # Create 4x4 matrix with rotation in top-left 3x3
            matrix = tf.concat([
                tf.concat([rot_mat, tf.zeros((3, 1))], axis=1),
                tf.constant([[0.0, 0.0, 0.0, 1.0]])
            ], axis=0)
        else:
            raise ValueError("Either quaternion or rotation matrix must be provided.")

        # Update translation column using tensor operations
        # Extract the rotation part (first 3 columns)
        rotation_part = matrix[:, :3]
        
        # Create translation column [x, y, z, 1]
        translation_col = tf.concat([position, tf.constant([1.0])], axis=0)
        translation_col = tf.expand_dims(translation_col, axis=1)
        
        # Combine rotation and translation
        matrix = tf.concat([rotation_part, translation_col], axis=1)

        return matrix

    def integrate_quaternion(self, q, omega, dt): # TODO: debug?
        """
        Integrate quaternion given angular velocity using TensorFlow
        The rotation is executed along the body fixed coordinate axes of the rotating object,
        not the world coordinate axes.
        
        Args:
            q: Current quaternion tensor [w, x, y, z]
            omega: Angular velocity vector tensor [wx, wy, wz] (rad/s)
            dt: Time step (scalar tensor or float)
        
        Returns:
            New integrated quaternion tensor [w, x, y, z]
        """
        # Convert inputs to tensors if needed
        q = tf.convert_to_tensor(q, dtype=tf.float32)
        omega = tf.convert_to_tensor(omega, dtype=tf.float32)
        dt = tf.convert_to_tensor(dt, dtype=tf.float32)
        
        # Calculate magnitude of angular velocity
        omega_magnitude = tf.norm(omega)
        
        # Handle very small rotations
        def no_rotation():
            return q
        
        def quaternion_multiply(q1, q2):
            """Multiply two quaternions in TensorFlow
            
            Args:
                q1: First quaternion tensor [w, x, y, z]
                q2: Second quaternion tensor [w, x, y, z]
            
            Returns:
                Product quaternion tensor [w, x, y, z]
            """
            w1, x1, y1, z1 = tf.unstack(q1, axis=-1)
            w2, x2, y2, z2 = tf.unstack(q2, axis=-1)
            
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            return tf.stack([w, x, y, z], axis=-1)
        
        def apply_rotation():
            # Create rotation quaternion for this time step
            angle = omega_magnitude * dt
            axis = omega / omega_magnitude
            
            # Rotation quaternion: q_rot = [cos(θ/2), sin(θ/2)*axis]
            half_angle = angle / 2.0
            cos_half = tf.cos(half_angle)
            sin_half = tf.sin(half_angle)
            
            q_rot = tf.concat([
                tf.expand_dims(cos_half, 0),
                sin_half * axis
            ], axis=0)
            
            # Apply rotation: q_new = q_rot * q (quaternion multiplication)
            q_new = quaternion_multiply(q_rot, q)
            
            # Normalize to prevent drift
            q_normalized = q_new / tf.norm(q_new)
            return q_normalized
        
        # Use conditional to handle small rotations
        result = tf.cond(
            omega_magnitude < 1e-12,
            no_rotation,
            apply_rotation
        )
        
        return result


def generate_rand_rot(degrees=(0, 0, 0)):
    """
    Generate a random rotation matrix.

    Args:
        degrees (tuple): Maximum rotation angles in degrees for roll, pitch, yaw.
    """
    roll = np.random.uniform(-degrees[0], degrees[0])
    pitch = np.random.uniform(-degrees[1], degrees[1])
    yaw = np.random.uniform(-degrees[2], degrees[2])

    # Convert to radians
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    euler_degrees = np.array([roll, pitch, yaw])
    euler_angles = np.array([roll_rad, pitch_rad, yaw_rad])
    quaternion = euler2quat(roll_rad, pitch_rad, yaw_rad, axes="sxyz")

    rotation_matrix = quat2mat(quaternion)
    return rotation_matrix, quaternion, euler_angles, euler_degrees


def generate_rand_pos(max_distance=1.0):
    """
    Generate a random position vector within a specified distance.

    Args:
        max_distance (float): Maximum distance from the origin.
    """
    x = np.random.uniform(-max_distance, max_distance)
    y = np.random.uniform(-max_distance, max_distance)
    z = np.random.uniform(-max_distance, max_distance)
    return np.array([x, y, z])

