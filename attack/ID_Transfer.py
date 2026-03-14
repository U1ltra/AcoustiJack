import tensorflow as tf
from math import cos, sin
from utils.sort_TF import *
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import quat2mat, mat2quat


# Generate 3D trajectory for the attacker
def optimize_3d_coord(
    tracker_A,
    tracker_B,
    verts_A,
    bbox_B,
    w2c,
    cam_model,
    lr=0.1,
    iteration=30,
    delta_location=0.3,
    static_camera=False,
):
    """
    Perform gradient descent on the 3D coordinates of the attacker w.r.t. to the adversarial loss
    Input:
    - tracker_A: KalmanFilterTracker object for the attacker
    - tracker_B: KalmanFilterTracker object for the victim
    - verts_A: attacker's 3D bounding box vertices (8 corner representation)
    - bbox_B: victim's current KF predicted states in the form of bbox
    - w2c: world to camera transformation matrix
    - cam_model: image_geometry.PinholeCameraModel instance
    - lr: learning rate alpha
    - iteration: max number of iterations
    - delta_location: limit the relative intra-frame movement for the attacker, ensuring physical realizability
    Output:
    - A (x,y,z) point in Gazebo World that represents where the attacker should move (center displacement)
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    center_displacement = tf.Variable([0.0, 0.0])
    z_B_next_pred = update_predict(
        tracker_B, convert_bbox_to_z(bbox_B)
    )  # Victim's KF prediction for the next time step
    bbox_B_curr_pred = tracker_B.predict_no_trace()[0]

    for it in range(iteration):
        # Stop if the displacement is too large
        if (
            tf.sqrt(center_displacement[0] ** 2 + center_displacement[1] ** 2)
            >= delta_location
        ):
            break

        with tf.GradientTape() as tape:
            tape.watch(center_displacement)
            verts_A_shifted = shift_verts(verts_A, center_displacement)
            bbox_A = get_2d_bbox_tf(verts_A_shifted, w2c, cam_model, static_camera)
            z_A = convert_to_z(bbox_A)  # Problem!!!
            z_predicted = update_predict(tracker_A, z_A)
            bbox_A_predicted = convert_to_bbox(z_predicted)  # Print and check
            iou_loss_1 = iou(bbox_A, bbox_B_curr_pred)
            iou_loss_2 = iou(bbox_A_predicted, convert_to_bbox(z_B_next_pred))
            loss = -(iou_loss_1 + iou_loss_2) + (
                distance_loss(bbox_A, bbox_B_curr_pred)
                + distance_loss_x(z_predicted, z_B_next_pred)
            )
            # Change distance loss to non-x

        grads = tape.gradient(loss, [center_displacement])
        optimizer.apply_gradients(zip(grads, [center_displacement]))

    # print(f"Iteration used: {it}")

    return center_displacement.numpy()


def update_predict(tracker, z, s=2400, r=1.5):
    """
    Return the KF prior prediction if we update the current tracker with z
    Input:
        - tracker: KalmanFilterTracker object
        - z: the state vector tf.Tensor([x, y, s, r]), shape=(4, 1)
                or tf.Tensor([x,y]), shape=(2,1)
        - s: scale of the bounding box (w * h)
        - r: aspect ratio of the bounding box
    Output:
        Prior prediction for the next time step
    """
    if z.shape[0] == 2:
        z = tf.concat([z, [[s], [r]]], axis=0)
    elif z.shape[0] == 3:
        z = tf.concat([z, [r]], axis=0)
    elif z.shape[0] == 4:
        z = tf.reshape(z, [4, 1])
    # Update step
    dim_x = tracker.kf.dim_x
    x = tracker.kf.x
    y = tracker.kf.y
    R = tracker.kf.R
    H = tracker.kf.H
    P = tracker.kf.P
    y = z - tf.matmul(H, x)
    PHT = tf.matmul(P, tf.transpose(H))
    S = tf.matmul(H, PHT) + R
    SI = tf.linalg.inv(S)
    K = tf.matmul(PHT, SI)
    x = x + tf.matmul(K, y)
    I_KH = tf.eye(dim_x) - tf.matmul(K, H)
    P = tf.matmul(tf.matmul(I_KH, P), tf.transpose(I_KH)) + tf.matmul(
        tf.matmul(K, R), tf.transpose(K)
    )
    # Prediction Step
    F = tracker.kf.F
    x = tf.matmul(F, x)
    return x


def shift_verts(verts, delta):
    shift = tf.pad(delta, tf.constant([[0, 1]]))
    shift = tf.reshape(shift, [1, 3])
    shifts = tf.repeat(shift, [8], 0)
    verts = verts + shifts
    return verts


def get_2d_bbox_tf(verts, w2c, camera_model, static_camera=False):
    """
    Get 2D bounding box from 3D vertices
    Input:
    - verts: 8 * 3 tensor of 3D vertices, shape=(8, 3), dtype=tf.float32
    - w2c: world to camera transformation matrix, shape=(4, 4), dtype=tf.float32
    - camera_model: image_geometry.PinholeCameraModel instance
    """

    def project_single_vert(vert):
        camera_coord = world_to_camera(vert, w2c, static_camera)
        pixel_coord, _ = project_box_3d_to_2d(camera_coord, camera_model)
        return pixel_coord

    # Map over all vertices using tf.map_fn
    pixel_coords = tf.map_fn(project_single_vert, verts)
    pixel_coords = tf.squeeze(pixel_coords)

    # Calculate min/max using tf operations
    x_min = tf.reduce_min(pixel_coords[:, 0])
    x_max = tf.reduce_max(pixel_coords[:, 0])
    y_min = tf.reduce_min(pixel_coords[:, 1])
    y_max = tf.reduce_max(pixel_coords[:, 1])

    return tf.stack([x_min, y_min, x_max, y_max])


def convert_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return tf.stack([x, y, s, r])


def convert_to_bbox(x):
    s, r = x[2], x[3]
    w = tf.sqrt(s * r)
    h = s / w
    return tf.stack([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0])


def iou(bb_a, bb_b):
    xx1 = max(bb_a[0], bb_b[0])
    yy1 = max(bb_a[1], bb_b[1])
    xx2 = min(bb_a[2], bb_b[2])
    yy2 = min(bb_a[3], bb_b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_a[2] - bb_a[0]) * (bb_a[3] - bb_a[1])
        + (bb_b[2] - bb_b[0]) * (bb_b[3] - bb_b[1])
        - wh
    )
    return o


def distance_loss(bb_a, bb_b):
    """
    Defined as the squared Euclidean distance of the two central points
    divided by square of the diagnal length of the smallest enclosing box
    covering the two boxes.
    """
    w_a = bb_a[2] - bb_a[0]
    w_b = bb_b[2] - bb_b[0]
    h_a = bb_a[3] - bb_a[1]
    h_b = bb_b[3] - bb_b[1]
    x_a = bb_a[2] + w_a / 2.0
    x_b = bb_b[2] + w_b / 2.0
    y_a = bb_a[3] + h_a / 2.0
    y_b = bb_b[3] + h_b / 2.0
    diag_a = tf.sqrt(tf.pow(w_a, 2) + tf.pow(h_a, 2))
    diag_b = tf.sqrt(tf.pow(w_b, 2) + tf.pow(h_b, 2))
    distance_sq = tf.pow(x_a - x_b, 2) + tf.pow(y_a - y_b, 2)
    loss = distance_sq / tf.pow(tf.sqrt(distance_sq) + diag_a / 2 + diag_b / 2, 2)
    return loss


def distance_loss_x(x_a, x_b):
    """
    Defined as the squared Euclidean distance of the two central points
    divided by square of the diagnal length of the smallest enclosing box
    covering the two boxes.
    """
    w_a = x_a[2] * x_a[3]
    w_b = x_b[2] * x_b[3]
    h_a = x_a[2] / w_a
    h_b = x_b[2] / w_b
    diag_a = tf.sqrt(tf.pow(w_a, 2) + tf.pow(h_a, 2))
    diag_b = tf.sqrt(tf.pow(w_b, 2) + tf.pow(h_b, 2))
    distance_sq = tf.pow(x_a[0] - x_b[0], 2) + tf.pow(x_a[1] - x_b[1], 2)
    loss = distance_sq / tf.pow(tf.sqrt(distance_sq) + diag_a / 2 + diag_b / 2, 2)
    return loss


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = tf.cast(tf.expand_dims(bb_gt, 0), tf.float32)
    bb_test = tf.cast(tf.expand_dims(bb_test, 1), tf.float32)

    xx1 = tf.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = tf.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = tf.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = tf.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = tf.maximum(0.0, xx2 - xx1)
    h = tf.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


"""
3D-2D conversion functions
"""


## Camera -> Pixel ##
def project_box_3d_to_2d(box_params, camera_model):
    """
    Project 3D box (camera coordinate) to 2D image coordinates using PinholeCameraModel

    Args:
        box_params: dict containing:
            - position: [x, y, z]
            - dimensions: [width, height, length]
            - quaternion: [x, y, z, w]
        camera_model: image_geometry.PinholeCameraModel instance

    Returns:
        pts_2d: 2D points of the box corners (with dim) or center
    """
    # Extract parameters
    if isinstance(
        box_params, dict
    ):  # TODO: Check if this is correct. Could use the transform matrix instead
        position = tf.cast(box_params["position"], tf.float32)
        dimensions = tf.cast(box_params["dimensions"], tf.float32)
        qx, qy, qz, qw = tf.cast(box_params["quaternion"], tf.float32)
        R = tf.cast(quat2mat([qw, qx, qy, qz]), tf.float32)
        R = tf.cast(R, tf.float32)

        # Get box corners in model coordinates
        w, h, l = tf.unstack(dimensions)
        corners = tf.stack(
            [
                tf.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]),
                tf.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]),
                tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]),
            ]
        )

        # Transform to camera coordinates
        corners_3d = tf.linalg.matmul(R, corners)
        corners_3d = corners_3d + tf.expand_dims(position, 1)
        corners_3d = tf.transpose(corners_3d)
    else:
        # Assume box is a point
        corners_3d = tf.expand_dims(tf.cast(box_params, tf.float32), 0)

    # Project points using camera model
    def project_point(point):
        """
        point: 3D point (tensor of shape [3])
        Returns: (u, v) pixel coordinates
        """
        P = camera_model.projection_matrix()
        P = tf.cast(P, tf.float32)

        x, y, z = tf.unstack(point)
        point_homogeneous = tf.concat([[x, -y, -z, 1.0]], axis=1)
        projected = tf.linalg.matmul(P, tf.expand_dims(point_homogeneous, 1))

        x, y, w = tf.unstack(tf.squeeze(projected))
        uv = tf.stack([x / w, y / w])
        return tf.where(tf.abs(w) > 1e-10, uv, tf.fill(uv.shape, float("nan")))

    pts_2d = tf.map_fn(lambda x: tf.cast(project_point(x), tf.float32), corners_3d)

    bbox = tf.stack(
        [
            tf.reduce_min(pts_2d[:, 0]),
            tf.reduce_min(pts_2d[:, 1]),
            tf.reduce_max(pts_2d[:, 0]),
            tf.reduce_max(pts_2d[:, 1]),
        ]
    )

    return pts_2d, bbox


## World -> Camera ##
def world_to_camera(point_world, w2c, static_camera=False):
    """Tranform first, then align axes"""

    w2c = tf.cast(w2c, tf.float32)
    # Add 1 to the point to make it homogeneous
    point_world = tf.concat([point_world, [1.0]], axis=0)
    point_camera = tf.linalg.matmul(w2c, tf.expand_dims(point_world, 1))
    point_camera = tf.squeeze(point_camera)

    # use tf to [point_camera[1], point_camera[2], point_camera[0]]
    if static_camera:
        point_camera = tf.stack(
            [
                -tf.gather(point_camera, 1),
                tf.gather(point_camera, 2),
                -tf.gather(point_camera, 0),
            ]
        )
    else:
        point_camera = tf.stack(
            [
                tf.gather(point_camera, 1),
                tf.gather(point_camera, 2),
                tf.gather(point_camera, 0),
            ]
        )
    return point_camera


## Camera -> World ##
def project_box_camera_to_world(box_params, c2w, static_camera=False):
    """
    Project 3D box from camera coordinates to world coordinates

    Args:
        box_params: dict containing:
            - position: [x, y, z]
            - dimensions: [width, height, length]
            - quaternion: [x, y, z, w]
        c2w: transformation matrix from camera to world coordinates
        static_camera: somehow directly spawning a camera model has
            a different coordinate system compared to a camera on a PX4 UAV ...
            This whole camera thing just a mess ...

    Returns:
        corners: 3D coordinates of the box corners in world frame
        center_world: 3D coordinates of the box center in world frame
    """
    x, y, z = box_params["position"]
    w, h, l = box_params["dimensions"]
    qx, qy, qz, qw = box_params["quaternion"]

    center_camera = np.array(box_params["position"], dtype=np.float32)
    c2w = np.array(c2w, dtype=np.float32)

    corners = np.zeros((8, 3), dtype=np.float32)
    corners[:, 0] = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners[:, 1] = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners[:, 2] = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    corners = corners + center_camera

    def camera_to_world(det, c2w):
        """Align axes first, then rotate"""
        # 1. Align axes first
        if static_camera:
            point_camera = [-det[2], -det[0], det[1]]
        else:
            point_camera = [det[2], det[0], det[1]]
        point_aligned = np.array(point_camera).reshape(3, 1)

        # 2. Then apply transformation matrix
        point_world = np.dot(
            c2w,
            np.concatenate([point_aligned, np.ones((1, 1))], axis=0, dtype=np.float32),
        )

        return point_world[:3, 0]

    # Transform to world coordinates
    center_world = camera_to_world(center_camera, c2w)
    corners = np.array([camera_to_world(det, c2w) for det in corners])

    return corners, center_world


## Get Camera's World Pose ##
def transform_pose(base_pose, relative_pose):
    """
    Transform pose from relative to world coordinates

    Args:
        base_position: [x, y, z, qx, qy, qz, qw] base pose in world frame
        relative_pose: [x, y, z, qx, qy, qz, qw] relative pose in base frame

    Returns:
        world_pose: [x, y, z, qx, qy, qz, qw] pose in world frame
    """
    # Extract positions and orientations
    base_pos = np.array(base_pose[:3])
    rel_pos = relative_pose[:3]

    # Convert orientations to rotation matrices
    qx, qy, qz, qw = base_pose[3:]
    R_base = quat2mat([qw, qx, qy, qz])
    qx, qy, qz, qw = relative_pose[3:]
    R_rel = quat2mat([qw, qx, qy, qz])

    # Transform position to world frame:
    # First rotate the relative position vector by vehicle's orientation
    # Then add vehicle's position
    world_pos = R_base @ rel_pos + base_pos

    # Transform orientation to world frame:
    # Combine rotation matrices
    R_world = R_base @ R_rel

    # Convert rotation matrix Rworld back to euler angles, using Rotation library
    qw, qx, qy, qz = mat2quat(R_world)

    return np.array([*world_pos, qx, qy, qz, qw])


def camera_stability_analysis(pose_history):
    """
    Analyze the stability of the camera pose
    Args:
        pose_history: list of camera poses [[x, y, z, qx, qy, qz, qw, roll, pitch, yaw, time_stamp], ...]
    Returns:
        stability: list of stability values for each pose
    """

    history = np.array(pose_history)
    poses = history[:, :3]
    orientations = history[:, 7:10]
    time_stamps = history[:, -1]

    # Calculate the distance between each pose
    distances = np.diff(poses, axis=0)
    # Calculate the difference in orientation between each pose
    orientation_diffs = np.diff(orientations, axis=0)
    time_spans = np.diff(time_stamps) / 1e9  # Convert time to seconds

    linear_velocities = distances / time_spans.reshape(-1, 1)
    angular_velocities = orientation_diffs / time_spans.reshape(-1, 1)

    v_norm, ang_norm = np.linalg.norm(linear_velocities, axis=1), np.linalg.norm(
        angular_velocities, axis=1
    )
    v_std, ang_std = np.std(v_norm), np.std(ang_norm)
    v_mean, ang_mean = np.mean(v_norm), np.mean(ang_norm)
    time_span = np.mean(time_spans)

    return v_std, ang_std, v_mean, ang_mean, time_span

def get_facing_orientation_gazebo(pos1, pos2, current_orientation=None, ret_format='q'):
    """
    Calculate the orientation (as a quaternion) for object 2 (at pos2) 
    to face object 1 (at pos1), with option to calculate relative rotation.
    
    Args:
        pos1: Position of object 1 [x, y, z]
        pos2: Position of object 2 [x, y, z]
        current_orientation: Current orientation of object 2 as [roll, pitch, yaw] in radians
                            or as a quaternion [w, x, y, z]
        ret_format: Format of the return value, 'q' for quaternion, 'e' for euler angles,
                   'delta_e' for delta Euler angles from current orientation
    
    Returns:
        Orientation in the requested format
    """
    # Calculate the direction vector from object 2 to object 1
    direction = np.array(pos1) - np.array(pos2)
    
    # Check if positions are too close
    if np.linalg.norm(direction) < 1e-10:
        # Objects are at the same position, return identity quaternion
        if ret_format == 'q':
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.0, 0.0, 0.0])
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Reference forward vector for Gazebo Harmonic is +X axis
    ref_forward = np.array([1, 0, 0])
    
    # Calculate target orientation
    # Using the rotation from reference to target
    target_rot = R.from_rotvec(
        np.cross(ref_forward, direction) * 
        np.arccos(np.clip(np.dot(ref_forward, direction), -1.0, 1.0))
    )
    
    # If direction and reference are nearly parallel or anti-parallel, handle special cases
    dot_product = np.dot(direction, ref_forward)
    if abs(abs(dot_product) - 1.0) < 1e-10:
        if dot_product > 0:
            # Same direction, no rotation needed
            target_rot = R.from_quat([0, 0, 0, 1])  # Identity rotation
        else:
            # Opposite direction, rotate 180 degrees around Z axis (up in Gazebo)
            target_rot = R.from_euler('xyz', [0, 0, np.pi])
    
    # If current orientation is provided, calculate the delta rotation
    if current_orientation is not None:
        if len(current_orientation) == 4:  # Quaternion [w,x,y,z]
            current_rot = R.from_quat([current_orientation[1], current_orientation[2], 
                                       current_orientation[3], current_orientation[0]])
        else:  # Euler angles [roll, pitch, yaw]
            current_rot = R.from_euler('xyz', current_orientation)
        
        # Calculate the delta rotation (what rotation to apply to current to reach target)
        # This is target * inverse(current)
        delta_rot = target_rot * current_rot.inv()
        
        if ret_format == 'delta_e':
            # Return delta as Euler angles
            return delta_rot.as_euler('xyz')
        elif ret_format == 'e':
            # Return target as Euler angles
            return target_rot.as_euler('xyz')
        else:  # 'q'
            # Return target as quaternion [w,x,y,z]
            quat = target_rot.as_quat()
            return np.array([quat[3], quat[0], quat[1], quat[2]])
    
    # Return the requested format
    if ret_format == 'e':
        return target_rot.as_euler('xyz')
    else:  # 'q'
        quat = target_rot.as_quat()
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # [w,x,y,z]