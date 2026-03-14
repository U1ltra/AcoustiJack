import tensorflow as tf
import numpy as np
import time

class CoordinateTransformation:
    def __init__(self):
        # Pre-compute static corner templates
        self.corner_template_raceway = tf.constant(
            [
                [2.0, 2.0, 0.75],
                [-2.0, 2.0, 0.75],
                [-2.0, 2.0, 0.75],
                [2.0, 2.0, 0.75],
                [2.0, -2.0, 0.75],
                [-2.0, -2.0, 0.75],
                [-2.0, -2.0, 0.75],
                [2.0, -2.0, 0.75],
            ],
            dtype=tf.float32,
        )

        self.corner_template_default = tf.constant(
            [
                [0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0],
                [0.5, -0.5, 1.0],
                [-0.5, -0.5, 1.0],
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
            ],
            dtype=tf.float32,
        )

    def get_vertices_vectorized(self, world_pose, world_file):
        """Vectorized vertex computation"""
        if world_file == "raceway" or world_file == "urban_car":
            corners = self.corner_template_raceway
        else:
            corners = self.corner_template_default

        # Add world pose to all corners at once
        return corners + tf.expand_dims(world_pose, 0)

    def project_vertices_batch(self, vertices_3d, w2c, camera_projection_matrix):
        """
        Vectorized projection of 3D vertices to 2D pixels

        Args:
            vertices_3d: [8, 3] tensor of 3D vertices
            w2c_inv: [4, 4] inverse world-to-camera matrix
            camera_projection_matrix: [3, 4] camera projection matrix
        """
        ## Transform to camera coordinates (vectorized) ##
        vertices_homogeneous = tf.concat(
            [vertices_3d, tf.ones([tf.shape(vertices_3d)[0], 1], dtype=tf.float32)],
            axis=1,
        )  # [8, 4]

        # Apply camera transformation
        camera_coords = tf.linalg.matmul(
            w2c, vertices_homogeneous, transpose_b=True
        )  # [4, 8]
        camera_coords = tf.transpose(camera_coords)[:, :3]  # [8, 3]
        
        # Apply world-to-camera transformation
        camera_coords = tf.linalg.matmul(w2c, vertices_homogeneous, transpose_b=True)  # [4, 8]
        camera_coords = tf.transpose(camera_coords)[:, :3]  # [8, 3]
        
        # Apply coordinate system transformation to match world_to_camera function
        x, y, z = tf.unstack(camera_coords, axis=1)  # [8], [8], [8]
        
        # Dynamic camera: [y, z, x]  
        camera_coords_corrected = tf.stack([y, z, x], axis=1)  # [8, 3]

        ## Transform to camera pixel coordinates (vectorized) ##
        # flip y and z
        x, y, z = tf.unstack(camera_coords_corrected, axis=1)
        camera_coords_corrected = tf.stack([x, -y, -z], axis=1)  # [8, 3]

        # Add homogeneous coordinate
        camera_homogeneous = tf.concat(
            [
                camera_coords_corrected,
                tf.ones([tf.shape(camera_coords_corrected)[0], 1], dtype=tf.float32),
            ],
            axis=1,
        )  # [8, 4]

        # Project to image coordinates (vectorized)
        projected = tf.linalg.matmul(
            camera_projection_matrix, camera_homogeneous, transpose_b=True
        )  # [3, 8]
        projected = tf.transpose(projected)  # [8, 3]

        # Normalize by homogeneous coordinate
        x_proj, y_proj, w = tf.unstack(projected, axis=1)

        # Handle division by zero
        valid_mask = tf.abs(w) > 1e-10
        u = tf.where(valid_mask, x_proj / w, tf.fill(tf.shape(x_proj), float("nan")))
        v = tf.where(valid_mask, y_proj / w, tf.fill(tf.shape(y_proj), float("nan")))

        pixels_2d = tf.stack([u, v], axis=1)  # [8, 2]

        return pixels_2d

    def get_2d_bbox_vectorized(self, vertices_3d, w2c, camera_projection_matrix):
        """
        Optimized 2D bbox computation using vectorized operations
        """
        start_ = time.time()
        pixels_2d = self.project_vertices_batch(
            vertices_3d, w2c, camera_projection_matrix
        )
        proj_end_ = time.time()

        # Filter out invalid projections
        valid_mask = tf.logical_not(tf.reduce_any(tf.math.is_nan(pixels_2d), axis=1))
        valid_pixels = tf.boolean_mask(pixels_2d, valid_mask)
        filter_end_ = time.time()

        # Handle case where no valid pixels (shouldn't happen in normal operation)
        def compute_bbox():
            x_coords = valid_pixels[:, 0]
            y_coords = valid_pixels[:, 1]
            return tf.stack(
                [
                    tf.reduce_min(x_coords),
                    tf.reduce_min(y_coords),
                    tf.reduce_max(x_coords),
                    tf.reduce_max(y_coords),
                ]
            )

        def default_bbox():
            return tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)

        bbox = tf.cond(tf.shape(valid_pixels)[0] > 0, compute_bbox, default_bbox)

        return bbox

    def apply_gimbal_rotation_optimized(
        self, ex_mat, in_mat, atk_states, dt, frame_idx=None
    ):
        """
        Optimized gimbal rotation with pre-computed matrices and vectorized operations
        """
        start = time.time()
        world_file = atk_states["world_file"]

        # Get object centers
        victim_3d_center = atk_states["victim_3d_center"]
        attacker_3d_center = atk_states["attacker_3d_center"]

        # Use sampled velocities if available
        if "sampled_trajectory" in atk_states and frame_idx is not None:
            sampled_trajectory = atk_states["sampled_trajectory"]
            victim_velocity = sampled_trajectory["victim_velocities"][frame_idx]
            attacker_velocity = sampled_trajectory["attacker_velocities"][frame_idx]
        else:
            victim_velocity = atk_states["victim_velocity"]
            attacker_velocity = atk_states["attacker_velocity"]

        dt = tf.convert_to_tensor(dt, dtype=tf.float32)

        # Calculate new positions
        victim_3d_center_at_dt = victim_3d_center + victim_velocity * dt
        attacker_3d_center_at_dt = attacker_3d_center + attacker_velocity * dt

        # Get vertices using vectorized approach
        start = time.time()
        victim_3d_vertices = self.get_vertices_vectorized(
            victim_3d_center_at_dt, world_file
        )
        attacker_3d_vertices = self.get_vertices_vectorized(
            attacker_3d_center_at_dt, world_file
        )
        get_ver_end = time.time()
        # print(f"Vertex computation time: {get_ver_end - start:.6f} seconds")

        # Pre-compute inverse matrix and camera projection matrix
        w2c_inv = tf.linalg.inv(ex_mat)
        camera_projection_matrix = tf.cast(in_mat, tf.float32)
        prep_end = time.time()
        # print(f"Pre-computation time: {prep_end - get_ver_end:.6f} seconds")

        # Get 2D bboxes using vectorized operations
        victim_2d_bbox = self.get_2d_bbox_vectorized(
            victim_3d_vertices, w2c_inv, camera_projection_matrix
        )
        attacker_2d_bbox = self.get_2d_bbox_vectorized(
            attacker_3d_vertices, w2c_inv, camera_projection_matrix
        )
        bbox_end = time.time()
        # print(f"2D bbox computation time: {bbox_end - prep_end:.6f} seconds")

        # Apply scaling (vectorized)
        def scale_bbox(bbox, wscale=0.7, hscale=1.0):
            x1, y1, x2, y2 = tf.unstack(bbox)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1) * wscale, (y2 - y1) * hscale
            return tf.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        victim_2d_scaled = scale_bbox(victim_2d_bbox)
        attacker_2d_scaled = scale_bbox(attacker_2d_bbox)
        scale_end = time.time()
        # print(f"Scaling time: {scale_end - bbox_end:.6f} seconds")

        return {
            "victim_2d_bbox": victim_2d_scaled,
            "attacker_2d_bbox": attacker_2d_scaled,
        }




class CoordinateTransformation2:
    def __init__(self):
        # Pre-compute static corner templates
        self.corner_template_raceway = tf.constant(
            [
                [2.0, 2.0, 0.75],
                [-2.0, 2.0, 0.75],
                [-2.0, 2.0, 0.75],
                [2.0, 2.0, 0.75],
                [2.0, -2.0, 0.75],
                [-2.0, -2.0, 0.75],
                [-2.0, -2.0, 0.75],
                [2.0, -2.0, 0.75],
            ],
            dtype=tf.float32,
        )

        self.corner_template_default = tf.constant(
            [
                [0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [-0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0],
                [0.5, -0.5, 1.0],
                [-0.5, -0.5, 1.0],
                [-0.5, -0.5, 1.0],
                [0.5, -0.5, 1.0],
            ],
            dtype=tf.float32,
        )

    def get_vertices_batch(self, world_poses, world_file):
        """Batched vertex computation for multiple objects"""
        if world_file == "raceway" or world_file == "urban_car":
            corners = self.corner_template_raceway
        else:
            corners = self.corner_template_default

        # world_poses: [N, 3] where N is number of objects (2 in this case)
        # corners: [8, 3]
        # Result: [N, 8, 3]
        return corners[None, :, :] + world_poses[:, None, :]

    def project_vertices_batch_multi(self, vertices_3d_batch, w2c, camera_projection_matrix):
        """
        Vectorized projection of multiple objects' 3D vertices to 2D pixels

        Args:
            vertices_3d_batch: [N, 8, 3] tensor of 3D vertices for N objects
            w2c: [4, 4] world-to-camera matrix
            camera_projection_matrix: [3, 4] camera projection matrix
            
        Returns:
            pixels_2d_batch: [N, 8, 2] tensor of 2D pixel coordinates
        """
        batch_size, num_vertices, _ = tf.unstack(tf.shape(vertices_3d_batch))
        
        # Reshape for batch processing: [N*8, 3] -> [N*8, 4]
        vertices_flat = tf.reshape(vertices_3d_batch, [-1, 3])
        vertices_homogeneous = tf.concat(
            [vertices_flat, tf.ones([tf.shape(vertices_flat)[0], 1], dtype=tf.float32)],
            axis=1,
        )  # [N*8, 4]

        # Apply camera transformation (batched)
        camera_coords = tf.linalg.matmul(
            w2c, vertices_homogeneous, transpose_b=True
        )  # [4, N*8]
        camera_coords = tf.transpose(camera_coords)[:, :3]  # [N*8, 3]
        
        # Apply coordinate system transformation
        x, y, z = tf.unstack(camera_coords, axis=1)  # [N*8], [N*8], [N*8]
        camera_coords_corrected = tf.stack([y, z, x], axis=1)  # [N*8, 3]

        # Flip y and z
        x, y, z = tf.unstack(camera_coords_corrected, axis=1)
        camera_coords_corrected = tf.stack([x, -y, -z], axis=1)  # [N*8, 3]

        # Add homogeneous coordinate
        camera_homogeneous = tf.concat(
            [
                camera_coords_corrected,
                tf.ones([tf.shape(camera_coords_corrected)[0], 1], dtype=tf.float32),
            ],
            axis=1,
        )  # [N*8, 4]

        # Project to image coordinates (batched)
        projected = tf.linalg.matmul(
            camera_projection_matrix, camera_homogeneous, transpose_b=True
        )  # [3, N*8]
        projected = tf.transpose(projected)  # [N*8, 3]

        # Normalize by homogeneous coordinate
        x_proj, y_proj, w = tf.unstack(projected, axis=1)

        # Handle division by zero
        valid_mask = tf.abs(w) > 1e-10
        u = tf.where(valid_mask, x_proj / w, tf.fill(tf.shape(x_proj), float("nan")))
        v = tf.where(valid_mask, y_proj / w, tf.fill(tf.shape(y_proj), float("nan")))

        pixels_2d = tf.stack([u, v], axis=1)  # [N*8, 2]
        
        # Reshape back to [N, 8, 2]
        pixels_2d_batch = tf.reshape(pixels_2d, [batch_size, num_vertices, 2])

        return pixels_2d_batch

    def get_2d_bbox_batch(self, vertices_3d_batch, w2c, camera_projection_matrix):
        """
        Optimized 2D bbox computation for multiple objects using vectorized operations
        
        Args:
            vertices_3d_batch: [N, 8, 3] tensor of 3D vertices for N objects
            
        Returns:
            bboxes: [N, 4] tensor of 2D bounding boxes
        """
        pixels_2d_batch = self.project_vertices_batch_multi(
            vertices_3d_batch, w2c, camera_projection_matrix
        )  # [N, 8, 2]
        
        # Process each object's bbox in parallel
        batch_size = tf.shape(pixels_2d_batch)[0]
        
        def compute_single_bbox(pixels_2d):
            """Compute bbox for a single object"""
            # Filter out invalid projections
            valid_mask = tf.logical_not(tf.reduce_any(tf.math.is_nan(pixels_2d), axis=1))
            valid_pixels = tf.boolean_mask(pixels_2d, valid_mask)
            
            def compute_bbox():
                x_coords = valid_pixels[:, 0]
                y_coords = valid_pixels[:, 1]
                return tf.stack([
                    tf.reduce_min(x_coords),
                    tf.reduce_min(y_coords),
                    tf.reduce_max(x_coords),
                    tf.reduce_max(y_coords),
                ])
            
            def default_bbox():
                return tf.constant([0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
            
            return tf.cond(tf.shape(valid_pixels)[0] > 0, compute_bbox, default_bbox)
        
        # Use tf.map_fn to apply bbox computation to each object in parallel
        bboxes = tf.map_fn(
            compute_single_bbox, 
            pixels_2d_batch, 
            fn_output_signature=tf.TensorSpec(shape=[4], dtype=tf.float32),
            parallel_iterations=10
        )
        
        return bboxes

    def scale_bbox_batch(self, bboxes, wscale=0.7, hscale=1.0):
        """Apply scaling to multiple bboxes simultaneously"""
        # bboxes: [N, 4]
        x1, y1, x2, y2 = tf.unstack(bboxes, axis=1)  # Each is [N]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1) * wscale, (y2 - y1) * hscale
        return tf.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    def apply_gimbal_rotation_optimized(
        self, ex_mat, in_mat, atk_states, dt, frame_idx=None
    ):
        """
        Optimized gimbal rotation with batched processing of victim and attacker
        """
        start = time.time()
        world_file = atk_states["world_file"]

        # Get object centers
        victim_3d_center = atk_states["victim_3d_center"]
        attacker_3d_center = atk_states["attacker_3d_center"]

        # Use sampled velocities if available
        if "sampled_trajectory" in atk_states and frame_idx is not None:
            sampled_trajectory = atk_states["sampled_trajectory"]
            victim_velocity = sampled_trajectory["victim_velocities"][frame_idx]
            attacker_velocity = sampled_trajectory["attacker_velocities"][frame_idx]
        else:
            victim_velocity = atk_states["victim_velocity"]
            attacker_velocity = atk_states["attacker_velocity"]

        dt = tf.convert_to_tensor(dt, dtype=tf.float32)

        # Calculate new positions for both objects
        victim_3d_center_at_dt = victim_3d_center + victim_velocity * dt
        attacker_3d_center_at_dt = attacker_3d_center + attacker_velocity * dt
        
        # Stack positions for batched processing: [2, 3]
        world_poses = tf.stack([victim_3d_center_at_dt, attacker_3d_center_at_dt], axis=0)

        # Get vertices for both objects at once: [2, 8, 3]
        start_vertex = time.time()
        vertices_3d_batch = self.get_vertices_batch(world_poses, world_file)
        get_ver_end = time.time()
        print(f"Batched vertex computation time: {get_ver_end - start_vertex:.6f} seconds")

        # Pre-compute matrices
        w2c_inv = tf.linalg.inv(ex_mat)
        camera_projection_matrix = tf.cast(in_mat, tf.float32)
        prep_end = time.time()
        print(f"Pre-computation time: {prep_end - get_ver_end:.6f} seconds")

        # Get 2D bboxes for both objects in one call: [2, 4]
        bboxes_2d = self.get_2d_bbox_batch(
            vertices_3d_batch, w2c_inv, camera_projection_matrix
        )
        bbox_end = time.time()
        print(f"Batched 2D bbox computation time: {bbox_end - prep_end:.6f} seconds")

        # Apply scaling to both bboxes at once: [2, 4]
        bboxes_2d_scaled = self.scale_bbox_batch(bboxes_2d)
        scale_end = time.time()
        print(f"Batched scaling time: {scale_end - bbox_end:.6f} seconds")

        # Extract individual results
        victim_2d_scaled = bboxes_2d_scaled[0]
        attacker_2d_scaled = bboxes_2d_scaled[1]

        total_time = scale_end - start
        print(f"Total batched processing time: {total_time:.6f} seconds")

        return {
            "victim_2d_bbox": victim_2d_scaled,
            "attacker_2d_bbox": attacker_2d_scaled,
        }