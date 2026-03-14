import tensorflow as tf
tf.random.set_seed(42)  # Set random seed for reproducibility

from utils.sort_TF import KalmanBoxTracker, convert_bbox_to_z, convert_x_to_bbox


class MockTracker:
    """Implementation of SORT-like surrogate tracker for optimization with state restoration"""
    
    def __init__(self):
        self.initial_states = {}  # Store initial states for restoration
        self.current_states = {}  # Working states that get modified
    
    def add(self, kf: KalmanBoxTracker, track_id: int) -> None:
        """Add a kf states to the tracker and save initial state"""
        def extract_kf_states(kf: KalmanBoxTracker):
            if isinstance(kf.kf.x, (tf.Tensor, tf.Variable)):
                # If x is a TensorFlow tensor, extract its values
                return (
                    int(kf.kf.dim_x),
                    kf.kf.x.numpy(),      # Current state (7, 1)
                    kf.kf.P.numpy(),      # State covariance (7, 7) 
                    kf.kf.F.numpy(),      # State transition matrix (7, 7)
                    kf.kf.H.numpy(),      # Measurement matrix (4, 7)
                    kf.kf.R.numpy(),      # Measurement noise covariance (4, 4)
                    kf.kf.Q.numpy()       # Process noise covariance (7, 7)
                )
            else:
                # If x is a numpy array, return directly
                return (
                    kf.kf.dim_x,
                    kf.kf.x,              # Current state (7, 1)
                    kf.kf.P,              # State covariance (7, 7)
                    kf.kf.F,              # State transition matrix (7, 7)
                    kf.kf.H,              # Measurement matrix (4, 7)
                    kf.kf.R,              # Measurement noise covariance (4, 4)
                    kf.kf.Q               # Process noise covariance (7, 7)
                )

        if track_id not in self.current_states:
            states = extract_kf_states(kf)
            dim_x, x, P, F, H, R, Q = states

            if track_id in self.initial_states:
                raise ValueError(f"Track ID {track_id} already exists in tracker.")
            
            # Store initial states (immutable references)
            self.initial_states[track_id] = {
                'dim_x': tf.constant(dim_x, dtype=tf.int32),
                'x_init': tf.constant(x, dtype=tf.float32),
                'P_init': tf.constant(P, dtype=tf.float32),
                'F': tf.constant(F, dtype=tf.float32),
                'H': tf.constant(H, dtype=tf.float32),
                'R': tf.constant(R, dtype=tf.float32),
                'Q': tf.constant(Q, dtype=tf.float32)
            }
            
            # Create working variables (will be reset each iteration)
            self.current_states[track_id] = {
                'x': tf.Variable(x, dtype=tf.float32, trainable=False),
                'P': tf.Variable(P, dtype=tf.float32, trainable=False)
            }
        else:
            current = self.current_states[track_id]
            current['x'].assign(kf.kf.x)
            current['P'].assign(kf.kf.P)
    
    def reset_all_states(self):
        """Reset all tracker states to their initial values"""
        for track_id in self.current_states:
            self.reset_state(track_id)
    
    def reset_state(self, track_id: int):
        """Reset a specific tracker state to its initial value"""
        if track_id not in self.initial_states:
            raise ValueError(f"Track ID {track_id} not found")
        
        initial = self.initial_states[track_id]
        current = self.current_states[track_id]
        
        # Reset to initial values
        current['x'].assign(initial['x_init'])
        current['P'].assign(initial['P_init'])
    
    # @tf.function
    def predict_step(self, track_id: int):
        """
        Perform only the prediction step of Kalman filter
        
        Args:
            track_id: ID of the tracker
            
        Returns:
            predicted_x: predicted state after prediction step
        """
        if track_id not in self.initial_states:
            raise ValueError(f"Track ID {track_id} not found")
        
        # Get initial states (constants)
        initial = self.initial_states[track_id]
        F = initial['F']
        Q = initial['Q']
        dim_x = initial['dim_x']
        
        # Get current working states (variables)
        current = self.current_states[track_id]
        x = current['x']
        P = current['P']
        
        # Handle edge case: if width becomes negative, set width velocity to 0
        width_velocity_idx = 6  # assuming index 6 is width velocity
        area_prediction = x[6, 0] + x[2, 0]  # width_velocity + width
        
        # If predicted area <= 0, set width velocity to 0
        x_corrected = tf.cond(
            area_prediction <= 0,
            lambda: tf.tensor_scatter_nd_update(x, [[width_velocity_idx, 0]], [0.0]),
            lambda: x
        )
        
        # Update x if correction was needed
        x.assign(x_corrected)
        
        # x = Fx (state prediction)
        x_pred = tf.matmul(F, x)
        
        # P = FPF' + Q (covariance prediction)
        P_pred = tf.matmul(tf.matmul(F, P), tf.transpose(F)) + Q
        
        # Update stored states
        x.assign(x_pred)
        P.assign(P_pred)
        
        return x_pred, convert_x_to_bbox(x_pred)
    
    # @tf.function
    def update_step(self, track_id: int, bbox, s=2400.0, r=1.5):
        """
        Perform only the update step of Kalman filter
        
        Args:
            track_id: ID of the tracker
            bbox: measurement tensor, shape (4,1) or (2,1)
            s: default scale (w*h) if bbox converted z is incomplete
            r: default aspect ratio if bbox converted  z is incomplete
            
        Returns:
            updated_x: updated state after measurement incorporation
        """
        if track_id not in self.initial_states:
            raise ValueError(f"Track ID {track_id} not found")
        
        # Get initial states (constants)
        initial = self.initial_states[track_id]
        H = initial['H']
        R = initial['R']
        dim_x = initial['dim_x']
        
        # Get current working states (variables)
        current = self.current_states[track_id]
        x = current['x']
        P = current['P']
        
        z = convert_bbox_to_z(bbox)
        
        # Ensure z has correct shape and complete measurements
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        
        if tf.rank(z) == 1:
            z = tf.expand_dims(z, axis=1)  # Make it (n, 1)
        
        # Complete missing measurements
        if tf.shape(z)[0] == 2:
            z = tf.concat([z, [[s], [r]]], axis=0)
        elif tf.shape(z)[0] == 3:
            z = tf.concat([z, [[r]]], axis=0)
        
        # Ensure z is (4, 1)
        z = tf.ensure_shape(z, [4, 1])
        
        # Kalman Filter Update Step
        # y = z - Hx (innovation/residual)
        y = z - tf.matmul(H, x)
        
        # S = HPH' + R (innovation covariance)
        PHT = tf.matmul(P, tf.transpose(H))
        S = tf.matmul(H, PHT) + R
        
        # K = PH'S^-1 (Kalman gain)
        SI = tf.linalg.inv(S)
        K = tf.matmul(PHT, SI)
        
        # x = x + Ky (state update)
        x_updated = x + tf.matmul(K, y)
        
        # P = (I - KH)P(I - KH)' + KRK' (covariance update - Joseph form)
        I_KH = tf.eye(dim_x, dtype=tf.float32) - tf.matmul(K, H)
        P_updated = tf.matmul(tf.matmul(I_KH, P), tf.transpose(I_KH)) + \
                   tf.matmul(tf.matmul(K, R), tf.transpose(K))
        
        # Update current states
        x.assign(x_updated)
        P.assign(P_updated)
        
        return x_updated

    @tf.function
    def predict_and_update(self, track_id: int, z=None, s=2400.0, r=1.5):
        """
        Perform standard Kalman filter cycle: predict then update (if measurement available)
        
        Args:
            track_id: ID of the tracker
            z: measurement tensor, shape (4,1) or (2,1) or None for predict-only
            s: default scale (w*h) if z is incomplete
            r: default aspect ratio if z is incomplete
            
        Returns:
            final_state: state after prediction (and update if z provided)
        """
        # First do prediction
        predicted_state = self.predict_step(track_id)
        
        # Then optionally do update if measurement is available
        if z is not None:
            updated_state = self.update_step(track_id, z, s, r)
            return updated_state
        else:
            return convert_x_to_bbox(predicted_state)
    
    @tf.function 
    def predict_only(self, track_id: int):
        """Perform only prediction step without update"""
        return self.predict_step(track_id)
    
    @tf.function
    def update_and_predict(self, track_id: int, z, s=2400.0, r=1.5):
        """Perform both update and prediction steps in correct order"""
        return self.predict_and_update(track_id, z=z, s=s, r=r)
    
    @tf.function
    def get_current_state(self, track_id: int):
        """Get current state vector for a tracker"""
        if track_id not in self.current_states:
            raise ValueError(f"Track ID {track_id} not found")
        return self.current_states[track_id]['x'], convert_x_to_bbox(self.current_states[track_id]['x'])
    
    @tf.function
    def get_current_covariance(self, track_id: int):
        """Get current covariance matrix for a tracker"""
        if track_id not in self.current_states:
            raise ValueError(f"Track ID {track_id} not found")
        return self.current_states[track_id]['P']
    
    @tf.function
    def iou_matching(self, iou_matrix, iou_threshold=0.5):
        # Find best matches above threshold
        victim_det_best_tracker = tf.argmax(iou_matrix[0, :])  # Best tracker for victim detection
        attacker_det_best_tracker = tf.argmax(iou_matrix[1, :])  # Best tracker for attacker detection
        
        victim_det_best_iou = tf.reduce_max(iou_matrix[0, :])
        attacker_det_best_iou = tf.reduce_max(iou_matrix[1, :])
        
        # Check if victim detection matches victim tracker (ID=10)
        victim_det_to_victim_tracker = tf.logical_and(
            tf.equal(victim_det_best_tracker, 0),  # matches tracker 0 (victim)
            tf.greater(victim_det_best_iou, iou_threshold)
        )
        
        # Check if victim detection matches attacker tracker (ID=20)
        victim_det_to_attacker_tracker = tf.logical_and(
            tf.equal(victim_det_best_tracker, 1),  # matches tracker 1 (attacker)
            tf.greater(victim_det_best_iou, iou_threshold)
        )
        
        # Check if attacker detection matches victim tracker
        attacker_det_to_victim_tracker = tf.logical_and(
            tf.equal(attacker_det_best_tracker, 0),  # matches tracker 0 (victim)
            tf.greater(attacker_det_best_iou, iou_threshold)
        )
        
        # Check if attacker detection matches attacker tracker
        attacker_det_to_attacker_tracker = tf.logical_and(
            tf.equal(attacker_det_best_tracker, 1),  # matches tracker 1 (attacker)
            tf.greater(attacker_det_best_iou, iou_threshold)
        )
        
        # Resolve conflicts: if both detections want same tracker, give to higher IoU
        victim_tracker_conflict = tf.logical_and(
            victim_det_to_victim_tracker, 
            attacker_det_to_victim_tracker
        )
        attacker_tracker_conflict = tf.logical_and(
            victim_det_to_attacker_tracker, 
            attacker_det_to_attacker_tracker
        )
        
        # Handle victim tracker conflict
        victim_tracker_gets_victim_det = tf.cond(
            victim_tracker_conflict,
            lambda: tf.greater(iou_matrix[0, 0], iou_matrix[1, 0]),  # victim_det vs attacker_det IoU with victim_tracker
            lambda: victim_det_to_victim_tracker
        )
        victim_tracker_gets_attacker_det = tf.logical_and(
            attacker_det_to_victim_tracker,
            tf.logical_not(victim_tracker_gets_victim_det)
        )
        
        # Handle attacker tracker conflict
        attacker_tracker_gets_attacker_det = tf.cond(
            attacker_tracker_conflict,
            lambda: tf.greater(iou_matrix[1, 1], iou_matrix[0, 1]),  # attacker_det vs victim_det IoU with attacker_tracker
            lambda: attacker_det_to_attacker_tracker
        )
        attacker_tracker_gets_victim_det = tf.logical_and(
            victim_det_to_attacker_tracker,
            tf.logical_not(attacker_tracker_gets_attacker_det)
        )
        return (
            victim_tracker_gets_victim_det,
            victim_tracker_gets_attacker_det,
            attacker_tracker_gets_attacker_det,
            attacker_tracker_gets_victim_det
        )

class MockTrackerSiamRPN:
    """TensorFlow-compatible implementation of SiamRPN tracker"""
    
    def __init__(self, tracker, cfg, env_conf=0.4):
        self.cfg = cfg
        self.env_conf = env_conf
        
        # Convert numpy arrays to TensorFlow constants
        self.anchors = tf.constant(tracker.anchors, dtype=tf.float32)  # score_size*score_size*anchor_scale_ratio_num x 4
        self.window = tf.constant(tracker.window, dtype=tf.float32)
        
        # Configuration parameters
        self.context_amount = tf.constant(cfg.TRACK.CONTEXT_AMOUNT, dtype=tf.float32)
        self.exemplar_size = tf.constant(cfg.TRACK.EXEMPLAR_SIZE, dtype=tf.float32)
        self.instance_size = tf.constant(cfg.TRACK.INSTANCE_SIZE, dtype=tf.float32)
        self.penalty_k = tf.constant(cfg.TRACK.PENALTY_K, dtype=tf.float32)
        self.window_influence = tf.constant(cfg.TRACK.WINDOW_INFLUENCE, dtype=tf.float32)
        self.lr = tf.constant(cfg.TRACK.LR, dtype=tf.float32)
        
        # State variables (trainable=False since we update them manually)
        self.center_pos = tf.Variable(
            [0.0, 0.0], dtype=tf.float32, trainable=False, name="center_pos"
        )  # center position (x, y)
        self.size = tf.Variable(
            [0.0, 0.0], dtype=tf.float32, trainable=False, name="size"
        )  # size (w, h)
        
        # Image shape constants
        self.img_shape = tf.constant([1080, 1920], dtype=tf.float32)  # height, width
        
        # Store initial states (will be set when reset_state is first called)
        self.initial_center_pos = None
        self.initial_size = None
        
        self.frame = 1
    
    def reset_state(self, center_pos=None, size=None):
        """Reset tracker state to initial values"""
        if center_pos is not None:
            self.center_pos.assign(center_pos)
            # only works if running in eager mode without @tf.function (not graph mode)
            if isinstance(center_pos, tf.Tensor):
                center_pos = center_pos.numpy()
            self.initial_center_pos = tf.constant(center_pos, dtype=tf.float32)
        elif self.initial_center_pos is not None:
            self.center_pos.assign(self.initial_center_pos)
            
        if size is not None:
            self.size.assign(size)
            if isinstance(size, tf.Tensor):
                size = size.numpy()
            self.initial_size = tf.constant(size, dtype=tf.float32)
        elif self.initial_size is not None:
            self.size.assign(self.initial_size)

    @tf.function
    def predict(self, victim_bbox=None, attacker_bbox=None):
        """
        TensorFlow-compatible prediction method
        
        Args:
            victim_bbox: Optional ground truth bbox in (x1,y1,x2,y2) format
            attacker_bbox: Optional ground truth bbox in (x1,y1,x2,y2) format
            
        Returns:
            dict with 'bbox', 'best_score', 'best_pscore'
        """
        # Add small epsilon to prevent division by zero
        eps = tf.constant(1e-8, dtype=tf.float32)
        
        # Calculate template and search region sizes with safety checks
        w_z = self.size[0] + self.context_amount * tf.reduce_sum(self.size)
        h_z = self.size[1] + self.context_amount * tf.reduce_sum(self.size)
        
        # Ensure positive values
        w_z = tf.maximum(w_z, eps)
        h_z = tf.maximum(h_z, eps)
        
        s_z = tf.sqrt(w_z * h_z)  # template size
        s_z = tf.maximum(s_z, eps)  # Ensure positive
        
        scale_z = self.exemplar_size / s_z  # scale change
        s_x = s_z * (self.instance_size / self.exemplar_size)  # search region size
        
        # Generate dummy outputs (in real implementation, these would come from neural network)
        num_anchors = tf.shape(self.anchors)[0]

        # Initialize scores with default value of 0.4
        score = tf.ones((num_anchors,), dtype=tf.float32) * self.env_conf
        if victim_bbox is not None or attacker_bbox is not None:
            score, anchor_centers = self._scores(score, victim_bbox, attacker_bbox, scale_z)
        
        delta = tf.zeros((4, num_anchors), dtype=tf.float32)
        # Convert delta predictions to bounding boxes using anchors
        pred_bbox = self._convert_bbox_tf(delta, self.anchors)
        # pred_bbox is in the search region coordinates not the original image coordinates
        # victim_bbox and attacker_bbox are in the original image coordinates
        # need conversion for both center and wh
        if victim_bbox is not None:
            pred_bbox = self._pred_bboxes(pred_bbox, victim_bbox, anchor_centers, eps, scale_z)
        
        # Helper functions with safety checks
        def change(r):
            r_safe = tf.maximum(r, eps)  # Ensure positive
            return tf.maximum(r_safe, 1.0 / r_safe)
        
        def sz(w, h):
            w_safe = tf.maximum(w, eps)
            h_safe = tf.maximum(h, eps)
            pad = (w_safe + h_safe) * 0.5
            return tf.sqrt((w_safe + pad) * (h_safe + pad))
        
        # Scale penalty with safety checks
        pred_sz = sz(pred_bbox[2, :], pred_bbox[3, :])
        target_sz = sz(self.size[0] * scale_z, self.size[1] * scale_z)
        target_sz = tf.maximum(target_sz, eps)
        s_c = change(pred_sz / target_sz)
        
        # Aspect ratio penalty with safety checks
        size_0_safe = tf.maximum(self.size[0], eps)
        size_1_safe = tf.maximum(self.size[1], eps)
        pred_w_safe = tf.maximum(pred_bbox[2, :], eps)
        pred_h_safe = tf.maximum(pred_bbox[3, :], eps)
        
        target_ratio = size_0_safe / size_1_safe
        pred_ratio = pred_w_safe / pred_h_safe
        r_c = change(target_ratio / pred_ratio)
        
        penalty = tf.exp(-(r_c * s_c - 1.0) * self.penalty_k)
        pscore = penalty * score
        
        # Window penalty
        pscore = pscore * (1.0 - self.window_influence) + self.window * self.window_influence
            
        # Find best prediction
        best_idx = tf.argmax(pscore)
        
        # Get best bbox relative to crop
        bbox = pred_bbox[:, best_idx] / tf.maximum(scale_z, eps)
        lr_factor = penalty[best_idx] * score[best_idx] * self.lr
        
        # Box center in entire image
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        
        # Smooth bbox - blend previous size with new prediction
        width = self.size[0] * (1.0 - lr_factor) + bbox[2] * lr_factor
        height = self.size[1] * (1.0 - lr_factor) + bbox[3] * lr_factor

        # Ensure positive dimensions
        width = tf.maximum(width, eps)
        height = tf.maximum(height, eps)

        # Clip boundary
        cx, cy, width, height = self._bbox_clip_tf(cx, cy, width, height, self.img_shape)
        
        # Final safety check - ensure no NaN values
        cx = tf.where(tf.math.is_nan(cx), self.center_pos[0], cx)
        cy = tf.where(tf.math.is_nan(cy), self.center_pos[1], cy)
        width = tf.where(tf.math.is_nan(width), self.size[0], width)
        height = tf.where(tf.math.is_nan(height), self.size[1], height)
        
        # Update state
        self.center_pos.assign([cx, cy])
        self.size.assign([width, height])
        
        # Convert to x1y1x2y2 format
        bbox_output = tf.stack([
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0
        ])
        
        # Final NaN check (remove the error raising in graph mode)
        # Instead of raising error, return fallback values
        bbox_output = tf.where(
            tf.math.is_nan(bbox_output),
            tf.stack([
                self.center_pos[0] - self.size[0] / 2.0,
                self.center_pos[1] - self.size[1] / 2.0,
                self.center_pos[0] + self.size[0] / 2.0,
                self.center_pos[1] + self.size[1] / 2.0
            ]),
            bbox_output
        )
        
        best_score = score[best_idx]
        best_pscore = pscore[best_idx]
        # print("Best score:", best_score.numpy(), "Best pscore:", best_pscore.numpy())
        
        return {
            'bbox': bbox_output,
            'best_score': best_score,
            'best_pscore': best_pscore
        }
        
    @tf.function
    def _bbox_to_center_relative(self, bbox, scale_z):
        # Convert bbox from (x1,y1,x2,y2) to center coordinates relative to current position
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        rel_x = center_x - self.center_pos[0]
        rel_y = center_y - self.center_pos[1]
        return tf.stack([rel_x, rel_y]) * scale_z  # Scale to search region coordinates
        
    @tf.function
    def _scores(self, score, victim_bbox, attacker_bbox, scale_z):
        # Extract anchor centers (anchors format: [cx, cy, w, h])
        anchor_centers = self.anchors[:, :2]  # Shape: (num_anchors, 2)
        
        # Number of closest anchors to consider
        k = 5
        if victim_bbox is not None:
            victim_center = self._bbox_to_center_relative(victim_bbox, scale_z)
            # Calculate distances from victim center to all anchor centers
            victim_distances = tf.norm(anchor_centers - victim_center[tf.newaxis, :], axis=1)
            
            # Find the k smallest distances and their indices
            # top_k returns (values, indices) where values are sorted in descending order
            # We want ascending order, so we negate the distances
            _, victim_closest_indices = tf.nn.top_k(-victim_distances, k=tf.minimum(k, tf.shape(victim_distances)[0]))
            # Set score to 1.0 for the closest k anchors to victim
            score = tf.tensor_scatter_nd_update(
                score,
                tf.expand_dims(victim_closest_indices, 1),
                tf.ones(tf.shape(victim_closest_indices)[0], dtype=tf.float32) * 1.0
            )
        
        if attacker_bbox is not None:
            attacker_center = self._bbox_to_center_relative(attacker_bbox, scale_z)
            attacker_distances = tf.norm(anchor_centers - attacker_center[tf.newaxis, :], axis=1)
            
            _, attacker_closest_indices = tf.nn.top_k(-attacker_distances, k=tf.minimum(k, tf.shape(attacker_distances)[0]))
            # Set score to 0.8 for the closest k anchors to attacker
            score = tf.tensor_scatter_nd_update(
                score,
                tf.expand_dims(attacker_closest_indices, 1),
                tf.ones(tf.shape(attacker_closest_indices)[0], dtype=tf.float32) * 0.9
            )
        return score, anchor_centers
    
    @tf.function
    def _pred_bboxes(self, pred_bbox, victim_bbox, anchor_centers, eps, scale_z):
        # Get victim bbox dimensions and area. Need to convert to search region scale
        # victim_width = (victim_bbox[2] - victim_bbox[0]) * scale_z
        # victim_height = (victim_bbox[3] - victim_bbox[1]) * scale_z
        # victim_area = victim_width * victim_height
        init_area = self.initial_size[0] * self.initial_size[1] * (scale_z ** 2)
        
        # Find the closest anchor to victim
        victim_center = self._bbox_to_center_relative(victim_bbox, scale_z)
        victim_distances = tf.norm(anchor_centers - victim_center[tf.newaxis, :], axis=1)
        victim_closest_idx = tf.argmin(victim_distances)
        
        # Get closest anchor's predicted bbox dimensions and area
        closest_pred_width = pred_bbox[2, victim_closest_idx]
        closest_pred_height = pred_bbox[3, victim_closest_idx]
        closest_pred_area = closest_pred_width * closest_pred_height
        
        # Calculate single scale factor based on area ratio
        # scale^2 * closest_area = victim_area
        # scale = sqrt(victim_area / closest_area)
        area_scale = tf.sqrt(init_area / tf.maximum(closest_pred_area, eps))
        
        # Apply uniform scaling to all predicted bbox dimensions
        pred_bbox = tf.stack([
            pred_bbox[0, :],  # Keep center x unchanged
            pred_bbox[1, :],  # Keep center y unchanged
            pred_bbox[2, :] * area_scale,   # Scale all widths by same factor
            pred_bbox[3, :] * area_scale    # Scale all heights by same factor
        ], axis=0)
        return pred_bbox
    
    @tf.function
    def _bbox_clip_tf(self, cx, cy, width, height, boundary):
        """TensorFlow version of bbox clipping"""
        cx = tf.clip_by_value(cx, 0.0, boundary[1])
        cy = tf.clip_by_value(cy, 0.0, boundary[0])
        width = tf.clip_by_value(width, 10.0, boundary[1])
        height = tf.clip_by_value(height, 10.0, boundary[0])
        return cx, cy, width, height
    
    @tf.function
    def _convert_bbox_tf(self, delta, anchor):
        """
        TensorFlow-compatible version of bbox conversion
        
        Args:
            delta: shape (4, num_anchors) - predicted offsets
            anchor: shape (num_anchors, 4) - anchor boxes [x, y, w, h]
            
        Returns:
            converted_bbox: shape (4, num_anchors) - converted bounding boxes
        """
        # Convert delta predictions to actual bounding box coordinates
        # delta format: [dx, dy, dw, dh] where dx,dy are offsets and dw,dh are log-space size changes
        
        # Center coordinates: add relative offset scaled by anchor size
        cx = delta[0, :] * anchor[:, 2] + anchor[:, 0]  # dx * anchor_w + anchor_cx
        cy = delta[1, :] * anchor[:, 3] + anchor[:, 1]  # dy * anchor_h + anchor_cy
        
        # Size: exponential of log-space predictions scaled by anchor size
        w = tf.exp(delta[2, :]) * anchor[:, 2]  # exp(dw) * anchor_w
        h = tf.exp(delta[3, :]) * anchor[:, 3]  # exp(dh) * anchor_h
        
        # Stack to create output tensor
        converted_bbox = tf.stack([cx, cy, w, h], axis=0)
        
        return converted_bbox
        
