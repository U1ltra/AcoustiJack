# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        best_pscore = pscore[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score,
                'best_pscore': best_pscore
               }

# class SiamRPNTrackerWithHeatmap(SiamRPNTracker):
#     """Extended SiamRPN tracker with heatmap visualization capabilities"""
    
    def track_with_insight(self, img, show_heatmap=False, show_analysis=False, save_path=None):
        """
        Enhanced tracking method that generates heatmaps and preserves all individual predictions
        
        Args:
            img: Input image
            show_heatmap: Whether to display the heatmap
            show_analysis: Whether to visualize detailed prediction analysis
            save_path: Path to save the heatmap image
            
        Returns:
            Dictionary containing bbox, best_score, and heatmap data, detailed prediction data
        """
        # Standard tracking computation
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        # Convert scores and bboxes
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        # Apply penalties (same as original)
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore_no_window = penalty * score
        pscore = pscore_no_window * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        heatmap_data = self.generate_heatmaps(score, pscore, penalty, img, scale_z)
        
        if show_heatmap:
            self.visualize_heatmaps(img, heatmap_data, save_path)  

        # Convert predictions to original image coordinates
        predictions_data = self.organize_predictions(
            pred_bbox, score, penalty, pscore_no_window, pscore, 
            scale_z, img.shape[:2]
        )

        if show_analysis:
            self.visualize_predictions(predictions_data, img, save_path)

        # Continue with standard tracking
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # Update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        final_bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = score[best_idx]
        
        return {
            'bbox': final_bbox,
            'best_score': best_score,
            'best_penalized_score': pscore[best_idx],
            'heatmap_data': heatmap_data,
            'predictions_data': predictions_data,
            'best_prediction_idx': best_idx
        }

    def organize_predictions(self, pred_bbox, raw_scores, penalties, 
                           pscore_no_window, final_scores, scale_z, img_shape):
        """
        Organize all predictions with their locations and scores
        
        Returns a structured array of all predictions
        """
        num_predictions = pred_bbox.shape[1]
        
        # Convert predicted boxes to original image coordinates
        # pred_bbox is in search crop coordinates relative to center
        pred_bbox_scaled = pred_bbox / scale_z
        
        # Convert to absolute image coordinates
        pred_centers_x = pred_bbox_scaled[0, :] + self.center_pos[0]
        pred_centers_y = pred_bbox_scaled[1, :] + self.center_pos[1]
        pred_widths = pred_bbox_scaled[2, :]
        pred_heights = pred_bbox_scaled[3, :]
        
        # Create structured array with all prediction information
        predictions = []
        for i in range(num_predictions):
            # Calculate anchor information for this prediction
            anchor_idx = i % self.anchor_num
            spatial_idx = i // self.anchor_num
            spatial_y = spatial_idx // self.score_size
            spatial_x = spatial_idx % self.score_size
            
            pred = {
                'prediction_idx': i,
                'anchor_idx': anchor_idx,
                'spatial_x': spatial_x,
                'spatial_y': spatial_y,
                'anchor_center_x': self.anchors[i, 0] + self.center_pos[0],
                'anchor_center_y': self.anchors[i, 1] + self.center_pos[1],
                'predicted_center_x': pred_centers_x[i],
                'predicted_center_y': pred_centers_y[i],
                'predicted_width': pred_widths[i],
                'predicted_height': pred_heights[i],
                'raw_score': raw_scores[i],
                'penalty': penalties[i],
                'score_with_penalty': pscore_no_window[i],
                'final_score': final_scores[i],
                'bbox': [pred_centers_x[i] - pred_widths[i]/2,
                        pred_centers_y[i] - pred_heights[i]/2,
                        pred_widths[i], pred_heights[i]]
            }
            predictions.append(pred)
        
        return {
            'predictions': predictions,
            'num_predictions': num_predictions,
            'img_shape': img_shape,
            'current_target_center': self.center_pos.copy(),
            'current_target_size': self.size.copy()
        }

    def get_predictions_at_point(self, predictions_data, x, y, radius=None, 
                               score_type='final_score', return_closest=True):
        """
        Get predictions at or near a specific point
        
        Args:
            predictions_data: Data from organize_predictions()
            x, y: Point coordinates in original image
            radius: Search radius (if None, returns only closest)
            score_type: Which score to consider ('raw_score', 'final_score', etc.)
            return_closest: If True and multiple found, return closest to point
            
        Returns:
            Dictionary with prediction information
        """
        predictions = predictions_data['predictions']
        
        if radius is None:
            # Find the closest prediction by center distance
            distances = []
            for pred in predictions:
                dist = np.sqrt((pred['predicted_center_x'] - x)**2 + 
                             (pred['predicted_center_y'] - y)**2)
                distances.append(dist)
            
            closest_idx = np.argmin(distances)
            closest_pred = predictions[closest_idx]
            
            return {
                'type': 'closest',
                'prediction': closest_pred,
                'distance': distances[closest_idx],
                'query_point': (x, y)
            }
        
        else:
            # Find all predictions within radius
            candidates = []
            for pred in predictions:
                dist = np.sqrt((pred['predicted_center_x'] - x)**2 + 
                             (pred['predicted_center_y'] - y)**2)
                if dist <= radius:
                    candidates.append((pred, dist))
            
            if not candidates:
                # No predictions within radius, return closest
                return self.get_predictions_at_point(predictions_data, x, y, None, score_type, True)
            
            if return_closest:
                # Return closest among candidates
                best_pred, best_dist = min(candidates, key=lambda x: x[1])
                return {
                    'type': 'closest_in_radius',
                    'prediction': best_pred,
                    'distance': best_dist,
                    'query_point': (x, y),
                    'radius': radius,
                    'candidates_count': len(candidates)
                }
            else:
                # Return highest scoring among candidates
                best_pred, best_dist = max(candidates, key=lambda x: x[0][score_type])
                return {
                    'type': 'max_score_in_radius',
                    'prediction': best_pred,
                    'distance': best_dist,
                    'query_point': (x, y),
                    'radius': radius,
                    'candidates_count': len(candidates),
                    'score_type': score_type
                }

    def get_max_score_in_region(self, predictions_data, x, y, radius, score_type='final_score'):
        """
        Get the prediction with maximum score within a circular region
        """
        return self.get_predictions_at_point(
            predictions_data, x, y, radius, score_type, return_closest=False
        )

    def generate_heatmaps(self, raw_scores, final_scores, penalties, img, scale_z, aggregate_method='max'):
        """Generate heatmap data for visualization"""
        
        # Reshape scores to spatial grid
        # The scores are arranged as [anchor1_pos1, anchor1_pos2, ..., anchor2_pos1, ...]
        # We need to average across anchors for each spatial position
        total_positions = self.score_size * self.score_size
        
        # Reshape to [num_anchors, spatial_positions]
        raw_scores_reshaped = raw_scores.reshape(self.anchor_num, total_positions)
        final_scores_reshaped = final_scores.reshape(self.anchor_num, total_positions)
        penalties_reshaped = penalties.reshape(self.anchor_num, total_positions)
        
        # Average across anchors to get spatial heatmaps
        if aggregate_method == 'max':
            raw_heatmap = np.max(raw_scores_reshaped, axis=0).reshape(self.score_size, self.score_size)
            final_heatmap = np.max(final_scores_reshaped, axis=0).reshape(self.score_size, self.score_size)
            penalty_heatmap = np.max(penalties_reshaped, axis=0).reshape(self.score_size, self.score_size)
        elif aggregate_method == 'mean':
            raw_heatmap = np.mean(raw_scores_reshaped, axis=0).reshape(self.score_size, self.score_size)
            final_heatmap = np.mean(final_scores_reshaped, axis=0).reshape(self.score_size, self.score_size)
            penalty_heatmap = np.mean(penalties_reshaped, axis=0).reshape(self.score_size, self.score_size)
        else:
            raise ValueError("Unsupported aggregation method. Use 'max' or 'mean'.")
                
        # Calculate the search region in original image coordinates
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        # Search region bounds in original image
        search_size = round(s_x)
        left = max(0, int(self.center_pos[0] - search_size // 2))
        top = max(0, int(self.center_pos[1] - search_size // 2))
        right = min(img.shape[1], left + search_size)
        bottom = min(img.shape[0], top + search_size)
        
        return {
            'raw_heatmap': raw_heatmap,
            'final_heatmap': final_heatmap,
            'penalty_heatmap': penalty_heatmap,
            'search_bounds': (left, top, right, bottom),
            'scale_factor': scale_z
        }

    def visualize_heatmaps(self, img, heatmap_data, save_path=None):
        """Visualize the heatmaps"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with search region
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image with Search Region')
        
        # Draw search region rectangle
        left, top, right, bottom = heatmap_data['search_bounds']
        search_rect = Rectangle((left, top), right-left, bottom-top, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(search_rect)
        
        # Draw current tracking box
        bbox = [self.center_pos[0] - self.size[0]/2, 
                self.center_pos[1] - self.size[1]/2,
                self.size[0], self.size[1]]
        track_rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                              linewidth=2, edgecolor='green', facecolor='none')
        axes[0, 0].add_patch(track_rect)
        axes[0, 0].set_xlim(0, img.shape[1])
        axes[0, 0].set_ylim(img.shape[0], 0)
        
        # Raw classification scores
        im1 = axes[0, 1].imshow(heatmap_data['raw_heatmap'], cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title('Raw Classification Scores')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Penalty heatmap
        im2 = axes[1, 0].imshow(heatmap_data['penalty_heatmap'], cmap='viridis', interpolation='bilinear')
        axes[1, 0].set_title('Penalty Map (Scale + Aspect Ratio)')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Final scores (after penalties and windowing)
        im3 = axes[1, 1].imshow(heatmap_data['final_heatmap'], cmap='hot', interpolation='bilinear')
        axes[1, 1].set_title('Final Scores (After Penalties + Windowing)')
        plt.colorbar(im3, ax=axes[1, 1])
        
        # Mark the best location on final heatmap
        best_y, best_x = np.unravel_index(np.argmax(heatmap_data['final_heatmap']), 
                                         heatmap_data['final_heatmap'].shape)
        axes[1, 1].plot(best_x, best_y, 'r*', markersize=15, markeredgecolor='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # plt.show()

    def overlay_heatmap_on_image(self, img, heatmap_data, alpha=0.6, save_path=None):
        """Overlay the final heatmap on the original image"""
        
        # Get search region
        left, top, right, bottom = heatmap_data['search_bounds']
        
        # Resize heatmap to match search region size
        search_h, search_w = bottom - top, right - left
        resized_heatmap = cv2.resize(heatmap_data['final_heatmap'], (search_w, search_h))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = ((resized_heatmap - resized_heatmap.min()) / 
                             (resized_heatmap.max() - resized_heatmap.min()) * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = img.copy()
        overlay[top:bottom, left:right] = cv2.addWeighted(
            img[top:bottom, left:right], 1-alpha,
            heatmap_colored, alpha, 0
        )
        
        # Draw tracking box
        bbox = [int(self.center_pos[0] - self.size[0]/2), 
                int(self.center_pos[1] - self.size[1]/2),
                int(self.size[0]), int(self.size[1])]
        cv2.rectangle(overlay, (bbox[0], bbox[1]), 
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return overlay
    
    def analyze_prediction_distribution(self, predictions_data, score_type='final_score', 
                                      top_k=None):
        """
        Analyze the distribution of predictions and their scores
        
        Args:
            predictions_data: Data from organize_predictions()
            score_type: Which score to analyze
            top_k: Return only top k predictions (None for all)
            
        Returns:
            Dictionary with analysis results
        """
        predictions = predictions_data['predictions']
        
        # Extract scores and locations
        scores = [pred[score_type] for pred in predictions]
        centers = [(pred['predicted_center_x'], pred['predicted_center_y']) for pred in predictions]
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        top_predictions = [predictions[i] for i in sorted_indices]
        top_scores = [scores[i] for i in sorted_indices]
        
        # Calculate statistics
        score_stats = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'top_k_mean': np.mean(top_scores) if top_scores else 0
        }
        
        return {
            'top_predictions': top_predictions,
            'top_scores': top_scores,
            'score_statistics': score_stats,
            'score_type': score_type,
            'total_predictions': len(predictions)
        }
    
    def visualize_predictions(self, predictions_data, img, save_path="predictions_visualization.png", 
                            show_top_k=50, score_type='final_score'):
        """
        Visualize predictions on the image
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Get top predictions for visualization
        analysis = self.analyze_prediction_distribution(
            predictions_data, score_type, top_k=show_top_k
        )
        
        # Plot 1: Original image with top predictions
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Top {show_top_k} Predictions ({score_type})')
        
        # Draw current tracking box
        current_bbox = [
            predictions_data['current_target_center'][0] - predictions_data['current_target_size'][0]/2,
            predictions_data['current_target_center'][1] - predictions_data['current_target_size'][1]/2,
            predictions_data['current_target_size'][0],
            predictions_data['current_target_size'][1]
        ]
        track_rect = Rectangle((current_bbox[0], current_bbox[1]), current_bbox[2], current_bbox[3],
                              linewidth=3, edgecolor='lime', facecolor='none', label='Current Track')
        axes[0].add_patch(track_rect)
        
        # Draw top predictions with color coding by score
        scores = analysis['top_scores']
        if scores:
            norm_scores = (np.array(scores) - min(scores)) / (max(scores) - min(scores) + 1e-8)
            
            for i, (pred, norm_score) in enumerate(zip(analysis['top_predictions'], norm_scores)):
                # Color from blue (low) to red (high)
                color = plt.cm.hot(norm_score)
                
                # Draw predicted box
                bbox = pred['bbox']
                rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                               linewidth=1, edgecolor=color, facecolor='none', alpha=0.7)
                axes[0].add_patch(rect)
                
                # Draw center point
                axes[0].plot(pred['predicted_center_x'], pred['predicted_center_y'], 
                           'o', color=color, markersize=3, alpha=0.8)
                
                # Label top 5
                if i < 5:
                    axes[0].text(pred['predicted_center_x']+5, pred['predicted_center_y']-5, 
                               f'{i+1}', color='white', fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
        
        axes[0].set_xlim(0, img.shape[1])
        axes[0].set_ylim(img.shape[0], 0)
        axes[0].legend()
        
        # Plot 2: Score distribution
        all_scores = [pred[score_type] for pred in predictions_data['predictions']]
        axes[1].hist(all_scores, bins=50, alpha=0.7, color='blue')
        axes[1].axvline(analysis['score_statistics']['mean'], color='red', linestyle='--', 
                       label=f'Mean: {analysis["score_statistics"]["mean"]:.4f}')
        axes[1].axvline(max(all_scores), color='green', linestyle='--', 
                       label=f'Max: {max(all_scores):.4f}')
        axes[1].set_xlabel(f'{score_type}')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Score Distribution')
        axes[1].legend()
        
        # Plot 3: Spatial distribution of top predictions
        axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Anchor Centers vs Predicted Centers')
        
        # Show anchor centers vs predicted centers for top predictions
        for i, pred in enumerate(analysis['top_predictions'][:20]):  # Show top 20
            # Anchor center
            axes[2].plot(pred['anchor_center_x'], pred['anchor_center_y'], 
                        's', color='blue', markersize=4, alpha=0.6)
            # Predicted center
            axes[2].plot(pred['predicted_center_x'], pred['predicted_center_y'], 
                        'o', color='red', markersize=4, alpha=0.8)
            # Arrow from anchor to prediction
            axes[2].arrow(pred['anchor_center_x'], pred['anchor_center_y'],
                         pred['predicted_center_x'] - pred['anchor_center_x'],
                         pred['predicted_center_y'] - pred['anchor_center_y'],
                         head_width=3, head_length=3, fc='yellow', ec='yellow', alpha=0.6)
        
        # Legend
        axes[2].plot([], [], 's', color='blue', label='Anchor Centers', markersize=6)
        axes[2].plot([], [], 'o', color='red', label='Predicted Centers', markersize=6)
        axes[2].legend()
        axes[2].set_xlim(0, img.shape[1])
        axes[2].set_ylim(img.shape[0], 0)
        
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # plt.show()


if __name__ == "__main__":
    # Initialize tracker with your model
    tracker = SiameseTracker(model)

    # Initialize with first frame
    tracker.init(first_frame, initial_bbox)

    # Track subsequent frames with heatmap visualization
    for frame in video_frames:
        result = tracker.track_with_heatmap(frame, show_heatmap=True)
        
        # Optional: Create overlay visualization
        overlay = tracker.overlay_heatmap_on_image(frame, result['heatmap_data'])
        
        # Your tracking logic here...

