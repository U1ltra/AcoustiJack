import os
import cv2
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from argparse import Namespace
from collections import defaultdict
from typing import List, Tuple, Optional

from scipy import stats
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# list directories under a given path
def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


# list files under a given path
def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_log(log_path):
    switch = False
    switch_frame = None
    max_delta_vic = None
    stability = None
    success_count = 0

    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Attacker switched: True" in line:
                success_count += 1
                switch = success_count > 0
                switch_frame = 0
            # if "Attacker switched at frame" in line:
            #     switch = True
            #     switch_frame = int(line.replace("Attacker switched at frame ", ""))
            #     switch_frame = switch_frame
            # if "ID switch detected at frame" in line and not switch:
            #     switch = True
            #     line = line.split("|")[0]
            #     switch_frame = int(line.replace("ID switch detected at frame ", ""))
            #     switch_frame = switch_frame
            if "Maximum victim delta speed:" in line:
                max_delta_vic = float(line.split(" ")[-1])
            if "Camera stability analysis: " in line:
                stability = line.replace("Camera stability analysis: ", "").strip()
                stability = stability.replace("(", "").replace(")", "").split(", ")
                stability = [
                    float(i) for i in stability
                ]  # v_std, ang_std, v_mean, ang_mean

    return switch, switch_frame, max_delta_vic, stability


def reconstruct_args(args_path) -> Namespace:
    with open(args_path, "r") as f:
        args_str = f.read()
    keys, values = [], []
    items = args_str.replace("Namespace(", "").rstrip(")").split("=")
    for item in items[1:-1]:
        if "[" in item:
            value, key = item.split("], ")
            values.append(value.replace("[", "").split(", "))
            keys.append(key)
        else:
            value, key = item.split(", ")
            values.append(value.replace("'", ""))
            keys.append(key)
    keys.insert(0, items[0])
    values.insert(len(values), items[-1])

    args = Namespace()
    for key, value in zip(keys, values):
        if isinstance(value, list):
            value = [float(i) for i in value]
        # if only contains numbers, convert to float
        elif value.isnumeric():
            value = float(value)
        setattr(args, key, value)

    return args


def get_bbx_center(bbx):
    center_x = (bbx[:, 0] + bbx[:, 2]) / 2
    center_y = (bbx[:, 1] + bbx[:, 3]) / 2
    return np.stack((center_x, center_y), axis=1)


def get_iou(bbx1, bbx2):
    x1 = np.maximum(bbx1[:, 0], bbx2[:, 0])
    y1 = np.maximum(bbx1[:, 1], bbx2[:, 1])
    x2 = np.minimum(bbx1[:, 2], bbx2[:, 2])
    y2 = np.minimum(bbx1[:, 3], bbx2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (bbx1[:, 2] - bbx1[:, 0]) * (bbx1[:, 3] - bbx1[:, 1])
    area2 = (bbx2[:, 2] - bbx2[:, 0]) * (bbx2[:, 3] - bbx2[:, 1])

    return intersection / (area1 + area2 - intersection)


def get_delta_scale(bbx1, bbx2):
    area1 = (bbx1[:, 2] - bbx1[:, 0]) * (bbx1[:, 3] - bbx1[:, 1])
    area2 = (bbx2[:, 2] - bbx2[:, 0]) * (bbx2[:, 3] - bbx2[:, 1])
    return (area2 - area1) / area1


def get_bins(data: np.ndarray):
    data = data.flatten()
    n = len(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (n ** (1 / 3))
    data_range = np.max(data) - np.min(data)

    return int(np.ceil(data_range / bin_width))


def get_camera_poses(main_timestamp: np.ndarray, camera_poses: np.ndarray):
    """
    main_timestemp: list of timestamps of the main camera
    camera_poses: list of camera poses in format [x, y, z, qx, qy, qz, qw, yaw, pitch, roll, timestamp]
    """
    poses = []
    for ts in main_timestamp:
        idx = np.argmin(np.abs(camera_poses[:, -1] - ts))
        poses.append(camera_poses[idx, :])

    return np.array(poses)


def get_polar_coords(x1, y1, x2, y2, x3, y3):
    # Vector from target 1 to vehicle (reference line)
    ref_vector = np.array([x3 - x1, y3 - y1])
    # Vector from target 1 to target 2
    target_vector = np.array([x2 - x1, y2 - y1])

    # Calculate radius (distance from target 1 to target 2)
    r = np.linalg.norm(target_vector)

    # Calculate angle between vectors
    # Get angle using arctan2 (handles all quadrants correctly)
    angle = np.arctan2(target_vector[1], target_vector[0]) - np.arctan2(
        ref_vector[1], ref_vector[0]
    )

    # Normalize angle to [-π, π]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    # radians to degrees
    deg_angle = np.degrees(angle)

    return r, angle, deg_angle  # Returns (radius, angle in radians)


def images_to_video(image_list, output_path, fps=30):
    """
    Convert a list of cv2 images to a video file.
    
    Args:
        image_list: List of cv2 images (numpy arrays)
        output_path: Path where to save the video (e.g., 'output.mp4')
        fps: Frames per second for the output video
    """
    if not image_list:
        raise ValueError("Image list is empty")
    
    # Get dimensions from the first image
    height, width, channels = image_list[0].shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each image to the video
    for image in image_list:
        # Ensure all images have the same dimensions
        if image.shape != (height, width, channels):
            image = cv2.resize(image, (width, height))
        video_writer.write(image)
    
    # Release the VideoWriter
    video_writer.release()
    print(f"Video saved to {output_path}")

def shift_image(frame, shift_x, shift_y):
    # Calculate shift needed to move obj2 center to obj1 center
    # shift_x = int(obj1[0] - obj2[0])
    # shift_y = int(obj1[1] - obj2[1])

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Create transformation matrix for translation
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply the shift
    shifted_frame = cv2.warpAffine(
        frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    return shifted_frame

def draw_heatmap(coordinates, values, resolution=100, method='linear', save_path=None, 
                 search_region=None, target_region=None, atk_region=None):
    """
    Draw a heatmap from coordinate pairs and values.
    
    Parameters:
    coordinates: list of tuples [(x,y), ...] - coordinate pairs
    values: list of numbers [v, ...] - values corresponding to coordinates
    resolution: int - grid resolution for interpolation
    method: str - interpolation method ('linear', 'nearest', 'cubic')
    search_region: list - bounding box as [x_min, y_min, x_max, y_max]
                  e.g., [-5, -5, 5, 5] for a square from (-5,-5) to (5,5)
    target_region: list - bounding box as [x_min, y_min, x_max, y_max]
    """
    
    # Convert to numpy arrays
    coords = np.array(coordinates)
    vals = np.array(values)
    
    # Extract x and y coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create a grid for interpolation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate values onto the grid
    zi = griddata((x, y), vals, (xi, yi), method=method)
    
    # Create the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(zi, extent=[x_min, x_max, y_min, y_max], 
                        origin='lower', cmap='viridis', aspect='auto')
    
    # Add colorbar
    plt.colorbar(heatmap, label='Value')
    
    # Overlay original points
    plt.scatter(x, y, c=vals, cmap='viridis', s=50, 
               edgecolors='white', linewidth=1, alpha=0.8)
    
    # Draw search region box if provided
    if search_region is not None:
        x_min_box, y_min_box, x_max_box, y_max_box = search_region
        
        # Create rectangle corners
        box_x = [x_min_box, x_max_box, x_max_box, x_min_box, x_min_box]
        box_y = [y_min_box, y_min_box, y_max_box, y_max_box, y_min_box]
        
        # Draw the box
        plt.plot(box_x, box_y, 'r-', linewidth=2, alpha=0.8, label='Search Region')
        
        # Optionally add corner points
        # corner_x = [x_min_box, x_max_box, x_max_box, x_min_box]
        # corner_y = [y_min_box, y_min_box, y_max_box, y_max_box]
        # plt.scatter(corner_x, corner_y, c='red', s=30, marker='s', alpha=0.8)
        
        plt.legend(loc="lower right")
    # Draw target region box if provided
    if target_region is not None:
        x_min_box, y_min_box, x_max_box, y_max_box = target_region
        
        # Create rectangle corners
        box_x = [x_min_box, x_max_box, x_max_box, x_min_box, x_min_box]
        box_y = [y_min_box, y_min_box, y_max_box, y_max_box, y_min_box]
        
        # Draw the box
        plt.plot(box_x, box_y, 'b-', linewidth=2, alpha=0.8, label='Target Region')
        
        plt.legend(loc="lower right")
    # Draw attack region box if provided
    if atk_region is not None:
        x_min_box, y_min_box, x_max_box, y_max_box = atk_region
        
        # Create rectangle corners
        box_x = [x_min_box, x_max_box, x_max_box, x_min_box, x_min_box]
        box_y = [y_min_box, y_min_box, y_max_box, y_max_box, y_min_box]
        
        # Draw the box
        plt.plot(box_x, box_y, 'r:', linewidth=2, alpha=0.8, label='Attacker Region')
        
        plt.legend(loc="lower right")
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Heatmap from Coordinate Pairs')
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_multiple_lines_with_confidence_area(x_values, *y_value_sets, labels=None, colors=None, 
                                           confidence_level=0.95, use_minmax=False, 
                                           smooth_method=None, smooth_window=3, interpolation_points=100, fig_name="plot"):
    """
    Create a line plot showing mean values for duplicate x values,
    with shaded confidence intervals or min-max areas for multiple y-value sets.
    
    Parameters:
    x_values: array-like, x-axis values (may contain duplicates)
    *y_value_sets: multiple array-like objects, each containing y-axis values (same length as x_values)
    labels: list of strings, labels for each y-value set (optional)
    colors: list of colors for each y-value set (optional)
    confidence_level: float, confidence level for intervals (default 0.95 for 95% CI)
    use_minmax: bool, if True use min-max instead of confidence intervals (default False)
    smooth_method: str, smoothing method ('rolling', 'gaussian', 'interpolate', None)
    smooth_window: int, window size for rolling average or gaussian smoothing (default 3)
    interpolation_points: int, number of points for interpolation smoothing (default 100)
    """
    # Default colors if not provided
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create default labels if not provided
    if labels is None:
        labels = [f'Series {i+1}' for i in range(len(y_value_sets))]
    
    # Validate inputs
    if not y_value_sets:
        raise ValueError("At least one y-value set must be provided")
    
    for i, y_vals in enumerate(y_value_sets):
        if len(x_values) != len(y_vals):
            raise ValueError(f"Length mismatch: x_values has {len(x_values)} elements, y_value_set {i+1} has {len(y_vals)} elements")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    grouped_results = []
    
    # Process each y-value set
    for i, y_values in enumerate(y_value_sets):
        # Create DataFrame for easier grouping
        df = pd.DataFrame({'x': x_values, 'y': y_values})
        
        # Group by x values and calculate statistics
        if use_minmax:
            grouped = df.groupby('x')['y'].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()
            grouped['lower'] = grouped['min']
            grouped['upper'] = grouped['max']
            area_label = 'Min-Max Range'
        else:
            grouped = df.groupby('x')['y'].agg(['mean', 'std', 'count']).reset_index()
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  # Standard error of mean
            grouped['margin_error'] = stats.t.ppf(1 - alpha/2, grouped['count'] - 1) * grouped['sem']
            grouped['lower'] = grouped['mean'] - grouped['margin_error']
            grouped['upper'] = grouped['mean'] + grouped['margin_error']
            
            # Handle cases where we don't have enough data points for t-distribution
            # For single data points, use the point estimate (no interval)
            single_point_mask = grouped['count'] == 1
            grouped.loc[single_point_mask, 'lower'] = grouped.loc[single_point_mask, 'mean']
            grouped.loc[single_point_mask, 'upper'] = grouped.loc[single_point_mask, 'mean']
            
            area_label = f'{int(confidence_level*100)}% ConfInt'
        
        # Sort by x values for proper line plotting
        grouped = grouped.sort_values('x')
        grouped_results.append(grouped)
        
        # Get color for this series
        color = colors[i % len(colors)]
        
        # Apply smoothing if requested
        x_plot = grouped['x'].values
        mean_plot = grouped['mean'].values
        lower_plot = grouped['lower'].values
        upper_plot = grouped['upper'].values
        
        if smooth_method == 'rolling' and len(grouped) >= smooth_window:
            # Rolling window smoothing
            mean_smooth = pd.Series(mean_plot).rolling(window=smooth_window, center=True, min_periods=1).mean().values
            lower_smooth = pd.Series(lower_plot).rolling(window=smooth_window, center=True, min_periods=1).mean().values
            upper_smooth = pd.Series(upper_plot).rolling(window=smooth_window, center=True, min_periods=1).mean().values
            mean_plot, lower_plot, upper_plot = mean_smooth, lower_smooth, upper_smooth
            area_label += ' (Rolling)'
            
        elif smooth_method == 'gaussian' and len(grouped) >= 3:
            # Gaussian smoothing
            sigma = smooth_window / 3.0  # Convert window to sigma
            mean_plot = gaussian_filter1d(mean_plot, sigma=sigma)
            lower_plot = gaussian_filter1d(lower_plot, sigma=sigma)
            upper_plot = gaussian_filter1d(upper_plot, sigma=sigma)
            area_label += ' (Gaussian)'
            
        elif smooth_method == 'interpolate' and len(grouped) >= 4:
            # Spline interpolation for smoother curves
            x_min, x_max = x_plot.min(), x_plot.max()
            x_interp = np.linspace(x_min, x_max, interpolation_points)
            
            # Create interpolation functions
            mean_interp = interp1d(x_plot, mean_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
            lower_interp = interp1d(x_plot, lower_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
            upper_interp = interp1d(x_plot, upper_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            x_plot = x_interp
            mean_plot = mean_interp(x_interp)
            lower_plot = lower_interp(x_interp)
            upper_plot = upper_interp(x_interp)
            area_label += ' (Interpolated)'
        
        # Plot the mean line
        if smooth_method == 'interpolate':
            plt.plot(x_plot, mean_plot, '-', linewidth=2, label=f'{labels[i]} (Mean)', color=color)
        else:
            # plt.plot(x_plot, mean_plot, '-', linewidth=2, label=f'{labels[i]} (Mean)', marker='o', color=color)
            plt.plot(x_plot, mean_plot, '-', linewidth=2, label=f'{labels[i]} (Mean)', color=color)
        
        # Fill the confidence interval or min-max area
        plt.fill_between(x_plot, lower_plot, upper_plot, 
                         alpha=0.2, color=color, label=f'{labels[i]} ({area_label})')
        
    plt.axhline(y=0.4, color='grey', linestyle='--', linewidth=3, alpha=0.7, label='Centered Attacker')
    
    # Add labels and title
    plt.xlabel('Normalized Distance')
    plt.ylabel('Tracking Response Value')
    # title = 'Multiple Line Plots with Mean Values and '
    # if use_minmax:
    #     title += 'Min-Max Areas'
    # else:
    #     title += f'{int(confidence_level*100)}% Confidence Intervals'
    # if smooth_method:
    #     title += f' ({smooth_method.title()} Smoothed)'
    title = 'Tracking Performance with Off-Centered Victim'
    plt.title(title)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{fig_name}.png", dpi=300, bbox_inches='tight')
    
    return grouped_results

def shift_distance(x, y, center_x, center_y, width, height):
    """
    Returns:
    < 1.0: inside the box
    = 1.0: on the box boundary  
    > 1.0: outside the box
    """
    # Distance from center to point, normalized by half-dimensions
    dx_norm = abs(x - center_x) / (width / 2)
    dy_norm = abs(y - center_y) / (height / 2)
    
    # Box distance is the maximum of the two normalized distances
    return max(dx_norm, dy_norm)

def create_scatter_heatmap(coordinates, values):
    """
    Alternative approach using scatter plot with color mapping.
    Better for irregular coordinate distributions.
    """
    if len(coordinates) != len(values):
        raise ValueError("Coordinates and values lists must have the same length")
    
    # Group values by coordinates and calculate averages
    coord_values = defaultdict(list)
    for coord, val in zip(coordinates, values):
        coord_values[coord].append(val)
    
    # Calculate averaged values for each unique coordinate
    unique_coords = []
    averaged_values = []
    for coord, vals in coord_values.items():
        unique_coords.append(coord)
        averaged_values.append(np.mean(vals))
    
    x_coords = [coord[0] for coord in unique_coords]
    y_coords = [coord[1] for coord in unique_coords]
    
    # Create scatter plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_coords, y_coords, c=averaged_values, 
                        cmap='viridis', s=100, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Scatter Heatmap from Coordinate Pairs')
    ax.grid(True, alpha=0.3)

    # Save the figure
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"scatter_heatmap-{curr_time}.pdf"
    plt.savefig(f"results/{fig_name}", dpi=300, bbox_inches="tight")
    
    # return fig, ax

def create_interpolated_heatmap(coordinates, values, method='cubic', resolution=100):
    """
    Create a smooth interpolated heatmap for irregular coordinates.
    
    Args:
        coordinates: List of (x, y) tuples
        values: List of corresponding values
        method: Interpolation method ('linear', 'cubic', 'rbf')
        resolution: Grid resolution for interpolation
    """
    from scipy import interpolate
    
    if len(coordinates) != len(values):
        raise ValueError("Coordinates and values lists must have the same length")
    
    # Group values by coordinates and calculate averages
    coord_values = defaultdict(list)
    for coord, val in zip(coordinates, values):
        coord_values[coord].append(val)
    
    # Calculate averaged values for each unique coordinate
    unique_coords = []
    averaged_values = []
    for coord, vals in coord_values.items():
        unique_coords.append(coord)
        averaged_values.append(np.mean(vals))
    
    x_coords = np.array([coord[0] for coord in unique_coords])
    y_coords = np.array([coord[1] for coord in unique_coords])
    z_values = np.array(averaged_values)
    
    # Create interpolation grid
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    XI, YI = np.meshgrid(xi, yi)
    
    # Perform interpolation
    if method == 'rbf':
        # Radial Basis Function interpolation
        rbf = interpolate.Rbf(x_coords, y_coords, z_values, function='gaussian')
        ZI = rbf(XI, YI)
    else:
        # Linear or cubic interpolation
        ZI = interpolate.griddata((x_coords, y_coords), z_values, (XI, YI), method=method, fill_value=np.nan)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot interpolated surface
    im = ax.imshow(ZI, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap='viridis', alpha=0.8, aspect='auto')
    
    # Overlay original points
    # scatter = ax.scatter(x_coords, y_coords, c=z_values, cmap='viridis', 
    #                     s=50, edgecolors='white', linewidth=1, alpha=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Interpolated Heatmap ({method.upper()} interpolation)')
    ax.grid(True, alpha=0.3)
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"scatter_heatmap-{curr_time}.pdf"
    plt.savefig(f"results/{fig_name}", dpi=300, bbox_inches="tight")
    
    # return fig, ax

def convert_bbox(bbox, format1, format2, wscale=1.0, hscale=1.0):
    """
    Convert bounding box from one format to another.
    Supported formats: 'x1y1x2y2', 'x1y1wh', 'xywh'.
    """
    if format1 == "x1y1x2y2":
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
    elif format1 == "x1y1wh":
        x1, y1, w, h = bbox
        x, y = x1 + w / 2, y1 + h / 2
    elif format1 == "xywh":
        x, y, w, h = bbox
    else:
        raise ValueError(f"Invalid format: {format1}")
    
    # Apply scaling if needed
    w *= wscale
    h *= hscale

    if format2 == "x1y1x2y2":
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    elif format2 == "x1y1wh":
        return [x - w / 2, y - h / 2, w, h]
    elif format2 == "xywh":
        return [x, y, w, h]
    else:
        raise ValueError(f"Invalid format: {format2}")
    
def convert_bbox_tf(bbox, format1, format2, wscale=1.0, hscale=1.0):
    """Convert bounding box from one format to another using TensorFlow
    
    Args:
        bbox: Tensor of shape [4] containing bounding box coordinates
        format1: Source format string ('x1y1x2y2', 'x1y1wh', 'xywh')
        format2: Target format string ('x1y1x2y2', 'x1y1wh', 'xywh')
    
    Returns:
        Converted bounding box tensor of shape [4]
    
    Supported formats:
        - 'x1y1x2y2': [x1, y1, x2, y2] (top-left, bottom-right corners)
        - 'x1y1wh': [x1, y1, w, h] (top-left corner, width, height)
        - 'xywh': [x, y, w, h] (center coordinates, width, height)
    """
    # Convert to tensor
    bbox = tf.squeeze(bbox)
    
    # Validate formats
    valid_formats = ['x1y1x2y2', 'x1y1wh', 'xywh']
    if format1 not in valid_formats:
        raise ValueError(f"Invalid format1: {format1}. Must be one of {valid_formats}")
    if format2 not in valid_formats:
        raise ValueError(f"Invalid format2: {format2}. Must be one of {valid_formats}")
    
    # Convert from format1 to intermediate xywh format
    if format1 == "x1y1x2y2":
        x1, y1, x2, y2 = tf.unstack(bbox)
        w = x2 - x1
        h = y2 - y1
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
    elif format1 == "x1y1wh":
        x1, y1, w, h = tf.unstack(bbox)
        x = x1 + w / 2.0
        y = y1 + h / 2.0
    elif format1 == "xywh":
        x, y, w, h = tf.unstack(bbox)
    
    # Apply scaling if needed
    w *= wscale
    h *= hscale
    
    # Convert from intermediate xywh format to format2
    if format2 == "x1y1x2y2":
        return tf.stack([x - w/2, y - h/2, x + w/2, y + h/2])
    elif format2 == "x1y1wh":
        return tf.stack([x - w/2, y - h/2, w, h])
    elif format2 == "xywh":
        return tf.stack([x, y, w, h])
    
def get_iou_tf(bbx1, bbx2):
    """Calculate Intersection over Union (IoU) for bounding boxes
    
    Args:
        bbx1: Tensor of shape [N, 4] with bounding boxes in format [x1, y1, x2, y2]
        bbx2: Tensor of shape [N, 4] with bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        iou: Tensor of shape [N] containing IoU values for each pair of boxes
    """
    # Convert inputs to tensors
    bbx1 = tf.convert_to_tensor(bbx1, dtype=tf.float32)
    bbx2 = tf.convert_to_tensor(bbx2, dtype=tf.float32)
    
    # Extract coordinates
    x1 = tf.maximum(bbx1[:, 0], bbx2[:, 0])
    y1 = tf.maximum(bbx1[:, 1], bbx2[:, 1])
    x2 = tf.minimum(bbx1[:, 2], bbx2[:, 2])
    y2 = tf.minimum(bbx1[:, 3], bbx2[:, 3])
    
    # Calculate intersection area
    intersection_width = tf.maximum(0.0, x2 - x1)
    intersection_height = tf.maximum(0.0, y2 - y1)
    intersection = intersection_width * intersection_height
    
    # Calculate areas of both bounding boxes
    area1 = (bbx1[:, 2] - bbx1[:, 0]) * (bbx1[:, 3] - bbx1[:, 1])
    area2 = (bbx2[:, 2] - bbx2[:, 0]) * (bbx2[:, 3] - bbx2[:, 1])
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU, avoiding division by zero
    iou = tf.where(
        union > 0,
        intersection / union,
        0.0
    )
    
    return iou

def visualize_boxes(boxes, labels, save_path, img_size=(10, 10)):
    fig, ax = plt.subplots(figsize=img_size)
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, label, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_xlim(min(b[0] for b in boxes)-20, max(b[2] for b in boxes)+20)
    ax.set_ylim(min(b[1] for b in boxes)-20, max(b[3] for b in boxes)+20)
    ax.invert_yaxis()  # Image coordinates: origin at top-left
    plt.axis('equal')
    # plt.show()
    plt.savefig(save_path)
    plt.close()

def visualize_bbox_trace_video(
    trace: np.ndarray,
    output_path: str,
    frame_size: Tuple[int, int] = (1920, 1080),
    fps: float = 30.0,
    bbox_colors: Optional[List[Tuple[int, int, int]]] = None,
    bbox_labels: Optional[List[str]] = None,
    thickness: int = 3,
    font_scale: float = 0.8,
    show_trail: bool = True,
    trail_length: int = 5,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Create a video visualization of bounding box traces.
    
    Args:
        trace: np.ndarray of shape [N, 4, 4] where:
               - N = number of frames
               - First 4 = 4 different bounding boxes 
               - Last 4 = bbox coordinates (x1, y1, x2, y2)
        output_path: Path to save the output video
        frame_size: (width, height) of the output video frames
        fps: Frames per second for the output video
        bbox_colors: List of BGR colors for each bbox type. If None, uses default colors.
        bbox_labels: List of labels for each bbox type. If None, uses default labels.
        thickness: Line thickness for bounding boxes
        font_scale: Font scale for labels
        show_trail: Whether to show trailing positions of bbox centers
        trail_length: Number of previous positions to show in trail
        background_color: BGR color for background
    """
    # print(f"Visualizing bounding box trace with shape: {trace.shape}")
    N, num_bboxes, coords = trace.shape
    assert num_bboxes == 4 and coords == 4, f"Expected shape [N, 4, 4], got {trace.shape}"
    
    # Default colors (BGR format for OpenCV)
    if bbox_colors is None:
        bbox_colors = [
            (0, 255, 0),    # Green - victim_2d_pred
            (255, 0, 0),    # Blue - attacker_2d_pred  
            (0, 255, 255),  # Yellow - victim_2d_rotated
            (255, 0, 255),  # Magenta - attacker_2d_rotated
        ]
    
    # Default labels
    if bbox_labels is None:
        bbox_labels = [
            "Victim Pred",
            "Attacker Pred", 
            "Victim Rotated",
            "Attacker Rotated"
        ]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    # Calculate center trails for each bbox
    centers_history = []
    for frame_idx in range(N):
        frame_centers = []
        for bbox_idx in range(num_bboxes):
            x1, y1, x2, y2 = trace[frame_idx, bbox_idx]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            frame_centers.append((center_x, center_y))
        centers_history.append(frame_centers)
    
    # print(f"Creating video with {N} frames at {fps} FPS...")
    # print(f"Output: {output_path}")
    # print(f"Frame size: {frame_size}")
    
    for frame_idx in range(N):
        # Create black background
        frame = np.full((frame_size[1], frame_size[0], 3), background_color, dtype=np.uint8)
        
        # Draw trails if enabled
        if show_trail and frame_idx > 0:
            for bbox_idx in range(num_bboxes):
                trail_start = max(0, frame_idx - trail_length)
                for trail_idx in range(trail_start, frame_idx):
                    # Calculate alpha for trail fade effect
                    alpha = (trail_idx - trail_start + 1) / trail_length * 0.5
                    trail_color = tuple(int(c * alpha) for c in bbox_colors[bbox_idx])
                    
                    prev_center = centers_history[trail_idx][bbox_idx]
                    curr_center = centers_history[trail_idx + 1][bbox_idx] if trail_idx + 1 < len(centers_history) else prev_center
                    
                    cv2.line(frame, prev_center, curr_center, trail_color, max(1, thickness // 2))
                    cv2.circle(frame, curr_center, 2, trail_color, -1)
        
        # Draw current frame bounding boxes
        for bbox_idx in range(num_bboxes):
            x1, y1, x2, y2 = trace[frame_idx, bbox_idx]
            
            # Convert to integers and ensure valid coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp coordinates to frame bounds
            x1 = max(0, min(x1, frame_size[0] - 1))
            y1 = max(0, min(y1, frame_size[1] - 1))
            x2 = max(0, min(x2, frame_size[0] - 1))
            y2 = max(0, min(y2, frame_size[1] - 1))
            
            # Skip invalid bounding boxes
            if x2 <= x1 or y2 <= y1:
                continue
                
            color = bbox_colors[bbox_idx]
            
            # Draw bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw label
            label = bbox_labels[bbox_idx]
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Position label above the bbox, with background
            label_x = x1
            label_y = max(label_size[1] + 10, y1 - 10)
            
            # Draw label background
            cv2.rectangle(
                frame, 
                (label_x - 5, label_y - label_size[1] - 5),
                (label_x + label_size[0] + 5, label_y + 5),
                color, 
                -1
            )
            
            # Draw label text
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(
                frame, 
                label, 
                (label_x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                text_color, 
                thickness - 1
            )
        
        # Add frame information
        frame_info = f"Frame: {frame_idx + 1}/{N}"
        cv2.putText(
            frame, 
            frame_info, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
        
        # Write frame to video
        out.write(frame)
    
    # Release video writer
    out.release()
    # print(f"Video saved successfully to: {output_path}")

def calculate_angular_velocity(orientations, timestamps):
    """
    Calculate angular velocity from orientation data.
    
    Args:
        orientations: List of (roll, pitch, yaw) tuples in radians
        timestamps: List of timestamps in nanoseconds
    
    Returns:
        angular_velocities: List of (wx, wy, wz) tuples in rad/s
        avg_timestamps: List of average timestamps for each velocity sample
    """
    if len(orientations) < 2 or len(timestamps) < 2:
        return [], []
    
    # Convert to numpy arrays for easier computation
    angles = np.array(orientations)
    times = np.array(timestamps)
    times = times - times[0]  # Normalize to start from zero
    print(times)
    
    # Calculate time differences
    dt = np.diff(times)
    
    # Calculate angle differences, handling wrap-around for angles
    dtheta = np.diff(angles, axis=0)
    dtheta = np.where(dtheta > np.pi, dtheta - 2*np.pi, dtheta)
    dtheta = np.where(dtheta < -np.pi, dtheta + 2*np.pi, dtheta)
    
    # Calculate angular velocities
    angular_velocities = dtheta / dt[:, np.newaxis]
    
    # Calculate average timestamps for each velocity sample
    avg_timestamps = (times[:-1] + times[1:]) / 2
    
    return angular_velocities.tolist(), avg_timestamps.tolist()

def plot_angular_velocity(angular_velocities, timestamps, save_path="angular_velocity_plot.png"):
    """
    Plot angular velocity components and magnitude over time.
    
    Args:
        angular_velocities: List of (wx, wy, wz) tuples in rad/s
        timestamps: List of timestamps in seconds
    """
    import matplotlib.pyplot as plt
    
    if not angular_velocities:
        print("No data to plot")
        return
    
    # Convert to numpy array for easier indexing
    velocities = np.array(angular_velocities)
    times = np.array(timestamps)
    
    # Calculate magnitude
    magnitude = np.linalg.norm(velocities, axis=1)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot components
    ax1.plot(times, velocities[:, 0], 'r-', label='ωx (roll rate)')
    ax1.plot(times, velocities[:, 1], 'g-', label='ωy (pitch rate)')
    ax1.plot(times, velocities[:, 2], 'b-', label='ωz (yaw rate)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Angular Velocity Components')
    
    # Plot magnitude
    ax2.plot(times, magnitude, 'k-', linewidth=2, label='|ω| (magnitude)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Speed (rad/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Angular Velocity Magnitude')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def get_vertices(world_pose):
    w, h, l = 2.0, 2.0, 2.0  # meters. dimensions of the pedestrian in gazebo
    corners = np.zeros((8, 3), dtype=np.float32)
    corners[:, 0] = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners[:, 1] = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners[:, 2] = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    corners = corners + world_pose
    return corners

def get_color(idx):
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue (OpenCV uses BGR)
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),  # Dark blue
        (0, 128, 0),  # Dark green
        (0, 0, 128),  # Dark red
        (128, 128, 0),  # Teal
    ]
    return colors[idx % len(colors)]


def get_direction_region(vx, vy):
    """
    Classify velocity vector [vx, vy] into 4 regions of 90 degrees each.
    
    Region 0: -45° to 45° (positive x-axis direction, right/east)
    Region 1: 45° to 135° (positive y-axis direction, up/north)
    Region 2: 135° to 225° (negative x-axis direction, left/west)
    Region 3: 225° to 315° (negative y-axis direction, down/south)
    
    Args:
        vx, vy: velocity components
        
    Returns:
        int: region index (0-3)
    """
    # Handle zero velocity case
    if vx == 0 and vy == 0:
        return 0  # or raise an exception, depending on your needs
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = math.atan2(vy, vx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize angle to 0-360 range
    if angle_deg < 0:
        angle_deg += 360
    
    # Shift by 45 degrees so that region 0 is centered on x-axis
    shifted_angle = angle_deg + 45
    if shifted_angle >= 360:
        shifted_angle -= 360
    
    # Convert to region index (each region is 90 degrees)
    region = int(shifted_angle / 90)
    
    return region


def plot_lines(
    data: np.ndarray,
    x_axis: np.ndarray,
    labels: list,
    axis_names: list,
    fig_name: str,
    line_styles: list,
    colors: list,
):
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    # Basic line plot
    plt.figure(figsize=(10, 6))  # Set figure size (optional)

    for i in range(len(data)):
        plt.plot(
            x_axis[i],
            data[i],
            label=labels[i],
            linestyle=line_styles[i],
            color=colors[i],
            linewidth=5,
            alpha=0.7,
        )

    plt.xlabel(f"Threshold")
    plt.ylabel(axis_names[1])
    # plt.title('Line Plot')
    plt.grid(True)  # Add grid (optional)
    plt.legend()  # Show legend (optional)

    # Save the figure
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"{fig_name}-{curr_time}.pdf"
    plt.savefig(f"results/dist1/{fig_name}", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_bins(data: np.ndarray, bins: int, labels: list, lim: list, fig_name: str):
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    plt.figure(figsize=(10, 6))  # Set figure size (optional)
    for i in range(len(data)):
        counts, bin_edges = np.histogram(data[i], bins=bins[i])
        counts = counts / sum(counts)
        plt.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            align="edge",
            alpha=0.3,
            label=labels[i],
        )

    plt.xlabel("Scale Change")
    plt.ylabel("Percentage")
    plt.xlim(*lim[0])
    plt.ylim(*lim[1])
    # plt.title('Histogram Plot')
    plt.grid(True)  # Add grid (optional)
    plt.legend()  # Show legend (optional)

    # Save the figure
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"{fig_name}-{curr_time}.pdf"
    plt.savefig(f"results/dist1/{fig_name}", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_polar_scatter(
    r: np.ndarray, theta: np.ndarray, vbar: np.ndarray, label: str, fig_name: str
):
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    # Create a polar subplot
    fig = plt.subplot(projection="polar")
    r = np.array(r)
    theta = np.array(theta)
    vbar = np.array(vbar)

    # Create scatter plot
    scatter = plt.scatter(
        theta,
        r,
        c=vbar,  # color based on percentages
        cmap="RdYlBu",  # color map (also 'viridis', 'coolwarm', etc.)
        s=75,  # size of dots
        alpha=0.5,
    )  # transparency

    # Add a colorbar
    cbar = plt.colorbar(scatter, pad=0.07)
    cbar.set_label(label)

    plt.ylim(0, 40)  # set radius limits
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("", pad=20)
    plt.rgrids(np.arange(1, max(r) + 1, 5), angle=0)  # customize radial grid
    plt.thetagrids(np.arange(0, 360, 45))  # customize angular grid

    # Save the figure
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"{fig_name}-{curr_time}.pdf"
    plt.savefig(f"results/{fig_name}", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def plot_boxes(data: np.ndarray, labels: list, axis_names: list, fig_name: str):
    """plot box plot"""
    plt.rcParams.update(
        {
            "font.size": 20,  # Increased from 16
            "axes.labelsize": 22,  # Increased from 18
            "axes.titlesize": 22,  # Increased from 18
            "xtick.labelsize": 20,  # Increased from 16
            "ytick.labelsize": 20,  # Increased from 16
            "legend.fontsize": 20,  # Increased from 16
            "boxplot.boxprops.linewidth": 2.0,  # Thicker box lines
            "boxplot.whiskerprops.linewidth": 2.0,  # Thicker whisker lines
            "boxplot.medianprops.linewidth": 2.5,  # Thicker median line
            "boxplot.flierprops.markersize": 8,  # Larger outlier points
        }
    )

    plt.figure(figsize=(10, 6))  # Set figure size (optional)
    plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.3,
        meanprops={"linestyle": "--", "linewidth": 2.5},
        medianprops={"linewidth": 2.5},
    )
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    # plt.title('Box Plot')
    plt.grid(True)  # Add grid (optional)

    # Save the figure
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    fig_name = f"{fig_name}-{curr_time}.pdf"
    plt.savefig(f"results/{fig_name}", dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def get_plots1(traces, exp_names, line_stypes, colors):
    """
    Plots to understand the performance of the tracking algorithm
    """
    precision_to_plot = []
    prec_x = []
    success_to_plot = []
    suc_x = []
    scale_to_plot = []
    scale_x = []
    for t in traces:
        precision_to_plot.append(t.results["vic_trial_prec"])
        prec_x.append(t.results["prec_threshold"])
        success_to_plot.append(t.results["vic_trial_iou"])
        suc_x.append(t.results["iou_threshold"])
        scale_to_plot.append(t.results["vic_d_scale_flat"])
        scale_x.append(100)

    labels = [name.split("-")[-1] for name in exp_names]

    plot_lines(
        precision_to_plot,
        prec_x,
        labels,
        ["", "Precision Plot"],
        "precision",
        line_stypes,
        colors,
    )
    plot_lines(
        success_to_plot,
        suc_x,
        labels,
        ["", "Success Plot"],
        "success",
        line_stypes,
        colors,
    )
    plot_bins(
        scale_to_plot[:2], scale_x[:2], labels[:2], [[-0.7, 0.7], [0, 0.1]], "scale1"
    )
    plot_bins(
        scale_to_plot[2:5], scale_x[2:5], labels[2:5], [[-0.7, 0.7], [0, 0.1]], "scale2"
    )
    plot_bins(
        scale_to_plot[5:], scale_x[5:], labels[5:], [[-0.7, 0.7], [0, 0.1]], "scale3"
    )


def get_plot2(traces):
    """
    Plots to understand factors affecting the attack performance
    """

    polar_r = []
    polar_theta = []
    polar_vbar_error = []
    polar_vbar_iou1 = []
    polar_vbar_iou2 = []
    for trace in traces:
        polar_r.extend(trace.results["p_r"])
        polar_theta.extend(trace.results["p_angle"])
        polar_vbar_error.extend(trace.results["p_vbar_error"])
        polar_vbar_iou1.extend(trace.results["p_vbar_iou1"])
        polar_vbar_iou2.extend(trace.results["p_vbar_iou2"])

    plot_polar_scatter(
        polar_r, polar_theta, polar_vbar_error, "Center Error", "polar_error"
    )
    plot_polar_scatter(polar_r, polar_theta, polar_vbar_iou1, "IoU1", "polar_iou1")
    plot_polar_scatter(polar_r, polar_theta, polar_vbar_iou2, "IoU2", "polar_iou2")


def get_plot3(traces):

    atk_perimeter = []
    vic_perimeter = []
    atk_fail_error = []
    vic_fail_error = []
    atk_switch_error = []
    vic_switch_error = []
    fail_polar_r = []
    fail_polar_theta = []
    fail_polar_error = []

    polar_r_camera = []
    polar_theta_camera = []
    polar_error_camera = []
    fail_polar_r_camera = []
    fail_polar_theta_camera = []

    for trace in traces:
        atk_perimeter.extend(trace.results["atk_perimeter"])
        vic_perimeter.extend(trace.results["vic_perimeter"])
        atk_fail_error.extend(trace.results["atk_fail_error"])
        vic_fail_error.extend(trace.results["vic_fail_error"])
        atk_switch_error.extend(trace.results["at_switch_atk_error"])
        vic_switch_error.extend(trace.results["at_switch_vic_error"])
        fail_polar_r.extend(trace.results["p_r_fail"])
        fail_polar_theta.extend(trace.results["p_angle_fail"])
        fail_polar_error.extend(trace.results["p_vbar_error_fail"])

        polar_r_camera.extend(trace.results["p_r_camera"])
        polar_theta_camera.extend(trace.results["p_angle_camera"])
        polar_error_camera.extend(trace.results["p_vbar_error"])
        fail_polar_r_camera.extend(trace.results["p_r_camera_fail"])
        fail_polar_theta_camera.extend(trace.results["p_angle_camera_fail"])

    atk_perimeter = np.concatenate(atk_perimeter)
    vic_perimeter = np.concatenate(vic_perimeter)
    # atk_fail_error = np.concatenate(atk_fail_error)
    # vic_fail_error = np.concatenate(vic_fail_error)
    # atk_switch_error = np.concatenate(atk_switch_error)
    # vic_switch_error = np.concatenate(vic_switch

    print(f"Average attacker perimeter: {np.mean(atk_perimeter)}")
    print(f"Average victim perimeter: {np.mean(vic_perimeter)}")
    print(f"Average attacker error at failure: {np.mean(atk_fail_error)}")
    print(f"Average victim error at failure: {np.mean(vic_fail_error)}")
    print(f"Average attacker error at switch: {np.mean(atk_switch_error)}")
    print(f"Average victim error at switch: {np.mean(vic_switch_error)}")
    print(f"Max attacker error at failure: {np.max(atk_fail_error)}")
    print(f"Max victim error at failure: {np.max(vic_fail_error)}")
    print(f"Max attacker error at switch: {np.max(atk_switch_error)}")
    print(f"Max victim error at switch: {np.max(vic_switch_error)}")
    print(f"Min attacker error at failure: {np.min(atk_fail_error)}")
    print(f"Min victim error at failure: {np.min(vic_fail_error)}")
    print(f"Min attacker error at switch: {np.min(atk_switch_error)}")
    print(f"Min victim error at switch: {np.min(vic_switch_error)}")

    # plot_polar_scatter(polar_r, polar_theta, polar_error, "Center Error", "polar_error")

    # plot_polar_scatter(fail_polar_r, fail_polar_theta, fail_polar_error, "Center Error", "polar_error")
    # plot_polar_scatter(fail_polar_r_camera, fail_polar_theta_camera, fail_polar_error, "Center Error", "polar_error_camera")
    # plot_polar_scatter(polar_r_camera, polar_theta_camera, polar_error_camera, "Center Error", "polar_error_camera_switch")

    # print(len(polar_error_camera), len(fail_polar_error))
    # randomly select from fail_polar_error to match the length of polar_error_camera
    np.random.seed(0)
    idx = np.random.choice(
        len(fail_polar_error), len(polar_error_camera), replace=False
    )
    fail_polar_error = np.array(fail_polar_error)[idx]
    plot_boxes(
        [polar_error_camera, fail_polar_error],
        ["Success", "Failure"],
        ["", "Victim Bbx. Disp."],
        "vic_displacement",
    )


def get_plot4(traces, exp_names):
    switch_frame = []
    labels = []
    scales = ["3.0°", "6.0°"]
    s_idx = 0
    for i, t in enumerate(traces):
        t_switch = [
            frame * 10 for frame in t.results["trial_switch_frame"] if frame != 0
        ]
        switch_frame.append(t_switch)
        labels.append(exp_names[i].split("-")[-1][6:])
        if "gimbal" in exp_names[i]:
            labels[-1] = str(scales[s_idx])
            s_idx += 1

    for i, name in enumerate(exp_names):
        print(i, " ", name, " ", np.mean(switch_frame[i]), np.std(switch_frame[i]))
    plot_boxes(
        switch_frame, labels, ["", "Success Frame #"], "Success Frame Distribution"
    )

def off_eval_kcf(sim_state, trial_path, trial_num, exp_name, kcf_tracker):
    frame0 = sim_state.get_frame(0)
    init_region = convert_bbox(frame0["10_det_uav"], "x1y1x2y2", "x1y1wh")
    img_list = [f"{trial_path}/raw_images/image_{i}.jpg" for i in range(len(sim_state))]
    
    kcf_tracker.prepare_region_file(*init_region)
    img_list = kcf_tracker.prepare_images_file(img_list)
    results = kcf_tracker.run_tracker()
    if results is None:
        raise ValueError(f"No results found for trial {trial_num+1:03d} in {exp_name}")
    
    last_pred = convert_bbox(results[-1], "x1y1wh", "x1y1x2y2")
    frame_last = sim_state.get_frame(len(sim_state) - 1)
    last_atk_region = convert_bbox(frame_last["20_det_uav"], "x1y1x2y2", "xywh")
    # check if the center of last atk region is within the last predicted region
    center_x, center_y = last_atk_region[0], last_atk_region[1]
    pred_x1, pred_y1, pred_x2, pred_y2 = last_pred
    if pred_x1 <= center_x <= pred_x2 and pred_y1 <= center_y <= pred_y2:
        success_count += 1
        print(f"Trial {trial_num+1:03d} success")
        
    all_images = []
    for idx in range(len(sim_state)):
        frame = sim_state.get_frame(idx)
        img = frame["img"].copy()
        pred_region = convert_bbox(results[idx], "x1y1wh", "x1y1x2y2")
        it_trace = frame["it_trace"]
        pred_vic = it_trace[0]
        pred_atk = it_trace[1]
        rot_vic = it_trace[2]
        rot_atk = it_trace[3]
        cv2.rectangle(img, (int(pred_region[0]), int(pred_region[1])),
                        (int(pred_region[2]), int(pred_region[3])),
                        (0, 255, 0), 2)
        cv2.rectangle(img, (int(pred_vic[0]), int(pred_vic[1])),
                        (int(pred_vic[2]), int(pred_vic[3])),
                        (255, 0, 0), 2)
        cv2.rectangle(img, (int(pred_atk[0]), int(pred_atk[1])),
                        (int(pred_atk[2]), int(pred_atk[3])),
                        (0, 0, 255), 2)
        cv2.rectangle(img, (int(rot_vic[0]), int(rot_vic[1])),
                        (int(rot_vic[2]), int(rot_vic[3])),
                        (255, 255, 0), 2)
        cv2.rectangle(img, (int(rot_atk[0]), int(rot_atk[1])),
                        (int(rot_atk[2]), int(rot_atk[3])),
                        (0, 255, 255), 2)
        cv2.putText(img, f"Frame: {idx+1}/{len(sim_state)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # label the rectangles
        cv2.putText(img, "Pred", (int(pred_region[0]), int(pred_region[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "Vic Det", (int(pred_vic[0]), int(pred_vic[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, "Atk Det", (int(pred_atk[0]), int(pred_atk[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img, "Vic Pred", (int(rot_vic[0]), int(rot_vic[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(img, "Atk Pred", (int(rot_atk[0]), int(rot_atk[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        all_images.append(img)

    video_path = f"{trial_path}/videos/tracker_comparison.mp4"
    images_to_video(all_images, video_path, fps=10)
