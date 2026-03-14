import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_raw_response(yaml_path):
    """Load raw response data and tracking context saved from C++"""
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)

    tracking_contexts = {}
    
    # Load tracking context
    frame_idx = 0
    while True:
        key = f'frame_{frame_idx}'
        frame_node = fs.getNode(key)
        if frame_node.empty():
            break
        tc = {
            'frame_idx': int(frame_node.getNode("frame_idx").real()) if not frame_node.getNode("frame_idx").empty() else 0,
            'resize_image': frame_node.getNode("resize_image").real(),
            'center_x': frame_node.getNode("center_x").real(),
            'center_y': frame_node.getNode("center_y").real(),
            'window_width': int(frame_node.getNode("window_width").real()),
            'window_height': int(frame_node.getNode("window_height").real()),
            'cell_size': int(frame_node.getNode("cell_size").real()),
            'current_scale': frame_node.getNode("current_scale").real(),
            'confidence': frame_node.getNode("confidence").real(),
            'max_response_x': frame_node.getNode("max_response_x").real(),
            'max_response_y': frame_node.getNode("max_response_y").real(),
            'response_map': frame_node.getNode("response_map").mat(),
            'bbox_x1': frame_node.getNode("bbox_x1").real(),
            'bbox_y1': frame_node.getNode("bbox_y1").real(),
            'bbox_width': frame_node.getNode("bbox_width").real(),
            'bbox_height': frame_node.getNode("bbox_height").real(),
        }
        tc.update(extract_tracking_context(tc))
        tracking_contexts[f"frame_{frame_idx}"] = tc
        frame_idx += 1
    tracking_contexts['n_frames'] = frame_idx

    fs.release()
    return tracking_contexts

def extract_tracking_context(tracker_state):
    """
    Extract spatial context from response map and tracker state
    
    Args:
        response_map: Raw response map from KCF tracker
        tracker_state: Dictionary containing tracker information
            - 'center_x', 'center_y': Current target center in image coordinates
            - 'window_width', 'window_height': Search window size in pixels
            - 'cell_size': HOG cell size (typically 4)
            - 'current_scale': Current scale factor
            - 'confidence': max response
            - 'response_map': raw response map from KCF tracker
    """
    # find the coordinate of the maximum response
    # response_map = tracker_state['response_map']
    # max_response_idx = np.unravel_index(np.argmax(response_map), response_map.shape)

    # Response map dimensions (in HOG cells)
    resp_height, resp_width = tracker_state['response_map'].shape
    
    # Convert to pixel dimensions
    pixel_width = resp_width * tracker_state['cell_size']
    pixel_height = resp_height * tracker_state['cell_size']
    
    # Search window boundaries in image coordinates
    center_x, center_y = tracker_state['center_x'], tracker_state['center_y']

    resize_image = tracker_state['resize_image']
    if resize_image:
        center_x *= 2
        center_y *= 2
        pixel_width *= 2
        pixel_height *= 2
    
    # Account for current scale
    scaled_width = pixel_width * tracker_state['current_scale']
    scaled_height = pixel_height * tracker_state['current_scale']
    
    # Search window boundaries
    search_left = int(center_x - scaled_width / 2)
    search_top = int(center_y - scaled_height / 2)
    search_right = int(center_x + scaled_width / 2)
    search_bottom = int(center_y + scaled_height / 2)
    
    return {
        'search_bounds': (search_left, search_top, search_right, search_bottom),
        'pixel_size': (scaled_width, scaled_height),
        'response_size': (resp_width, resp_height),
        'cell_size': tracker_state['cell_size'],
        'scale': tracker_state['current_scale']
    }

def recenter_response_map(response_map):
    # equivalent to np.fft.fftshift
    # new_map = np.zeros_like(response_map)
    # resp_height, resp_width = response_map.shape
    # center_x = resp_width // 2
    # center_y = resp_height // 2
    # for y in range(resp_height):
    #     for x in range(resp_width):
    #         new_x = (x + center_x) % resp_width
    #         new_y = (y + center_y) % resp_height
    #         new_map[new_y, new_x] = response_map[y, x]

    new_map = np.fft.fftshift(response_map)
    return new_map


def response_to_image_coordinates(response_map, spatial_context):
    """
    Convert response map coordinates to image pixel coordinates
    """
    resp_height, resp_width = response_map.shape
    search_left, search_top, search_right, search_bottom = spatial_context['search_bounds']
    
    # Create coordinate grids
    x_coords = np.linspace(search_left, search_right, resp_width)
    y_coords = np.linspace(search_top, search_bottom, resp_height)
    
    return x_coords, y_coords

def response_at_coordinates(response_map, spatial_context, x, y):
    """
    Get response value at specific image coordinates
    """
    x_coords, y_coords = response_to_image_coordinates(response_map, spatial_context)

    # Ensure coordinates are within bounds
    # if x < np.min(x_coords) or x > np.max(x_coords) or y < np.min(y_coords) or y > np.max(y_coords):
    #     return 0.0
    
    # Find closest coordinates in the response map
    x_idx = np.argmin(np.abs(x_coords - x))
    y_idx = np.argmin(np.abs(y_coords - y))

    recentered_map = recenter_response_map(response_map)

    return recentered_map[y_idx, x_idx]

def max_response_in_box(response_map, spatial_context, x, y, box_size=11):
    """
    Get maximum response value in a box around specific image coordinates
    
    Args:
        response_map: The response map array
        spatial_context: Spatial context for coordinate conversion
        x, y: Center coordinates of the box (in image coordinates)
        box_size: Size of the box in image coordinate units (default 11 for 11x11 pixels)
    
    Returns:
        Maximum response value in the box
    """
    x_coords, y_coords = response_to_image_coordinates(response_map, spatial_context)
    
    # Define box boundaries in image coordinates
    half_box = box_size // 2
    box_x_min = x - half_box
    box_x_max = x + half_box
    box_y_min = y - half_box
    box_y_max = y + half_box
    
    # Find indices in response map that fall within the box
    x_mask = (x_coords >= box_x_min) & (x_coords <= box_x_max)
    y_mask = (y_coords >= box_y_min) & (y_coords <= box_y_max)
    
    # Get the indices that are within the box
    x_indices = np.where(x_mask)[0]
    y_indices = np.where(y_mask)[0]
    
    # If no points fall within the box, return the closest point
    if len(x_indices) == 0 or len(y_indices) == 0:
        x_idx = np.argmin(np.abs(x_coords - x))
        y_idx = np.argmin(np.abs(y_coords - y))
        recentered_map = recenter_response_map(response_map)
        return recentered_map[y_idx, x_idx]
    
    # Extract the box region from response map
    recentered_map = recenter_response_map(response_map)
    box_region = recentered_map[np.ix_(y_indices, x_indices)]
    
    # Return maximum value in the box
    return np.max(box_region)

def overlay_heatmap_on_image(image, response_map, spatial_context, alpha=0.3, threshold=0.3):
    """
    Overlay response heatmap on original image
    
    Args:
        image: Original image (BGR or RGB)
        response_map: Response map from tracker
        spatial_context: Output from extract_tracking_context()
        alpha: Transparency of overlay (0=transparent, 1=opaque)
        threshold: Only show responses above this percentile
    """
    
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR (OpenCV format) and convert to RGB
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image.copy()
    
    # Get image coordinates
    x_coords, y_coords = response_to_image_coordinates(response_map, spatial_context)
    
    # Normalize response map
    response_norm = (response_map - np.min(response_map)) / (np.max(response_map) - np.min(response_map))
    
    # Apply threshold
    threshold_value = np.percentile(response_norm, threshold * 100)
    # response_masked = np.where(response_norm > threshold_value, response_norm, 0)
    response_masked = response_norm.copy()
    
    # Create overlay
    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    
    # 1. Original image with search window
    axes[0].imshow(display_image)
    search_left, search_top, search_right, search_bottom = spatial_context['search_bounds']
    search_rect = Rectangle((search_left, search_top), 
                          search_right - search_left, 
                          search_bottom - search_top,
                          linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    axes[0].add_patch(search_rect)
    axes[0].set_title('Original Image + Search Window')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    
    # 2. Heatmap overlay
    axes[1].imshow(display_image)
    
    # Create meshgrid for proper overlay
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Only overlay where we have significant response
    overlay = axes[1].contourf(X, Y, response_masked, levels=10, alpha=alpha, cmap='jet')
    plt.colorbar(overlay, ax=axes[1], label='Response Strength')
    
    # Mark peak location
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    peak_x = x_coords[peak_idx[1]]
    peak_y = y_coords[peak_idx[0]]
    # axes[1].plot(peak_x, peak_y, 'r*', markersize=15, label='Peak Response')
    # axes[1].legend()
    # axes[1].set_title('Heatmap Overlay on Image')
    # axes[1].set_xlabel('X (pixels)')
    # axes[1].set_ylabel('Y (pixels)')
    
    # 3. Response map alone (zoomed)
    im3 = axes[2].imshow(response_map, cmap='jet', extent=[search_left, search_right, search_bottom, search_top])
    axes[2].plot(peak_x, peak_y, 'r*', markersize=15)
    plt.colorbar(im3, ax=axes[2], label='Response')
    axes[2].set_title('Response Map (Search Window)')
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    return fig, (peak_x, peak_y)

def create_tracking_visualization_sequence(image_sequence, response_sequence, tracker_states):
    """
    Create visualization for a sequence of frames showing tracking evolution
    """
    n_frames = len(image_sequence)
    
    fig, axes = plt.subplots(2, min(n_frames, 6), figsize=(20, 8))
    if n_frames == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(n_frames, 6)):
        # Top row: Image with search window and peak
        axes[0, i].imshow(cv2.cvtColor(image_sequence[i], cv2.COLOR_BGR2RGB))
        
        spatial_context = extract_tracking_context(response_sequence[i], tracker_states[i])
        x_coords, y_coords = response_to_image_coordinates(response_sequence[i], spatial_context)
        
        # Search window
        search_left, search_top, search_right, search_bottom = spatial_context['search_bounds']
        search_rect = Rectangle((search_left, search_top), 
                              search_right - search_left, 
                              search_bottom - search_top,
                              linewidth=1, edgecolor='red', facecolor='none')
        axes[0, i].add_patch(search_rect)
        
        # Peak location
        peak_idx = np.unravel_index(np.argmax(response_sequence[i]), response_sequence[i].shape)
        peak_x = x_coords[peak_idx[1]]
        peak_y = y_coords[peak_idx[0]]
        axes[0, i].plot(peak_x, peak_y, 'r*', markersize=10)
        
        # Target center
        axes[0, i].plot(tracker_states[i]['center_x'], tracker_states[i]['center_y'], 'bo', markersize=8)
        
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].set_xlim(search_left - 50, search_right + 50)
        axes[0, i].set_ylim(search_bottom + 50, search_top - 50)  # Flip Y for image coordinates
        
        # Bottom row: Response maps
        im = axes[1, i].imshow(response_sequence[i], cmap='jet')
        axes[1, i].plot(peak_idx[1], peak_idx[0], 'r*', markersize=10)
        axes[1, i].set_title(f'Response Map {i+1}')
        
        # Add colorbar for first response map
        if i == 0:
            plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    return fig

def analyze_tracking_displacement(response_map, spatial_context, previous_center):
    """
    Analyze the displacement and movement pattern
    """
    # Find peak in response map
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    
    # Convert to image coordinates
    x_coords, y_coords = response_to_image_coordinates(response_map, spatial_context)
    peak_x = x_coords[peak_idx[1]]
    peak_y = y_coords[peak_idx[0]]
    
    # Calculate displacement from search center
    search_center_x = (spatial_context['search_bounds'][0] + spatial_context['search_bounds'][2]) / 2
    search_center_y = (spatial_context['search_bounds'][1] + spatial_context['search_bounds'][3]) / 2
    
    displacement_x = peak_x - search_center_x
    displacement_y = peak_y - search_center_y
    displacement_magnitude = np.sqrt(displacement_x**2 + displacement_y**2)
    
    # If we have previous center, calculate inter-frame motion
    # if previous_center is not None:
    #     motion_x = peak_x - previous_center[0]
    #     motion_y = peak_y - previous_center[1]
    #     motion_magnitude = np.sqrt(motion_x**2 + motion_y**2)
    # else:
    #     motion_x = motion_y = motion_magnitude = 0
    
    return {
        'peak_location': (peak_x, peak_y),
        'search_center': (search_center_x, search_center_y),
        'displacement': (displacement_x, displacement_y, displacement_magnitude),
        # 'inter_frame_motion': (motion_x, motion_y, motion_magnitude),
        'response_strength': np.max(response_map)
    }


def overlay_analysis(image, tracker_state):
    try:
        # Extract spatial context
        spatial_context = extract_tracking_context(tracker_state)
        
        # Create overlay visualization
        # image = cv2.imread("current_frame.jpg")
        fig, peak_location = overlay_heatmap_on_image(image, recenter_response_map(tracker_state['response_map']), spatial_context)
        
        # Analyze displacement
        # displacement_info = analyze_tracking_displacement(tracker_state['response_map'], spatial_context, None)
        
        # print("Tracking Analysis:")
        # print(f"Peak found at: ({peak_location[0]:.1f}, {peak_location[1]:.1f})")
        # print(f"Displacement from search center: {displacement_info['displacement'][2]:.1f} pixels")
        # print(f"Response strength: {displacement_info['response_strength']:.3f}")
        
        # plt.show()
        save_path = "tracking_overlay.png"
        fig.savefig(save_path)
        return 
        
    except Exception as e:
        print(f"Demo failed: {e}")
