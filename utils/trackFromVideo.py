import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

def process_video(video_path, output_dir=None, confidence=0.5):
    # Load the YOLO model
    model = YOLO("yolov8n.pt")
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output directory is provided
    writer = None
    if output_dir:
        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a proper output filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        
        # Initialize the writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Verify the writer was created properly
        if not writer.isOpened():
            print(f"Error: Could not create output video at {output_path}")
            writer = None
    
    # Process each frame
    frame_number = 0
    start_time = time.time()
    
    # Dictionary to store tracking data for analysis
    tracking_data = {}
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_number += 1
        if frame_number % 10 == 0:  # Status update every 10 frames
            elapsed = time.time() - start_time
            fps_processing = frame_number/elapsed if elapsed > 0 else 0
            print(f"Processed {frame_number}/{frame_count} frames ({frame_number/frame_count*100:.1f}%) - {fps_processing:.1f} FPS")
        if frame_number % 2 == 0: # Skip every other frame for faster processing
            continue
        
        # Run YOLO tracking instead of just detection
        # track() maintains object IDs across frames
        results = model.track(frame, conf=confidence, persist=True, tracker="/home/jiaruili/Documents/github/uav-attacks/utils/bytetrack.yaml")
        
        # Check if tracking was successful
        if results[0].boxes.id is not None:
            # Get tracked object IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Get bounding boxes
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Get class information
            classes = results[0].boxes.cls.cpu().numpy()
            class_names = results[0].names
            
            # Store tracking data for each object
            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                box = boxes[i]
                class_id = int(classes[i])
                class_name = class_names[class_id]
                
                # Calculate center point of the box
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # Store the data
                if track_id not in tracking_data:
                    tracking_data[track_id] = {
                        'class': class_name,
                        'frames': [],
                        'positions': [],
                    }
                
                tracking_data[track_id]['frames'].append(frame_number)
                tracking_data[track_id]['positions'].append((center_x, center_y))
        
        # Get the annotated frame with tracking visualization
        annotated_frame = results[0].plot()
        
        # Write the frame to output video if specified
        if writer:
            writer.write(annotated_frame)
    
    # Release resources
    video.release()
    if writer:
        writer.release()
    
    # Print tracking summary
    print("\nTracking Summary:")
    print(f"Total tracked objects: {len(tracking_data)}")
    for track_id, data in tracking_data.items():
        print(f"ID {track_id} ({data['class']}): Appeared in {len(data['frames'])} frames")
    
    # Save tracking data if output directory is provided
    if output_dir:
        import json
        # Convert numpy values to native Python types for JSON serialization
        for track_id in tracking_data:
            tracking_data[track_id]['positions'] = [
                (float(x), float(y)) for x, y in tracking_data[track_id]['positions']
            ]
        
        tracking_json_path = os.path.join(output_dir, f"{video_name}_tracking_data.json")
        with open(tracking_json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Tracking data saved to {tracking_json_path}")
    
    print(f"Finished processing {frame_number} frames in {time.time() - start_time:.2f} seconds")
    if writer:
        print(f"Output saved to {output_path}")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Track objects in a video using YOLO")
    parser.add_argument("video_path", help="Path to the input .mov file")
    parser.add_argument("--o", "--output", dest="output_dir", help="Directory to save the output video (optional)")
    parser.add_argument("--c", "--confidence", dest="confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Process the video
    process_video(args.video_path, args.output_dir, args.confidence)