import os
import subprocess

class KCFTracker:
    def __init__(self, kcf_path="/home/jiaruili/Documents/github/trackingAtk/kcf-master"):
        self.kcf_path = kcf_path
        self.executable_path = os.path.join(kcf_path, "build", "kcf_vot")
        self.region_file = "region.txt"
        self.images_file = "images.txt"
        self.output_file = "output.txt"
    
    def prepare_region_file(self, x1, y1, width, height):
        with open(self.region_file, 'w') as f:
            f.write(f"{x1},{y1},{width},{height}\n")
        # print(f"Created {self.region_file}: {x},{y},{width},{height}")
    
    def prepare_images_file(self, image_list=None):
        if image_list:
            images = image_list
        else:
            raise ValueError("Either image_folder or image_list must be provided")
        # Write absolute paths to images.txt
        with open(self.images_file, 'w') as f:
            for img_path in images:
                f.write(f"{os.path.abspath(img_path)}\n")
        # print(f"Created {self.images_file} with {len(images)} images")
        return images

    def run_tracker(self):
        # print(f"Running KCF tracker: {self.executable_path}")
        try:
            # Run the executable
            result = subprocess.run([self.executable_path], 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            
            # print("Tracker output:")
            # print(result.stdout)
            
            if result.stderr:
                print("Tracker errors:")
                print(result.stderr)
            
            # Check if output file was created
            if os.path.exists(self.output_file):
                # print(f"Tracking results saved to: {self.output_file}")
                return self.read_results()
            else:
                print("Warning: No output file generated")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running tracker: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None
        
    def read_results(self):
        if not os.path.exists(self.output_file):
            return None
        
        results = []
        with open(self.output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse bounding box coordinates
                    coords = [float(x) for x in line.split(',')]
                    if len(coords) == 4:
                        results.append(tuple(coords))
        
        # print(f"Read {len(results)} tracking results")
        return results