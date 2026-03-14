import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess

import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from collections import defaultdict

from attack.gimbal_atk import SiameseTracker, convert_bbox, get_color
from utils.eval.helper import *
from utils.sort import Sort
from utils.kcf_resp import overlay_analysis, extract_tracking_context, load_raw_response, response_at_coordinates, max_response_in_box

class KCFTracker:
    def __init__(self, kcf_path="/home/jiaruili/Documents/github/trackingAtk/kcf-master"):
        self.kcf_path = kcf_path
        self.executable_path = os.path.join(kcf_path, "build", "kcf_vot")
        self.region_file = "region.txt"
        self.images_file = "images.txt"
        self.output_file = "output.txt"
    
    def prepare_region_file(self, x, y, width, height):
        with open(self.region_file, 'w') as f:
            f.write(f"{x},{y},{width},{height}\n")
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


class SiameseTracker:
    """Siamese tracker for the attack simulation"""

    def __init__(self):
        self.tracker = self.load_siamese_tracker(
            "/home/jiaruili/Documents/github/trackingAtk/models/siamrpn_alex_dwxcorr/config.yaml",
            "/home/jiaruili/Documents/github/trackingAtk/models/siamrpn_alex_dwxcorr/model.pth",
        )  # siamrpn with alexnet backbone

    def load_siamese_tracker(self, config_file, snapshot_file, use_cuda=True):
        # Load config
        cfg.merge_from_file(config_file)
        cfg.CUDA = torch.cuda.is_available() and use_cuda
        device = torch.device("cuda" if cfg.CUDA else "cpu")

        # Load model weights
        model = ModelBuilder()
        model.load_state_dict(
            torch.load(snapshot_file, map_location=lambda storage, loc: storage.cpu())
        )
        model.eval().to(device)
        tracker = build_tracker(model)

        return tracker

    def init(self, image, init_bbox):
        self.tracker.init(image, init_bbox)

    def track(self, image, show_heatmap=False, show_analysis=False):
        outputs = self.tracker.track_with_insight(image, show_heatmap=show_heatmap, show_analysis=show_analysis, save_path="heatmap.png")
        bbox = list(map(float, outputs["bbox"]))
        score = float(outputs["best_score"])
        pscore = float(outputs["best_penalized_score"])
        heatmap = outputs["heatmap_data"]
        preds_data = outputs["predictions_data"]
        return bbox, score, pscore, heatmap, preds_data


class ExpTrace:
    id2name = {
        "10": "victim",
        "20": "attacker",
    }
    
    trace_files = [
        "default.sdf",
        "social_force.toml",
        "log.txt",
        # main loop
        "main_timestamp_history.npy",
        ## attacker machine
        "atker_machine/20_history.npy",
        "atker_machine/10_history.npy",
        "atker_machine/20_pred_history.npy",
        "atker_machine/10_pred_history.npy",
        "atker_machine/20_world_history.npy",
        "atker_machine/10_world_history.npy",
        "atker_machine/center_displacement_history.npy",
        "atker_machine/displacement_history.npy",
        "atker_machine/displacement_coord_history.npy",
        "atker_machine/id_switch_history.npy",
        "atker_machine/inv_mat_history.npy",
        "atker_machine/proj_matrix_history.npy",
        "atker_machine/world_corners_history.npy",
        ## victim uav
        "vic_vehicle/20_history.npy",
        "vic_vehicle/10_history.npy",
        "vic_vehicle/20_pred_history.npy",
        "vic_vehicle/10_pred_history.npy",
        "vic_vehicle/20_world_history.npy",
        "vic_vehicle/10_world_history.npy",
        "vic_vehicle/id_switch_history.npy",
        "vic_vehicle/error_history.npy",
        # social force
        "distance_history.npy",
        "state_history.npy",
        "timestamp_history.npy",
        # misc
        "camera0_mainloop_pose_history.npy",
    ]

    def __init__(self, exp_path):
        self.exp_path = exp_path
        # load experiment arguments from log/args.txt
        args_path = os.path.join(exp_path, "log", "args.txt")
        self.args = reconstruct_args(args_path)

        self.results = dict()
        self.parse_logs()

    def parse_logs(self):
        self.results["trial_paths"] = []
        self.results["trial_names"] = []
        self.results["trial_switch"] = []
        self.results["trial_switch_frame"] = []
        self.results["trial_max_delta_vic"] = []
        self.results["trial_stability"] = []

        dirs = list_dirs(self.exp_path)#[:35]
        for i in range(len(dirs) - 1):
            # i = i + 50
            path = os.path.join(self.exp_path, f"trial_{i+1:03d}")
            self.results["trial_paths"].append(path)
            self.results["trial_names"].append(f"trial_{i+1:03d}")
            switch, switch_frame, max_delta_vic, stability = parse_log(
                os.path.join(path, "log.txt")
            )
            self.results["trial_switch"].append(switch)
            if not switch:
                switch_frame = 0
            self.results["trial_switch_frame"].append(switch_frame)
            if max_delta_vic is not None:
                self.results["trial_max_delta_vic"].append(max_delta_vic)
            else:
                self.results["trial_max_delta_vic"].append(0)
            if stability is not None:
                self.results["trial_stability"].append(stability)
            else:
                self.results["trial_stability"].append([0, 0, 0, 0])

    def analyze_trial(self):
        self.results["ideal_switch"] = []
        self.results["ideal_switch_score"] = []
        self.results["ideal_high_score_switch"] = []
        self.results["skipped"] = []
        self.results["center_error"] = []
        self.results["final_pred_atk_error"] = []
        self.results["last_frame"] = []
        self.results["shifted_max_responses"] = []
        self.results["shifted_max_responses_penalized"] = []
        self.results["shifted_pred_error"] = []
        self.results["shifted_vic_responses"] = []
        self.results["shifted_atk_responses"] = []
        self.results["shifted_normalized_dist"] = []
        self.results["shifted_normalized_pairs"] = []

        self.results["trial_paths"] = self.results["trial_paths"][:50]
        start_time = time.time()
        for i, path in enumerate(self.results["trial_paths"]):
            print(f"Analyzing trial {i+1}/{len(self.results['trial_paths'])}: {path}")
            try:
                # atk_gt = np.load(os.path.join(path, "atker_machine/20_history.npy"))
                # vic_gt = np.load(os.path.join(path, "atker_machine/10_history.npy"))
                # atk_pred = np.load(os.path.join(path, "atker_machine/20_pred_history.npy"))
                # vic_pred = np.load(os.path.join(path, "atker_machine/10_pred_history.npy"))
                # atk_world = np.load(os.path.join(path, "atker_machine/20_world_history.npy"))
                # vic_world = np.load(os.path.join(path, "atker_machine/10_world_history.npy"))
                # atk_center_displacement = np.load(os.path.join(path, "atker_machine/center_displacement_history.npy"))
                # atk_displacement = np.load(os.path.join(path, "atker_machine/displacement_history.npy"))
                # atk_displacement_coord = np.load(os.path.join(path, "atker_machine/displacement_coord_history.npy"))
                # atk_id_switch = np.load(os.path.join(path, "atker_machine/id_switch_history.npy"))
                vic_id_switch = np.load(os.path.join(path, "vic_vehicle/id_switch_history.npy"))
                main_loop_timestamp = np.load(os.path.join(path, "main_timestamp_history.npy"))
                camera_pose = np.load(os.path.join(path, "camera0_mainloop_pose_history.npy"))

                atk_gt_uav = np.load(os.path.join(path, "vic_vehicle/20_history.npy"))
                vic_gt_uav = np.load(os.path.join(path, "vic_vehicle/10_history.npy"))
                atk_pred_uav = np.load(os.path.join(path, "vic_vehicle/20_pred_history.npy"))
                vic_pred_uav = np.load(os.path.join(path, "vic_vehicle/10_pred_history.npy"))

                # inv_mat_history = np.load(os.path.join(path, "atker_machine/inv_mat_history.npy"))
                # world_corners_history = np.load(os.path.join(path, "atker_machine/world_corners_history.npy"))
                # proj_matrix_history = np.load(os.path.join(path, "atker_machine/proj_matrix_history.npy"))
            except:
                print(f"Error loading data from {path}")
                continue
            
            # check the optimal position
            if len(main_loop_timestamp) == 0:
                continue

            imgs_path = "/home/jiaruili/Documents/github/uav-attacks/utils/eval/imgs"
            imgs_list = []

            cap = cv2.VideoCapture(os.path.join(path, "raw_images.mp4"))
            cap_idx = cv2.VideoCapture(os.path.join(path, "raw_images.mp4"))
            os.makedirs(os.path.join(path, "ideal_switch"), exist_ok=True)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 2
            if video_length + frame_count != len(vic_gt_uav):
                print(f"Video length {video_length} does not match gt length {len(vic_gt_uav)}")
                continue

            init_vic_center = None
            final_atk_center = None
            init_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                vic_box, atk_box = vic_gt_uav[frame_count], atk_gt_uav[frame_count]
                vic_box = convert_bbox(vic_box, "x1y1x2y2", "xywh")
                vic_box = [vic_box[0], vic_box[1], vic_box[2]*2, vic_box[3]*2] # double the size for diff Gazebo models due to size issues
                vic_box = convert_bbox(vic_box, "xywh", "x1y1wh")
                atk_box = convert_bbox(atk_box, "x1y1x2y2", "x1y1wh")

                vic_box = np.append(vic_box, 1)
                atk_box = np.append(atk_box, 1)

                if frame_count == 100:
                    # tracker = SiameseTracker()
                    # tracker.init(frame, vic_box[:4])

                    # last_frame = video_length - 1
                    # cap_idx.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
                    # ret, frame_ = cap_idx.read()
                    # if not ret:
                    #     raise ValueError(f"Could not read frame {frame_count} from {path}")
                    # last_vic_box, last_atk_box = vic_gt_uav[last_frame], atk_gt_uav[last_frame]
                    # last_atk_box = convert_bbox(last_atk_box, "x1y1x2y2", "xywh")
                    # init_vic_box = convert_bbox(vic_box[:4], "x1y1wh", "xywh")
                    # offset_x = init_vic_box[0] - last_atk_box[0]
                    # offset_y = init_vic_box[1] - last_atk_box[1]
                    # # shift the last frame to the initial victim box
                    # shifted_last_frame = shift_image(frame_, offset_x, offset_y)

                    # siamBbox, siamScore, siamHeatmap = tracker.track(shifted_last_frame, show_heatmap=True)
                    # tracker.tracker.overlay_heatmap_on_image(
                    #     shifted_last_frame, siamHeatmap, save_path="overlay_heatmap.png"
                    # )
                    # raise ValueError(f"Tracker did not return results for {path}")
                    # frame = shifted_last_frame.copy()

                    x_coords = np.arange(-421, 421, 20)
                    y_coords = np.arange(-421, 421, 20)

                    # Create all possible (x, y) pairs
                    pairs = []
                    for x in x_coords:
                        for y in y_coords:
                            pairs.append((x, y))

                    shift_rec_img_list = []
                    for i, (x, y) in enumerate(pairs):
                        if i % 100 == 0:
                            print(f"Processing shifted image {i+1}/{len(pairs)}")

                        tracker = SiameseTracker()
                        tracker.init(frame, vic_box[:4])

                        # if i % 100 == 0:
                        #     print(f"Processing shifted image {i+1}/{len(pairs)}")
                        # shift the image
                        shifted_img = shift_image(frame, x, y)
                        # run tracker
                        siamBbox, siamScore, pscore, heatmap, preds_data = tracker.track(shifted_img, show_heatmap=False, show_analysis=False)
                        self.results["shifted_max_responses"].append(siamScore)
                        self.results["shifted_max_responses_penalized"].append(pscore)

                        # process error and responses
                        pred_box = siamBbox
                        pred_box_xywh = convert_bbox(pred_box, "x1y1wh", "xywh")
                        vic_box_xywh = convert_bbox(vic_box[:4], "x1y1wh", "xywh")
                        shifted_vic_box_xywh = np.array(vic_box_xywh) + np.array([x, y, 0, 0])
                        self.results["shifted_pred_error"].append(
                            np.linalg.norm(
                                np.array(pred_box_xywh[:2]) - np.array(shifted_vic_box_xywh[:2])
                            )
                        )

                        # print(pscore)
                        # print(np.max(heatmap["final_heatmap"]))
                        # # score_ = tracker.tracker.get_heatmap_value_at_point(
                        # #     heatmap, pred_box_xywh[0], pred_box_xywh[1]
                        # # )["value"]
                        # score_ = tracker.tracker.get_max_heatmap_value_in_region(
                        #     heatmap, shifted_vic_box_xywh[0], shifted_vic_box_xywh[1], radius=100
                        # )["max_value"]
                        # # score_ should be close to siamScore
                        # assert np.isclose(score_, siamScore, atol=1e-3), f"Score mismatch: {score_} vs {siamScore}"
                        # raise ValueError(f"Score mismatch: {score_} vs {siamScore}")

                        closest_best = tracker.tracker.get_max_score_in_region(
                                preds_data, shifted_vic_box_xywh[0], shifted_vic_box_xywh[1], radius=20
                            )
                        self.results["shifted_vic_responses"].append(
                            closest_best["prediction"]["final_score"]
                        )
                        # shifted_last_atk_box = np.array(last_atk_box) + np.array([x + offset_x, y + offset_y, 0, 0])
                        # self.results["shifted_atk_responses"].append(
                        #     max_response_in_box(
                        #         tc["response_map"], tc, shifted_last_atk_box[0], shifted_last_atk_box[1], box_size=81 # why 81? Just a detailed choice in our cases which is considered a close enough hijacking
                        #     )
                        # )

                        # Mahalanobis-style
                        # norm_x, norm_y = x/(tc["search_bounds"][2] - tc["search_bounds"][0]), y/(tc["search_bounds"][3] - tc["search_bounds"][1])
                        # self.results["shifted_normalized_dist"].append(
                        #     np.linalg.norm(np.array([norm_x, norm_y]))
                        # )
                        # Chebyshev distance
                        self.results["shifted_normalized_dist"].append(
                            shift_distance(
                                x, y, 0, 0,
                                heatmap["search_bounds"][2] - heatmap["search_bounds"][0],
                                heatmap["search_bounds"][3] - heatmap["search_bounds"][1]
                            )
                        )
                        # self.results["shifted_normalized_pairs"].append(
                        #     # (
                        #     #     x/((heatmap["search_bounds"][2] - heatmap["search_bounds"][0]) / 2),
                        #     #     y/((heatmap["search_bounds"][3] - heatmap["search_bounds"][1]) / 2)
                        #     # )
                        #     (x, y)
                        # )

                        # visualize the results
                        if True:
                            cv2.rectangle(
                                shifted_img,
                                (int(pred_box[0]), int(pred_box[1])),
                                (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])),
                                (0, 255, 0), 2
                            )
                            cv2.putText(
                                shifted_img,
                                f"Predicted Box: {pred_box}",
                                (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )
                            shifted_vic_box = convert_bbox(shifted_vic_box_xywh, "xywh", "x1y1wh")
                            cv2.rectangle(
                                shifted_img,
                                (int(shifted_vic_box[0]), int(shifted_vic_box[1])),
                                (int(shifted_vic_box[0] + shifted_vic_box[2]), int(shifted_vic_box[1] + shifted_vic_box[3])),
                                (255, 0, 0), 2
                            )
                            cv2.putText(
                                shifted_img,
                                f"Shifted Victim Box: {shifted_vic_box}",
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )
                            shift_rec_img_list.append(shifted_img)
                            # cv2.imwrite(
                            #     f"frame_{frame_count:04d}.jpg",
                            #     shifted_img
                            # )
                    images_to_video(
                        shift_rec_img_list,
                        "shifted.mp4",
                        fps=10,
                    )

                    # visualize tracking contexts for different shifts
                    search_bounds = heatmap["search_bounds"]
                    vic_box_xywh = convert_bbox(vic_box[:4], "x1y1wh", "xywh")
                    zero_centered_search_bounds = [
                        search_bounds[0] - vic_box_xywh[0],
                        search_bounds[1] - vic_box_xywh[1],
                        search_bounds[2] - vic_box_xywh[0],
                        search_bounds[3] - vic_box_xywh[1]
                    ]

                    target_region = [
                        -vic_box_xywh[2] / 2,
                        -vic_box_xywh[3] / 2,
                        vic_box_xywh[2] / 2,
                        vic_box_xywh[3] / 2
                    ]
                    atk_box_xywh = convert_bbox(atk_box[:4], "x1y1wh", "xywh")
                    atk_region = [
                        -atk_box_xywh[2] / 2,
                        -atk_box_xywh[3] / 2,
                        atk_box_xywh[2] / 2,
                        atk_box_xywh[3] / 2
                    ]
                    vic_to_atk_offset = [ 
                        vic_box_xywh[0] - atk_box_xywh[0],
                        vic_box_xywh[1] - atk_box_xywh[1]
                    ] # v-a because we are shifting the image instead of the objects
                    atk_region = [
                        atk_region[0] + vic_to_atk_offset[0],
                        atk_region[1] + vic_to_atk_offset[1],
                        atk_region[2] + vic_to_atk_offset[0],
                        atk_region[3] + vic_to_atk_offset[1]
                    ]

                    # this work with only one trial
                    # draw_heatmap(
                    #     pairs, self.results["shifted_max_responses"], resolution=100, save_path="siam_shifted_heatmap.png", search_region=zero_centered_search_bounds, target_region=target_region, atk_region=atk_region
                    # )
                    # draw_heatmap(
                    #     pairs, self.results["shifted_max_responses_penalized"], resolution=100, save_path="siam_shifted_penalized_heatmap.png", search_region=zero_centered_search_bounds, target_region=target_region, atk_region=atk_region
                    # )
                    # draw_heatmap(
                    #     pairs, self.results["shifted_pred_error"], resolution=100, save_path="siam_shifted_pred_error_heatmap.png", search_region=zero_centered_search_bounds, target_region=target_region, atk_region=atk_region
                    # )
                    # draw_heatmap(
                    #     pairs, self.results["shifted_vic_responses"], resolution=100, save_path="siam_shifted_vic_responses_heatmap.png", search_region=zero_centered_search_bounds, target_region=target_region, atk_region=atk_region
                    # )

                    # draw_heatmap(
                    #     pairs, self.results["shifted_atk_responses"], resolution=100, save_path="shifted_atk_responses_heatmap.png", search_region=zero_centered_search_bounds, target_region=target_region
                    # )
                    # plot_multiple_lines_with_confidence_area(
                    #     self.results["shifted_normalized_dist"],
                    #     self.results["shifted_max_responses"],
                    #     self.results["shifted_vic_responses"],
                    #     labels=["Shifted Max Responses", "Shifted Victim Responses"],
                    #     use_minmax=True,
                    # )
                    break
                    raise ValueError(f"Tracker did not return results for {path}")

                frame_count += 1
            cap.release()
            cap_idx.release()

            if i % 10 == 0:
                print(f"Processed {i+1}/{len(self.results['trial_paths'])} trials. Time elapsed: {time.time() - start_time:.2f} seconds")
                np.savez(
                    os.path.join(self.exp_path, "log", f"siam_results_{i+1:03d}.npz"),
                    shifted_max_responses=self.results["shifted_max_responses"],
                    shifted_max_responses_penalized=self.results["shifted_max_responses_penalized"],
                    shifted_pred_error=self.results["shifted_pred_error"],
                    shifted_vic_responses=self.results["shifted_vic_responses"],
                    shifted_atk_responses=self.results.get("shifted_atk_responses", []),
                    shifted_normalized_dist=self.results["shifted_normalized_dist"],
                    shifted_normalized_pairs=self.results.get("shifted_normalized_pairs", []),
                )

            continue

        # save the results
        np.savez(
            os.path.join(self.exp_path, "log", f"siam_results_{i+1:03d}.npz"),
            shifted_max_responses=self.results["shifted_max_responses"],
            shifted_max_responses_penalized=self.results["shifted_max_responses_penalized"],
            shifted_pred_error=self.results["shifted_pred_error"],
            shifted_vic_responses=self.results["shifted_vic_responses"],
            shifted_atk_responses=self.results.get("shifted_atk_responses", []),
            shifted_normalized_dist=self.results["shifted_normalized_dist"],
            shifted_normalized_pairs=self.results.get("shifted_normalized_pairs", []),
        )
        plot_multiple_lines_with_confidence_area(
            self.results["shifted_normalized_dist"],
            self.results["shifted_max_responses_penalized"],
            self.results["shifted_vic_responses"],
            labels=["Max Resp", "Victim Resp"],
            use_minmax=False,
            smooth_method="gaussian",
            smooth_window=10,
            fig_name=f"shifted_responses_plot",
        )
        
        print(f"Analysis completed in {time.time() - start_time:.2f} seconds")

    def __str__(self):
        # Print the results in a readable format
        result = ""
        # for i in range(len(self.results["trial_paths"])):
        #     result += f"Trial {self.results['trial_names'][i]}: \n"
        #     result += f"  Path: {self.results['trial_paths'][i]} \n"
        #     result += f"  Switch: {self.results['trial_switch'][i]} \n"
        #     result += f"  Max delta victim speed: {self.results['trial_max_delta_vic'][i]} \n"
        #     result += "\n"
        result += (
            f"Total number of attacker switches: {sum(self.results['trial_switch'])} \n"
        )
        result += f"Percentage of attacker switches: {sum(self.results['trial_switch'])/len(self.results['trial_switch'])} \n"
        # average switch frame
        avg_switch_frame = np.mean(self.results["trial_switch_frame"])
        result += f"Average switch frame: {avg_switch_frame} \n"
        # result += f"Average max delta victim speed: {np.mean(self.results['trial_max_delta_vic'])} \n"
        # result += f"Average stability: {np.mean(self.results['trial_stability'], axis=0)} \n"
        result += f"Average ideal switch: {np.sum(self.results['ideal_switch'])/len(self.results["trial_paths"])} \n"
        # result += f"Average ideal switch score: {np.mean(self.results['ideal_switch_score'])} \n"
        # result += f"Average ideal high score switch: {np.mean(self.results['ideal_high_score_switch'])} \n"
        result += f"Skipped trials: {len(self.results['skipped'])} \n"
        result += f"Average center error: {np.mean(self.results['center_error']) if self.results['center_error'] else 0} \n"
        result += f"Average final predicted attacker error: {np.mean(self.results['final_pred_atk_error']) if self.results['final_pred_atk_error'] else 0} \n"

        return result


if __name__ == "__main__":
    result_path = "~/Documents/exp/localityAtk"
    # result_path = "~/Documents/exp/gzFolMe2"
    result_path = os.path.expanduser(result_path)

    exp_names = [
        # "20250507-220045-followme_sot",
        # "20250508-002658-followme_sot_gimbal3",
        # "20250515-170733-locality_vuln_bothStand", # gimbal move start from frame 50, end before 100
        # "20250515-204458-locality_vuln_diffApp_static",
        "20250531-151129-smooth_diffApp", # gimbal move start from frame 100, end before 200
        # "20250531-172619-smooth_sameApp",
    ]

    traces = []
    for exp_name in exp_names:
        exp_path = os.path.join(result_path, exp_name)
        exp_trace = ExpTrace(exp_path)
        exp_trace.analyze_trial()
        traces.append(exp_trace)
        print(exp_name)
        print(exp_trace)
        print("\n")


   
    