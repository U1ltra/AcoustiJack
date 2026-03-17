import os
import cv2
import shutil
import tempfile
import subprocess
import numpy as np
from ultralytics import YOLO
from utils.sort import Sort
from utils.eval.helper import get_iou, convert_bbox
from utils.ecc import ECC
from dataclasses import dataclass

import torch
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from onboard.DaSiamRPN.code.net import SiamRPNvot, SiamRPNBIG
from onboard.DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track

from onboard.UCMCTrack.tracker.ucmc import UCMCTrack
from onboard.UCMCTrack.detector.mapper import Mapper
from onboard.UCMCTrack.demo import Detector

@dataclass
class TrackerConfig:
    tracker_type: str = "SORT" # "SORT", "SiamRPN", "KCF", "DaSiamRPN"
    track_conf_threshold: float = 0.3
    # SORT params
    yolo_model: str = "yolov8x.pt"
    det_conf_threshold: float = 0.3
    verbose: bool = False
    # SiamRPN params
    siamrpn_model: str = "/root/AcoustiJack/onboard/models/siamrpn_alex_dwxcorr/model.pth"
    siamrpn_config: str = "/root/AcoustiJack/onboard/models/siamrpn_alex_dwxcorr/config.yaml"
    # KCF params
    kcf_executable: str = "./kcf-master/build/kcf_vot"
    # DaSiamRPN params
    dasiamrpn_model: str = "/root/AcoustiJack/onboard/models/SiamRPNBIG.model"
    # UCMCTrack params
    wx: float = 5.0 # motion uncertainty in x-direction
    wy: float = 5.0 # motion uncertainty in y-direction
    vmax: float = 10.0 # maximum velocity
    a: float = 100.0 # assignment threshold
    cdt: float = 100.0 # deletion time
    high_score: float = 0.5 # high score threshold
    conf_thresh: float = 0.01 # confidence threshold for detection
    fps: float = 30.0 # frames per second

class OnboardTracker:
    def __init__(self, tracker_config: TrackerConfig):
        self.tracker_config = tracker_config

        if tracker_config.tracker_type == "SORT":
            self.tracker_type = "SORT"
            self.track_iou_threshold = tracker_config.track_conf_threshold
            self.det_conf_threshold = tracker_config.det_conf_threshold
            
            self._model = YOLO(tracker_config.yolo_model, verbose=tracker_config.verbose)
            self._tracker = Sort(max_age=3, min_hits=1, iou_threshold=self.track_iou_threshold)
            # print(f"YOLO.names: {self._model.names}")
        elif tracker_config.tracker_type == "SiamRPN":
            self.tracker_type = "SiamRPN"
            self.track_conf_threshold = tracker_config.track_conf_threshold
            
            self.cfg = cfg
            cfg.merge_from_file(tracker_config.siamrpn_config)
            cfg.CUDA = torch.cuda.is_available()
            device = torch.device("cuda" if cfg.CUDA else "cpu")
            
            self._model = ModelBuilder()
            self._model.load_state_dict(torch.load(tracker_config.siamrpn_model))
            self._model.eval().to(device)
            self._tracker = build_tracker(self._model)
            self._last_frame = None
            self._affine_matrix = [ # initial affine matrix as identity
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        elif tracker_config.tracker_type == "KCF":
            self.tracker_type = "KCF"
            self.track_conf_threshold = tracker_config.track_conf_threshold
            
            if not os.path.exists(tracker_config.kcf_executable):
                raise FileNotFoundError(f"KCF executable not found: {tracker_config.kcf_executable}")
            self._tracker = KCFTracker(tracker_config.kcf_executable)
        elif tracker_config.tracker_type == "DaSiamRPN":
            self.tracker_type = "DaSiamRPN"
            self.track_conf_threshold = tracker_config.track_conf_threshold
            
            # load net
            self._model = SiamRPNBIG()
            self._model.load_state_dict(torch.load(tracker_config.dasiamrpn_model))
            self._model.eval().cuda()
        elif tracker_config.tracker_type == "UCMCTrack":
            self.tracker_type = "UCMCTrack"
            self.det_conf_threshold = tracker_config.conf_thresh
            
            self._detector = Detector()
            self._tracker = UCMCTrack(
                a1=tracker_config.a,
                a2=tracker_config.a,
                wx=tracker_config.wx,
                wy=tracker_config.wy,
                vmax=tracker_config.vmax,
                max_age=tracker_config.cdt,
                fps=tracker_config.fps,
                dataset=None,
                high_score=tracker_config.high_score,
                use_cmc=False,
                detector=self._detector
            )
            self._last_frame = None
            self._affine_matrix = [ # initial affine matrix as identity
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
            self._affine_matrix = np.array(self._affine_matrix, dtype=np.float32)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_config.tracker_type}")
            
        self.init_vic_id, self.init_atk_id = None, None
        self.has_init = False

    def init_tracker(self, frame, init_vic_box, init_atk_box, cam_params=None):
        if self.tracker_type == "SORT":
            _, _, res, _ = self.track(frame, vic_gt_box=init_vic_box, atk_gt_box=init_atk_box)
            self.init_vic_id = self._tracker.find_best_matching_tracker(init_vic_box)
            self.init_atk_id = self._tracker.find_best_matching_tracker(init_atk_box)
            if self.init_vic_id == -1 or self.init_atk_id == -1:
                print("self.init_vic_id = ", self.init_vic_id)
                print("self.init_atk_id = ", self.init_atk_id)
                return
                raise ValueError("Failed to initialize tracker with given boxes.")
            self._tracker.kept_ids.append(self.init_vic_id)
            self._tracker.kept_ids.append(self.init_atk_id)
            self.has_init = True
            
        elif self.tracker_type == "SiamRPN":
            init_vic_box_xywh = convert_bbox(init_vic_box, format1="x1y1x2y2", format2="x1y1wh")
            init_atk_box_xywh = convert_bbox(init_atk_box, format1="x1y1x2y2", format2="x1y1wh")
            self._tracker.init(frame, init_vic_box_xywh)
            self.has_init = True
        elif self.tracker_type == "KCF":
            init_vic_box_xywh = convert_bbox(init_vic_box, format1="x1y1x2y2", format2="x1y1wh")
            init_atk_box_xywh = convert_bbox(init_atk_box, format1="x1y1x2y2", format2="x1y1wh")
            self._tracker.init(frame, init_vic_box_xywh)
            self.has_init = True
        elif self.tracker_type == "DaSiamRPN":
            init_vic_box_xywh = convert_bbox(init_vic_box, format1="x1y1x2y2", format2="xywh")
            cx, cy, w, h = init_vic_box_xywh
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            self._tracker = SiamRPN_init(frame, target_pos, target_sz, self._model)
            self.has_init = True
        elif self.tracker_type == "UCMCTrack":
            assert cam_params is not None, "Camera parameters must be provided for UCMCTrack."
            self._detector.load(cam_params)
            self.class_list = [0, 2]  # person, car
            self.frame_id = 0
            _, _, res, _ = self.track(frame, vic_gt_box=init_vic_box, atk_gt_box=init_atk_box)
            _, _, res, _ = self.track(frame, vic_gt_box=init_vic_box, atk_gt_box=init_atk_box) # UCMCTrack requires two hits to initialize a tracklet.

            # find the vic and atk IDs based on track results IoU with init boxes
            tracklets = [r[:4] for r in res]
            vic_boxes = [init_vic_box for _ in range(len(tracklets))]
            atk_boxes = [init_atk_box for _ in range(len(tracklets))]
            ious_vic = get_iou(np.array(tracklets), np.array(vic_boxes))
            ious_atk = get_iou(np.array(tracklets), np.array(atk_boxes))
            self.init_vic_id = res[np.argmax(ious_vic)][5] if len(ious_vic) > 0 else -1
            self.init_atk_id = res[np.argmax(ious_atk)][5] if len(ious_atk) > 0 else -1
            print(f"UCMCTrack init_vic_id: {self.init_vic_id}, init_atk_id: {self.init_atk_id}")
            if self.init_vic_id == -1 or self.init_atk_id == -1:
                raise ValueError("Failed to initialize tracker with given boxes.")
            self.has_init = True
            

    def track(self, frame, vic_gt_box=None, atk_gt_box=None):
        if self.tracker_type == "SORT":
            results = self._model(frame)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0]) # TODO: double the box?
                conf = float(box.conf[0])
                enlarged_box = convert_bbox([x1, y1, x2, y2], format1="x1y1x2y2", format2="x1y1x2y2", wscale=1.5, hscale=1.5)
                x1, y1, x2, y2 = map(float, enlarged_box)
                cls_id = int(box.cls[0])
                if conf >= self.det_conf_threshold and cls_id in [0, 2]: # only person and car
                # if conf >= self.det_conf_threshold:
                    detections.append([x1, y1, x2, y2, conf])
            
            # TODO: Fix yolo detection failure in sequential cars. Finetune can solve this problem.
            detections = []
            if vic_gt_box is not None:
                detections.append([*vic_gt_box, 1.0])
            if atk_gt_box is not None:
                detections.append([*atk_gt_box, 1.0])
                
            if len(detections) == 0:
                detections = np.empty((0, 5))
            else:
                detections = np.array(detections)
            tracked_results, pred_trks, matched = self._tracker.update(detections)
            return detections, pred_trks, tracked_results, matched
        elif self.tracker_type == "SiamRPN":
            # if self._last_frame is not None:
            #     # Use ECC to align the current frame with the last frame
            #     warp_matrix, aligned_frame = ECC(
            #         self._last_frame, frame, warp_mode=cv2.MOTION_AFFINE, max_iter=5, eps=1e-5, scale=0.1, align=False
            #     )
            #     self._affine_matrix = warp_matrix
            #     pass
            # self._last_frame = frame.copy()
            # center_pos = self._tracker.center_pos
            # x_new = (self._affine_matrix[0][0] * center_pos[0] + self._affine_matrix[0][1] * center_pos[1] + self._affine_matrix[0][2])
            # y_new = (self._affine_matrix[1][0] * center_pos[0] + self._affine_matrix[1][1] * center_pos[1] + self._affine_matrix[1][2])
            # self._tracker.center_pos = np.array([x_new, y_new])
            
            tracked_results = self._tracker.track(frame)
            bbox = list(map(float, convert_bbox(tracked_results['bbox'], format1="x1y1wh", format2="x1y1x2y2")))
            pred_trks = [bbox + [tracked_results['best_score'], tracked_results['best_pscore']]]
            return None, pred_trks, tracked_results, None
        elif self.tracker_type == "KCF":
            tracked_results = self._tracker.track(frame)
            if tracked_results is None:
                raise RuntimeError("KCF tracking failed, tracker not initialized or image not valid.")
            return None, [tracked_results], tracked_results, None
        elif self.tracker_type == "DaSiamRPN":
            tracked_results = SiamRPN_track(self._tracker, frame)  # track
            x, y = tracked_results['target_pos']
            w, h = tracked_results['target_sz']
            bbox = convert_bbox([x, y, w, h], format1="xywh", format2="x1y1x2y2")
            self._tracker = tracked_results
            return None, [bbox + [tracked_results['score']]], tracked_results, None
        elif self.tracker_type == "UCMCTrack":
            # if self._last_frame is not None:
            #     # Use ECC to align the current frame with the last frame
            #     warp_matrix, aligned_frame = ECC(
            #         self._last_frame, frame, warp_mode=cv2.MOTION_AFFINE, max_iter=10, eps=1e-5, scale=0.1, align=False
            #     )
            #     self._affine_matrix = warp_matrix
            #     pass
            # self._last_frame = frame.copy()
            detections = self._detector.get_dets(frame, self.det_conf_threshold, self.class_list)
            
            # TODO: Fix yolo detection failure in sequential cars. Finetune can solve this problem.
            detections = []
            if vic_gt_box is not None:
                detections.append(self._detector.box_to_det(vic_gt_box, detections))
            if atk_gt_box is not None:
                detections.append(self._detector.box_to_det(atk_gt_box, detections))
                
            det_results = []
            for det in detections:
                x1, y1 = det.bb_left, det.bb_top
                x2, y2 = det.bb_left + det.bb_width, det.bb_top + det.bb_height
                conf, obj_id = det.conf, det.track_id
                det_results.append([x1, y1, x2, y2, conf])
            self._tracker.update(detections, self.frame_id, self._affine_matrix)
            self.frame_id += 1
            tracked_results = []
            pred_trks = []
            for trk in detections:
                x1, y1 = trk.bb_left, trk.bb_top
                x2, y2 = trk.bb_left + trk.bb_width, trk.bb_top + trk.bb_height
                # detections not attached to a tracklet will have track_id = 0
                # otherwise, track_id starts from 1
                conf, obj_id = trk.conf, trk.track_id 
                tracked_results.append([x1, y1, x2, y2, conf, obj_id])
                pred_trks.append([x1, y1, x2, y2, obj_id, conf])

            return det_results, pred_trks, tracked_results, None

    def get_vic_atk_boxes(self, tracked_results):
        """Get the bounding boxes of the victim and attacker from the tracked results."""
        vic_box = [None, None, None, None]
        atk_box = [None, None, None, None]
        if self.tracker_type == "SORT":
            for obj in tracked_results:
                if int(obj[4]) == self.init_vic_id:
                    vic_box = obj[:4]
                    vic_box = list(map(float, convert_bbox(vic_box, format1="x1y1x2y2", format2="x1y1x2y2", wscale=0.8, hscale=0.8)))
                elif int(obj[4]) == self.init_atk_id:
                    atk_box = obj[:4]
                    atk_box = list(map(float, convert_bbox(atk_box, format1="x1y1x2y2", format2="x1y1x2y2", wscale=0.8, hscale=0.8)))
        elif self.tracker_type == "SiamRPN":
            if tracked_results["best_pscore"] >= self.track_conf_threshold:
                vic_box = list(map(float, convert_bbox(tracked_results['bbox'], format1="x1y1wh", format2="x1y1x2y2")))
        elif self.tracker_type == "KCF":
            vic_box = list(map(float, convert_bbox(tracked_results[:4], format1="x1y1wh", format2="x1y1x2y2")))
            atk_box = [None, None, None, None]
        elif self.tracker_type == "DaSiamRPN":
            x, y = tracked_results['target_pos']
            w, h = tracked_results['target_sz']
            vic_box = list(map(float, convert_bbox([x, y, w, h], format1="xywh", format2="x1y1x2y2")))
            atk_box = [None, None, None, None]
        elif self.tracker_type == "UCMCTrack":
            for obj in tracked_results:
                if int(obj[5]) == self.init_vic_id:
                    vic_box = obj[:4]
                elif int(obj[5]) == self.init_atk_id:
                    atk_box = obj[:4]

        return vic_box, atk_box
    
    def get_vic_atk_det_boxes(self, detections, gz_det, tracked_results):
        """
        Use the Gazebo detections to locate the victim and attacker boxes from the detections or tracked results.
        return [x1, y1, x2, y2, conf]
        """
        if self.tracker_type == "SORT":
            gz_det_vic, gz_det_atk = gz_det
            tracklets = [d[:4] for d in detections]
            if len(tracklets) == 0:
                return [None, None, None, None, 0], [None, None, None, None, 0]
            vic_boxes = [gz_det_vic for _ in range(len(tracklets))]
            atk_boxes = [gz_det_atk for _ in range(len(tracklets))]
            ious_vic = get_iou(np.array(tracklets), np.array(vic_boxes))
            ious_atk = get_iou(np.array(tracklets), np.array(atk_boxes))
            vic_box_w_conf = detections[np.argmax(ious_vic)] if len(ious_vic) > 0.05 else [None, None, None, None, 0]
            atk_box_w_conf = detections[np.argmax(ious_atk)] if len(ious_atk) > 0.05 else [None, None, None, None, 0]
        elif self.tracker_type == "SiamRPN":
            # For SiamRPN, we need to use the tracked_results to get the boxes
            vic_box = list(map(float, convert_bbox(tracked_results['bbox'], format1="x1y1wh", format2="x1y1x2y2")))
            vic_box_w_conf = vic_box + [tracked_results['best_pscore']]
            atk_box_w_conf = [None, None, None, None, 0]
        elif self.tracker_type == "KCF":
            # For KCF, we use the tracked_results directly
            vic_box = list(map(float, convert_bbox(tracked_results[:4], format1="x1y1wh", format2="x1y1x2y2")))
            vic_box_w_conf = vic_box + [tracked_results[4]]
            atk_box_w_conf = [None, None, None, None, 0]
        elif self.tracker_type == "DaSiamRPN":
            # For DaSiamRPN, we use the tracked_results directly
            x, y = tracked_results['target_pos']
            w, h = tracked_results['target_sz']
            vic_box = list(map(float, convert_bbox([x, y, w, h], format1="xywh", format2="x1y1x2y2")))
            vic_box_w_conf = vic_box + [tracked_results['score']]
            atk_box_w_conf = [None, None, None, None, 0]
        elif self.tracker_type == "UCMCTrack":
            gz_det_vic, gz_det_atk = gz_det
            tracklets = [d[:4] for d in detections]
            if len(tracklets) == 0:
                return [None, None, None, None, 0], [None, None, None, None, 0]
            vic_boxes = [gz_det_vic for _ in range(len(tracklets))]
            atk_boxes = [gz_det_atk for _ in range(len(tracklets))]
            ious_vic = get_iou(np.array(tracklets), np.array(vic_boxes))
            ious_atk = get_iou(np.array(tracklets), np.array(atk_boxes))
            vic_box_w_conf = detections[np.argmax(ious_vic)] if len(ious_vic) > 0.05 else [None, None, None, None, 0]
            atk_box_w_conf = detections[np.argmax(ious_atk)] if len(ious_atk) > 0.05 else [None, None, None, None, 0]

        return vic_box_w_conf, atk_box_w_conf
    
    def find_vic_atk_trackers(self):
        if not self.has_init:
            return None, None
            raise ValueError("Tracker has not been initialized yet.")
        vic_tracker = self._tracker.find_tracker_by_id(self.init_vic_id)
        atk_tracker = self._tracker.find_tracker_by_id(self.init_atk_id)
        return vic_tracker, atk_tracker

class KCFTracker:
    def __init__(self, executable_path):
        self.process = subprocess.Popen(
            [executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        # Wait for READY status
        self._read_status()
        self.initialized = False
        self.temp_dir = tempfile.mkdtemp()
        self.center = None
        self.size = None
        self.response = None

        self.init_image = None
        self.init_bbox = None

    def init(self, image, bbox):
        """Initialize tracker with image and bounding box
        Args:
            image: cv2 image (numpy array)
            bbox: tuple (x1, y1, width, height)
        Returns:
            tuple: (x1, y1, width, height) of initialized bbox
        """
        # Save image to temporary file
        temp_path = os.path.join(self.temp_dir, "init.jpg")
        cv2.imwrite(temp_path, image)
        
        # Send INIT command
        x1, y1, w, h = bbox
        command = f"INIT {temp_path} {x1} {y1} {w} {h}\n"
        self.process.stdin.write(command)
        self.process.stdin.flush()
        
        # Read response
        bbox_result = self._read_bbox()
        status = self._read_status()
        
        if status == "INITIALIZED":
            self.initialized = True
            self.center = [x1 + w / 2, y1 + h / 2]
            self.size = (w, h)
            self.response = bbox_result[-1]  # response value
            # if self.init_image is None or self.init_bbox is None:
            #     self.init_image = image.copy()
            #     self.init_bbox = bbox
            return bbox_result
        else:
            raise RuntimeError(f"Initialization failed: {status}")

    
    def track(self, image):
        """Track object in new image
        Args:
            image: cv2 image (numpy array)
        Returns:
            tuple: (x1, y1, width, height) of tracked bbox
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized")
            
        # Save image to temporary file
        temp_path = os.path.join(self.temp_dir, "track.jpg")
        cv2.imwrite(temp_path, image)
        
        # Send TRACK command
        command = f"TRACK {temp_path}\n"
        self.process.stdin.write(command)
        self.process.stdin.flush()
        
        # Read response
        bbox_result = self._read_bbox()
        status = self._read_status()
        
        if status == "TRACKED":
            # Update center and size
            x1, y1, w, h, resp = bbox_result
            self.center = [x1 + w / 2, y1 + h / 2]
            self.size = (w, h)
            self.response = resp
            return bbox_result
        else:
            raise RuntimeError(f"Tracking failed: {status}")
    
    def _read_bbox(self):
        """Read BBOX response from process"""
        line = self.process.stdout.readline().strip()
        if line.startswith("BBOX"):
            parts = line.split()
            # x1, y1, width, height, resp_value
            return (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
        else:
            raise RuntimeError(f"Expected BBOX, got: {line}")
    
    def _read_status(self):
        """Read STATUS response from process"""
        line = self.process.stdout.readline().strip()
        if line.startswith("STATUS"):
            return line.split(None, 1)[1]
        else:
            raise RuntimeError(f"Expected STATUS, got: {line}")
    
    def close(self):
        """Close the tracker process"""
        if self.process.poll() is None:
            self.process.stdin.write("QUIT\n")
            self.process.stdin.flush()
            self.process.wait()
        
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        self.close()