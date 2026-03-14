from ultralytics import YOLO
import numpy as np


class Model:
    # List of supported model names
    SUPPORTED_MODELS = [
        "yolov8m",
        "yolov8s",
        "yolov8l",
        "yolov7",
        "yolov6",
        "yolo11m"
    ]  # TODO: Add more models

    def __init__(self):
        self._model = None

    def load_model(self, model_version: str):
        """
        Load the YOLO model based on the specified version.

        Args:
            model_version (str): The model version to load.

        Returns:
            YOLO model object if the version is supported, otherwise None.
        """
        if model_version not in Model.SUPPORTED_MODELS:
            raise ValueError(
                f"Model version '{model_version}' is not supported. Available models: {Model.SUPPORTED_MODELS}"
            )

        try:
            model_path = f"{model_version}.pt"  # Assuming model file is named according to the version
            self._model = YOLO(model_path)
            print(f"Loaded {model_version} model.")
        except Exception as e:
            print(f"Error loading model {model_version}: {e}")
            return None

    def predict(self, image: np.ndarray, classes: list = None):
        """
        Perform object detection on the input image.

        Args:
            image (ndarray): The input image for detection.
            classes (list): Optional list of classes to detect.

        Returns:
            list: Detection results.
        """
        if self._model:
            results = self._model.predict(image, classes=classes)
            return results
        else:
            print("Model is not loaded.")
            return None

    def track(self, image: np.ndarray, persist: bool = True):
        """
        Perform multi-object tracking on the input image.

        Args:
            image (ndarray): The input image for tracking.
            persist (bool): Optional persist setting.

        Returns:
            list: Tracking results.
        """
        if self._model:
            results = self._model.track(image, persist=persist, classes=[0], conf=0.25)
            return results
        else:
            print("Model is not loaded.")
            return None

    def calculate_iou(self, boxA, boxB):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            boxA (tuple): Coordinates of the first box (x1, y1, x2, y2).
            boxB (tuple): Coordinates of the second box (x1, y1, x2, y2).

        Returns:
            float: The IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def calculate_accuracy(self, detections, ground_truths, threshold=0.5):
        """
        Calculate the detection accuracy based on IoU threshold.

        Args:
            detections (list): List of detected bounding boxes.
            ground_truths (list): List of ground truth bounding boxes.
            threshold (float): IoU threshold to count as a correct detection.

        Returns:
            float: Accuracy as the ratio of correct detections to total ground truths.
        """
        correct_detections = 0

        for gt in ground_truths:
            for detection in detections:
                iou = self.calculate_iou(detection, gt)
                if iou >= threshold:
                    correct_detections += 1
                    break

        accuracy = correct_detections / len(ground_truths) if ground_truths else 0
        return accuracy
