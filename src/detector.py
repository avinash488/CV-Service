from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)  # auto-downloads on first run

    def detect(self, image: np.ndarray) -> list:
        results = self.model(image, verbose=False)
        detections = []

        for box in results[0].boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

        return detections