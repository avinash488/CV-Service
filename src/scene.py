import numpy as np
from src.detector import ObjectDetector

class SceneAnalyzer:
    def __init__(self):
        self.detector = ObjectDetector()

    def get_zone(self, bbox: list, frame_width: int) -> str:
        x_center = (bbox[0] + bbox[2]) / 2
        third = frame_width / 3
        if x_center < third:
            return "left"
        elif x_center < 2 * third:
            return "center"
        return "right"

    def analyze(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)

        # Count objects per class
        class_counts = {}
        for d in detections:
            cls = d["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Assign spatial zones
        for d in detections:
            d["zone"] = self.get_zone(d["bbox"], w)

        # Flag crowding (more than 5 objects in frame)
        crowded = sum(class_counts.values()) > 5

        # Flag occlusion (any confidence below 0.5)
        occlusion = any(d["confidence"] < 0.5 for d in detections)

        return {
            "frame_size": {"width": w, "height": h},
            "total_objects": len(detections),
            "class_counts": class_counts,
            "crowded": crowded,
            "occlusion_detected": occlusion,
            "detections": detections
        }