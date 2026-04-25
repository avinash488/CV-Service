import cv2
import numpy as np
import pytest
from src.detector import ObjectDetector
from src.scene import SceneAnalyzer

# ── Unit tests ────────────────────────────────────────────
class TestObjectDetector:
    def setup_method(self):
        self.detector = ObjectDetector()
        self.sample = cv2.imread("tests/.fixtures/sample.jpg")

    def test_model_loads(self):
        assert self.detector.model is not None

    def test_detect_returns_list(self):
        result = self.detector.detect(self.sample)
        assert isinstance(result, list)

    def test_detection_schema(self):
        result = self.detector.detect(self.sample)
        if result:
            d = result[0]
            assert "class" in d
            assert "confidence" in d
            assert "bbox" in d
            assert len(d["bbox"]) == 4
            assert 0.0 <= d["confidence"] <= 1.0

class TestSceneAnalyzer:
    def setup_method(self):
        self.analyzer = SceneAnalyzer()
        self.sample = cv2.imread("tests/.fixtures/sample.jpg")

    def test_analyze_returns_dict(self):
        result = self.analyzer.analyze(self.sample)
        assert isinstance(result, dict)

    def test_analyze_required_keys(self):
        result = self.analyzer.analyze(self.sample)
        assert "total_objects" in result
        assert "class_counts" in result
        assert "crowded" in result
        assert "occlusion_detected" in result
        assert "detections" in result

    def test_zone_assignment(self):
        result = self.analyzer.analyze(self.sample)
        for d in result["detections"]:
            assert d["zone"] in ["left", "center", "right"]

    def test_crowded_flag_type(self):
        result = self.analyzer.analyze(self.sample)
        assert isinstance(result["crowded"], bool)