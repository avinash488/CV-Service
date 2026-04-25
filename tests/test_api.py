import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "yolov8n"

def test_detect_endpoint():
    with open("tests/.fixtures/sample.jpg", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "total_objects" in data
    assert "detections" in data

def test_analyze_endpoint():
    with open("tests/.fixtures/sample.jpg", "rb") as f:
        response = client.post(
            "/analyze",
            files={"file": ("sample.jpg", f, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "frame_size" in data
    assert "class_counts" in data
    assert "p1_bridge" in data

def test_detect_invalid_file():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code in [400, 422, 500]