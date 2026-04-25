import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from api.schema import DetectionResponse, SceneResponse, HealthResponse, Detection
from src.scene import SceneAnalyzer
from src.bridge import forward_to_p1

app = FastAPI(title="CV-Service", version="1.0.0")
analyzer = SceneAnalyzer()

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": "yolov8n"}

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid image file")

    scene = analyzer.analyze(frame)
    detections = [
        Detection(
            class_name=d["class"],
            confidence=d["confidence"],
            bbox=d["bbox"],
            zone=d["zone"]
        )
        for d in scene["detections"]
    ]
    return DetectionResponse(total_objects=scene["total_objects"], detections=detections)

@app.post("/analyze", response_model=SceneResponse)
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid image file")

    scene = analyzer.analyze(frame)
    p1_result = await forward_to_p1(scene)
    scene["p1_bridge"] = p1_result
    return scene