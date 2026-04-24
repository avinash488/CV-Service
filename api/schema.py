from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]
    zone: str

class DetectionResponse(BaseModel):
    total_objects: int
    detections: List[Detection]

class SceneResponse(BaseModel):
    frame_size: Dict[str, int]
    total_objects: int
    class_counts: Dict[str, int]
    crowded: bool
    occlusion_detected: bool
    detections: List[Dict[str, Any]]
    p1_bridge: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model: str