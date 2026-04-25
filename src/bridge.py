import httpx
import os
from dotenv import load_dotenv

load_dotenv()

P1_URL = os.getenv("P1_PREDICT_URL", "http://localhost:8000/predict")

async def forward_to_p1(scene_data: dict) -> dict:
    # Serialize scene summary as a text string for P1
    scene_text = (
        f"Objects detected: {scene_data['total_objects']}. "
        f"Classes: {', '.join(f'{k}: {v}' for k, v in scene_data['class_counts'].items())}. "
        f"Crowded: {scene_data['crowded']}. "
        f"Occlusion: {scene_data['occlusion_detected']}."
    )

    payload = {"text": scene_text}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(P1_URL, json=payload)
            response.raise_for_status()
            return {"p1_response": response.json(), "status": "success"}
    except httpx.ConnectError:
        return {"p1_response": None, "status": "p1_unavailable"}
    except httpx.TimeoutException:
        return {"p1_response": None, "status": "p1_timeout"}
    except Exception as e:
        return {"p1_response": None, "status": f"error: {str(e)}"}