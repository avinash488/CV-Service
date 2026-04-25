import cv2
import requests
import numpy as np
import argparse
import time

def run_demo(interval: float = 0.2):
    cap = cv2.VideoCapture(0)
    print("CV-Service Live Demo | Press Q to quit")
    print("Make sure CV-Service is running on http://localhost:8001")

    last_scene = {}
    last_request_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Send frame to /detect every `interval` seconds
        if current_time - last_request_time >= interval:
            _, img_encoded = cv2.imencode(".jpg", frame)
            try:
                response = requests.post(
                    "http://localhost:8001/detect",
                    files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                    timeout=2.0
                )
                if response.status_code == 200:
                    last_scene = response.json()
            except requests.exceptions.RequestException:
                pass
            last_request_time = current_time

        # Draw bounding boxes
        for d in last_scene.get("detections", []):
            x1, y1, x2, y2 = [int(c) for c in d["bbox"]]
            label = f"{d['class_name']} ({d['zone']}) {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # HUD overlay
        total = last_scene.get("total_objects", 0)
        cv2.putText(frame, f"Objects: {total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"CV-Service | {interval*1000:.0f}ms interval",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("CV-Service Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Seconds between API calls (default: 0.2)")
    args = parser.parse_args()
    run_demo(interval=args.interval)