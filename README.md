# CV-Service

Real-time computer vision microservice built with YOLOv8 and FastAPI.
Part of a 5-project AI Engineer portfolio. Connects to the MLOps pipeline (Project 1).

## Features
- Object detection with YOLOv8n
- Scene understanding (zones, crowding, occlusion detection)
- REST API with `/detect`, `/analyze`, `/health` endpoints
- Live HTTP bridge to Project 1 MLOps inference API
- Fully containerised with Docker Compose

## Tech Stack
`YOLOv8` `OpenCV` `PyTorch` `FastAPI` `Docker` `httpx` `pytest`

## Quick Start
```bash
docker-compose up --build
```
API docs available at `http://localhost:8001/docs`

## Demo
[Add your GIF here]

## Project Roadmap
This is Project 2 of 5 in an interconnected AI Engineer portfolio:
- ✅ Project 1 — MLOps Pipeline
- ✅ Project 2 — CV Service (this repo)
- 🔜 Project 3 — Multimodal RAG System
- 🔜 Project 4 — ML Evaluation Dashboard
- 🔜 Project 5 — Autonomous AI Agent