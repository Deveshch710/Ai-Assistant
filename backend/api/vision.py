from fastapi import APIRouter, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import io

# Initialize FastAPI router
router = APIRouter()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model (lightweight)

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects and faces in an uploaded image."""
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Run YOLO object detection
        results = model(image)

        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bounding_box": box.xyxy[0].tolist()
                })

        return {"status": "success", "detections": detections}

    except Exception as e:
        return {"status": "error", "message": str(e)}
