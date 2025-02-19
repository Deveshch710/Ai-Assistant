# backend/core/vision/vision_service.py

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class VisionService:
    """Service for object and face detection using YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the YOLOv8 model."""
        try:
            self.model = YOLO(self.model_path)
            print("✅ YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {str(e)}")

    def detect_objects(self, image_path: str):
        """Perform object and face detection on an image."""
        try:
            results = self.model(image_path)  # Run inference
            detections = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                class_names = [self.model.names[int(cls)] for cls in class_ids]  # Class labels

                # Store results
                for i in range(len(boxes)):
                    detections.append({
                        "class": class_names[i],
                        "confidence": float(confidences[i]),
                        "bounding_box": boxes[i].tolist()
                    })

            return {"status": "success", "detections": detections}

        except Exception as e:
            return {"status": "error", "message": str(e)}

# Initialize service (for testing)
if __name__ == "__main__":
    vision_service = VisionService()
    image_path = "test.jpg"  # Replace with an actual image path
    result = vision_service.detect_objects(image_path)
    print(result)
