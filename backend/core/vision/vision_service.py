# backend/core/vision/vision_service.py
# In this it have different differnet models i can use in object detection in yolo
# This file contanin all the things related to object detection
# This file is used to detect the object in image, video and live webcam
# To chnage it to better model change model_path in __init__ function
# THese are the differnet models i can use in this file
        # YOLOv8 Model Options:
        # - 'yolov8n.pt'  (Nano)  -> Fastest, least accurate (Default)
        # - 'yolov8s.pt'  (Small)  -> Slightly better accuracy
        # - 'yolov8m.pt'  (Medium) -> Balanced speed & accuracy
        # - 'yolov8l.pt'  (Large)  -> Higher accuracy, slower
        # - 'yolov8x.pt'  (X-Large) -> Best accuracy, slowest
        
        # Change model_path to switch between these models



import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class VisionService:
    """Service for object detection using YOLOv8 on images, videos, and live webcam."""

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
        """Perform object detection on an image."""
        try:
            results = self.model(image_path)  # Run inference
            detections = []

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                class_names = [self.model.names[int(cls)] for cls in class_ids]  # Class labels

                for i in range(len(boxes)):
                    detections.append({
                        "class": class_names[i],
                        "confidence": float(confidences[i]),
                        "bounding_box": boxes[i].tolist()
                    })

            return {"status": "success", "detections": detections}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def detect_objects_in_video(self, video_path: str, output_path: str = "output.mp4", show: bool = False):
        """Perform object detection on a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Error opening video file"}

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO object detection
            results = self.model(frame)

            # Draw bounding boxes
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                class_names = [self.model.names[int(cls)] for cls in class_ids]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    label = f"{class_names[i]}: {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)  # Save to output video file

            if show:
                cv2.imshow("Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return {"status": "success", "message": f"Processed video saved at {output_path}"}

    def detect_objects_live(self):
        """Perform real-time object detection using a webcam."""
        cap = cv2.VideoCapture(0)  # Open webcam

        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return

        print("🎥 Starting live object detection. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO object detection
            results = self.model(frame)

            # Draw bounding boxes
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                class_names = [self.model.names[int(cls)] for cls in class_ids]

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    label = f"{class_names[i]}: {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Live Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# Testing
if __name__ == "__main__":
    vision_service = VisionService()
    
    # Test image detection
    image_path = "test.jpg"
    print(vision_service.detect_objects(image_path))

    # Test video detection
    video_path = "test.mp4"
    vision_service.detect_objects_in_video(video_path, show=True)

    # Test live webcam detection
    vision_service.detect_objects_live()
