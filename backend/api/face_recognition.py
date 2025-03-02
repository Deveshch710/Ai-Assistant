# backend/api/face_recognition.py

import cv2
import face_recognition
import numpy as np
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io

# Create the router
router = APIRouter()

class FaceRecognitionService:
    """Service for face recognition using images stored in 'known_faces' directory."""

    def __init__(self, known_faces_dir: str = "backend/core/vision/known_faces"):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the 'known_faces' directory and encode them."""
        for image_path in self.known_faces_dir.glob("*.jpg"):  # Supports only JPG for now
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                self.known_encodings.append(encodings[0])
                self.known_names.append(image_path.stem)  # Use filename (without extension) as person's name
        
        print(f"✅ Loaded {len(self.known_names)} known faces.")

    def recognize_faces(self, image: np.ndarray):
        """Recognize faces in the given image."""
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        recognized_faces = []
        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"
            
            if True in matches:
                matched_idx = matches.index(True)
                name = self.known_names[matched_idx]
                
            recognized_faces.append({
                "name": name,
                "bounding_box": location
            })
        
        return {"status": "success", "faces": recognized_faces}

# Initialize service
face_service = FaceRecognitionService()

@router.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    """Endpoint to recognize faces in an uploaded image."""
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform face recognition
        result = face_service.recognize_faces(image)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"An error occurred: {str(e)}"}
        )

