# backend/core/vision/face_recognition_service.py

import face_recognition
import numpy as np
import cv2

class FaceRecognitionService:
   def __init__(self):
       self.known_face_encodings = []
       self.known_face_names = []

   def load_known_faces(self, known_faces: dict):
       """
       Load known faces from a dictionary of names and image paths.
       :param known_faces: dict with names as keys and image paths as values
       """
       for name, image_path in known_faces.items():
           image = face_recognition.load_image_file(image_path)
           encoding = face_recognition.face_encodings(image)[0]
           self.known_face_encodings.append(encoding)
           self.known_face_names.append(name)

   def recognize_faces(self, frame):
       """
       Recognize faces in a given frame.
       :param frame: image frame in which to recognize faces
       :return: list of names and face locations
       """
       rgb_frame = frame[:, :, ::-1]  # Convert BGR (OpenCV) to RGB
       face_locations = face_recognition.face_locations(rgb_frame)
       face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

       face_names = []
       for face_encoding in face_encodings:
           matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
           name = "Unknown"

           face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
           best_match_index = np.argmin(face_distances)
           if matches[best_match_index]:
               name = self.known_face_names[best_match_index]

           face_names.append(name)

       return face_names, face_locations

