import numpy as np
from deepface import DeepFace
import face_recognition
from .interfaces import IFaceModel

class DeepFaceAdapter(IFaceModel):
    def __init__(self, model_name: str = "VGG-Face", threshold: float = 0.40):
        self.model_name = model_name
        self.threshold = threshold

    def get_descriptors(self, image_path: str):
        try:
            objs = DeepFace.represent(img_path=image_path, model_name=self.model_name, enforce_detection=False, detector_backend="opencv")
            return [obj["embedding"] for obj in objs]
        except: return []

    def compare(self, d1, d2) -> bool:
        dist = 1 - (np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
        return dist <= self.threshold

    def get_name(self) -> str: return f"DeepFace_{self.model_name}"

class FaceRecognitionAdapter(IFaceModel):
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def get_descriptors(self, image_path: str):
        try:
            img = face_recognition.load_image_file(image_path)
            return face_recognition.face_encodings(img)
        except: return []

    def compare(self, d1, d2) -> bool:
        return face_recognition.face_distance([d1], d2)[0] <= self.threshold

    def get_name(self) -> str: return "FaceRecognition_Dlib"