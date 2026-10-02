import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf

# Allow this module to resolve its sibling imports whether run as
# `python src/...` or imported as `src.face_recognition` from main.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_detection import create_detector, detect_faces

# Resolve model paths relative to the project root so `python main.py`
# works from any directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "face_recognition_model.keras")
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "labels.json")
CONFIDENCE_THRESHOLD = 0.70

model = None
class_names = []
detector = create_detector()

def load_model():
    global model, class_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run capture_faces.py and train_cnn.py first."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

def recognize_face(frame):
    if model is None:
        load_model()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = []

    for (x, y, w, h) in detect_faces(gray, detector):
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100)).astype("float32") / 255.0
        face = face.reshape(1, 100, 100, 1)

        probabilities = model.predict(face, verbose=0)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        name = class_names[index] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
        results.append((name, confidence, (x, y, w, h)))

    return results
