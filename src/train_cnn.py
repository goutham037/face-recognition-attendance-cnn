import os
import sys
import json
import tensorflow as tf

# Allow `python src/train_cnn.py` from the project root to resolve the
# sibling `preprocess` module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_dataset

# Resolve `models` and `dataset` paths relative to the project root (the
# parent of `src/`) so training works no matter where the script is run.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
os.makedirs(MODEL_DIR, exist_ok=True)

X, y, class_names = load_dataset(DATASET_DIR)

num_classes = len(class_names)
if num_classes < 2:
    raise ValueError("Add at least two students before training.")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 100, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2, shuffle=True)

model.save(os.path.join(MODEL_DIR, "face_recognition_model.keras"))

with open(os.path.join(MODEL_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("Training completed.")
print("Classes:", class_names)
