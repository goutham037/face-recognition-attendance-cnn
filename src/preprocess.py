import os
import cv2
import numpy as np

IMAGE_SIZE = (100, 100)

def load_dataset(dataset_dir="dataset"):
    images = []
    labels = []
    class_names = []

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError("Dataset folder not found.")

    folders = sorted(
        f for f in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, f))
    )

    for label, folder in enumerate(folders):
        class_names.append(folder)
        folder_path = os.path.join(dataset_dir, folder)

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label)

    if not images:
        raise ValueError("No training images found.")

    X = np.array(images, dtype=np.float32) / 255.0
    X = X.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y = np.array(labels, dtype=np.int32)

    return X, y, class_names
