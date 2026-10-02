# Face Recognition Attendance System using CNN

A beginner-friendly academic project in Python that:

1. Captures student face images from a webcam.
2. Trains a small Convolutional Neural Network (CNN) to classify them.
3. Runs live webcam attendance that recognises registered students and
   writes a CSV log — one entry per student per day.

Stack: **Python**, **OpenCV** (Haar Cascade face detection) and
**TensorFlow / Keras** (CNN classifier).

---

## Project flow

```
Camera -> Face Detection (Haar) -> Preprocess (grayscale, 100x100)
       -> CNN -> Student Recognition -> Attendance CSV
```

## Project tree

```
Face_Recognition_Attendance_System_CNN/
├── main.py                   # live attendance loop
├── requirements.txt
├── README.md
├── dataset/                  # captured face images (per-student folders)
├── models/                   # trained Keras model + labels
├── attendance/               # attendance.csv output
└── src/
    ├── capture_faces.py      # Step 1 — capture 30 images per student
    ├── train_cnn.py          # Step 2 — train the CNN
    ├── face_detection.py     # Haar Cascade helper
    ├── face_recognition.py   # loads model, predicts a face
    ├── preprocess.py         # loads dataset into tensors
    ├── attendance_manager.py # writes attendance.csv
    └── database.py           # (optional) SQLite variant
```

## 1. Set up Python

**TensorFlow currently supports Python 3.9 – 3.12.** Python 3.13 is
not supported by stable TensorFlow releases yet, so install one of
the supported versions first.

Recommended: **Python 3.11**.

- Windows: `winget install Python.Python.3.11`
- or download from https://www.python.org/downloads/release/python-3119/

Check it works:

```bash
py -3.11 --version
```

## 2. Create a virtual environment and install dependencies

From the project root (Windows PowerShell):

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you get `ScriptNotFound` on `Activate.ps1`, use `venv\Scripts\activate.bat`
from `cmd.exe` instead.

## 3. Capture student faces (one student at a time)

```bash
python src/capture_faces.py
```

It will ask for the student's ID and name, open the webcam, and save
about 30 cropped grayscale face images to `dataset/<id>_<name>/`.

- Press **Q** to stop early.
- Repeat the command for every student you want to register.
- For good accuracy you want at least **2 students**, with varied head
  angles and lighting.

## 4. Train the CNN

```bash
python src/train_cnn.py
```

Trains for 15 epochs on the images in `dataset/`. On completion you'll
find:

- `models/face_recognition_model.keras` — the trained model
- `models/labels.json` — the class labels in training order

Re-run this step every time you add or remove students.

## 5. Run live attendance

```bash
python main.py
```

The webcam opens, faces are detected and classified, and recognised
students (confidence ≥ 0.70) are logged to:

```
attendance/attendance.csv
```

Only **one entry per student per day** is written. Press **Q** to quit.

CSV format:

```
Student,Date,Time,Status
1_Samhitha,2026-10-02,09:15:03,Present
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'preprocess'` or `'face_detection'`**
Fixed in this repo — every `src/*.py` that imports a sibling adds its
own folder to `sys.path`. If you still see it, make sure you pulled the
latest commit.

**`Could not open webcam`**
Close any app using the camera (Zoom, Teams, OBS, browser tabs). On
Windows, check Settings → Privacy → Camera and allow desktop apps.

**TensorFlow install fails with `No matching distribution`**
You're on Python 3.13 (or 3.8-). Install Python 3.11 as described
above and recreate the virtual environment with it.

**`FileNotFoundError: Trained model not found`**
You skipped Step 4. Run `python src/train_cnn.py` first.

**`ValueError: Add at least two students before training`**
Capture faces for at least two different students before training —
one-class classification doesn't make sense here.

**Low recognition accuracy**
Capture more images per student (edit the `count >= 30` limit in
`src/capture_faces.py`), vary head angles and lighting, and re-train.

## What's NOT included (intentionally)

- Anti-spoofing / liveness detection
- Encrypted storage of face images
- Role-based access control
- Audit logging

This is an **academic / educational** project. For a real deployment
you'd add consent, secure storage, access controls, anti-spoofing and
the appropriate privacy safeguards.

## License

MIT.
