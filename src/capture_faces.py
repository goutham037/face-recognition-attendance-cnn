import cv2
import os

# Resolve `dataset/` relative to the project root (the parent of `src/`)
# so the script works no matter which directory it is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

student_id = input("Enter student ID: ").strip()
student_name = input("Enter student name: ").strip().replace(" ", "_")

if not student_id or not student_name:
    raise SystemExit("Both student ID and name are required.")

save_dir = os.path.join(DATASET_DIR, f"{student_id}_{student_name}")
os.makedirs(save_dir, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and retry.")
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
print("Look at the camera. Press Q to stop.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Images: {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or count >= 30:
        break

camera.release()
cv2.destroyAllWindows()
print(f"Saved {count} face images in {save_dir}")
