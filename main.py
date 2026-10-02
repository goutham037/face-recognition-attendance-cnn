import cv2
from src.face_recognition import load_model, recognize_face
from src.attendance_manager import mark_attendance

def main():
    load_model()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Attendance system started. Press Q to quit.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        for name, confidence, (x, y, w, h) in recognize_face(frame):
            label = f"{name} ({confidence * 100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Face Recognition Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
