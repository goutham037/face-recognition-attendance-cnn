import csv
import os
from datetime import datetime

# Resolve attendance file relative to the project root (the parent of `src/`)
# so attendance is written to the correct location regardless of cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATTENDANCE_DIR = os.path.join(PROJECT_ROOT, "attendance")
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")

def mark_attendance(student):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    existing = set()

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add((row["Student"], row["Date"]))

    if (student, today) not in existing:
        new_file = not os.path.exists(ATTENDANCE_FILE)

        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Student", "Date", "Time", "Status"])
            writer.writerow([student, today, now, "Present"])

        print(f"Attendance marked: {student} - {today} {now}")
        return True

    return False
