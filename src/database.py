# Optional database module.
# The starter project uses CSV for simplicity.
# You can replace attendance_manager.py with MySQL/SQLite logic later.

import os
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_FILE = os.path.join(PROJECT_ROOT, "attendance", "attendance.db")

def get_connection():
    return sqlite3.connect(DB_FILE)

def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    connection.commit()
    connection.close()
