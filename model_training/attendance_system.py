"""
Simplified Attendance System - Returns: person_id, date, time
"""

import os
import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import numpy as np
import pickle
import json
import sqlite3
from typing import Optional, Dict

class SimpleAttendanceSystem:
    def __init__(self, dataset_path, model_name="Facenet", threshold=0.6, db_path="attendance.db"):
        """
        Simple attendance system returning person_id, date, time
        
        Args:
            dataset_path: Path to person image folders
            model_name: DeepFace model name
            threshold: Recognition threshold
            db_path: SQLite database path
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.threshold = threshold
        self.db_path = db_path
        self.embeddings_cache = {}
        self.cache_file = "embeddings_cache.pkl"
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Create SQLite database with persons table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Persons table - stores person_id and name mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attendance table - stores person_id, date, time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id),
                UNIQUE(person_id, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    def initialize(self):
        """Initialize system and load/build embeddings"""
        print("="*60)
        print("INITIALIZING ATTENDANCE SYSTEM")
        print("="*60)
        
        if not os.path.exists(self.dataset_path):
            return {"success": False, "error": "Dataset path not found"}
        
        persons = [d for d in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not persons:
            return {"success": False, "error": "No persons found in dataset"}
        
        # Register persons in database and get their IDs
        person_ids = self.register_persons(persons)
        
        # Load or build embeddings
        if os.path.exists(self.cache_file):
            print("Loading cached embeddings...")
            with open(self.cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"✓ Loaded {len(self.embeddings_cache)} embeddings")
        else:
            print("Building embeddings cache...")
            self.build_embeddings_cache()
        
        print("✓ System ready!")
        print("="*60)
        
        return {
            "success": True,
            "persons": [{"person_id": pid, "name": name} for name, pid in person_ids.items()],
            "total_persons": len(persons)
        }
    
    def register_persons(self, persons: list) -> dict:
        """
        Register persons in database and return name->person_id mapping
        
        Returns:
            dict: {"John": 1, "Sarah": 2, ...}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        person_ids = {}
        
        for person in persons:
            cursor.execute(
                "INSERT OR IGNORE INTO persons (name) VALUES (?)",
                (person,)
            )
            
            # Get the person_id
            cursor.execute("SELECT person_id FROM persons WHERE name = ?", (person,))
            person_id = cursor.fetchone()[0]
            person_ids[person] = person_id
        
        conn.commit()
        conn.close()
        
        print(f"\nRegistered persons:")
        for name, pid in person_ids.items():
            print(f"  • {name} (ID: {pid})")
        
        return person_ids
    
    def get_person_id(self, name: str) -> Optional[int]:
        """Get person_id from name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id FROM persons WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def build_embeddings_cache(self):
        """Build embeddings cache for all persons"""
        self.embeddings_cache = {}
        
        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            print(f"  Processing {person_name}...", end=" ")
            person_embeddings = []
            
            for img_file in os.listdir(person_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(person_path, img_file)
                
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if embedding_objs:
                        person_embeddings.append(embedding_objs[0]["embedding"])
                except:
                    continue
            
            if person_embeddings:
                self.embeddings_cache[person_name] = np.mean(person_embeddings, axis=0)
                print(f"✓ ({len(person_embeddings)} images)")
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        
        print(f"✓ Cache saved")
    
    def cosine_distance(self, emb1, emb2):
        """Calculate cosine distance"""
        return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def recognize_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Recognize person from OpenCV frame
        
        Returns:
            dict: {"success": bool, "person_name": str, "person_id": int}
        """
        try:
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            embedding_objs = DeepFace.represent(
                img_path=temp_path,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if not embedding_objs:
                return {"success": False, "error": "No face detected"}
            
            current_embedding = embedding_objs[0]["embedding"]
            
            best_match = None
            min_distance = float('inf')
            
            for person_name, stored_embedding in self.embeddings_cache.items():
                distance = self.cosine_distance(current_embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person_name
            
            if min_distance < self.threshold:
                person_id = self.get_person_id(best_match)
                return {
                    "success": True,
                    "person_name": best_match,
                    "person_id": person_id
                }
            
            return {"success": False, "error": "No matching person found"}
            
        except Exception as e:
            if os.path.exists('temp_frame.jpg'):
                os.remove('temp_frame.jpg')
            return {"success": False, "error": str(e)}
    
    def mark_attendance(self, person_id: int) -> Dict:
        """
        Mark attendance - stores only person_id, date, time
        
        Args:
            person_id: ID of the person
            
        Returns:
            dict: {
                "success": bool,
                "person_id": int,
                "date": str,
                "time": str,
                "already_marked": bool
            }
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_datetime = datetime.now()
            today = current_datetime.date()
            current_time = current_datetime.time()
            
            # Check if already marked today
            cursor.execute(
                "SELECT id FROM attendance WHERE person_id = ? AND date = ?",
                (person_id, today)
            )
            existing = cursor.fetchone()
            
            if existing:
                conn.close()
                return {
                    "success": True,
                    "already_marked": True,
                    "person_id": person_id,
                    "date": str(today),
                    "time": str(current_time.strftime('%H:%M:%S')),
                    "message": "Already marked today"
                }
            
            # Insert attendance record
            cursor.execute(
                "INSERT INTO attendance (person_id, date, time) VALUES (?, ?, ?)",
                (person_id, today, current_time)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "already_marked": False,
                "person_id": person_id,
                "date": str(today),
                "time": current_time.strftime('%H:%M:%S')
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_attendance_today(self) -> Dict:
        """Get today's attendance - returns person_id, date, time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute(
                "SELECT person_id, date, time FROM attendance WHERE date = ?",
                (today,)
            )
            
            records = cursor.fetchall()
            conn.close()
            
            return {
                "success": True,
                "date": str(today),
                "total_present": len(records),
                "records": [
                    {
                        "person_id": r[0],
                        "date": r[1],
                        "time": r[2]
                    }
                    for r in records
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def start_webcam_attendance(self):
        """Webcam mode with simplified output"""
        print("\n" + "="*60)
        print("WEBCAM ATTENDANCE MODE")
        print("="*60)
        print("Controls: Q=Quit | S=Show Today's Attendance")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        last_recognition = 0
        marked_today = set()
        
        print("✓ Webcam started. Waiting for faces...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Recognize every 45 frames
            if frame_count % 45 == 0 and (frame_count - last_recognition) > 45:
                result = self.recognize_from_frame(frame)
                
                if result["success"]:
                    person_id = result["person_id"]
                    person_name = result["person_name"]
                    
                    # Mark attendance
                    attendance = self.mark_attendance(person_id)
                    
                    if attendance["success"] and not attendance.get("already_marked"):
                        marked_today.add(person_name)
                        
                        # Print backend data format
                        print(f"\n{'='*60}")
                        print(f"✓ ATTENDANCE MARKED")
                        print(f"{'='*60}")
                        print(f"person_id: {attendance['person_id']}")
                        print(f"date:      {attendance['date']}")
                        print(f"time:      {attendance['time']}")
                        print(f"{'='*60}\n")
                        
                        last_recognition = frame_count
            
            # UI
            cv2.rectangle(display_frame, (0, 0), (640, 40), (50, 50, 50), -1)
            cv2.putText(display_frame, "Attendance System", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.rectangle(display_frame, (0, 440), (640, 480), (50, 50, 50), -1)
            cv2.putText(display_frame, f"Marked: {len(marked_today)}", 
                       (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            y_pos = 90
            for name in list(marked_today)[:5]:
                cv2.putText(display_frame, f"✓ {name}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 35
            
            cv2.imshow('Attendance System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                today = self.get_attendance_today()
                print("\n" + json.dumps(today, indent=2))
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam closed")


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    system = SimpleAttendanceSystem(
        dataset_path="dataset",
        model_name="Facenet",
        threshold=0.6
    )
    
    # Initialize
    init_result = system.initialize()
    print(json.dumps(init_result, indent=2))
    
    if init_result["success"]:
        # Start webcam
        system.start_webcam_attendance()
        
        # Get today's attendance
        today = system.get_attendance_today()
        print("\n" + json.dumps(today, indent=2))