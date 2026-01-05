import os
import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace
import time
import numpy as np
import pickle

class FastAttendanceSystem:
    def __init__(self, dataset_path, model_name="Facenet", threshold=0.6):
        """
        Optimized real-time attendance system
        
        Args:
            dataset_path: Path to folder containing person folders with images
            model_name: DeepFace model (Facenet is faster than Facenet512)
            threshold: Distance threshold for face matching
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.threshold = threshold
        self.attendance_records = []
        self.marked_today = set()
        self.embeddings_cache = {}
        self.cache_file = "embeddings_cache.pkl"
        
    def initialize(self):
        """Initialize and build face embeddings database"""
        print("="*60)
        print("INITIALIZING ATTENDANCE SYSTEM")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_path}")
        
        # Verify dataset
        if not os.path.exists(self.dataset_path):
            print(f"\n❌ Error: Dataset path '{self.dataset_path}' not found!")
            return False
        
        persons = [d for d in os.listdir(self.dataset_path) 
                  if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if not persons:
            print(f"\n❌ Error: No person folders found in '{self.dataset_path}'")
            return False
        
        print(f"\nFound {len(persons)} persons in database:")
        for person in persons:
            person_path = os.path.join(self.dataset_path, person)
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  • {person}: {len(images)} images")
        
        # Load or create embeddings cache
        print("\n" + "-"*60)
        if os.path.exists(self.cache_file):
            print("Loading cached embeddings...")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"✓ Loaded {len(self.embeddings_cache)} cached embeddings")
            except:
                print("⚠ Cache corrupted, rebuilding...")
                self.build_embeddings_cache()
        else:
            print("Building embeddings cache (this may take 1-2 minutes)...")
            self.build_embeddings_cache()
        
        print("-"*60)
        print("\n✓ System ready!")
        print("="*60 + "\n")
        return True
    
    def build_embeddings_cache(self):
        """Pre-compute embeddings for all faces in dataset"""
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
                    # Generate embedding
                    embedding_objs = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if embedding_objs:
                        person_embeddings.append(embedding_objs[0]["embedding"])
                
                except Exception as e:
                    continue
            
            if person_embeddings:
                # Store average embedding for this person
                self.embeddings_cache[person_name] = np.mean(person_embeddings, axis=0)
                print(f"✓ ({len(person_embeddings)} images)")
            else:
                print("❌ Failed")
        
        # Save cache
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        
        print(f"\n✓ Cache saved to '{self.cache_file}'")
    
    def cosine_distance(self, embedding1, embedding2):
        """Calculate cosine distance between two embeddings"""
        return 1 - np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def recognize_face_fast(self, frame):
        """
        Fast face recognition using pre-computed embeddings
        
        Args:
            frame: OpenCV frame from webcam
            
        Returns:
            tuple: (person_name, confidence) or (None, None)
        """
        try:
            # Detect and get embedding for current frame
            temp_img = 'temp_detection.jpg'
            cv2.imwrite(temp_img, frame)
            
            embedding_objs = DeepFace.represent(
                img_path=temp_img,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend='opencv'
            )
            
            # Clean up
            if os.path.exists(temp_img):
                os.remove(temp_img)
            
            if not embedding_objs:
                return None, None
            
            current_embedding = embedding_objs[0]["embedding"]
            
            # Compare with all cached embeddings
            best_match = None
            min_distance = float('inf')
            
            for person_name, stored_embedding in self.embeddings_cache.items():
                distance = self.cosine_distance(current_embedding, stored_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = person_name
            
            # Check if match is good enough
            if min_distance < self.threshold:
                confidence = (1 - min_distance) * 100
                return best_match, confidence
            
            return None, None
            
        except Exception as e:
            if os.path.exists('temp_detection.jpg'):
                os.remove('temp_detection.jpg')
            return None, None
    
    def mark_attendance(self, person_name, confidence):
        """Mark attendance for recognized person"""
        current_time = datetime.now()
        today = current_time.date()
        
        if person_name in self.marked_today:
            return False
        
        record = {
            'Name': person_name,
            'Date': today.strftime('%Y-%m-%d'),
            'Time': current_time.strftime('%H:%M:%S'),
            'Confidence': f"{confidence:.1f}%"
        }
        
        self.attendance_records.append(record)
        self.marked_today.add(person_name)
        
        print(f"\n{'='*60}")
        print(f"✓ ATTENDANCE MARKED")
        print(f"{'='*60}")
        print(f"Name:       {person_name}")
        print(f"Date:       {record['Date']}")
        print(f"Time:       {record['Time']}")
        print(f"Confidence: {record['Confidence']}")
        print(f"{'='*60}\n")
        
        return True
    
    def start_webcam_attendance(self):
        """Start optimized webcam attendance system"""
        print("\n" + "="*60)
        print("STARTING WEBCAM ATTENDANCE SYSTEM")
        print("="*60)
        print("Instructions:")
        print("  • Position your face in front of the camera")
        print("  • Hold still for 1-2 seconds for detection")
        print("  • Press 'Q' to quit")
        print("  • Press 'S' to save attendance")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam!")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        last_recognition_time = 0
        recognition_cooldown = 2  # 2 seconds between recognitions
        recognizing = False
        
        print("✓ Webcam started. Waiting for faces...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read from webcam")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            display_frame = frame.copy()
            current_time = time.time()
            
            # Recognize face every 45 frames (1.5 seconds) with cooldown
            if frame_count % 45 == 0 and (current_time - last_recognition_time) > recognition_cooldown:
                if not recognizing:
                    recognizing = True
                    
                    # Show "Detecting..." message
                    cv2.putText(display_frame, "Detecting...", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, (0, 255, 255), 2)
                    cv2.imshow('Attendance System', display_frame)
                    cv2.waitKey(1)
                    
                    # Perform recognition
                    person_name, confidence = self.recognize_face_fast(frame)
                    
                    if person_name:
                        marked = self.mark_attendance(person_name, confidence)
                        if marked:
                            last_recognition_time = current_time
                    
                    recognizing = False
            
            # Draw UI
            cv2.rectangle(display_frame, (0, 0), (640, 40), (50, 50, 50), -1)
            cv2.putText(display_frame, f"Attendance System | FPS: {fps:.0f}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            # Bottom bar
            cv2.rectangle(display_frame, (0, 440), (640, 480), (50, 50, 50), -1)
            cv2.putText(display_frame, f"Marked: {len(self.marked_today)}", 
                       (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 1)
            cv2.putText(display_frame, "Q: Quit | S: Save", 
                       (400, 465), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (200, 200, 200), 1)
            
            # Show marked names
            if self.marked_today:
                y_pos = 90
                for name in list(self.marked_today)[:5]:  # Show max 5
                    cv2.putText(display_frame, f"✓ {name}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 255, 0), 2)
                    y_pos += 35
            
            cv2.imshow('Attendance System', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                self.save_attendance()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Webcam closed")
    
    def save_attendance(self, filename='attendance.csv'):
        """Save attendance to CSV"""
        if not self.attendance_records:
            print("\n⚠ No attendance records to save")
            return
        
        df = pd.DataFrame(self.attendance_records)
        
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"\n✓ Attendance saved to '{filename}' ({len(self.attendance_records)} new records)")
    
    def display_attendance(self):
        """Display attendance records"""
        if not self.attendance_records:
            print("\n⚠ No attendance records")
            return
        
        print("\n" + "="*60)
        print("ATTENDANCE RECORDS")
        print("="*60)
        df = pd.DataFrame(self.attendance_records)
        print(df.to_string(index=False))
        print("="*60 + "\n")


if __name__ == "__main__":
    # Configuration - OPTIMIZED FOR SPEED
    DATASET_PATH = "dataset"
    MODEL_NAME = "Facenet"  # Faster than Facenet512
    THRESHOLD = 0.6  # Adjusted for Facenet model
    
    system = FastAttendanceSystem(
        dataset_path=DATASET_PATH,
        model_name=MODEL_NAME,
        threshold=THRESHOLD
    )
    
    if system.initialize():
        system.start_webcam_attendance()
        system.display_attendance()
        system.save_attendance()
    else:
        print("\n❌ System initialization failed.")