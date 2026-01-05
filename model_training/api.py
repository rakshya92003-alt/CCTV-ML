"""
Flask API - Returns: person_id, date, time
Ready to use - No changes needed
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
from datetime import datetime

# Import attendance system
from attendance_system import SimpleAttendanceSystem

app = Flask(__name__)
CORS(app)

# Initialize system
DATASET_PATH = "dataset"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the attendance system
print("Initializing attendance system...")
system = SimpleAttendanceSystem(
    dataset_path=DATASET_PATH,
    model_name="Facenet",
    threshold=0.6
)
init_result = system.initialize()

if not init_result["success"]:
    print(f"❌ Error: {init_result.get('error')}")
    print("Make sure 'dataset' folder exists with person folders inside")
    exit(1)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/mark-attendance', methods=['POST'])
def mark_attendance():
    """
    Mark attendance from image
    
    Request (JSON):
    {
        "image": "base64_encoded_image"
    }
    
    OR Form-data:
        file: image file
    
    Response:
    {
        "success": true,
        "person_id": 1,
        "date": "2024-12-19",
        "time": "15:30:45",
        "already_marked": false
    }
    """
    try:
        # Method 1: Base64 image (from JSON)
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            
            if not image_data:
                return jsonify({"success": False, "error": "No image data provided"}), 400
            
            # Decode base64
            image_bytes = base64.b64decode(image_data.split(',')[-1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Method 2: File upload (from form-data)
        elif 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Read image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        else:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400
        
        # Recognize person
        recognition = system.recognize_from_frame(frame)
        
        if not recognition["success"]:
            return jsonify(recognition), 400
        
        # Mark attendance
        person_id = recognition["person_id"]
        result = system.mark_attendance(person_id)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    """
    Get today's attendance
    
    Response:
    {
        "success": true,
        "date": "2024-12-19",
        "total_present": 3,
        "records": [
            {
                "person_id": 1,
                "date": "2024-12-19",
                "time": "09:00:00"
            }
        ]
    }
    """
    try:
        result = system.get_attendance_today()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/persons', methods=['GET'])
def get_persons():
    """
    Get all registered persons
    
    Response:
    {
        "success": true,
        "persons": [
            {"person_id": 1, "name": "John"},
            {"person_id": 2, "name": "Sarah"}
        ]
    }
    """
    try:
        import sqlite3
        conn = sqlite3.connect(system.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT person_id, name FROM persons ORDER BY person_id")
        persons = cursor.fetchall()
        conn.close()
        
        return jsonify({
            "success": True,
            "total": len(persons),
            "persons": [
                {"person_id": p[0], "name": p[1]}
                for p in persons
            ]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/recognize-only', methods=['POST'])
def recognize_only():
    """
    Only recognize person without marking attendance
    Useful for preview/testing
    
    Response:
    {
        "success": true,
        "person_id": 1,
        "person_name": "John"
    }
    """
    try:
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            image_bytes = base64.b64decode(image_data.split(',')[-1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif 'file' in request.files:
            file = request.files['file']
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            return jsonify({"success": False, "error": "No image provided"}), 400
        
        result = system.recognize_from_frame(frame)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ATTENDANCE API - Returns: person_id, date, time")
    print("="*60)
    print("Running on: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /api/health              - Health check")
    print("  POST /api/mark-attendance     - Mark attendance from image")
    print("  POST /api/recognize-only      - Just recognize (no marking)")
    print("  GET  /api/attendance/today    - Get today's attendance")
    print("  GET  /api/persons             - Get all persons")
    print("="*60)
    print("\nSystem initialized successfully!")
    print(f"Registered persons: {init_result['total_persons']}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)