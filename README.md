# Face Recognition Attendance System

A real-time face recognition attendance system using DeepFace AI with Flask REST API integration. Perfect for schools, offices, and events.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![DeepFace](https://img.shields.io/badge/DeepFace-AI-orange.svg)](https://github.com/serengil/deepface)

## Features

- **Real-time Face Recognition** - Detect and recognize faces from webcam
- **REST API** - Easy integration with any backend (Node.js, Django, PHP, etc.)
- **SQLite Database** - No complex database setup required for now
- **Fast Performance** - Cached embeddings for instant recognition
- **Duplicate Prevention** - One attendance per person per day
- **Simple Output** - Returns `person_id`, `date`, `time`
- **CORS Enabled** - Works with any frontend framework

## Demo

### Standalone Mode (Webcam)
```bash
python attendance_system.py
```

### API Mode
```bash
python api.py
```

**Sample Response:**
```json
{
  "success": true,
  "person_id": 1,
  "date": "2024-12-19",
  "time": "15:30:45",
  "already_marked": false
}
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for standalone mode)
- 4+ GB RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/face-attendance-system.git
cd face-attendance-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
flask==3.0.0
flask-cors==4.0.0
deepface==0.0.79
opencv-python==4.8.1.78
pandas==2.1.3
numpy==1.24.3
tf-keras==2.17.0
tensorflow==2.17.0
```

### Step 4: Setup Dataset

Create your dataset folder structure:

```
dataset/
├── person1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ... (20 images recommended)
├── person2/
│   └── ... (20 images)
├── person3/
│   └── ... (20 images)
└── person4/
    └── ... (20 images)
```

**Important:** 
- Use clear, well-lit photos
- Multiple angles per person
- Face should be clearly visible
- 15-20 images per person recommended

## Quick Start

### Option 1: Standalone Webcam Mode

```bash
python attendance_system.py
```

**Features:**
- Opens webcam automatically
- Detects faces in real-time
- Marks attendance automatically
- Displays results on screen

**Controls:**
- Press `Q` to quit
- Press `S` to show today's attendance

### Option 2: API Server Mode

```bash
python api.py
```

Server runs on `http://localhost:5000`

**Test the API:**
```bash
# Health check
curl http://localhost:5000/api/health

# Mark attendance with image
curl -X POST http://localhost:5000/api/mark-attendance \
  -F "file=@photo.jpg"
```

## API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-19T15:30:00"
}
```

---

#### 2. Mark Attendance
```http
POST /api/mark-attendance
```

**Request (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Request (Form-data):**
```
file: image.jpg
```

**Response:**
```json
{
  "success": true,
  "person_id": 1,
  "date": "2024-12-19",
  "time": "15:30:45",
  "already_marked": false
}
```

---

#### 3. Get Today's Attendance
```http
GET /api/attendance/today
```

**Response:**
```json
{
  "success": true,
  "date": "2024-12-19",
  "total_present": 3,
  "records": [
    {
      "person_id": 1,
      "date": "2024-12-19",
      "time": "09:00:00"
    },
    {
      "person_id": 2,
      "date": "2024-12-19",
      "time": "09:15:30"
    }
  ]
}
```

---

#### 4. Get All Persons
```http
GET /api/persons
```

**Response:**
```json
{
  "success": true,
  "total": 4,
  "persons": [
    {"person_id": 1, "name": "John"},
    {"person_id": 2, "name": "Sarah"},
    {"person_id": 3, "name": "Mike"},
    {"person_id": 4, "name": "Lisa"}
  ]
}
```

---

#### 5. Recognize Only (No Marking)
```http
POST /api/recognize-only
```

**Request:** Same as mark-attendance

**Response:**
```json
{
  "success": true,
  "person_id": 1,
  "person_name": "John"
}
```

## Project Structure

```
face-attendance-system/
├── attendance_system.py    # Core attendance logic
├── api.py                  # Flask REST API
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── dataset/               # Training images
│   ├── person1/
│   ├── person2/
│   ├── person3/
│   └── person4/
│
├── attendance.db          # SQLite database (auto-created)
├── embeddings_cache.pkl   # Cached face embeddings (auto-created)
└── uploads/               # Temporary uploads (auto-created)
```

## Configuration

### Model Selection

Edit `attendance_system.py` or `api.py`:

```python
system = SimpleAttendanceSystem(
    dataset_path="dataset",
    model_name="Facenet",      # Options: Facenet, Facenet512, VGG-Face, ArcFace
    threshold=0.6              # Lower = stricter (0.3-0.7)
)
```

**Model Comparison:**

| Model | Speed | Accuracy | Recommended For |
|-------|-------|----------|----------------|
| **Facenet** | Fast | Good | 4-10 people |
| **Facenet512** | Medium | Excellent | 10-50 people |
| **VGG-Face** | Medium | Good | General use |
| **ArcFace** | Slow | Excellent | High accuracy needs |

### Threshold Tuning

- **0.3-0.4**: Very strict (fewer false positives)
- **0.5-0.6**: Balanced (recommended)
- **0.7-0.8**: Lenient (may have false positives)

### Port Configuration

Change API port in `api.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to your port
```

## Database Schema

### `persons` Table
```sql
CREATE TABLE persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `attendance` Table
```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    date DATE NOT NULL,
    time TIME NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id),
    UNIQUE(person_id, date)
);
```

## Deployment

### Production with Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

### Docker

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api:app"]
```

**Build and run:**
```bash
docker build -t attendance-system .
docker run -p 5000:5000 -v $(pwd)/dataset:/app/dataset attendance-system
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name attendance.yourdomain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Troubleshooting

### "No module named 'tf_keras'"
```bash
pip install tf-keras
```

### "Could not open webcam"
- Check if webcam is connected
- Ensure no other app is using the webcam
- Try changing camera index in code: `cv2.VideoCapture(1)`

### "No face detected"
- Ensure good lighting
- Face should be clearly visible
- Try adjusting threshold
- Check if `enforce_detection=True` is too strict

### Slow recognition
- Use lighter model: `Facenet` instead of `Facenet512`
- Reduce image quality
- Process fewer frames: Change `frame_count % 45` to `frame_count % 60`

### "Dataset path not found"
```bash
# Create dataset folder
mkdir dataset
mkdir dataset/person1
# Add images to person1 folder
```

### Database locked error
- Close other connections to `attendance.db`
- Restart the application

---
