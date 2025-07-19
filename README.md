# Exam-Cheating-Detection-Application-Using-Python
AI-powered exam proctoring system using Python, MediaPipe, YOLOv8, and OpenCV. Detects cheating behavior in real-time including looking away, phone usage, and multiple faces.
# Exam Cheating Detection Application 📷🎓

A real-time cheating detection application built using Python, OpenCV, MediaPipe, and YOLOv8.  
It monitors students during online exams and flags suspicious behaviors like:

- Looking away from the screen
- Using a phone
- Presence of multiple faces and people

## 💡 Features
- Head pose estimation using MediaPipe Face Mesh
- YOLOv8 object detection for phone/book detection
- Face detection to flag multiple people in frame
- Auto-calibration before session starts
- Live percentage summary of misconduct events

## 📦 Requirements
Install dependencies using:
pip install -r requirements.txt
