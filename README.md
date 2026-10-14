# Real-Time AI Face Detection

A real-time face detection project built using Python, OpenCV, and YOLOv8. The application detects faces from a webcam or video source and displays bounding boxes, confidence scores, face count, and FPS.

## Features
- Real-time face detection
- Webcam and video input support
- Bounding boxes around detected faces
- Confidence scores for each detection
- Real-time face count
- FPS display

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLOv8
- Computer Vision

## Requirements
- Python
- OpenCV
- Ultralytics
- YOLOv8 face detection model (`yolov8n-face.pt`)

## Installation

Install the required libraries:

```bash
pip install opencv-python ultralytics
```

## How to Run
1. Clone or download this repository.
2. Place `yolov8n-face.pt` in the project folder.
3. Run the Python script:

```bash
python face_detection.py
```

Replace `face_detection.py` with your actual Python filename if it differs.

Press **ESC** to close the application.

## Future Improvements
- Enhanced low-light face detection
- Face tracking across video frames
- Support for additional video sources

## Disclaimer
This project is intended for educational and computer vision experimentation. Use cameras and video feeds only with appropriate authorization.
