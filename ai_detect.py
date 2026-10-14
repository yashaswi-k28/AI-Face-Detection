import cv2
from ultralytics import YOLO
import time

# =========================
# CONFIGURATION
# =========================

# Choose input source:
# 0 -> Webcam
# "video.mp4" -> Video file
# "rtsp://user:pass@ip:port/stream" -> CCTV
VIDEO_SOURCE = 1

CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =========================
# LOAD MODEL
# =========================
print("[INFO] Loading YOLOv8 Face Model...")
model = YOLO("yolov8n-face.pt")

# =========================
# VIDEO CAPTURE
# =========================
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("[ERROR] Unable to access video source")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Improve low-light performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # YOLO Face Detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    face_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            face_count += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(frame, f"Face {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
    prev_time = curr_time

    # Display Info
    cv2.putText(frame, f"Faces: {face_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.putText(frame, f"FPS: {fps}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("AI Face Detection - Real Time", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Application Closed")
