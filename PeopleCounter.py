import numpy as np
import cv2
import cvzone
import time
from ultralytics import YOLO
from sort import *

# Load model and files
model = YOLO('../yolo-weights/yolov8l.pt')
cap = cv2.VideoCapture('../Videos/people.mp4')
mask = cv2.imread('mask-1.png')
graphics = cv2.imread("graphics-1.png", cv2.IMREAD_UNCHANGED)
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Lines for counting
limitUp = [103, 161, 296, 161]
limitDown = [527, 489, 735, 489]

# Counters
countUp, countDown = [], []

# Class names (COCO)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Drawing lines
def draw_lines(img):
    cv2.line(img, tuple(limitUp[:2]), tuple(limitUp[2:]), (0, 0, 255), 2)
    cv2.line(img, tuple(limitDown[:2]), tuple(limitDown[2:]), (0, 0, 255), 2)

# Check crossings
def check_crossing(img, cx, cy, track_id):
    tolerance = 15

    if limitUp[0] < cx < limitUp[2] and limitUp[1] - tolerance < cy < limitUp[1] + tolerance:
        if track_id not in countUp:
            countUp.append(track_id)
            cv2.line(img, tuple(limitUp[:2]), tuple(limitUp[2:]), (0, 255, 0), 3)

    if limitDown[0] < cx < limitDown[2] and limitDown[1] - tolerance < cy < limitDown[1] + tolerance:
        if track_id not in countDown:
            countDown.append(track_id)
            cv2.line(img, tuple(limitDown[:2]), tuple(limitDown[2:]), (0, 255, 0), 3)

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img_region = cv2.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, graphics, (730, 260))
    results = model(img_region, stream=True, verbose=False)

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if classNames[cls] == "person" and conf >= 0.3:
                current_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_detection))

    draw_lines(img)
    tracked_objects = tracker.update(detections)

    for x1, y1, x2, y2, track_id in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, (x1, y1, x2, y2, track_id))
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(img, (cx, cy), 2, (0, 150, 255), 2, cv2.FILLED)
        cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=2, colorR=(200, 200, 200), colorC=(100, 200, 255))
        check_crossing(img, cx, cy, track_id)

    # Display counts
    cv2.putText(img, str(len(countUp)), (929, 345), cv2.FONT_HERSHEY_SIMPLEX, 3, (120, 200, 50), 3)
    cv2.putText(img, str(len(countDown)), (1191, 345), cv2.FONT_HERSHEY_SIMPLEX, 3, (120, 200, 50), 3)

    cv2.imshow("frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f'Person_count_{int(time.time())}.jpg'
        cv2.imwrite(filename, img)
        print("Screenshot saved!")

cap.release()
cv2.destroyAllWindows()
