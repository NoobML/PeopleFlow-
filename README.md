# PeopleFlow 👣

**Real-time People Counting using YOLOv8 + SORT tracking**

Detects and counts people crossing virtual lines in a video using deep learning and object tracking.

---

## 🎯 Features

* Detects **people** using YOLOv8.
* Tracks them using **SORT algorithm**.
* Counts people moving **up and down** across defined lines.
* Supports **video input**, **custom mask**, and **graphics overlay**.
* Saves **screenshots** and shows live count.

---

## 🖼️ Demo
### Orignial
![People](people.gif)

### Detection
![People Detection](PeopleDetection.gif)
---


## 🧠 Model Info

* **Model**: YOLOv8 (large)
* **Detection Class**: `"person"` only
* **Tracking**: SORT (Simple Online and Realtime Tracker)
* **Line Crossing**: Custom-defined up/down lines with `tolerance`

---

## 📸 Controls

* Press `q`: Quit
* Press `s`: Save screenshot (`Person_count_<timestamp>.jpg`)

---

## 🔧 Requirements

* Python 3.8+
* OpenCV
* NumPy
* [Ultralytics](https://github.com/ultralytics/ultralytics) (for YOLOv8)
* cvzone

Install:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repo and navigate into it.
2. Place your **video file** in a folder (`./Videos/people.mp4`).
3. Add:

   * `mask-1.png`
   * `graphics-1.png`
4. Download [YOLOv8 weights](https://github.com/ultralytics/assets/releases) and place them in `./yolo-weights/yolov8l.pt`.
5. Download `sort.py` from [abewley/sort](https://github.com/abewley/sort) or use your own version.
6. Run:

```bash
python peopleCounter.py
```


---

## 🖼️ Visual Assets

### 🎯 Mask Used

![Mask](mask-1.png)

### ✨ Graphics Overlay

![Graphics](graphics-1.png)

---


PeopleCounter.py

PeopleDetection.gif
README.md

graphics-1.png

mask-1.png
people.gif
people.mp4
sort.py

## 📂 Folder Structure

```
PeopleFlow/
│
├── peopleCounter.py         # Main script: handles detection, tracking & counting
├── PeopleDetection.gif      # Sample GIF showing people detection in action
├── README.md                # Project overview, setup, and usage instructions
├── graphics-1.png           # UI overlay for display (e.g., zone borders, count text)
├── mask-1.png               # Binary mask for selecting region of interest (ROI)
├── People.gif               # Animated demo of people counting
├── People.mp4               # Video demo of people counting
├── sort.py                  # SORT algorithm for tracking detected people



```


## 📎 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [SORT](https://github.com/abewley/sort)
* [cvzone](https://github.com/cvzone/cvzone)


