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

## 📂 Folder Structure

```
PeopleFlow/
│
├── peopleCounter.py         # Main Python script
├── mask-1.png               # Binary mask for region of interest
├── graphics-1.png           # UI overlay (semi-transparent)
├── sort.py                  # SORT tracking algorithm (linked)
├── demo.gif                 # Side-by-side demo (with/without detection)
└── README.md
```

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

## 🖼️ Demo

<p align="center">
  <img src="demo.gif" width="600"/>
</p>

Left: Original | Right: Detection + Count overlay

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

## 📎 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [SORT](https://github.com/abewley/sort)
* [cvzone](https://github.com/cvzone/cvzone)


