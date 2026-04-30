# Badminton Shot Recognition & Refinement (SRA) Pipeline

This repository implements the **Shot Refinement Algorithm (SRA)** to detect hit moments in badminton broadcast videos. It combines **HD-A** (Heuristic or YOLO-based Action detection) with **HD-T** (Trajectory-based hit detection) to produce precise temporal shot intervals.
The dataset used is ShuttleSet, same as the Shot Classification algorithm.

---

## Requirements

- Python ≥ 3.10
- `ultralytics` (for YOLOv8 pipeline)
- `opencv-python`, `pandas`, `scipy`, `matplotlib`, `tqdm`
- Pre-computed Pose (`.pkl`) and Shuttle (`.npy`) data[cite: 6, 7]

```bash
pip install ultralytics opencv-python pandas scipy matplotlib tqdm
```

---

## Pipeline Architectures

The system supports two pipelines for **HD-A** (Action Detection):

### 1. Heuristic Pipeline (`main.py`)
Uses hand-crafted pose features to detect swings. It calculates four primary signals from keypoints:
*   **Wrist Acceleration:** Frame-to-frame velocity gradients.[cite: 4]
*   **Arm Extension:** Shoulder-elbow-wrist angles.[cite: 4]
*   **Ankle Displacement:** Detects foot-planting/lunging.[cite: 4]
*   **Bbox Area Rate:** Identifies explosive lunges or jumps.[cite: 4]

### 2. YOLO Pipeline (`main_2.py`)
Uses a trained **YOLOv8-cls** model to classify player crops into `swing` or `not_swing`. It has a **Dual-Stage Peak Suppression** strategy to find hit moments by using both YOLO detections and trajectory peaks.[cite: 2, 5]

---

## Core Modules

### Shot Refinement Algorithm (`SRA`)
It resolves detections using the following:
*   **Segment Building:** Merges consecutive action detections by the same player.[cite: 4, 5]
*   **Intersection:** Matches action segments against **HD-T** trajectory peaks.[cite: 4, 5]
*   **Refinement:** If multiple peaks exist in a segment, it picks the one closest to the action confidence peak; if no peak exists, it uses the action peak itself.[cite: 4, 5]

### Trajectory Smoothing (`HD-T`)
Uses a three-step process to clean raw shuttle detections. This is the same method used in shuttle tracking.

---

## Training the YOLO Classifier (`training.py`)

To use the YOLO pipeline, you must first train the swing classifier:

1.  **Dataset Construction:** Extracts positive crops (centered on ground-truth hits) and negative crops (random rally movement).[cite: 6]
2.  **Fine-tuning:** Trains `yolov8s-cls` for 30 epochs on the extracted crops.[cite: 6]

```bash
python training.py
```

---

## Execution & Evaluation

### Running Inference
Run either the heuristic or YOLO pipeline to generate predictions and metrics.

```bash
# For Heuristic Features
python heuristic_main.py

# For YOLO-based Detection
python yolo_main.py
```

### Metrics & Reporting
The pipeline evaluates performance using **t-IoU (Temporal Intersection over Union)** at thresholds of **0.5, 0.85, and 0.95**.[cite: 2]
*   **Precision/Recall/F1:** Calculated based on matched shot intervals.
*   **Outputs:** Results are saved to `results/` as CSV logs, PNG plots, and a `predictions.json` file.[cite: 1, 2]

---

## Data Structure required

```text
data/
├── MATCH_DB/                 # Ground Truth (ShuttleSet format)
│   └── <match_name>/
│       └── set1.csv          # Columns: rally, ball_round, time, frame_num...
└── PKL_ROOT/                 # Pre-computed features
    └── <match_name>/
        ├── pose_out/
        │   └── poses_trimmed.pkl
        └── shuttle_out/
            └── shuttle_trimmed.npy
```[cite: 6, 7]