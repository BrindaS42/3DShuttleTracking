# Badminton Shuttlecock 3D Trajectory & Shot Classification

Reconstruct the full **X, Y, Z trajectory** of a badminton shuttlecock from a single monocular broadcast camera, then classify shot types using fused 2D+3D features.

---

## Requirements

- Python ≥ 3.10
- TrackNetV3 weights + `predict.py` → place in `tracknet_weights/`
- ShuttleSet dataset CSVs → place in `shuttleset/set/<match_name>/`
- Match videos → place in `test_assets/`

```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

```
test_assets/<match>.mp4
        │
        ├─► 01_trim_and_calibrate.py  ──► P.npy, trimmed_temp.mp4
        ├─► 02_detect_shuttle.py      ──► shuttle_trimmed.npy
        ├─► 03_estimate_pose.py       ──► poses.pkl
        ├─► 04_reconstruct_trajectory.py ──► traj_3d_trimmed.npy
        │
        ├─► 07_split_matchwise.py     ──► train/val/test_data.pkl
        └─► 08_5D_train_pipeline.py   ──► best_bst_5d_global.pt
```

---

## Modules

### Module 1 : Court Calibration (`01_trim_and_calibrate.py`)

Reads each match video from `test_assets/`, trims it to only the frames containing rally action (±60 frames around each hit), and runs an interactive calibration to compute the camera projection matrix **P** (3×4).

During calibration, you click **6 court landmarks** on a broadcast frame in this order:

| # | Point |
|---|-------|
| 1 | Near-Left corner |
| 2 | Near-Right corner |
| 3 | Far-Right corner |
| 4 | Far-Left corner |
| 5 | Left net post tip |
| 6 | Right net post tip |

Click the **inner edge** of white court lines, and the **very top** of the net post cap. The optimiser independently solves `fx`, `fy`, `cx`, `cy` : no square-pixel assumption : then polishes with Levenberg-Marquardt.

**Outputs:** `calib_out/P.npy`, `K.npy`, `rvec.npy`, `tvec.npy`, `trimmed_temp.mp4`

---

### Module 2 : Shuttle Detection (`02_detect_shuttle.py`)

Runs **TrackNetV3** on the trimmed video to detect the shuttlecock pixel position `(u, v)` in every frame. Raw detections are saved without any gap-filling so downstream modules can apply context-aware cleaning.

**Outputs:** `shuttle_out/shuttle_trimmed.npy` : shape `(N, 2)`, `NaN` where undetected

---

### Module 3 : Pose Estimation (`03_estimate_pose.py`)

Detects both players per frame using a custom YOLOv8 player detector + RTMPose keypoints. Each player's ankle pixels are back-projected onto the court floor (z = 0) using the calibrated P matrix to obtain 3D floor positions. Players are assigned *near* (lower in frame) and *far* (upper in frame) roles, with state reset at each rally boundary.

**Outputs:** `poses.pkl` : list of `PoseFrame` objects (one per trimmed frame), each with `near` and `far` player keypoints and floor positions

---

### Module 4 : 3D Trajectory Reconstruction (`04_reconstruct_trajectory.py`)

Reconstructs the full 3D trajectory for each shot using a **physics optimiser**. The shuttle is modelled under gravity and quadratic air drag:

```
d²x/dt² = g − Cd · ‖v‖² · v
```

The optimiser (L-BFGS-B) finds 7 parameters : initial position `x₀`, velocity `v₀`, drag `Cd` : by minimising:

```
L = σ·Lr  +  ‖x(0) − xH‖²  +  ‖x(tR) − xR‖²  +  dOut²
```

where `Lr` is reprojection error, `xH`/`xR` are hitter/receiver positions from Module 3, and `dOut` penalises landing outside the court.

**Coordinate system:** Origin = near-left corner | X = width (0→6.7 m) | Y = length (0→13.4 m) | Z = height

**Outputs:** `traj_out/<match>_traj_3d_trimmed.npy`, `_traj_2d_trimmed.npy`, overlay video

---

## Shot Classification Training

### Step 5 : Dataset Preparation (`07_split_matchwise.py`)

Validates all processed matches and performs an **80/10/10 match-wise split** (train/val/test) to prevent data leakage across matches. For each shot, a 100-frame window centred on the hit frame is extracted, containing:

- **Pose features:** 17 COCO keypoints + 19 bone vectors per player → `(100, 2, 72)`
- **2D shuttle:** normalised `(u, v)` → `(100, 2)`
- **3D shuttle:** world-space `(X, Y, Z)` → `(100, 3)`
- **Label:** 35 stroke types (side × shot type from ShuttleSet)

**Outputs:** `weights/dataset_cache/train_data.pkl`, `val_data.pkl`, `test_data.pkl`

### Step 6 : 5D Fusion Training (`08_5D_train_pipeline.py`)

Trains the **BST-CG-AP** model with fused 2D+3D shuttle input (`s5d` : 5 channels). Uses AdamW with cosine warmup scheduling, label smoothing, and early stopping (patience = 25).

Evaluation metrics: Top-1 Accuracy, Top-2 Accuracy, Macro-F1, Min-F1.

```bash
# Ensure bst_5d.py exists in stroke_classification/model/ before running
python 08_5D_train_pipeline.py
```

**Outputs:** `weights/checkpoints/best_bst_5d_global.pt`

---

## Running the Full Pipeline

```bash
python 01_trim_and_calibrate.py    # Trim videos + court calibration
python 02_detect_shuttle.py        # Shuttle detection
python 03_estimate_pose.py         # Player pose estimation
python 04_reconstruct_trajectory.py # 3D trajectory reconstruction
python 07_split_matchwise.py       # Prepare dataset splits
python 08_5D_train_pipeline.py     # Train shot classifier
```

---

## Reference

```bibtex
@inproceedings{liu2022monotrack,
  title  = {MonoTrack: Shuttle trajectory reconstruction from monocular badminton video},
  author = {Liu, Paul and Wang, Jui-Hsien},
  booktitle = {CVPR},
  year   = {2022}
}
```

```bibtex
@article{2502.21085,
  author = {Authors of BST},
  title = {BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports},
  journal = {arXiv preprint arXiv:2502.21085},
  year = {2025},
  url = {https://arxiv.org/abs/2502.21085}
}
```

**TrackNetV3**
> Huang et al., TrackNet: A Deep Learning Network for Tracking High-speed
> and Tiny Objects in Sports Applications. arXiv:1907.03698

**RTMPose**
> MMPose Contributors. OpenMMLab Pose Estimation Toolbox and Benchmark.
> https://github.com/openmmlab/mmpose, 2020.
