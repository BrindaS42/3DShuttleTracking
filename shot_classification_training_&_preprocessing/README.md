# MonoTrack — Shuttlecock 3-D Trajectory Reconstruction

Reconstruct the full **X, Y, Z trajectory** of a badminton shuttlecock from a **single monocular broadcast camera**, with no stereo rig and no manual annotation beyond a one-time court calibration.

Based on **MonoTrack** (Liu & Wang, CVPR 2022), extended with an independent `fx / fy` intrinsic optimiser, player tracking and a fully modular codebase.

---

## How it works

```
broadcast frame ──► Module 1 ──► P matrix (3×4)
                                      │
video clip ──────► Module 2 ──► shuttle_2d.npy  (u,v) per frame
                │                     │
                └────► Module 3 ──► poses.pkl    player 3-D positions
                                      │
              ┌───────────────────────┘
              ▼
          Module 4  ──  physics optimiser
              │         minimises: σ·Lr + ‖x(0)−xH‖² + ‖x(tR)−xR‖² + dOut²
              ▼
       trajectory_3d.npy   (N_frames × 3)  world-space X, Y, Z
       output_annotated.mp4
       trajectory_plots.png
       accuracy_report.txt
```

**Module 1** clicks 6 visible court landmarks on one static frame and solves the camera projection matrix P (3 × 4) via PnP with an independent `fx / fy / cx / cy` optimiser — no square-pixel assumption.

**Module 2** calls TrackNetV3 to detect the shuttle pixel position `(u, v)` in every frame, then applies light Gaussian smoothing and short-gap interpolation.

**Module 3** runs pose detection to find both players' ankle positions per frame, then back-projects them onto the court floor to get 3-D player positions used as physics priors.

**Module 4** integrates a gravity + quadratic-drag ODE and optimises 7 parameters — initial position `x₀`, initial velocity `v₀`, and drag coefficient `Cd` — by minimising the reprojection loss plus player-position and out-of-court penalties.

---

## Project structure

```
project/
│
├── config.py                      ← edit once: all paths + court constants
│
├── court_calibration.py   ← interactive P-matrix estimation
├── shuttle_detection.py   ← TrackNetV3 wrapper + post-processing
├── pose_estimation.py     ← player tracking
├── trajectory.py          ← physics optimiser + annotated video
│
├── run_pipeline.py                ← one-command orchestrator
├── requirements.txt
│
├── tracknet_weights/             
│   ├── predict.py
│   ├── tracknet_best.pt
│   └── inpaintnet_best.pt
│
├── frame.jpg                      ← one clear broadcast frame (for M1)
├── clip.mp4                       ← your shot video (for M2–M4)
│
├── calib_out/                     ← created by M1
│   ├── P.npy
│   ├── K.npy
│   ├── rvec.npy
│   ├── tvec.npy
│   └── calib_result.png
│
├── shuttle_out/                   ← created by M2
│   ├── shuttle_2d.npy
│   ├── shuttle_detection.mp4
│   └── detection_stats.txt
│
├── pose_out/                      ← created by M3
│   ├── poses.pkl
│   └── pose_debug.mp4
│
└── traj_out/                      ← created by M4
    ├── trajectory_3d.npy
    ├── output_annotated.mp4
    ├── trajectory_plots.png
    └── accuracy_report.txt
```

---

## Requirements

- **Python ≥ 3.10**
- **TrackNetV3** weights + `predict.py` (obtain from the [TrackNetV3 repo](https://github.com/qaz812345/TrackNetV3))
- A clear static broadcast frame (`frame.jpg`) for the one-time calibration
- A video clip of the shot or rally (`clip.mp4`)

---

## Installation

### Step 1 — clone and install core dependencies

```bash
git clone <your-repo-url>
cd monotrack
pip install -r requirements.txt
```
>Reaname tha name as tracknet_weights & add 

```bash
cp /path/to/tracknet_best.pt    tracknet_weights/
cp /path/to/inpaintnet_best.pt  tracknet_weights/
cp /path/to/predict.py          tracknet_weights/
```

### Step 2 — install PyTorch (TrackNetV3 needs it)

Pick the build matching your hardware from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick start

### Run everything end-to-end

```bash
python run_pipeline.py \
    --video  clip.mp4 \
    --image  frame.jpg \
    --hitter near
```

### Re-run only the trajectory optimiser (modules 1–3 already done)

```bash
python run_pipeline.py \
    --video  clip.mp4 \
    --hitter near \
    --only_traj
```

### Run each module individually

```bash
# M1 — one-time calibration on any clear broadcast frame
python module1_court_calibration.py --image frame.jpg --out_dir calib_out

# M2 — shuttle detection on the shot clip
python module2_shuttle_detection.py --video clip.mp4 \
    --tracknet tracknet_weights --out_dir shuttle_out

# M3 — player pose estimation
python module3_pose_estimation.py --video clip.mp4 \
    --calib_dir calib_out --out_dir pose_out

# M4 — 3-D trajectory reconstruction
python module4_trajectory.py --video clip.mp4 \
    --calib_dir  calib_out \
    --shuttle_dir shuttle_out \
    --pose_dir   pose_out \
    --hitter     near \
    --hit_frame  0 \
    --out_dir    traj_out
```

---

## Module reference

### Module 1 — Court calibration

**Script:** `module1_court_calibration.py`

Opens an interactive matplotlib window on a broadcast frame. You click 6 landmarks in order, then the module solves the 3 × 4 projection matrix P.

**The 6 annotation points**

| # | Name | Where to click |
|---|------|----------------|
| 1 | Near-Left corner | Inner-edge intersection of the white lines at the near-left baseline corner |
| 2 | Near-Right corner | Same, near-right |
| 3 | Far-Right corner | Inner-edge intersection at the far-right baseline corner |
| 4 | Far-Left corner | Same, far-left |
| 5 | Left net post tip | Top-centre of the yellow pole cap on the left |
| 6 | Right net post tip | Top-centre of the yellow pole cap on the right |

> **Click the inner edge of white court lines**, not the outer edge. Lines are ~40 mm wide; outer vs inner is a 3–5 px annotation error that propagates into Z.
>
> **For net posts**, zoom in first with the toolbar. Click the very top of the metal cap — NOT the net tape, which sits 2–3 cm lower.

**How the intrinsics optimiser works**

Standard calibration assumes square pixels (`fx = fy`). Broadcast telephoto lenses rarely satisfy this. Even a 0.3% aspect difference causes ~5 px systematic error on off-floor points (the net posts). The optimiser uses `scipy.optimize.minimize` with L-BFGS-B to find `fx`, `fy`, `cx`, `cy` independently, then runs a Levenberg-Marquardt sub-pixel polish with `cv2.solvePnPRefineLM`.

**Outputs**

| File | Description |
|------|-------------|
| `calib_out/P.npy` | 3 × 4 projection matrix (float64, normalised so P[2,3] = 1) |
| `calib_out/K.npy` | 3 × 3 intrinsic matrix |
| `calib_out/rvec.npy` | 3 × 1 rotation vector |
| `calib_out/tvec.npy` | 3 × 1 translation vector |
| `calib_out/calib_result.png` | Visual overlay: clicks vs reprojections + court wireframe |

**Expected reprojection error**

| Error | Quality |
|-------|---------|
| < 2 px | Excellent |
| 2–5 px | Good |
| 5–10 px | Acceptable |
| > 10 px | Re-annotate |

---

### Module 2 — Shuttle detection

**Script:** `module2_shuttle_detection.py`

Wraps TrackNetV3's `predict.py` as a subprocess, parses the output CSV into a clean `(N_frames, 2)` NumPy array, applies post-processing, and saves for Module 4.

**Post-processing steps**

1. **Gaussian smoothing** (`σ = 1.2`) on detected positions only — reduces sub-pixel jitter from TrackNet without affecting NaN (undetected) frames.
2. **Gap interpolation** — linearly fills gaps of ≤ 5 consecutive undetected frames. Longer gaps (occlusion, shuttle out of frame) are left as NaN and handled correctly by the physics optimiser.

**Detection quality guide**

| Detection rate | Impact on Z reconstruction |
|---------------|---------------------------|
| > 80% | Excellent |
| 60–80% | Good |
| < 60% | Poor — Z accuracy will degrade |

**Outputs**

| File | Description |
|------|-------------|
| `shuttle_out/shuttle_2d.npy` | `(N_frames, 2)` float64; `NaN` = undetected |
| `shuttle_out/shuttle_detection.mp4` | Debug video with green dot + amber trail |
| `shuttle_out/detection_stats.txt` | Detection rate and quality grade |

---

### Module 3 — Pose estimation

**Script:** `pose_estimation.py`

Estimates both players' positions per frame and back-projects their averaged ankle pixels onto the court floor plane (z = 0) using the calibrated P matrix.

**Backend selection (automatic)**

The module tries backends in this order and uses the first available:

| Priority | Backend | Notes |
|----------|---------|-------|
| 1 | RTMPose-m (mmpose) | Best accuracy. Needs `mim install`. |
| 2 | MediaPipe Pose | CPU-friendly. `pip install mediapipe`. |
| 3 | Disabled | Player priors turned off in M4. Still works. |

Override with `--backend rtmpose|mediapipe|auto|none`.

**Why ankle keypoints?**

The ankle is the closest visible body part to the floor. Back-projecting the average of both ankles to z = 0 gives the player's floor position. Adding `PLAYER_HEIGHT / 2` (default 1.75 m / 2 = 0.875 m) gives the body centre, which the physics optimiser uses as a prior for where the shuttle starts and ends each shot.

**Player assignment** (near vs far) is determined by vertical pixel position: the player with the larger y-coordinate (lower in the frame) is the near player.

**Outputs**

| File | Description |
|------|-------------|
| `pose_out/poses.pkl` | `list[PoseFrame]` — one entry per video frame with ankle pixels and 3-D floor positions |
| `pose_out/pose_debug.mp4` | Video with ankle circles (green = near, blue = far) and court-coordinate labels |

---

### 3-D trajectory reconstruction

**Script:** `trajectory.py`

The core of the pipeline. Reconstructs the full 3-D trajectory by solving a constrained nonlinear optimisation problem over the physics of shuttle flight.

**Physics model**

The shuttle is modelled as a particle under gravity and quadratic air drag:

```
d²x/dt² = g − Cd · ‖v‖² · v
```

where `g = (0, 0, −9.81) m/s²` and `Cd` is the drag coefficient (optimised per shot since feathers degrade during a rally).

**Optimisation**

The optimiser (L-BFGS-B → Levenberg-Marquardt polish) finds 7 parameters:

```
params = [x₀(3),  v₀(3),  log(Cd)]
```

by minimising the full MonoTrack loss (paper Eq. 4):

```
L = σ · Lr  +  ‖x(0) − xH‖²  +  ‖x(tR) − xR‖²  +  dOut²
```

| Term | Meaning |
|------|---------|
| `Lr` | Mean squared reprojection error: project 3-D trajectory back to 2-D and compare against TrackNet detections |
| `‖x(0) − xH‖²` | Hitter position prior from Module 3 |
| `‖x(tR) − xR‖²` | Receiver position prior from Module 3 |
| `dOut²` | Distance the shuttle lands outside the court (0 if it lands inside) |
| `σ = 1/‖P‖²` | Scale factor that balances pixel-space and world-space terms |

**Constraints enforced as soft penalties**

- Initial height `z₀ ∈ [0, 3.0]` m
- Initial speed `‖v₀‖ ≤ 120 m/s` (≈ 432 kph)
- Starting position on the hitter's half of the court
- Initial velocity directed toward the opponent

**Outputs**

| File | Description |
|------|-------------|
| `traj_out/trajectory_3d.npy` | `(N_frames, 3)` float64 — world-space X, Y, Z per frame |
| `traj_out/output_annotated.mp4` | Original video with X/Y/Z overlay + projected amber trail |
| `traj_out/trajectory_plots.png` | 4-panel: 3-D view, bird-eye (colour = height), Z over time, reprojection error |
| `traj_out/accuracy_report.txt` | Full numeric report: error stats, speed, net clearance, quality grade |

**Coordinate system**

```
Origin: near-left corner of the doubles court

X ──►  court width   (0 → 6.7 m)
Y ──►  court length  (0 → 13.4 m)
Z  ↑   height above floor  (0 = floor)
```

---

## Reference

```bibtex
@inproceedings{liu2022monotrack,
  title     = {MonoTrack: Shuttle trajectory reconstruction from
               monocular badminton video},
  author    = {Liu, Paul and Wang, Jui-Hsien},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

**TrackNetV3**
> Huang et al., TrackNet: A Deep Learning Network for Tracking High-speed
> and Tiny Objects in Sports Applications. arXiv:1907.03698

**RTMPose**
> MMPose Contributors. OpenMMLab Pose Estimation Toolbox and Benchmark.
> https://github.com/openmmlab/mmpose, 2020.