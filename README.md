# Monocular Badminton Analytics System

This is an end-to-end deep learning and physics-informed system designed to extract professional-grade 3D analytics from single-camera badminton footage. By combining computer vision, physics modeling, and sequence learning, it enables detailed analysis of shuttle trajectories, player positioning, and stroke classification without requiring specialized multi-camera setups.

---

## Objective

The goal of this project is to make advanced sports analytics accessible without expensive infrastructure. Traditional systems such as multi-camera tracking solutions are costly and complex. CourtClip achieves comparable analytical depth using a single broadcast video feed, providing:

- 3D shuttle trajectory reconstruction  
- Player movement and positioning insights  
- Automated shot segmentation and classification
  
---

## System Overview

The application is built as a pipeline of four interconnected modules:

### 1. Detection and Pose Tracking
- Court detection and geometric calibration using Roboflow workflows  
- Extraction of player skeletal keypoints using RTMPose via Hugging Face APIs & trained model to detect player only at the court
- Estimation of camera parameters (intrinsics and extrinsics) using detected court landmarks  

### 2. Shuttlecock Tracking
- Dedicated shuttle detection model based on TrackNet architecture  
- Robust tracking under high speed, motion blur and occlusion scenarios  

### 3. Physics-Informed 3D Reconstruction

The system reconstructs shuttle motion in 3D space by mapping 2D detections into world coordinates using a physics-based model:

- Incorporates gravity and quadratic air drag  
- Uses nonlinear optimization to fit trajectories  
- Produces physically consistent 3D flight paths  

### 4. Stroke Classification (5D Fusion Model)
- Transformer-based architecture (BST-CG)  
- Combines:
  - 2D spatial features  
  - 3D trajectory data  
- Improves classification accuracy using multimodal input  

---

## Installation and Setup

### Prerequisites
- Python 3.10+
- Roboflow API Key
- Hugging Face API Token

### Setup
```bash
git clone https://github.com/BrindaS42/3DShuttleTracking.git
cd 3DShuttleTracking
cd full_webapp_deployed
pip install -r requirements.txt
```

### Environment Configuration
Create a .env file in the root directory:
```bash
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_WORKFLOW_ID=your_workflow
HF_TOKEN=your_hugging_face_token
HF_SHUTTLE_SPACE = briii6/shuttle_detactor
```
### Run Application 
```bash
gradio app.py
```
The application will be available at:
http://127.0.0.1:7860

---

## 🚀 How to Use

- Step 1: Detection & Pose
Upload your video. Click "Run Heavy Analysis". The system will automatically:
Calibrate the court and calculate reprojection errors.  
Track the shuttlecock and both players (Near/Far).

- Step 2: Shot Segmentation
Define the boundaries of each shot. You can use the Manual Navigator slider to find the exact frame where a racket hits the shuttle and click "Mark Current Frame as Hit".  

- Step 3: Trajectory Estimation
Click "Estimate Trajectory". The physics engine will solve for the 3D flight path between the marked hits, providing a 3D visualization and error report (in pixels).  

- Step 4: Shot Classification
Run the "Analyze Shot Types" task. The system utilizes the 5D fusion model to identify the stroke type for every segmented interval in the rally.

---

## Technical Stack

- Frontend/UI: Gradio, Huggingface_hub
- Computer Vision: OpenCV, Roboflow Inference SDK
- Deep Learning: PyTorch, RTMPose, TrackNetV3
- Optimization: SciPy (L-BFGS-B)

--- 

## Supported Stroke Types
- Net Shot
- Return Net
- Smash
- Wrist Smash
- Lob
- Defensive Return Lob
- Clear
- Drive
- Back-court Drive
- Drop
- Passive Drop
- Push
- Rush (Kill)
- Defensive Return Drive
- Cross-court Net Shot
- Short Service
- Long Service
- Unknown
