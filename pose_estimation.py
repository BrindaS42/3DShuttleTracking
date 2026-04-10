"""
module3_pose_estimation.py
══════════════════════════
Player pose estimation (RTMPose) → ankle pixels → 3-D floor positions
+ Stores full skeleton data for Module 5 (BST Stroke Classification).
"""

import argparse
import pickle
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    VIDEO_PATH, POSE_OUT, 
    PLAYER_HEIGHT, COURT_W as W, COURT_L as L,
)
from court_calibration import load_calibration, backproject_to_floor

@dataclass
class PlayerPose:
    left_ankle_px:  Optional[list] = None   # [u, v]
    right_ankle_px: Optional[list] = None   # [u, v]
    keypoints:      Optional[list] = None   # NEW: Full 17 COCO keypoints for BST
    bbox:           Optional[list] = None   # NEW: Player bounding box [x1, y1, x2, y2]
    floor_pos_3d:   Optional[list] = None   # [x, y, 0.0]
    body_pos_3d:    Optional[list] = None   # [x, y, PLAYER_HEIGHT/2]
    confidence:     float          = 0.0

@dataclass
class PoseFrame:
    frame_idx: int
    near: PlayerPose = field(default_factory=PlayerPose)
    far:  PlayerPose = field(default_factory=PlayerPose)

# COCO indices
_L_ANKLE, _R_ANKLE = 15, 16
_L_HIP,   _R_HIP   = 11, 12

def load_pose_backend():
    """Directly loads RTMPose-m."""
    try:
        import torch
        import mmdet, mmpose
        from mmpose.apis import init_model
        from mmdet.apis import init_detector
        from mmpose.registry import TRANSFORMS
        from mmdet.datasets.transforms import PackDetInputs
        
        TRANSFORMS.register_module(name='PackDetInputs', module=PackDetInputs, force=True)
        
        det_config = os.path.join(os.path.dirname(mmdet.__file__), '.mim', 'configs', 'rtmdet', 'rtmdet_tiny_8xb32-300e_coco.py')
        pose_config = os.path.join(os.path.dirname(mmpose.__file__), '.mim', 'configs', 'body_2d_keypoint', 'rtmpose', 'coco', 'rtmpose-m_8xb256-420e_coco-256x192.py')

        # ── DYNAMICALLY ASSIGN GPU ──
        device = "cuda" if torch.cuda.is_available() else "cpu"

        det = init_detector(det_config, "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth", device=device)
        
        pipeline = det.cfg.test_dataloader.dataset.pipeline
        det.cfg.test_dataloader.dataset.pipeline = [p for p in pipeline if 'LoadAnnotations' not in p.get('type', '')]
        
        pose = init_model(pose_config, "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth", device=device)
        return det, pose, "rtmpose"
    except Exception as e:
        print(f"[Pose] RTMPose Error: {e}")
        sys.exit(1)

def _add_3d(player: PlayerPose, K, rvec, tvec) -> PlayerPose:
    """
    Standardized ankle-to-floor mapping.
    """
    ankles = [px for px in [player.left_ankle_px, player.right_ankle_px] if px is not None]
    if not ankles: 
        return player

    u_avg, v_avg = np.mean(ankles, axis=0)
    fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
    
    if fp is not None:
        x, y = float(np.clip(fp[0], -1.0, W + 1.0)), float(np.clip(fp[1], -1.0, L + 1.0))
        player.floor_pos_3d = [x, y, 0.0]
        player.body_pos_3d  = [x, y, float(PLAYER_HEIGHT / 2.0)]
    return player

def estimate_poses(frames, K, rvec, tvec, det_m, pose_m) -> list:
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    results = []

    for fi, frame in enumerate(frames):
        pf = PoseFrame(fi)
        det_res = inference_detector(det_m, frame)
        boxes = det_res.pred_instances.bboxes.cpu().numpy()
        scores = det_res.pred_instances.scores.cpu().numpy()
        labels = det_res.pred_instances.labels.cpu().numpy()
        
        valid = (scores > 0.40) & (labels == 0)
        persons = boxes[valid]
        h, w = frame.shape[:2]
        mid_y = h / 2

        # PRESERVED: Original width filtering boundaries
        near_cands = [b for b in persons if (b[1]+b[3])/2 >= mid_y and w*0.15 < (b[0]+b[2])/2 < w*0.85]
        far_cands  = [b for b in persons if (b[1]+b[3])/2 < mid_y and w*0.15 < (b[0]+b[2])/2 < w*0.85]

        for candidates, p_attr in [(near_cands, 'near'), (far_cands, 'far')]:
            if candidates:
                best_box = max(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                res = inference_topdown(pose_m, frame, bboxes=best_box.reshape(1, 4))
                if res:
                    inst = res[0].pred_instances
                    kps, confs = inst.keypoints[0], inst.keypoint_scores[0]
                    p_obj = getattr(pf, p_attr)
                    
                    # DATA FOR BST: All joints and the bounding box
                    p_obj.keypoints = kps.tolist()
                    p_obj.bbox      = best_box.tolist()
                    
                    p_obj.confidence = float(np.mean(confs[[_L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE]]))
                    if confs[_L_ANKLE] > 0.3: p_obj.left_ankle_px = kps[_L_ANKLE].tolist()
                    if confs[_R_ANKLE] > 0.3: p_obj.right_ankle_px = kps[_R_ANKLE].tolist()

        pf.near = _add_3d(pf.near, K, rvec, tvec)
        pf.far  = _add_3d(pf.far, K, rvec, tvec)
        results.append(pf)
    return results

# Updated function in pose_estimation.py

def estimate_poses_batched(frames, K, rvec, tvec, det_m, pose_m, batch_size=8) -> list:
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    results = []
    total_frames = len(frames)
    
    # Process in batches to maximize GPU utilization
    for i in range(0, total_frames, batch_size):
        batch = frames[i : i + batch_size]
        print(f"[Pose] Processing batch {i//batch_size + 1}/{(total_frames+batch_size-1)//batch_size}...", end='\r')
        
        # 1. Batched Detection: MMDetection can take a list of frames
        det_results = inference_detector(det_m, batch)
        
        for j, frame in enumerate(batch):
            f_idx = i + j
            pf = PoseFrame(f_idx)
            
            # Extract detection data for this specific frame
            inst = det_results[j].pred_instances
            boxes = inst.bboxes.cpu().numpy()
            scores = inst.scores.cpu().numpy()
            labels = inst.labels.cpu().numpy()
            
            valid = (scores > 0.40) & (labels == 0)
            persons = boxes[valid]
            h, w = frame.shape[:2]
            mid_y = h / 2

            # Apply your specific width/height filtering
            near_cands = [b for b in persons if (b[1]+b[3])/2 >= mid_y and w*0.15 < (b[0]+b[2])/2 < w*0.85]
            far_cands  = [b for b in persons if (b[1]+b[3])/2 < mid_y and w*0.15 < (b[0]+b[2])/2 < w*0.85]

            for candidates, p_attr in [(near_cands, 'near'), (far_cands, 'far')]:
                if candidates:
                    best_box = max(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    # 2. Top-down pose inference
                    res = inference_topdown(pose_m, frame, bboxes=best_box.reshape(1, 4))
                    if res:
                        p_inst = res[0].pred_instances
                        kps, confs = p_inst.keypoints[0], p_inst.keypoint_scores[0]
                        p_obj = getattr(pf, p_attr)
                        p_obj.keypoints = kps.tolist()
                        p_obj.bbox      = best_box.tolist()
                        
                        # Indices for hips and ankles
                        p_obj.confidence = float(np.mean(confs[[11, 12, 15, 16]]))
                        if confs[15] > 0.3: p_obj.left_ankle_px = kps[15].tolist()
                        if confs[16] > 0.3: p_obj.right_ankle_px = kps[16].tolist()

            pf.near = _add_3d(pf.near, K, rvec, tvec)
            pf.far  = _add_3d(pf.far, K, rvec, tvec)
            results.append(pf)
            
    print(f"\n[Pose] Finished {total_frames} frames.")
    return results

def render_pose_debug(frames, poses, fps, out_path):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame, pf in zip(frames, poses):
        out = frame.copy()
        for p, clr, tag in [(pf.near, (0,255,0), "NEAR"), (pf.far, (0,0,255), "FAR")]:
            ankles = [px for px in [p.left_ankle_px, p.right_ankle_px] if px is not None]
            for px in ankles: cv2.circle(out, (int(px[0]), int(px[1])), 5, clr, -1)
            
            if p.floor_pos_3d and ankles:
                u_avg, v_avg = np.mean(ankles, axis=0)
                v_text = f"{tag}: ({int(u_avg)},{int(v_avg)})px -> ({p.floor_pos_3d[0]:.2f}, {p.floor_pos_3d[1]:.2f})m"
                cv2.putText(out, v_text, (int(u_avg)-50, int(v_avg)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        writer.write(out)
    writer.release()

def load_poses(pose_dir: str) -> list:
    path = Path(pose_dir) / "poses.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No poses found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def get_player_3d(poses: list, frame_idx: int, side: str, use_floor: bool = True) -> Optional[np.ndarray]:
    if frame_idx >= len(poses):
        return None
    pf = poses[frame_idx]
    player = pf.near if side == "near" else pf.far
    pos = player.floor_pos_3d if use_floor else player.body_pos_3d
    if pos is None:
        return None
    return np.array(pos, dtype=np.float64)

def save_poses(poses, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "poses.pkl", "wb") as f: pickle.dump(poses, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(VIDEO_PATH))
    ap.add_argument("--calib_dir", default="calib_out")
    ap.add_argument("--out_dir", default=str(POSE_OUT))
    args = ap.parse_args()

    P, K, rvec, tvec = load_calibration(args.calib_dir)
    cap = cv2.VideoCapture(args.video)
    fps, frames = cap.get(cv2.CAP_PROP_FPS) or 30.0, []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()

    det_m, pose_m, _ = load_pose_backend()
    poses = estimate_poses(frames, K, rvec, tvec, det_m, pose_m)
    save_poses(poses, Path(args.out_dir))
    render_pose_debug(frames, poses, fps, str(Path(args.out_dir) / "pose_debug.mp4"))