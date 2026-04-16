"""
module3_pose_estimation.py
══════════════════════════
Player pose estimation (RTMPose).
Saves RAW 2D pixels (keypoints, bounding boxes, ankle coords) to .pkl.
Calculates 3D floor positions ON-THE-FLY only when explicitly requested.
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
    keypoints:      Optional[list] = None   # Full 17 COCO keypoints for BST
    bbox:           Optional[list] = None   # Player bounding box [x1, y1, x2, y2]
    floor_pos_3d:   Optional[list] = None   # [Legacy/Optional Cache]
    body_pos_3d:    Optional[list] = None   # [Legacy/Optional Cache]
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

def get_iou(bb1, bb2):
    """Calculates Intersection over Union for tracking persistence."""
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return intersection_area / float(bb1_area + bb2_area - intersection_area)

def is_valid_player(kpts, confs, bbox, court_poly, last_bbox=None):
    """
    Refined logic:
    - If we have a 'last_bbox', we prioritize overlap (IOU) even if they leave the court.
    - If NO 'last_bbox' (starting fresh), they MUST have ankles inside the court.
    """
    if court_poly is None: return True
    
    # 1. Check for temporal persistence (Tracking during jumps)
    if last_bbox is not None:
        if get_iou(bbox, last_bbox) > 0.15: # Overlap check
            return True
        # If no overlap, check if they are very close (fast movement)
        c1 = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        c2 = [(last_bbox[0]+last_bbox[2])/2, (last_bbox[1]+last_bbox[3])/2]
        if np.linalg.norm(np.array(c1) - np.array(c2)) < 150:
            return True

    # 2. Seeding logic: If not tracked, they must be on the court floor
    for idx in [15, 16]: 
        if confs[idx] > 0.3:
            pt = (float(kpts[idx][0]), float(kpts[idx][1]))
            if cv2.pointPolygonTest(court_poly, pt, False) >= 0:
                return True
    return False

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
                    
                    # Store RAW 2D Data only
                    p_obj.keypoints = kps.tolist()
                    p_obj.bbox      = best_box.tolist()
                    
                    p_obj.confidence = float(np.mean(confs[[_L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE]]))
                    if confs[_L_ANKLE] > 0.3: p_obj.left_ankle_px = kps[_L_ANKLE].tolist()
                    if confs[_R_ANKLE] > 0.3: p_obj.right_ankle_px = kps[_R_ANKLE].tolist()

        # REMOVED: pf.near = _add_3d(...)  -> Keeping .pkl strictly 2D
        results.append(pf)
    return results


def estimate_poses_batched(frames, court_poly, K, rvec, tvec, det_m, pose_m, batch_size=32) -> list:
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    results = []
    
    # Persistence memory
    last_near = None # Stores last [x1, y1, x2, y2]
    last_far = None

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        det_results = inference_detector(det_m, batch)
        
        for j, frame in enumerate(batch):
            pf = PoseFrame(i + j)
            inst = det_results[j].pred_instances
            valid = (inst.scores.cpu().numpy() > 0.45) & (inst.labels.cpu().numpy() == 0)
            bboxes = inst.bboxes.cpu().numpy()[valid]
            
            if len(bboxes) == 0:
                results.append(pf); continue

            pose_res = inference_topdown(pose_m, frame, bboxes=bboxes)
            candidates = []
            for k, res in enumerate(pose_res):
                kpts, confs = res.pred_instances.keypoints[0], res.pred_instances.keypoint_scores[0]
                
                # Check if this detection matches either the 'Near' or 'Far' player history
                is_p_near = is_valid_player(kpts, confs, bboxes[k], court_poly, last_near)
                is_p_far  = is_valid_player(kpts, confs, bboxes[k], court_poly, last_far)

                if is_p_near or is_p_far:
                    candidates.append({
                        'kpts': kpts.tolist(), 'bbox': bboxes[k].tolist(),
                        'conf': float(np.mean(confs[[11, 12, 15, 16]]))
                    })

            # --- SMART ASSIGNMENT (NOT SORTING) ---
            if candidates:
                # 1. Assign Near: Find candidate closest to last_near OR highest Y
                if last_near is not None:
                    c_near = max(candidates, key=lambda x: get_iou(x['bbox'], last_near))
                else:
                    c_near = max(candidates, key=lambda x: x['bbox'][3]) # Highest Y-bottom
                
                pf.near.keypoints, pf.near.bbox = c_near['kpts'], c_near['bbox']
                last_near = c_near['bbox']
                
                # Remove assigned Near from Far consideration
                candidates = [c for c in candidates if c['bbox'] != c_near['bbox']]

            if candidates:
                # 2. Assign Far: Find candidate closest to last_far OR lowest remaining Y
                if last_far is not None:
                    # Prefer overlap with previous Far position to ignore static background people
                    c_far = max(candidates, key=lambda x: get_iou(x['bbox'], last_far))
                else:
                    c_far = min(candidates, key=lambda x: x['bbox'][3]) # Lowest Y-bottom
                
                if last_far is None or get_iou(c_far['bbox'], last_far) > 0:
                    pf.far.keypoints, pf.far.bbox = c_far['kpts'], c_far['bbox']
                    last_far = c_far['bbox']

            results.append(pf)
    return results


def render_pose_debug(frames, poses, fps, out_path, court_poly=None, K=None, rvec=None, tvec=None):
    """
    Improved debug renderer to verify spatial filtering.
    - Draws the 4-corner court polygon.
    - Draws bounding boxes for Near/Far players.
    - Displays 3D world coordinates if calibration is provided.
    """
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    for i, (frame, pf) in enumerate(zip(frames, poses)):
        out = frame.copy()
        
        # 1. Draw the Court Polygon (The 'Allowed' Zone)
        if court_poly is not None:
            cv2.polylines(out, [court_poly.astype(np.int32)], True, (0, 255, 255), 2)

        # 2. Draw Players
        for p, clr, tag in [(pf.near, (0, 255, 0), "NEAR"), (pf.far, (0, 0, 255), "FAR")]:
            if p.bbox is not None:
                # Draw Bounding Box
                x1, y1, x2, y2 = map(int, p.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), clr, 2)
                cv2.putText(out, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
                
                # Draw Ankles
                ankles = [px for px in [p.left_ankle_px, p.right_ankle_px] if px is not None]
                for px in ankles:
                    cv2.circle(out, (int(px[0]), int(px[1])), 5, clr, -1)
                
                # Draw 3D Position Text
                if ankles and K is not None and rvec is not None and tvec is not None:
                    u_avg, v_avg = np.mean(ankles, axis=0)
                    fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
                    if fp is not None:
                        v_text = f"({fp[0]:.2f}, {fp[1]:.2f})m"
                        cv2.putText(out, v_text, (int(u_avg)-30, int(v_avg)+25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                    
        writer.write(out)
    writer.release()


def load_poses(pose_dir: str) -> list:
    path = Path(pose_dir) / "poses.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No poses found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_player_3d(poses: list, frame_idx: int, side: str, K=None, rvec=None, tvec=None, use_floor: bool = True) -> Optional[np.ndarray]:
    """
    Retrieves the player's 2D ankle positions and dynamically calculates the 3D projection.
    Requires K, rvec, and tvec to perform the on-the-fly math.
    """
    if frame_idx >= len(poses):
        return None
    pf = poses[frame_idx]
    player = pf.near if side == "near" else pf.far

    # If camera calibration matrices are provided, calculate 3D on the fly
    if K is not None and rvec is not None and tvec is not None:
        ankles = [px for px in [player.left_ankle_px, player.right_ankle_px] if px is not None]
        if ankles:
            u_avg, v_avg = np.mean(ankles, axis=0)
            fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
            
            if fp is not None:
                x = float(np.clip(fp[0], -1.0, W + 1.0))
                y = float(np.clip(fp[1], -1.0, L + 1.0))
                
                if use_floor:
                    return np.array([x, y, 0.0], dtype=np.float64)
                else:
                    return np.array([x, y, float(PLAYER_HEIGHT / 2.0)], dtype=np.float64)
                    
    # Fallback to legacy behavior if the object already had it computed
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
    ap.add_argument("--calib_dir", default="results/calib_out")
    ap.add_argument("--out_dir", default=str(POSE_OUT))
    args = ap.parse_args()

    # 1. Load Calibration and generate the 2D Court Polygon
    # We use the first 4 WORLD_PTS (the floor corners) defined in config.py
    from court_calibration import load_calibration, project_to_pixel
    from config import WORLD_PTS 
    
    P, K, rvec, tvec = load_calibration(args.calib_dir)
    # Project 3D floor corners (0,0,0) to 2D pixels (u,v)
    court_poly = project_to_pixel(WORLD_PTS[:4], P).astype(np.int32)
    
    # 2. Load Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()

    # 3. Run Batched Estimation with the Persistence Logic
    det_m, pose_m, _ = load_pose_backend()
    
    # Pass the court_poly to filter out spectators behind the baseline
    poses = estimate_poses_batched(
        frames, 
        court_poly, 
        K, rvec, tvec, 
        det_m, pose_m, 
        batch_size=32
    )
    
    # 4. Save and Render
    out_path = Path(args.out_dir)
    save_poses(poses, out_path)
    
    print(f"Rendering debug video with polygon verification to {out_path}...")
    render_pose_debug(
        frames, 
        poses, 
        fps, 
        str(out_path / "pose_debug_verified.mp4"), 
        court_poly=court_poly, 
        K=K, rvec=rvec, tvec=tvec
    )