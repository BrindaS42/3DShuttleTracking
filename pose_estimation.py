"""
pose_estimation.py
══════════════════
Player pose estimation (RTMPose) gated by a custom YOLOv8 badminton-player detector.

Pipeline per frame:
  1. RTMDet                   →  finds all person boxes (players + umpires + crowd)
  2. Custom YOLOv8 (best.pt)  →  finds ONLY active badminton players
  3. Box Matching             →  Finds the RTMDet box that matches the YOLOv8 player box
  4. RTMPose                  →  estimates pose on the matched RTMDet boxes
  5. Near / Far assignment    →  same court-poly + persistence logic as before
"""

import argparse
import pickle
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    VIDEO_PATH, POSE_OUT,
    PLAYER_HEIGHT, COURT_W as W, COURT_L as L,
)
from court_calibration import load_calibration, backproject_to_floor

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerPose:
    left_ankle_px:  Optional[list] = None   # [u, v]
    right_ankle_px: Optional[list] = None   # [u, v]
    keypoints:      Optional[list] = None   # Full 17 COCO keypoints
    bbox:           Optional[list] = None   # [x1, y1, x2, y2]
    floor_pos_3d:   Optional[list] = None   # Legacy / optional cache
    body_pos_3d:    Optional[list] = None   # Legacy / optional cache
    confidence:     float          = 0.0

@dataclass
class PoseFrame:
    frame_idx: int
    near: PlayerPose = field(default_factory=PlayerPose)
    far:  PlayerPose = field(default_factory=PlayerPose)

# COCO keypoint indices
_L_ANKLE, _R_ANKLE = 15, 16
_L_HIP,   _R_HIP   = 11, 12


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_player_detector(weights_path: str):
    """
    Load the custom YOLOv8 badminton-player detector.
    Returns an Ultralytics YOLO model.
    weights_path – path to best.pt
    """
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        print(f"[PlayerDet] Loaded YOLOv8 weights from {weights_path}")
        return model
    except Exception as e:
        print(f"[PlayerDet] Failed to load YOLOv8: {e}")
        sys.exit(1)


def load_pose_backend():
    """Load RTMDet-tiny + RTMPose-m."""
    try:
        import torch
        import mmdet, mmpose
        from mmpose.apis import init_model
        from mmdet.apis import init_detector
        from mmpose.registry import TRANSFORMS
        from mmdet.datasets.transforms import PackDetInputs

        TRANSFORMS.register_module(name='PackDetInputs', module=PackDetInputs, force=True)

        det_config = os.path.join(
            os.path.dirname(mmdet.__file__), '.mim', 'configs', 'rtmdet',
            'rtmdet_tiny_8xb32-300e_coco.py')
        pose_config = os.path.join(
            os.path.dirname(mmpose.__file__), '.mim', 'configs',
            'body_2d_keypoint', 'rtmpose', 'coco',
            'rtmpose-m_8xb256-420e_coco-256x192.py')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        det = init_detector(
            det_config,
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_tiny_8xb32-300e_coco/"
            "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
            device=device)

        pipeline = det.cfg.test_dataloader.dataset.pipeline
        det.cfg.test_dataloader.dataset.pipeline = [
            p for p in pipeline if 'LoadAnnotations' not in p.get('type', '')]

        pose = init_model(
            pose_config,
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth",
            device=device)

        return det, pose, "rtmpose"
    except Exception as e:
        print(f"[Pose] RTMPose Error: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def get_iou(bb1, bb2) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x_left   = max(bb1[0], bb2[0])
    y_top    = max(bb1[1], bb2[1])
    x_right  = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / float(area1 + area2 - inter + 1e-6)


def get_matching_rtmdet_boxes(rtmdet_boxes: np.ndarray,
                              yolo_boxes: np.ndarray) -> np.ndarray:
    """
    For every player detected by YOLO, find the RTMDet box that overlaps with it.
    This ensures we only feed confirmed player boxes (from RTMDet) into RTMPose.
    """
    if len(yolo_boxes) == 0 or len(rtmdet_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    matched_rtm_boxes = []
    
    for ybox in yolo_boxes:
        best_iou = 0
        best_rbox = None
        
        for rbox in rtmdet_boxes:
            iou = get_iou(ybox, rbox)
            if iou > best_iou:
                best_iou = iou
                best_rbox = rbox
                
        # If we found an RTMDet box that matches the YOLO player box
        if best_rbox is not None and best_iou > 0.1:
            matched_rtm_boxes.append(best_rbox)

    if matched_rtm_boxes:
        return np.array(matched_rtm_boxes, dtype=np.float32)
    return np.empty((0, 4), dtype=np.float32)


def is_valid_player(kpts, confs, bbox, court_poly, last_bbox=None) -> bool:
    """
    Spatial / temporal validity check.
    - With a last_bbox   → allow persistence via IoU / proximity (handles jumps)
    - Without last_bbox  → player ankles must be inside the court polygon (seeding)
    """
    if court_poly is None:
        return True

    if last_bbox is not None:
        if get_iou(bbox, last_bbox) > 0.15:
            return True
        c1 = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        c2 = np.array([(last_bbox[0] + last_bbox[2]) / 2,
                        (last_bbox[1] + last_bbox[3]) / 2])
        if np.linalg.norm(c1 - c2) < 150:
            return True

    for idx in [_L_ANKLE, _R_ANKLE]:
        if confs[idx] > 0.3:
            pt = (float(kpts[idx][0]), float(kpts[idx][1]))
            if cv2.pointPolygonTest(court_poly, pt, False) >= 0:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Core estimation functions
# ─────────────────────────────────────────────────────────────────────────────

def estimate_poses_batched(frames: list,
                           court_poly,
                           K, rvec, tvec,
                           det_m, pose_m,
                           player_det_m=None,          
                           player_det_conf: float = 0.30,  
                           player_iou_gate: float = 0.25,  
                           batch_size: int = 32,
                           first_hit_indices=None,
                           start_idx: int = 0) -> list:
    """
    Batched pose estimation with YOLOv8 player gating.
    Uses RTMDet to get standard person boxes, uses custom YOLO to find the players,
    matches them together, and runs RTMPose on the matched RTMDet boxes.
    """
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown

    results      = []
    last_near    = None
    last_far     = None

    if first_hit_indices is None:
        first_hit_indices = set()

    for i in range(0, len(frames), batch_size):
        batch       = frames[i: i + batch_size]
        det_results = inference_detector(det_m, batch)   # RTMDet on whole batch

        for j, frame in enumerate(batch):
            global_idx = start_idx + i + j

            # ── Rally boundary reset ──────────────────────────────────────────
            if global_idx in first_hit_indices:
                last_near = None
                last_far  = None

            pf = PoseFrame(global_idx)

            # ── Step 1: RTMDet boxes (persons only) ──────────────────────────
            inst  = det_results[j].pred_instances
            valid = ((inst.scores.cpu().numpy() > 0.45) &
                     (inst.labels.cpu().numpy() == 0))
            rtmdet_boxes = inst.bboxes.cpu().numpy()[valid]   # shape (N,4)

            if len(rtmdet_boxes) == 0:
                results.append(pf)
                continue

            # ── Step 2: YOLOv8 gating to find matching RTMDet boxes ───────────
            if player_det_m is not None:
                yolo_results = player_det_m.predict(
                    frame,
                    conf=player_det_conf,
                    classes=[0],          
                    verbose=False
                )
                if yolo_results and len(yolo_results[0].boxes):
                    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()  
                else:
                    yolo_boxes = np.empty((0, 4), dtype=np.float32)

                # Fetch ONLY the RTMDet boxes that perfectly align with YOLO player boxes
                bboxes = get_matching_rtmdet_boxes(rtmdet_boxes, yolo_boxes)
            else:
                bboxes = rtmdet_boxes

            if len(bboxes) == 0:
                results.append(pf)
                continue

            # ── Step 3: RTMPose on Matched RTMDet boxes ───────────────────────
            pose_res   = inference_topdown(pose_m, frame, bboxes=bboxes)
            candidates = []

            for k, res in enumerate(pose_res):
                kpts  = res.pred_instances.keypoints[0]
                confs = res.pred_instances.keypoint_scores[0]

                is_p_near = is_valid_player(kpts, confs, bboxes[k],
                                            court_poly, last_near)
                is_p_far  = is_valid_player(kpts, confs, bboxes[k],
                                            court_poly, last_far)

                if is_p_near or is_p_far:
                    candidates.append({
                        'kpts': kpts.tolist(),
                        'bbox': bboxes[k].tolist(),
                        'conf': float(np.mean(confs[[_L_HIP, _R_HIP,
                                                     _L_ANKLE, _R_ANKLE]]))
                    })

            # ── Step 4: Near / Far assignment ─────────────────────────────────
            if candidates:
                if last_near is not None:
                    c_near = max(candidates,
                                 key=lambda x: get_iou(x['bbox'], last_near))
                else:
                    c_near = max(candidates, key=lambda x: x['bbox'][3])

                pf.near.keypoints = c_near['kpts']
                pf.near.bbox      = c_near['bbox']
                last_near         = c_near['bbox']
                candidates = [c for c in candidates if c['bbox'] != c_near['bbox']]

            if candidates:
                if last_far is not None:
                    c_far = max(candidates,
                                key=lambda x: get_iou(x['bbox'], last_far))
                else:
                    c_far = min(candidates, key=lambda x: x['bbox'][3])

                pf.far.keypoints = c_far['kpts']
                pf.far.bbox      = c_far['bbox']
                last_far         = c_far['bbox']

            results.append(pf)

    return results


# kept for backwards compatibility / standalone use without gating
def estimate_poses(frames, K, rvec, tvec, det_m, pose_m) -> list:
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown
    results = []

    for fi, frame in enumerate(frames):
        pf = PoseFrame(fi)
        det_res = inference_detector(det_m, frame)
        boxes  = det_res.pred_instances.bboxes.cpu().numpy()
        scores = det_res.pred_instances.scores.cpu().numpy()
        labels = det_res.pred_instances.labels.cpu().numpy()

        valid   = (scores > 0.40) & (labels == 0)
        persons = boxes[valid]
        h, w    = frame.shape[:2]
        mid_y   = h / 2

        near_cands = [b for b in persons
                      if (b[1] + b[3]) / 2 >= mid_y
                      and w * 0.15 < (b[0] + b[2]) / 2 < w * 0.85]
        far_cands  = [b for b in persons
                      if (b[1] + b[3]) / 2 < mid_y
                      and w * 0.15 < (b[0] + b[2]) / 2 < w * 0.85]

        for candidates, p_attr in [(near_cands, 'near'), (far_cands, 'far')]:
            if candidates:
                best_box = max(candidates,
                               key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                res = inference_topdown(pose_m, frame,
                                        bboxes=best_box.reshape(1, 4))
                if res:
                    inst      = res[0].pred_instances
                    kps, cnfs = inst.keypoints[0], inst.keypoint_scores[0]
                    p_obj     = getattr(pf, p_attr)
                    p_obj.keypoints = kps.tolist()
                    p_obj.bbox      = best_box.tolist()
                    p_obj.confidence = float(
                        np.mean(cnfs[[_L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE]]))
                    if cnfs[_L_ANKLE] > 0.3:
                        p_obj.left_ankle_px  = kps[_L_ANKLE].tolist()
                    if cnfs[_R_ANKLE] > 0.3:
                        p_obj.right_ankle_px = kps[_R_ANKLE].tolist()

        results.append(pf)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Debug rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_pose_debug(frames, poses, fps, out_path,
                      court_poly=None, K=None, rvec=None, tvec=None):
    h, w   = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame, pf in zip(frames, poses):
        out = frame.copy()

        if court_poly is not None:
            cv2.polylines(out, [court_poly.astype(np.int32)], True, (0, 255, 255), 2)

        for p, clr, tag in [(pf.near, (0, 255, 0), "NEAR"),
                             (pf.far,  (0, 0, 255), "FAR")]:
            if p.bbox is not None:
                x1, y1, x2, y2 = map(int, p.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), clr, 2)
                cv2.putText(out, tag, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)

                ankles = [px for px in [p.left_ankle_px, p.right_ankle_px]
                          if px is not None]
                for px in ankles:
                    cv2.circle(out, (int(px[0]), int(px[1])), 5, clr, -1)

                if ankles and K is not None:
                    u_avg, v_avg = np.mean(ankles, axis=0)
                    fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec,
                                             target_z=0.0)
                    if fp is not None:
                        cv2.putText(out, f"({fp[0]:.2f}, {fp[1]:.2f})m",
                                    (int(u_avg) - 30, int(v_avg) + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)
        writer.write(out)
    writer.release()


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_poses(poses, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "poses.pkl", "wb") as f:
        pickle.dump(poses, f)


def load_poses(pose_dir: str) -> list:
    path = Path(pose_dir) / "poses.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No poses found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_player_3d(poses, frame_idx, side, K=None, rvec=None, tvec=None,
                  use_floor=True) -> Optional[np.ndarray]:
    if frame_idx >= len(poses):
        return None
    pf     = poses[frame_idx]
    player = pf.near if side == "near" else pf.far

    if K is not None and rvec is not None and tvec is not None:
        ankles = [px for px in [player.left_ankle_px, player.right_ankle_px]
                  if px is not None]
        if ankles:
            u_avg, v_avg = np.mean(ankles, axis=0)
            fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
            if fp is not None:
                x = float(np.clip(fp[0], -1.0, W + 1.0))
                y = float(np.clip(fp[1], -1.0, L + 1.0))
                z = 0.0 if use_floor else float(PLAYER_HEIGHT / 2.0)
                return np.array([x, y, z], dtype=np.float64)

    pos = player.floor_pos_3d if use_floor else player.body_pos_3d
    return np.array(pos, dtype=np.float64) if pos is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        default=str(VIDEO_PATH))
    ap.add_argument("--calib_dir",    default="results/calib_out")
    ap.add_argument("--out_dir",      default=str(POSE_OUT))
    ap.add_argument("--player_weights", default="best.pt",
                    help="Path to custom YOLOv8 badminton player weights")
    ap.add_argument("--no_player_gate", action="store_true",
                    help="Disable YOLOv8 gating (use all RTMDet detections)")
    args = ap.parse_args()

    from court_calibration import load_calibration, project_to_pixel
    from config import WORLD_PTS

    P, K, rvec, tvec = load_calibration(args.calib_dir)
    court_poly = project_to_pixel(WORLD_PTS[:4], P).astype(np.int32)

    cap    = cv2.VideoCapture(args.video)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    det_m, pose_m, _ = load_pose_backend()

    player_det_m = None
    if not args.no_player_gate:
        player_det_m = load_player_detector(args.player_weights)

    test_first_hits = {0}

    poses = estimate_poses_batched(
        frames,
        court_poly,
        K, rvec, tvec,
        det_m, pose_m,
        player_det_m   = player_det_m,
        batch_size     = 32,
        first_hit_indices = test_first_hits,
        start_idx      = 0
    )

    out_path = Path(args.out_dir)
    save_poses(poses, out_path)
    render_pose_debug(frames, poses, fps,
                      str(out_path / "pose_test_single_rally.mp4"),
                      court_poly=court_poly, K=K, rvec=rvec, tvec=tvec)