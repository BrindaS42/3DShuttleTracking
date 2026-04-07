"""
shot_classification.py
══════════════════════
Optimized Rally Analyzer: Hit-Centered context windows, Edge Padding,
Hitter Alternation, and English-Chinese stroke reporting.

Key Fixes:
1. Synchronized normalize_rally_data signature to use H_court.
2. Implemented Edge-Padding for hits near start/end to prevent "Unknown" collapse.
3. Integrated English translation map for match reporting.
4. Synchronized cache paths with reconstruct.py (marked_hits.json & shuttle_2d.npy)
"""

import argparse
import logging
import pickle
import sys
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

# ── BST repo on sys.path ──────────────────────────────────────────────────────
BST_ROOT = Path(__file__).parent / "stroke_classification"
if str(BST_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(BST_ROOT.resolve()))

from config import (
    VIDEO_PATH, SHUTTLE_OUT, POSE_OUT, CALIB_OUT_DIR,
    TRACKNET_DIR, COURT_W, COURT_L,
    WORLD_PTS, TRAJ_OUT  # Added TRAJ_OUT to sync with reconstruct.py
)
from court_calibration import load_calibration
from shuttle_detection import run_tracknet, parse_tracknet_csv, render_debug_video
from pose_estimation import load_pose_backend, estimate_poses_batched, render_pose_debug
from stroke_classification.preparing_data.shuttleset_dataset import (
    get_merged_stroke_types, get_bone_pairs, make_seq_len_same, create_bones
)
from stroke_classification.model.bst import BST_CG_AP

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
#  MAPPING & LOGGING
# ══════════════════════════════════════════════════════════════════════════════

STROKE_MAP = {
    "放小球": "Net Shot", "擋小球": "Net Block", "殺球": "Smash",
    "點扣": "Wrist Smash", "挑球": "Lob", "防守回挑": "Defensive Lob",
    "長球": "Clear", "平球": "Drive", "後場抽平球": "Back-court Drive",
    "切球": "Drop", "過渡切球": "Passive Drop", "推球": "Push",
    "撲球": "Rush", "防守回抽": "Defensive Drive", "勾球": "Cross-court Net Shot",
    "發短球": "Short Service", "發長球": "Long Service", "未知球種": "Unknown"
}

def translate_stroke(chinese_stroke):
    """Splits prefix (Top/Bottom) and translates the stroke name."""
    if "_" in chinese_stroke:
        prefix, name = chinese_stroke.split("_")
        return f"{prefix}_{STROKE_MAP.get(name, name)}"
    return STROKE_MAP.get(chinese_stroke, chinese_stroke)

def setup_logging(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(out_dir / "rally_analysis.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("RallyAnalyzer")

# ══════════════════════════════════════════════════════════════════════════════
#  HOMOGRAPHY & GAP FILL
# ══════════════════════════════════════════════════════════════════════════════

_FLOOR_CORNER_IDX = [0, 1, 2, 3] # Near-L, Near-R, Far-R, Far-L
_COURT_DST = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)

def build_court_homography(K, rvec, tvec) -> np.ndarray:
    floor_pts_world = WORLD_PTS[_FLOOR_CORNER_IDX, :3].astype(np.float64)
    img_pts, _ = cv2.projectPoints(floor_pts_world, rvec, tvec, K, distCoeffs=None)
    img_pts = img_pts.reshape(-1, 2).astype(np.float32)
    H, _ = cv2.findHomography(img_pts, _COURT_DST)
    return H.astype(np.float64)

def pixel_to_court_norm(u, v, H) -> np.ndarray:
    p_h = H @ np.array([u, v, 1.0], dtype=np.float64)
    return (p_h[:2] / p_h[2]).astype(np.float32)

def fill_linear_shuttle_gaps(shuttle_2d, max_gap=30):
    filled = shuttle_2d.copy()
    valid_indices = np.where(~np.any(np.isnan(shuttle_2d), axis=1))[0]
    if len(valid_indices) < 2: return filled
    for i in range(len(valid_indices) - 1):
        idx_s, idx_e = valid_indices[i], valid_indices[i + 1]
        gap = idx_e - idx_s
        if 1 < gap <= max_gap:
            filled[idx_s:idx_e + 1] = np.linspace(shuttle_2d[idx_s], shuttle_2d[idx_e], gap + 1)
    return filled

# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

_L_ANKLE, _R_ANKLE = 15, 16

def normalize_rally_data(poses, shuttle_2d, v_w, v_h, n_frames, H_court):
    joints  = np.zeros((n_frames, 2, 17, 2), dtype=np.float32)
    pos     = np.zeros((n_frames, 2, 2),     dtype=np.float32)
    shuttle = np.zeros((n_frames, 2),        dtype=np.float32)

    valid_sh = ~np.any(np.isnan(shuttle_2d[:n_frames]), axis=1)
    shuttle[valid_sh] = shuttle_2d[:n_frames][valid_sh]
    shuttle[:, 0] /= v_w
    shuttle[:, 1] /= v_h

    detected_j = 0
    for i, pf in enumerate(poses[:n_frames]):
        for p_idx, player in enumerate([pf.far, pf.near]):
            if player.keypoints and player.bbox:
                kps = np.array(player.keypoints, dtype=np.float32)
                bx  = np.array(player.bbox,      dtype=np.float32)
                dist = max(float(np.linalg.norm(bx[2:] - bx[:2])), 1e-6)
                center = (bx[:2] + bx[2:]) / 2
                joints[i, p_idx] = (kps - bx[:2]) / dist - (center - bx[:2]) / dist
                if p_idx == 1: detected_j += 1 # Tracking Far player joints

                la, ra = kps[_L_ANKLE], kps[_R_ANKLE]
                ankles = [px for px in [la, ra] if not (px[0] == 0.0 and px[1] == 0.0)]
                u_avg, v_avg = np.mean(ankles, axis=0) if ankles else ((bx[0]+bx[2])/2, bx[3])
                pos[i, p_idx] = pixel_to_court_norm(u_avg, v_avg, H_court)

    return joints, pos, shuttle, detected_j

# ══════════════════════════════════════════════════════════════════════════════
#  UI & RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def mark_hits_ui(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    hits, f_idx = [], 0
    print("\n[UI] Space/Right=next  Left=prev  m=mark  d=delete last  q=done")
    while True:
        display = frames[f_idx].copy()
        cv2.putText(display, f"Frame: {f_idx}/{total-1}  Hits: {len(hits)}", (20, 40), 2, 1, (255,255,255), 2)
        if f_idx in hits: cv2.circle(display, (display.shape[1]-30, 30), 12, (0,0,255), -1)
        cv2.imshow("Mark Hits", display)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord(' '), 83, 0): f_idx = min(f_idx+1, total-1)
        elif key == 81: f_idx = max(f_idx-1, 0)
        elif key == ord('m') and f_idx not in hits: hits.append(f_idx)
        elif key == ord('d') and hits: hits.pop()
        elif key == ord('q'): break
    cv2.destroyAllWindows()
    return sorted(hits)

def dump_annotated_rally(frames, global_shuttle, global_poses, hit_results, out_path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bp = get_bone_pairs('coco')
    hit_map = {item['Frame']: (item['Stroke'], item['Hitter']) for item in hit_results}
    cur_stroke, cur_hitter = "N/A", "N/A"
    for i, frame in enumerate(frames):
        out = frame.copy()
        if i in hit_map: cur_stroke, cur_hitter = hit_map[i]
        if i < len(global_shuttle) and not np.any(np.isnan(global_shuttle[i])):
            cv2.circle(out, tuple(global_shuttle[i].astype(int)), 6, (0,255,0), -1)
        pf = global_poses[i]
        for p, clr, lbl in [(pf.near,(0,255,0),"NEAR"),(pf.far,(0,0,255),"FAR")]:
            if p.keypoints:
                kps = np.array(p.keypoints).astype(int)
                for s, e in bp: cv2.line(out, tuple(kps[s]), tuple(kps[e]), clr, 2)
                cv2.putText(out, lbl, tuple(kps[0]), 0, 0.6, clr, 2)
        cv2.rectangle(out, (10,h-90), (500,h-10), (0,0,0), -1)
        cv2.putText(out, f"Hitter: {cur_hitter.upper()}", (20,h-60), 0, 0.9, (255,255,255), 2)
        cv2.putText(out, f"Type  : {cur_stroke}", (20,h-25), 0, 0.9, (0,255,0), 2)
        writer.write(out)
    writer.release()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test_assets/half_rally.mp4")
    ap.add_argument("--weight", default="weight/bst_CG_AP_JnB_bone_between_2_hits_with_max_limits_seq_100_merged.pt")
    ap.add_argument("--first_hitter", choices=["near","far"], default="near")
    ap.add_argument("--annotate", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path("results/classif_out")
    logger = setup_logging(out_dir)

    # 1. Hit Marking (Synced with reconstruct.py TRAJ_OUT)
    hits_cache = Path(TRAJ_OUT) / "marked_hits.json"
    hit_frames = []
    if hits_cache.exists():
        if input(f"\n[?] Reuse hits at {hits_cache}? (y/n): ").lower() == 'y':
            with open(hits_cache) as f: hit_frames = json.load(f)
            logger.info(f"Loaded {len(hit_frames)} hits.")
    if not hit_frames:
        hit_frames = mark_hits_ui(args.video)
        hits_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(hits_cache, "w") as f: json.dump(hit_frames, f)

    cap = cv2.VideoCapture(args.video)
    v_w, v_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    P, K, rvec, tvec = load_calibration(CALIB_OUT_DIR)
    H_court = build_court_homography(K, rvec, tvec)

    # 2. Global Pose Estimation
    pose_cache = Path(POSE_OUT) / "poses.pkl"
    if pose_cache.exists() and input(f"[?] Use cached poses at {pose_cache}? (y/n): ").lower() == 'y':
        with open(pose_cache, "rb") as f: global_poses = pickle.load(f)
    else:
        cap2 = cv2.VideoCapture(args.video); frames_list = []
        while True:
            ret, f = cap2.read()
            if not ret: break
            frames_list.append(f)
        cap2.release()
        det_m, pose_m, _ = load_pose_backend()
        global_poses = estimate_poses_batched(frames_list, K, rvec, tvec, det_m, pose_m, batch_size=16)
        with open(pose_cache, "wb") as f: pickle.dump(global_poses, f)

    # 3. Shuttle Detection (Synced with reconstruct.py SHUTTLE_OUT)
    shuttle_cache = Path(SHUTTLE_OUT) / "shuttle_2d.npy"
    if shuttle_cache.exists() and input(f"[?] Use cached shuttle at {shuttle_cache}? (y/n): ").lower() == 'y':
        global_shuttle = np.load(shuttle_cache)
    else:
        csv_path = run_tracknet(args.video, TRACKNET_DIR, SHUTTLE_OUT, eval_mode="average")
        global_shuttle = fill_linear_shuttle_gaps(parse_tracknet_csv(csv_path, args.video))
        np.save(shuttle_cache, global_shuttle)
        render_debug_video(args.video, global_shuttle, str(Path(SHUTTLE_OUT) / "shuttle_debug.mp4"))

    # 4. BST Inference
    logger.info("Normalising data for BST...")
    joints, pos_norm, shuttle_norm, detected_j = normalize_rally_data(global_poses, global_shuttle, v_w, v_h, n_total, H_court)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bone_pairs = get_bone_pairs('coco')
    net = BST_CG_AP(in_dim=72, seq_len=100, n_class=25).to(device)
    net.load_state_dict(torch.load(args.weight, map_location=device, weights_only=True))
    net.eval()

    hitter_cycle = [args.first_hitter, "far" if args.first_hitter == "near" else "near"]
    hit_results = []

    logger.info("Running BST Inference (Edge-Padded Context)...")
    for i, hit_idx in enumerate(hit_frames):
        current_hitter = hitter_cycle[i % 2]
        
        # 1. Define strict boundaries for THIS specific shot
        prev_h = hit_frames[i-1] if i > 0 else max(0, hit_idx - 25)
        next_h = hit_frames[i+1] if i < len(hit_frames)-1 else min(n_total, hit_idx + 25)
        
        # 2. Extract context window [hit-50, hit+50]
        start, end = hit_idx - 50, hit_idx + 50
        
        # 3. Create isolated tensors (initialized to zero)
        j_win = np.zeros((100, 2, 17, 2), dtype=np.float32)
        p_win = np.zeros((100, 2, 2),     dtype=np.float32)
        s_win = np.zeros((100, 2),        dtype=np.float32)

        # 4. Fill ONLY the relevant segment [prev_h, next_h] into the 100-frame window
        for offset in range(-50, 50):
            abs_frame = hit_idx + offset
            target_idx = offset + 50
            
            # Strict Isolation: Only use data between the previous and next hits
            if prev_h <= abs_frame <= next_h and 0 <= abs_frame < n_total:
                j_win[target_idx] = joints[abs_frame]
                p_win[target_idx] = pos_norm[abs_frame]
                s_win[target_idx] = shuttle_norm[abs_frame]

        # Calculate actual length of the visible movement for the model's forward pass
        real_len = (next_h - prev_h)
        
        # 5. Collate and Bone creation
        bone_pairs = get_bone_pairs('coco')
        bones = create_bones(j_win, bone_pairs)
        hp_in = np.concatenate((j_win, bones), axis=-2)
        
        with torch.no_grad():
            hp_t = torch.tensor(hp_in).unsqueeze(0).to(device).view(1, 100, 2, -1)
            sh_t = torch.tensor(s_win).unsqueeze(0).to(device)
            ps_t = torch.tensor(p_win).unsqueeze(0).to(device)
            
            logits = net(hp_t, sh_t, ps_t, torch.tensor([100]).to(device))
            
            probs = torch.softmax(logits, dim=1)
            sorted_indices = torch.argsort(probs, dim=1, descending=True)[0]

        expected_side = hitter_cycle[i % 2]
        all_types = get_merged_stroke_types()
        final_pred_idx = sorted_indices[0].item() # Fallback
        found_valid = False

        # --- STEP 2: General Filter (Side Match & Skip Unknown) ---
        if not found_valid:
            for idx in sorted_indices:
                cand_idx = idx.item()
                cand_e = translate_stroke(all_types[cand_idx])
                
                # Ignore 'Unknown' unless it's literally the only choice
                if "Unknown" in cand_e and len(sorted_indices) > 1:
                    continue
                
                cand_prefix = "near" if "Bottom" in cand_e else "far"
                if cand_prefix == expected_side:
                    final_pred_idx = cand_idx
                    found_valid = True
                    break

        # Final Translation
        e_stroke = translate_stroke(all_types[final_pred_idx])
        
        hit_results.append({
            'Hit #': i+1, 'Frame': hit_idx, 
            'Hitter': expected_side, 'Stroke': e_stroke
        })
        logger.info(f" Hit {i+1:2d} | f{hit_idx:3d} | Hitter: {expected_side:4s} | Result: {e_stroke}")

    # 5. Report & Video
    df = pd.DataFrame(hit_results)
    df.to_excel(out_dir / "rally_summary.xlsx", index=False)
    if args.annotate:
        cap3 = cv2.VideoCapture(args.video); flist = []
        while True:
            ret, f = cap3.read()
            if not ret: break
            flist.append(f)
        cap3.release()
        dump_annotated_rally(flist, global_shuttle, global_poses, hit_results, out_dir / "rally_annotated.mp4", fps)
    logger.info("Done.")

if __name__ == "__main__":
    main()