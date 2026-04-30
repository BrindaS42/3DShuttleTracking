import os
import sys
import logging
import pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import the backend and the batched function
from pose_estimation import load_pose_backend, load_player_detector, estimate_poses_batched

# ── Path to your custom YOLOv8 badminton-player weights ──────────────────────
PLAYER_WEIGHTS = Path(r"C:/Users/brind/Desktop/academics/sem6/DL/project/3DShuttleTracking/weights/pose.pt")

class EmptyPoseFrame:
    near = None
    far = None

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)
    return logger

def load_frame_counts(filepath):
    """Parses the video_frame_counts.txt file into a dictionary mapping."""
    counts = {}
    if not os.path.exists(filepath):
        return counts
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip header or empty lines
            if not line or "total_frames" in line or line.startswith("[source"):
                continue
            # Split by tab as seen in your provided file
            parts = line.split('\t')
            if len(parts) == 2:
                counts[parts[0].strip()] = int(parts[1].strip())
    return counts

def custom_pose_draw(frame, pf_obj):
    if not pf_obj:
        return
    players = []
    if hasattr(pf_obj, 'near') and pf_obj.near is not None:
        players.append(("Near", pf_obj.near))
    if hasattr(pf_obj, 'far') and pf_obj.far is not None:
        players.append(("Far", pf_obj.far))

    coco_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for label, person in players:
        kpts = (np.array(person.keypoints)
                if hasattr(person, 'keypoints') and person.keypoints is not None
                else None)
        if kpts is not None and kpts.shape[0] >= 17:
            for p1, p2 in coco_pairs:
                x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            for x, y in kpts[:, :2]:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

def main():
    base_output_dir = Path("test_shuttlenet")
    counts_file = "video_frame_counts.txt"
    
    if not base_output_dir.exists():
        print(f"[!] {base_output_dir} folder not found.")
        return

    # Load the n_total mapping from your provided text file
    frame_counts = load_frame_counts(counts_file)

    # ── Load models once ─────────────────────────────────────────────────────
    print("[Init] Loading RTMDet + RTMPose …")
    det_m, pose_m, _ = load_pose_backend()

    print(f"[Init] Loading custom YOLOv8 player detector from {PLAYER_WEIGHTS} …")
    player_det_m = load_player_detector(PLAYER_WEIGHTS) if PLAYER_WEIGHTS.exists() else None

    # Process each match folder directly from test_shuttlenet
    for match_dir in sorted(base_output_dir.iterdir()):
        if not match_dir.is_dir():
            continue
            
        match_folder = match_dir.name
        trimmed_video_path = match_dir / "trimmed_temp.mp4"
        
        if not trimmed_video_path.exists():
            print(f"[Skip] {match_folder}: No trimmed_temp.mp4 found.")
            continue

        # Look up n_total using the folder name as the key (mapping expects .mp4 extension)
        n_total = frame_counts.get(f"{match_folder}.mp4", 1_000_000)
        
        logger = setup_logger(f"Pose_{match_folder}", match_dir / "pose.log")
        logger.info(f"=== Processing: {match_folder} (Total Original Frames: {n_total}) ===")

        # ── Build first_hit_indices from CSV annotations ──────────────────────
        set_dir = Path("shuttleset/set") / match_folder
        dfs = []
        if set_dir.exists():
            for set_file in os.listdir(set_dir):
                if set_file.startswith("set") and set_file.endswith(".csv"):
                    df = pd.read_csv(set_dir / set_file, encoding='utf-8')
                    if 'frame_nur' in df.columns:
                        df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
                    df['set_file'] = set_file
                    dfs.append(df)

        if not dfs:
            logger.error(f"No CSVs found for {match_folder}. Skipping.")
            continue

        gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)

        grouped_rallies = (gt_df.sort_values(by=["set_file", "rally", "frame_num"])
                           .groupby(["set_file", "rally"]))

        first_hit_indices = set()
        current_trimmed_len = 0

        for _, group in grouped_rallies:
            first_hit = int(group["frame_num"].min())
            last_hit  = int(group["frame_num"].max())

            start_f = max(0, first_hit - 60)
            end_f   = min(n_total, last_hit + 61) # Utilizing n_total from mapping

            first_hit_relative = current_trimmed_len + (first_hit - start_f)
            first_hit_indices.add(first_hit_relative)
            current_trimmed_len += (end_f - start_f)

        # ── Calibration ───────────────────────────────────────────────────────
        from court_calibration import load_calibration, project_to_pixel
        from config import WORLD_PTS
        P, K, rvec, tvec = load_calibration(match_dir)
        court_poly = project_to_pixel(WORLD_PTS[:4], P).astype(np.int32)

        # ── Main processing loop (Using trimmed_temp.mp4) ─────────────────────
        cap = cv2.VideoCapture(str(trimmed_video_path))
        trimmed_poses = []
        batch_frames = []
        current_idx = 0

        pbar = tqdm(total=current_trimmed_len, desc=f"Processing {match_folder}", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret: break

            batch_frames.append(frame)
            current_idx += 1
            is_next_frame_new_rally = current_idx in first_hit_indices

            if len(batch_frames) == 32 or (is_next_frame_new_rally and len(batch_frames) > 0):
                batch_results = estimate_poses_batched(
                    batch_frames, court_poly, K, rvec, tvec, det_m, pose_m,
                    player_det_m=player_det_m, player_det_conf=0.30, player_iou_gate=0.25,
                    batch_size=len(batch_frames), first_hit_indices=first_hit_indices,
                    start_idx=current_idx - len(batch_frames)
                )
                trimmed_poses.extend(batch_results)
                pbar.update(len(batch_frames))
                batch_frames.clear()

        # Flush remaining frames
        if batch_frames:
            batch_results = estimate_poses_batched(
                batch_frames, court_poly, K, rvec, tvec, det_m, pose_m,
                player_det_m=player_det_m, batch_size=len(batch_frames),
                first_hit_indices=first_hit_indices, start_idx=current_idx - len(batch_frames)
            )
            trimmed_poses.extend(batch_results)
            pbar.update(len(batch_frames))

        pbar.close()
        cap.release()

        # ── Save and Render Debug Video ───────────────────────────────────────
        pose_cache = match_dir / "poses.pkl"
        with open(pose_cache, "wb") as f:
            pickle.dump(trimmed_poses, f)
        
        vid_out = match_dir / f"{match_folder}_pose_debug.mp4"
        cap = cv2.VideoCapture(str(trimmed_video_path))
        fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for idx in range(len(trimmed_poses)):
            ret, frame = cap.read()
            if not ret: break
            custom_pose_draw(frame, trimmed_poses[idx])
            if idx in first_hit_indices:
                cv2.putText(frame, "RALLY START: RESET STATE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out.write(frame)
        cap.release()
        out.release()

if __name__ == "__main__":
    main()