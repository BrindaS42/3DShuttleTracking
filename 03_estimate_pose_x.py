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
from pose_estimation_x import load_pose_backend, estimate_poses_batched

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

def custom_pose_draw(frame, pf_obj):
    if not pf_obj: return
    players = []
    if hasattr(pf_obj, 'near') and pf_obj.near is not None: players.append(("Near", pf_obj.near))
    if hasattr(pf_obj, 'far') and pf_obj.far is not None: players.append(("Far", pf_obj.far))

    coco_pairs = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

    for label, person in players:
        kpts = np.array(person.keypoints) if hasattr(person, 'keypoints') and person.keypoints is not None else None
        if kpts is not None and kpts.shape[0] >= 17:
            # Draw Skeleton
            for p1, p2 in coco_pairs:
                x1, y1, x2, y2 = int(kpts[p1][0]), int(kpts[p1][1]), int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0: 
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 100, 100), 2)
            
            # Draw joints
            for x, y in kpts[:, :2]:
                if x > 0 and y > 0: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            # Change this line in 03_estimate_pose.py
            l_ankle = kpts[15][:2] if (kpts.shape[1] > 2 and kpts[15][2] > 0.1) else kpts[15][:2]
            r_ankle = kpts[16][:2] if (kpts.shape[1] > 2 and kpts[16][2] > 0.1) else kpts[16][:2]
            
            valid_ankles = [a for a in [l_ankle, r_ankle] if a is not None and a[0]>0 and a[1]>0]
            for ax, ay in valid_ankles:
                cv2.circle(frame, (int(ax), int(ay)), 6, (0, 255, 0), -1)
                text = f"{label} Ankle: ({int(ax)}, {int(ay)})"
                cv2.putText(frame, text, (int(ax)-20, int(ay)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    asset_dir = Path("test_assets")
    if not asset_dir.exists():
        print("[!] test_assets folder not found.")
        return

    for video_path in asset_dir.glob("*.mp4"):
        match_folder = video_path.stem
        base_out = Path("test_shuttlenet") / match_folder
        trimmed_video_path = base_out / "trimmed_temp.mp4"
        pose_dir = base_out / "pose_out"
        pose_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(f"Pose_{match_folder}", pose_dir / "pose.log")
        logger.info(f"=== Processing Pose Estimation: {match_folder} ===")

        set_dir = Path("shuttleset/set") / match_folder
        dfs = []
        for set_file in os.listdir(set_dir):
            if set_file.startswith("set") and set_file.endswith(".csv"):
                df = pd.read_csv(set_dir / set_file, encoding='utf-8')
                if 'frame_nur' in df.columns: 
                    df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
                
                df['set_file'] = set_file 
                dfs.append(df)
        
        if not dfs:
            logger.error("No CSVs found. Skipping.")
            continue

        gt_df = pd.concat(dfs).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)
        
        grouped_rallies = gt_df.sort_values(by=["set_file", "rally", "frame_num"]).groupby(["set_file", "rally"])
        first_hit_indices = set()
        current_trimmed_len = 0
        cap_orig = cv2.VideoCapture(str(video_path))
        n_total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_orig.release()

        for _, group in grouped_rallies:
            first_hit = int(group["frame_num"].min())
            last_hit = int(group["frame_num"].max())
            
            # Matching the math in 01_trim_and_calibrate.py
            start_f = max(0, first_hit - 60)
            end_f = min(n_total, last_hit + 61)
            
            # Frame index relative to the start of the trimmed video
            first_hit_relative = current_trimmed_len + (first_hit - start_f)
            first_hit_indices.add(first_hit_relative)
            
            # Calculate the actual number of frames this rally contributes
            rally_duration = end_f - start_f
            current_trimmed_len += rally_duration

        # Now current_trimmed_len should be ~50,000, not 3 million.
        # pbar = tqdm(total=current_trimmed_len, desc="Rally-Aware Estimation", unit="frame")

        # Load Calibration for Polygon Filtering
        calib_dir = base_out / "calib_out"
        from court_calibration import load_calibration, project_to_pixel
        from config import WORLD_PTS
        P, K, rvec, tvec = load_calibration(calib_dir)
        court_poly = project_to_pixel(WORLD_PTS[:4], P).astype(np.int32)

        # Initialize Models
        det_m, pose_m, _ = load_pose_backend()
        cap = cv2.VideoCapture(str(trimmed_video_path))
        
        trimmed_poses = []
        batch_frames = []
        current_idx = 0
        
        pbar = tqdm(total=current_trimmed_len, desc="Rally-Aware Estimation", unit="frame")
        
        # Processing loop
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            batch_frames.append(frame)
            current_idx += 1

            # This ensures we don't carry 'state' across a rally boundary
            is_next_frame_new_rally = current_idx in first_hit_indices
            
            if len(batch_frames) == 128 or (is_next_frame_new_rally and len(batch_frames) > 0):
                # Call the batched estimator from pose_estimation.py
                # We pass first_hit_indices so it knows when to ignore 'last_near' / 'last_far'
                batch_results = estimate_poses_batched(
                    batch_frames, 
                    court_poly, 
                    K, rvec, tvec, 
                    det_m, pose_m, 
                    batch_size=len(batch_frames),
                    first_hit_indices=first_hit_indices,
                    start_idx=current_idx - len(batch_frames)
                )
                trimmed_poses.extend(batch_results)
                pbar.update(len(batch_frames))
                batch_frames.clear()

        pbar.close()
        cap.release()
        
        # Save results
        pose_cache = pose_dir / "poses.pkl"
        with open(pose_cache, "wb") as f:
            pickle.dump(trimmed_poses, f)
        logger.info(f"Saved {len(trimmed_poses)} frames to {pose_cache}")

        # Render debug video
        vid_out = pose_dir / f"{match_folder}_pose_debug.mp4"
        logger.info("Dumping annotated pose video...")
        cap = cv2.VideoCapture(str(trimmed_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for idx in range(len(trimmed_poses)):
            ret, frame = cap.read()
            if not ret: break
            custom_pose_draw(frame, trimmed_poses[idx])
            # Draw a marker if it's a "reset" frame
            if idx in first_hit_indices:
                cv2.putText(frame, "RALLY START: RESET STATE", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out.write(frame)
        
        cap.release()
        out.release()

if __name__ == "__main__":
    main()