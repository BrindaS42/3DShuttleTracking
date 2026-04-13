import os
import sys
import logging
import pickle
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm  # Added for progress bar

from pose_estimation import load_pose_backend, estimate_poses_batched

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

            # Annotate Ankles (Indices 15 & 16 in COCO)
            l_ankle = kpts[15][:2] if kpts[15][2] > 0.1 else None # Assuming index 2 is confidence
            r_ankle = kpts[16][:2] if kpts[16][2] > 0.1 else None
            
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

        if not trimmed_video_path.exists():
            logger.error(f"Trimmed video missing for {match_folder}. Run Step 1.")
            continue

        pose_cache = pose_dir / "poses_trimmed.pkl"
        if pose_cache.exists():
            logger.info("Pose cache exists. Loading...")
            with open(pose_cache, "rb") as f: trimmed_poses = pickle.load(f)
        else:
            logger.info("Extracting poses...")
            cap = cv2.VideoCapture(str(trimmed_video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            det_m, pose_m, _ = load_pose_backend()
            trimmed_poses = []
            batch_frames = []
            
            # Initialize the tqdm progress bar
            pbar = tqdm(total=total_frames, desc="Estimating Poses", unit="frame", dynamic_ncols=True)
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                batch_frames.append(frame)
                if len(batch_frames) == 32:
                    trimmed_poses.extend(estimate_poses_batched(batch_frames, None, None, None, det_m, pose_m, batch_size=32))
                    pbar.update(32)  # Update progress bar
                    batch_frames.clear()
                    
            if batch_frames:
                trimmed_poses.extend(estimate_poses_batched(batch_frames, None, None, None, det_m, pose_m, batch_size=len(batch_frames)))
                pbar.update(len(batch_frames))  # Update remaining frames
                
            pbar.close()
            cap.release()
            with open(pose_cache, "wb") as f: pickle.dump(trimmed_poses, f)

        vid_out = pose_dir / f"{match_folder}_pose_trimmed.mp4"
        if not vid_out.exists():
            logger.info("Dumping annotated pose video...")
            cap = cv2.VideoCapture(str(trimmed_video_path))
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(str(vid_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_idx < len(trimmed_poses): custom_pose_draw(frame, trimmed_poses[frame_idx])
                out.write(frame)
                frame_idx += 1
            cap.release()
            out.release()
            logger.info(f"Pose video saved to {vid_out}")

if __name__ == "__main__": main()