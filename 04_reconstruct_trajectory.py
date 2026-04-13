import os
import sys
import logging
import pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from court_calibration import load_calibration, backproject_to_floor
from shuttle_detection import clean_detections
from shot_classification import fill_linear_shuttle_gaps
from pose_estimation import get_player_3d
from trajectory import reconstruct, save_trajectory, make_plots

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

def custom_trajectory_overlay(trimmed_video_path, out_path, global_shuttle, master_traj_2d, master_traj_3d, frame_map, global_poses, K, rvec, tvec, fps, logger):
    logger.info("Generating Final Video Overlay with World Coords & Reprojection...")
    cap = cv2.VideoCapture(str(trimmed_video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        orig_f = frame_map[frame_idx] if frame_idx < len(frame_map) else -1
        if orig_f != -1:
            # Draw Player World Coords based on Ankles
            pf = global_poses[orig_f]
            for label, person in [("Near", getattr(pf, 'near', None)), ("Far", getattr(pf, 'far', None))]:
                if person and person.left_ankle_px and person.right_ankle_px:
                    u_avg, v_avg = np.mean([person.left_ankle_px, person.right_ankle_px], axis=0)
                    fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
                    if fp is not None:
                        text = f"{label} Pos: (X:{fp[0]:.2f}, Y:{fp[1]:.2f})"
                        cv2.putText(frame, text, (int(u_avg)-50, int(v_avg)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw 3D Shuttle Data
            shuttle_3d = master_traj_3d[orig_f]
            if not np.isnan(shuttle_3d[0]):
                cv2.putText(frame, f"Shuttle 3D: X:{shuttle_3d[0]:.2f} Y:{shuttle_3d[1]:.2f} Z:{shuttle_3d[2]:.2f}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw original detection (Blue) vs Reprojection (Red)
            orig_det = global_shuttle[orig_f]
            if not np.isnan(orig_det[0]):
                cv2.circle(frame, (int(orig_det[0]), int(orig_det[1])), 6, (255, 0, 0), 2) # Blue outline
            
            reproj_det = master_traj_2d[orig_f]
            if not np.isnan(reproj_det[0]):
                cv2.circle(frame, (int(reproj_det[0]), int(reproj_det[1])), 4, (0, 0, 255), -1) # Red fill

            # Draw trailing 2D trajectory
            for past_idx in range(max(0, orig_f - 15), orig_f + 1):
                px, py = master_traj_2d[past_idx]
                if not np.isnan(px) and not np.isnan(py):
                    intensity = int(255 * (1 - (orig_f - past_idx) / 15.0))
                    cv2.circle(frame, (int(px), int(py)), 3, (0, intensity, 255), -1)

        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()

def main():
    asset_dir = Path("test_assets")
    if not asset_dir.exists():
        print("[!] test_assets folder not found.")
        return

    for video_path in asset_dir.glob("*.mp4"):
        match_folder = video_path.stem
        base_out = Path("test_shuttlenet") / match_folder
        traj_dir = base_out / "traj_out"
        traj_dir.mkdir(parents=True, exist_ok=True)
        
        logger = setup_logger(f"Traj_{match_folder}", traj_dir / "traj_eval.log")
        logger.info(f"=== Processing Trajectory: {match_folder} ===")

        # Verify Prerequisites
        trimmed_video_path = base_out / "trimmed_temp.mp4"
        calib_dir = base_out / "calib_out"
        shuttle_cache = base_out / "shuttle_out" / "shuttle_trimmed.npy"
        pose_cache = base_out / "pose_out" / "poses_trimmed.pkl"

        if not all([p.exists() for p in [trimmed_video_path, shuttle_cache, pose_cache, calib_dir/"P.npy"]]):
            logger.error(f"Missing prerequisites for {match_folder}. Check Steps 1-3.")
            continue

        # Load resources
        P = np.load(calib_dir / "P.npy")
        _, K, rvec, tvec = load_calibration(str(calib_dir))
        trimmed_shuttle = np.load(shuttle_cache)
        with open(pose_cache, "rb") as f: trimmed_poses = pickle.load(f)

        cap = cv2.VideoCapture(str(video_path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Load GT to build frame map
        set_dir = Path("shuttleset/set") / match_folder
        dfs = []
        for set_file in os.listdir(set_dir):
            if set_file.startswith("set") and set_file.endswith(".csv"):
                df = pd.read_csv(set_dir / set_file, encoding='utf-8')
                if 'frame_nur' in df.columns: df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
                df['set_file'] = set_file
                dfs.append(df)
        gt_df = pd.concat(dfs, ignore_index=True).dropna(subset=["frame_num"])
        gt_df["frame_num"] = gt_df["frame_num"].astype(int)
        gt_df = gt_df.sort_values(by=["rally", "frame_num"]).reset_index(drop=True)

        if 'player_location_y' in gt_df.columns:
            valid_pos_mask = gt_df['player_location_y'].notna() & gt_df['opponent_location_y'].notna()
            gt_df.loc[valid_pos_mask, 'hitter_side_gt'] = np.where(gt_df.loc[valid_pos_mask, 'player_location_y'] < gt_df.loc[valid_pos_mask, 'opponent_location_y'], 'Top', 'Bottom')
        else: gt_df['hitter_side_gt'] = 'Unknown'

        frame_map = []
        required_frames = set()
        for _, group in gt_df.groupby(["set_file", "rally"]):
            first_hit, last_hit = int(group["frame_num"].min()), int(group["frame_num"].max())
            for f in range(max(0, first_hit - 60), min(n_total, last_hit + 61)):
                required_frames.add(f)
        frame_map = sorted(list(required_frames))
        n_trimmed = len(frame_map)

        # CLEAN SHUTTLE DATA HERE
        logger.info("Applying noise cleaning and linear gap filling to shuttle data...")
        trimmed_shuttle = clean_detections(trimmed_shuttle, str(trimmed_video_path))
        trimmed_shuttle = fill_linear_shuttle_gaps(trimmed_shuttle, max_gap=30)
        if len(trimmed_shuttle) < n_trimmed: 
            trimmed_shuttle = np.vstack([trimmed_shuttle, np.full((n_trimmed - len(trimmed_shuttle), 2), np.nan)])

        global_shuttle = np.full((n_total, 2), np.nan)
        for i, orig_f in enumerate(frame_map):
            if i < len(trimmed_shuttle): global_shuttle[orig_f] = trimmed_shuttle[i]

        safe_trimmed_poses = [pf if pf is not None else EmptyPoseFrame() for pf in trimmed_poses]
        global_poses = [EmptyPoseFrame()] * n_total
        for i, orig_f in enumerate(frame_map):
            if i < len(safe_trimmed_poses): global_poses[orig_f] = safe_trimmed_poses[i]

        master_traj_3d = np.full((n_total, 3), np.nan)
        master_traj_2d = np.full((n_total, 2), np.nan)
        master_reproj  = np.full(n_total, np.nan)
        
        # Reconstruction Loop with required Logging
        for rally_id, group in gt_df.groupby(["set_file", "rally"]):
            hits = group.to_dict('records')
            set_name = rally_id[0]
            r_num = rally_id[1]
            logger.info(f"[INFO] Source Video: {match_folder} | Set: {set_name} | Rally: {r_num} | Shots to process: {len(hits)}")
            
            for i in range(len(hits)):
                start_frame = hits[i]["frame_num"]
                end_frame = hits[i+1]["frame_num"] if i < len(hits)-1 else min(start_frame + 30, n_total)
                gt_side = hits[i].get("hitter_side_gt", "Unknown")
                
                if gt_side not in ["Top", "Bottom"]: continue
                h_side = "far" if gt_side == "Top" else "near"
                other_side = "near" if h_side == "far" else "far"
                
                shot_shuttle = global_shuttle[start_frame:end_frame]
                hitter_3d = get_player_3d(global_poses, start_frame, h_side, K, rvec, tvec, use_floor=False)
                recv_idx = min(end_frame - 1, n_total - 1)
                receiver_3d = get_player_3d(global_poses, recv_idx, other_side, K, rvec, tvec, use_floor=False)
                
                if hitter_3d is None or receiver_3d is None: continue
                    
                try:
                    res = reconstruct(shot_shuttle, P, hitter_3d, receiver_3d, fps, h_side)
                    if res["converged"]:
                        actual_len = min(start_frame + len(res["traj_3d"]), n_total) - start_frame
                        if actual_len > 0:
                            master_traj_3d[start_frame:start_frame+actual_len] = res["traj_3d"][:actual_len]
                            master_traj_2d[start_frame:start_frame+actual_len] = res["traj_2d_proj"][:actual_len]
                            master_reproj[start_frame:start_frame+actual_len] = res["reproj_err"][:actual_len]
                except Exception as e:
                    logger.error(f"  [!] Reconstruct failed at frame {start_frame}: {e}")

        logger.info("Extracting and saving trimmed trajectory data...")
        
        trimmed_traj_3d = np.full((n_trimmed, 3), np.nan)
        trimmed_traj_2d = np.full((n_trimmed, 2), np.nan)
        
        for trimmed_idx, orig_f in enumerate(frame_map):
            trimmed_traj_3d[trimmed_idx] = master_traj_3d[orig_f]
            trimmed_traj_2d[trimmed_idx] = master_traj_2d[orig_f]
            
        np.save(traj_dir / f"{match_folder}_traj_3d_trimmed.npy", trimmed_traj_3d)
        np.save(traj_dir / f"{match_folder}_traj_2d_trimmed.npy", trimmed_traj_2d)
        logger.info("Saved strictly 1:1 mapped trimmed trajectory arrays.")

        # Final Dump
        out_vid_path = traj_dir / f"{match_folder}_traj_overlay.mp4"
        custom_trajectory_overlay(trimmed_video_path, out_vid_path, global_shuttle, master_traj_2d, master_traj_3d, frame_map, global_poses, K, rvec, tvec, fps, logger)

if __name__ == "__main__": main()