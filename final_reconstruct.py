"""
final_reconstruct.py
════════════════════
Interactive end-to-end 3D Trajectory Reconstruction script for ShuttleSet videos.
"""

import os
import sys
import logging
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from config import TRACKNET_DIR
from court_calibration import load_calibration, backproject_to_floor
from shuttle_detection import run_tracknet, parse_tracknet_csv, clean_detections
from shuttle_detection import render_debug_video as dump_shuttle_video
from pose_estimation import load_pose_backend, estimate_poses_batched, get_player_3d
from shot_classification import fill_linear_shuttle_gaps

try:
    from trajectory import reconstruct, save_trajectory, make_plots
except ImportError:
    print("[!] Ensure 'trajectory.py' is in the root directory to run reconstruction.")
    sys.exit(1)

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(out_dir / "traj_eval.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("ReconstructEval")

class EmptyPoseFrame:
    near = None
    far = None

def create_rally_subset_video(video_path: str, gt_df: pd.DataFrame, n_total: int, out_path: str, logger: logging.Logger, buffer: int = 60, skip_extraction: bool = False):
    logger.info("Calculating required frames for selected rallies...")
    required_frames = set()
    
    for rally_id, group in gt_df.groupby(["set_file", "rally"]):
        first_hit = int(group["frame_num"].min())
        last_hit = int(group["frame_num"].max())
        for f in range(max(0, first_hit - buffer), min(n_total, last_hit + buffer + 1)):
            required_frames.add(f)
            
    sorted_frames = sorted(list(required_frames))
    max_frame = sorted_frames[-1] if sorted_frames else 0
    fast_lookup = set(sorted_frames)
    
    if skip_extraction:
        logger.info(f"Skipping video slicing. Mapped {len(sorted_frames)} frames instantly.")
        return sorted_frames
    
    logger.info(f"Total required frames: {len(sorted_frames)} out of {n_total}. Slicing video (Fast-Grab Mode)...")
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    curr_frame = 0
    while curr_frame <= max_frame:
        if curr_frame % 5000 == 0: logger.info(f"  -> Scanning frame {curr_frame} / {max_frame}...")
        if curr_frame in fast_lookup:
            ret, frame = cap.read()
            if ret: out.write(frame)
        else:
            ret = cap.grab()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 1)
        curr_frame += 1
        
    cap.release()
    out.release()
    logger.info(f"Temporary trimmed video created at {out_path}")
    return sorted_frames

def run_manual_calibration(video_path: str, calib_dir: str, frame_idx: int, logger: logging.Logger):
    logger.info(f"Extracting frame {frame_idx} for calibration...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error(f"Could not read video to extract calibration frame {frame_idx}.")
        return
        
    img_path = os.path.join(calib_dir, "calib_frame.jpg")
    cv2.imwrite(img_path, frame)
    
    logger.info("Launching manual calibration window...")
    cmd = [sys.executable, "court_calibration.py", "--image", img_path, "--out_dir", calib_dir]
    subprocess.run(cmd)

def draw_pose_frame(frame, pf_obj, coco_pairs, draw_coords=False, K=None, rvec=None, tvec=None):
    if not pf_obj: return
    players = []
    if hasattr(pf_obj, 'near') and pf_obj.near is not None: players.append(("Near", pf_obj.near))
    if hasattr(pf_obj, 'far') and pf_obj.far is not None: players.append(("Far", pf_obj.far))

    for label, person in players:
        kpts = None
        if hasattr(person, 'keypoints') and person.keypoints is not None: kpts = np.array(person.keypoints)

        if kpts is not None and kpts.ndim >= 2 and kpts.shape[0] >= 17:
            for p1, p2 in coco_pairs:
                x1, y1, x2, y2 = int(kpts[p1][0]), int(kpts[p1][1]), int(kpts[p2][0]), int(kpts[p2][1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y in kpts[:, :2]:
                if x > 0 and y > 0: cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

            if draw_coords and K is not None and rvec is not None and tvec is not None:
                ankles = [px for px in [person.left_ankle_px, person.right_ankle_px] if px is not None]
                if ankles:
                    u_avg, v_avg = np.mean(ankles, axis=0)
                    fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
                    if fp is not None:
                        valid_kpts = [k for k in kpts if k[0] > 0 and k[1] > 0]
                        if valid_kpts:
                            min_y = min([k[1] for k in valid_kpts])
                            avg_x = sum([k[0] for k in valid_kpts]) / len(valid_kpts)
                            text = f"{label}: (X:{fp[0]:.1f}, Y:{fp[1]:.1f})"
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (int(avg_x)-35, int(min_y)-th-15), (int(avg_x)-35+tw, int(min_y)-5), (0,0,0), -1)
                            cv2.putText(frame, text, (int(avg_x)-30, int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def dump_pose_video(video_path: str, trimmed_poses: list, out_path: str, K=None, rvec=None, tvec=None):
    print("Generating Pose Annotated Video (with Coordinates)...")
    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    coco_pairs = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx < len(trimmed_poses):
            draw_pose_frame(frame, trimmed_poses[frame_idx], coco_pairs, draw_coords=True, K=K, rvec=rvec, tvec=tvec)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def dump_trajectory_video(trimmed_video_path: str, out_path: Path, master_traj_2d: np.ndarray, frame_map: list, fps: float, w: int, h: int, logger: logging.Logger):
    logger.info("Generating 3D Trajectory Overlay Video...")
    cap = cv2.VideoCapture(trimmed_video_path)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        orig_f = frame_map[frame_idx] if frame_idx < len(frame_map) else -1
        if orig_f != -1:
            for past_idx in range(max(0, orig_f - 15), orig_f + 1):
                if past_idx < len(master_traj_2d):
                    px, py = master_traj_2d[past_idx]
                    if not np.isnan(px) and not np.isnan(py):
                        intensity = int(255 * (1 - (orig_f - past_idx) / 15.0))
                        cv2.circle(frame, (int(px), int(py)), 5, (0, intensity, 255), -1)
                        
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()

def load_shuttleset_gt(match_folder: str, scope: dict) -> pd.DataFrame:
    base_dir = Path("shuttleset/set") / match_folder
    dfs = []
    for set_file in scope["sets"]:
        df = pd.read_csv(base_dir / set_file, encoding='utf-8') 
        if 'frame_nur' in df.columns: df.rename(columns={'frame_nur': 'frame_num'}, inplace=True)
        df['set_file'] = set_file
        if scope["rallies"]: df = df[df['rally'].isin(scope["rallies"])]
        dfs.append(df)
        
    if not dfs: return pd.DataFrame()
    combined_df = pd.concat(dfs, ignore_index=True).dropna(subset=["frame_num"])
    combined_df["frame_num"] = combined_df["frame_num"].astype(int)
    
    if 'player_location_y' in combined_df.columns and 'opponent_location_y' in combined_df.columns:
        valid_pos_mask = combined_df['player_location_y'].notna() & combined_df['opponent_location_y'].notna()
        combined_df.loc[valid_pos_mask, 'hitter_side_gt'] = np.where(
            combined_df.loc[valid_pos_mask, 'player_location_y'] < combined_df.loc[valid_pos_mask, 'opponent_location_y'],
            'Top', 'Bottom'
        )
    else: combined_df['hitter_side_gt'] = 'Unknown'

    combined_df["gt_type"] = combined_df["type"].astype(str)
    combined_df["skip"] = combined_df["gt_type"] == "接不到"
    return combined_df.sort_values("frame_num").reset_index(drop=True)

def main():
    print("🏸 ShuttleSet 3D Trajectory Reconstruction Pipeline 🏸")
    match_folder = input("\nEnter the match folder name (e.g., Kento_MOMOTA...): ").strip()
    video_path = Path(f"test_assets/{match_folder}.mp4")
    if not video_path.exists():
        print(f"[!] Video not found at {video_path}")
        sys.exit(1)

    base_out = Path("test_shuttlenet") / match_folder
    shuttle_dir, pose_dir, traj_dir = base_out/"shuttle_out", base_out/"pose_out", base_out/"traj_out"
    for d in [shuttle_dir, pose_dir, traj_dir]: d.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(traj_dir)

    print("\n═ RUN SCOPE CONFIGURATION ═")
    avail_sets = sorted([f for f in os.listdir(Path("shuttleset/set") / match_folder) if f.startswith("set")])
    print("1. Process specific Set No. AND exact Rally Numbers\n2. Process specific Set No. (entire set)\n3. Process ALL sets")
    c = input("Select an option (1/2/3): ").strip()
    
    scope = {"sets": avail_sets, "rallies": None}
    if c in ['1', '2']:
        print(f"Available sets: {avail_sets}")
        set_choice = input("Enter Set filename: ").strip()
        scope["sets"] = [set_choice] if set_choice in avail_sets else avail_sets
        if c == '1': scope["rallies"] = [int(r.strip()) for r in input("Enter Rally numbers (comma separated): ").split(',')]

    gt_df = load_shuttleset_gt(match_folder, scope)
    if gt_df.empty:
        logger.error("No valid ground truth data found for the selected scope.")
        sys.exit(1)
    
    first_rally_num = gt_df["rally"].iloc[0]
    first_rally_df = gt_df[gt_df["rally"] == first_rally_num]
    
    first_hit_frame = int(first_rally_df["frame_num"].min())
    calib_target_frame = int(first_rally_df.iloc[1]["frame_num"]) if len(first_rally_df) > 1 else first_hit_frame

    logger.info(f"Loaded {len(gt_df)} strokes. 1st shot at {first_hit_frame}. Calibrating at {calib_target_frame} (2nd shot).")

    cap = cv2.VideoCapture(str(video_path))
    v_w, v_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    trimmed_video_path = str(base_out / "trimmed_temp.mp4")
    skip_trimming = False
    if Path(trimmed_video_path).exists():
        ans = input(f"\n[?] Found existing trimmed video at {trimmed_video_path}.\nSkip extraction and use existing file? (y/n): ").strip().lower()
        if ans == 'y': skip_trimming = True

    frame_map = create_rally_subset_video(str(video_path), gt_df, n_total, trimmed_video_path, logger, skip_extraction=skip_trimming)
    n_trimmed = len(frame_map)

    print("\n═ 3D CAMERA CALIBRATION ═")
    print("Homography is insufficient for 3D Reconstruction. Full 3x4 P matrix is required.")
    print("1. Use existing stored (cache in calib_out)")
    print("2. Recalculate using manual court calibration tool")
    homo_mode = input("Select an option (1/2): ").strip()

    calib_dir = base_out / "calib_out"
    calib_dir.mkdir(parents=True, exist_ok=True)

    if homo_mode == "2":
        logger.info("Opening manual calibration window via subprocess...")
        run_manual_calibration(str(video_path), str(calib_dir), calib_target_frame, logger)
        
    try:
        P = np.load(calib_dir / "P.npy")
        _, K, rvec, tvec = load_calibration(str(calib_dir))
    except Exception as e:
        logger.error(f"Failed to load P.npy from {calib_dir}. Did you complete calibration? {e}")
        sys.exit(1)

    print("\n═ SHUTTLE DETECTION ═")
    shuttle_cache = shuttle_dir / "shuttle_trimmed.npy"
    use_shuttle_cache = False
    if shuttle_cache.exists():
        if input(f"Cached raw shuttle detections found. Use it? (y/n): ").strip().lower() == 'y':
            use_shuttle_cache = True
            
    if use_shuttle_cache:
        trimmed_shuttle = np.load(shuttle_cache)
    else:
        logger.info("Running TrackNetV3 on TRIMMED video...")
        csv_path = run_tracknet(video_path=trimmed_video_path, weights_dir=str(TRACKNET_DIR), raw_out_dir=str(shuttle_dir), eval_mode="weight", batch_size=4, large_video=True)
        trimmed_shuttle = parse_tracknet_csv(csv_path, trimmed_video_path) 
        np.save(shuttle_cache, trimmed_shuttle)
        logger.info(f"Raw shuttle detections saved to {shuttle_cache}")

    logger.info("Cleaning false positive detections & filling gaps...")
    trimmed_shuttle = clean_detections(trimmed_shuttle, trimmed_video_path)
    trimmed_shuttle = fill_linear_shuttle_gaps(trimmed_shuttle, max_gap=30)

    if len(trimmed_shuttle) < n_trimmed: trimmed_shuttle = np.vstack([trimmed_shuttle, np.full((n_trimmed - len(trimmed_shuttle), 2), np.nan)])
    
    if input("Do you want to dump the annotated video for SHUTTLE DETECTION? (y/n): ").strip().lower() == 'y':
        dump_shuttle_video(trimmed_video_path, trimmed_shuttle[:n_trimmed], str(shuttle_dir / f"{match_folder}_shuttle_trimmed.mp4"))

    global_shuttle = np.full((n_total, 2), np.nan)
    for i, orig_f in enumerate(frame_map):
        if i < len(trimmed_shuttle): global_shuttle[orig_f] = trimmed_shuttle[i]

    print("\n═ POSE ESTIMATION ═")
    pose_cache = pose_dir / "poses_trimmed.pkl"
    use_pose_cache = False
    if pose_cache.exists():
        if input(f"Cached pose estimations found. Use it? (y/n): ").strip().lower() == 'y':
            use_pose_cache = True

    if use_pose_cache:
        with open(pose_cache, "rb") as f: trimmed_poses = pickle.load(f)
    else:
        logger.info("Extracting frames & Running RTMPose on TRIMMED video (Streaming Mode to save RAM)...")
        cap = cv2.VideoCapture(trimmed_video_path)
        det_m, pose_m, _ = load_pose_backend()
        trimmed_poses = []
        batch_frames = []
        batch_size = 32
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            batch_frames.append(frame)
            if len(batch_frames) == batch_size:
                batch_res = estimate_poses_batched(batch_frames, K, rvec, tvec, det_m, pose_m, batch_size=batch_size)
                trimmed_poses.extend(batch_res)
                batch_frames.clear()
                
        if len(batch_frames) > 0:
            batch_res = estimate_poses_batched(batch_frames, K, rvec, tvec, det_m, pose_m, batch_size=len(batch_frames))
            trimmed_poses.extend(batch_res)
            batch_frames.clear()
            
        cap.release()
        with open(pose_cache, "wb") as f: pickle.dump(trimmed_poses, f)
        
    if input("Do you want to dump the annotated video for POSE ESTIMATION? (y/n): ").strip().lower() == 'y':
        dump_pose_video(trimmed_video_path, trimmed_poses, str(pose_dir / f"{match_folder}_pose_trimmed.mp4"), K, rvec, tvec)

    safe_trimmed_poses = [pf if pf is not None else EmptyPoseFrame() for pf in trimmed_poses]
    global_poses = [EmptyPoseFrame()] * n_total
    for i, orig_f in enumerate(frame_map):
        if i < len(safe_trimmed_poses): global_poses[orig_f] = safe_trimmed_poses[i]

    print("\n═ 3D TRAJECTORY RECONSTRUCTION ═")
    
    master_traj_3d = np.full((n_total, 3), np.nan)
    master_traj_2d = np.full((n_total, 2), np.nan)
    master_reproj  = np.full(n_total, np.nan)
    
    v0_list, cd_list, x0_list = [], [], []
    all_converged = True

    for rally_id, group in gt_df.groupby(["set_file", "rally"]):
        hits = group.to_dict('records')
        logger.info(f"Reconstructing Rally {rally_id[1]} from {rally_id[0]} ({len(hits)} hits)...")
        
        for i in range(len(hits)):
            start_frame = hits[i]["frame_num"]
            end_frame = hits[i+1]["frame_num"] if i < len(hits)-1 else min(start_frame + 30, n_total)
            
            gt_side = hits[i]["hitter_side_gt"]
            if gt_side not in ["Top", "Bottom"]: continue
                
            h_side = "far" if gt_side == "Top" else "near"
            other_side = "near" if h_side == "far" else "far"
            
            shot_shuttle = global_shuttle[start_frame:end_frame]
            
            hitter_3d = get_player_3d(global_poses, start_frame, h_side, K, rvec, tvec, use_floor=False)
            
            recv_idx = min(end_frame - 1, n_total - 1)
            receiver_3d = get_player_3d(global_poses, recv_idx, other_side, K, rvec, tvec, use_floor=False)
            
            if hitter_3d is None or receiver_3d is None:
                logger.warning(f"  [!] Missing player pose bounds at frames {start_frame}-{recv_idx}. Skipping shot.")
                continue
                
            try:
                res = reconstruct(shot_shuttle, P, hitter_3d, receiver_3d, fps, h_side)
                if res["converged"]:
                    N_shot = len(res["traj_3d"])
                    safe_end = min(start_frame + N_shot, n_total)
                    actual_len = safe_end - start_frame
                    
                    if actual_len > 0:
                        master_traj_3d[start_frame:safe_end] = res["traj_3d"][:actual_len]
                        master_traj_2d[start_frame:safe_end] = res["traj_2d_proj"][:actual_len]
                        master_reproj[start_frame:safe_end] = res["reproj_err"][:actual_len]
                    
                all_converged = all_converged and res.get("converged", True)
                if "v0" in res: v0_list.append(res["v0"])
                if "Cd" in res: cd_list.append(res["Cd"])
                if "x0" in res: x0_list.append(res["x0"])
                
            except Exception as e:
                logger.error(f"  [!] Reconstruction failed for stroke at {start_frame}: {e}")

    combined_result = {
        "traj_3d": master_traj_3d, "traj_2d_proj": master_traj_2d,
        "reproj_err": master_reproj,
        "mean_reproj_err": float(np.nanmean(master_reproj)) if np.any(~np.isnan(master_reproj)) else 0.0,
        "n_frames": n_total, "n_valid": int(np.sum(~np.isnan(master_reproj))),
        "converged": all_converged,
        "v0": np.mean(v0_list, axis=0) if v0_list else np.zeros(3),
        "Cd": float(np.mean(cd_list)) if cd_list else 0.0,
        "x0": x0_list[0] if x0_list else np.zeros(3)
    }

    overall_mean = combined_result["mean_reproj_err"]
    overall_quality = ("EXCELLENT" if overall_mean < 2 else "GOOD" if overall_mean < 5 else "ACCEPTABLE" if overall_mean < 15 else "POOR")
    logger.info(f"\n=> [OVERALL] Average Reprojection Error: {overall_mean:.3f} px [{overall_quality}]")

    save_trajectory(combined_result, traj_dir)
    logger.info("Generating trajectory plots...")
    make_plots(combined_result, fps, str(traj_dir / f"{match_folder}_trajectory_plots.png"))

    if input("\nDump Final Trajectory Overlay Video? (y/n): ").strip().lower() == 'y':
        dump_trajectory_video(trimmed_video_path, traj_dir / f"{match_folder}_traj_overlay.mp4", master_traj_2d, frame_map, fps, v_w, v_h, logger)

if __name__ == "__main__": main()