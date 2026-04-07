"""
run_pipeline.py
═══════════════
Upgraded Orchestrator for MonoTrack.
Features:
- Extracts Frame 0 for calibration dynamically.
- Manual Hit Frame selection (with caching).
- Batched GPU inference for Shuttle & Poses.
- Strictly linear interpolation for shuttle gaps (no physics filtering).
- Parallelized 3D trajectory reconstruction for all shots in a rally.
- Direct console readout for pixel errors (No text files).
"""

import argparse
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import UI and Linear Fill from shot_classification
from shot_classification import mark_hits_ui, fill_linear_shuttle_gaps
from config import VIDEO_FPS

def process_single_shot(shot_idx, start_frame, end_frame, hitter_side, 
                        global_shuttle_2d, P, poses, fps, n_total_frames):
    """Worker function to reconstruct a single shot in parallel."""
    from trajectory import reconstruct
    from pose_estimation import get_player_3d
    
    # 1. Slice shuttle data for just this shot
    shot_shuttle = global_shuttle_2d[start_frame:end_frame]
    
    # 2. Get 3D Player Priors
    other_side = "far" if hitter_side == "near" else "near"
    hitter_3d = get_player_3d(poses, start_frame, hitter_side, use_floor=False)
    
    # Receiver prior is taken from the end of the shot
    recv_idx = min(end_frame - 1, len(poses) - 1)
    receiver_3d = get_player_3d(poses, recv_idx, other_side, use_floor=False)
    
    # 3. Run Optimization
    res = reconstruct(shot_shuttle, P, hitter_3d, receiver_3d, fps, hitter_side)
    
    # 4. Pad results back to global frame length
    global_traj_3d = np.full((n_total_frames, 3), np.nan, dtype=np.float64)
    global_traj_2d = np.full((n_total_frames, 2), np.nan, dtype=np.float64)
    global_reproj  = np.full(n_total_frames, np.nan, dtype=np.float64)
    
    if start_frame < n_total_frames:
        N_shot = len(res["traj_3d"])
        safe_end = min(start_frame + N_shot, n_total_frames)
        actual_len = safe_end - start_frame
        
        if actual_len > 0:
            global_traj_3d[start_frame:safe_end] = res["traj_3d"][:actual_len]
            global_traj_2d[start_frame:safe_end] = res["traj_2d_proj"][:actual_len]
            global_reproj[start_frame:safe_end]  = res["reproj_err"][:actual_len]
    
    res["global_traj_3d"] = global_traj_3d
    res["global_traj_2d"] = global_traj_2d
    res["global_reproj"]  = global_reproj
    
    return shot_idx, res


def main():
    ap = argparse.ArgumentParser(description="MonoTrack Full GPU Pipeline")

    ap.add_argument("--video",       default="test_assets/half_rally.mp4")
    ap.add_argument("--tracknet",    default="tracknet_weights")
    ap.add_argument("--first_hitter",choices=["near", "far"], default="near",
                    help="Who hits the first shot of the rally?")

    # Output directories
    ap.add_argument("--calib_dir",   default="results/calib_out")
    ap.add_argument("--shuttle_dir", default="results/shuttle_out")
    ap.add_argument("--pose_dir",    default="results/pose_out")
    ap.add_argument("--traj_dir",    default="results/traj_out")

    # GPU & Parallelization controls
    ap.add_argument("--tn_batch_size", type=int, default=4, help="TrackNet batch size")
    ap.add_argument("--pose_batch_size", type=int, default=4, help="RTMPose batch size")
    ap.add_argument("--traj_workers", type=int, default=4, help="Threads for 3D physics solver")

    # Flags
    ap.add_argument("--skip_calib",  action="store_true")
    ap.add_argument("--skip_shuttle",action="store_true")
    ap.add_argument("--skip_pose",   action="store_true")
    ap.add_argument("--only_traj",   action="store_true")

    args = ap.parse_args()
    if args.only_traj:
        args.skip_calib = args.skip_shuttle = args.skip_pose = True

    sep = "─" * 60

    # ── 0. Manual Hit Frame Selection ──────────────────────────────
    print(f"\n{sep}\nMODULE 0 — Hit Frame Selection\n{sep}")
    out_traj = Path(args.traj_dir)
    out_traj.mkdir(parents=True, exist_ok=True)
    hits_cache = out_traj / "marked_hits.json"
    hit_frames = []
    
    if hits_cache.exists():
        ans = input(f"[?] Reuse existing hits from {hits_cache}? (y/n): ").lower()
        if ans == 'y':
            with open(hits_cache, "r") as f:
                hit_frames = json.load(f)
            print(f"Loaded {len(hit_frames)} hits.")
    
    if not hit_frames:
        print("Opening UI to mark hits...")
        hit_frames = mark_hits_ui(args.video)
        with open(hits_cache, "w") as f:
            json.dump(hit_frames, f)
        print(f"Saved {len(hit_frames)} hits.")

    # Get total frames
    cap = cv2.VideoCapture(args.video)
    fps = float(VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── 1. Module 1: Calibration (Dynamic Frame 0) ──────────────────
    out_calib = Path(args.calib_dir)
    out_calib.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_calib:
        print(f"\n{sep}\nMODULE 1 — Court Calibration (Extracting Frame 0)\n{sep}")
        calib_img_path = out_calib / "calib_frame0.jpg"
        
        cap = cv2.VideoCapture(args.video)
        ret, frame0 = cap.read()
        cap.release()
        if not ret: raise RuntimeError("Failed to read video for calibration.")
        cv2.imwrite(str(calib_img_path), frame0)
        
        from court_calibration import calibrate, verify_and_save, save_calibration
        P, K, rvec, tvec, img_pts = calibrate(str(calib_img_path))
        verify_and_save(P, K, rvec, tvec, img_pts, str(calib_img_path), out_calib)
        save_calibration(P, K, rvec, tvec, out_calib)
    else:
        print("Module 1 skipped.")

    # ── 2. Module 2: Shuttle Detection (Batch + Linear Fill Only) ────
    out_shuttle = Path(args.shuttle_dir)
    out_shuttle.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_shuttle:
        print(f"\n{sep}\nMODULE 2 — Shuttle Detection (Batch GPU)\n{sep}")
        from shuttle_detection import run_tracknet, parse_tracknet_csv, save_shuttle, render_debug_video
        
        csv_path = run_tracknet(
            video_path=args.video, 
            weights_dir=args.tracknet, 
            raw_out_dir=str(out_shuttle),
            eval_mode="average",
            batch_size=args.tn_batch_size
        )
        
        shuttle_2d_raw = parse_tracknet_csv(csv_path, args.video)
        shuttle_2d_clean = fill_linear_shuttle_gaps(shuttle_2d_raw, max_gap=30)
        
        save_shuttle(shuttle_2d_clean, out_shuttle)
        render_debug_video(args.video, shuttle_2d_clean, str(out_shuttle / "shuttle_detection.mp4"))
    else:
        print("Module 2 skipped.")

    # ── 3. Module 3: Pose Estimation (Batched) ────────────────────────
    out_pose = Path(args.pose_dir)
    out_pose.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_pose:
        print(f"\n{sep}\nMODULE 3 — Pose Estimation (Batched GPU)\n{sep}")
        from court_calibration import load_calibration
        from pose_estimation import load_pose_backend, estimate_poses_batched, save_poses
        
        P, K, rvec, tvec = load_calibration(args.calib_dir)
        
        cap = cv2.VideoCapture(args.video)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()
        
        det_m, pose_m, _ = load_pose_backend()
        poses = estimate_poses_batched(frames, K, rvec, tvec, det_m, pose_m, batch_size=args.pose_batch_size)
        save_poses(poses, out_pose)
    else:
        print("Module 3 skipped.")

    # ── 4. Module 4: Trajectory (Parallelized) ────────────────────────
    print(f"\n{sep}\nMODULE 4 — Parallel 3D Trajectory Reconstruction\n{sep}")
    from court_calibration import load_calibration
    from shuttle_detection import load_shuttle
    from pose_estimation import load_poses
    
    # REMOVED: write_report is no longer imported
    from trajectory import render_annotated, save_trajectory, make_plots

    P, K, rvec, tvec = load_calibration(args.calib_dir)
    global_shuttle = load_shuttle(args.shuttle_dir)
    try:
        global_poses = load_poses(args.pose_dir)
    except FileNotFoundError:
        global_poses = []
        print("Poses not found, priors disabled.")

    shot_intervals = []
    hitter_cycle = [args.first_hitter, "far" if args.first_hitter == "near" else "near"]
    
    for i in range(len(hit_frames)):
        start = hit_frames[i]
        end = hit_frames[i+1] if i < len(hit_frames) - 1 else n_total_frames
        hitter = hitter_cycle[i % 2]
        shot_intervals.append((i, start, end, hitter))

    master_traj_3d = np.full((n_total_frames, 3), np.nan, dtype=np.float64)
    master_traj_2d = np.full((n_total_frames, 2), np.nan, dtype=np.float64)
    master_reproj  = np.full(n_total_frames, np.nan, dtype=np.float64)
    
    v0_list, cd_list, x0_list = [], [], []
    all_converged = True

    print(f"Executing {len(shot_intervals)} shots in parallel using {args.traj_workers} workers...\n")
    
    with ThreadPoolExecutor(max_workers=args.traj_workers) as executor:
        futures = {
            executor.submit(process_single_shot, idx, start, end, hitter, 
                            global_shuttle, P, global_poses, fps, n_total_frames): idx 
            for (idx, start, end, hitter) in shot_intervals
        }
        
        for future in as_completed(futures):
            idx, res = future.result()
            
            # --- CONSOLE READOUT PER SHOT ---
            mean_e = res.get("mean_reproj_err", 0.0)
            quality = ("EXCELLENT" if mean_e < 2  else
                       "GOOD"      if mean_e < 5  else
                       "ACCEPTABLE" if mean_e < 15 else "POOR")
            
            print(f"  ✓ Shot {idx+1:02d}/{len(shot_intervals)} | Mean reproj error: {mean_e:6.3f} px [{quality}]")
            # --------------------------------
            
            valid_mask = ~np.isnan(res["global_reproj"])
            master_traj_3d[valid_mask] = res["global_traj_3d"][valid_mask]
            master_traj_2d[valid_mask] = res["global_traj_2d"][valid_mask]
            master_reproj[valid_mask]  = res["global_reproj"][valid_mask]

            all_converged = all_converged and res.get("converged", True)
            if "v0" in res: v0_list.append(res["v0"])
            if "Cd" in res: cd_list.append(res["Cd"])
            if "x0" in res: x0_list.append(res["x0"])

    combined_result = {
        "traj_3d": master_traj_3d,
        "traj_2d_proj": master_traj_2d,
        "reproj_err": master_reproj,
        "mean_reproj_err": float(np.nanmean(master_reproj)) if np.any(~np.isnan(master_reproj)) else 0.0,
        "n_frames": n_total_frames,
        "n_valid": int(np.sum(~np.isnan(master_reproj))),
        "converged": all_converged,
        "v0": np.mean(v0_list, axis=0) if v0_list else np.zeros(3),
        "Cd": float(np.mean(cd_list)) if cd_list else 0.0,
        "x0": x0_list[0] if x0_list else np.zeros(3)
    }

    # --- FINAL COMBINED CONSOLE READOUT ---
    overall_mean = combined_result["mean_reproj_err"]
    overall_quality = ("EXCELLENT" if overall_mean < 2  else
                       "GOOD"      if overall_mean < 5  else
                       "ACCEPTABLE" if overall_mean < 15 else "POOR")
    print(f"\n  => [OVERALL RALLY] Average pixel error: {overall_mean:.3f} px [{overall_quality}]")
    # --------------------------------------

    save_trajectory(combined_result, out_traj)
    print("Generating trajectory plots...")
    make_plots(combined_result, fps, str(out_traj / "trajectory_plots.png"))
    
    cap = cv2.VideoCapture(args.video)
    frames_for_vid = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames_for_vid.append(f)
    cap.release()
    
    print("Rendering annotated video...")
    render_annotated(frames_for_vid, combined_result, global_shuttle, fps, str(out_traj / "output_annotated.mp4"))

    print(f"\n{sep}\nPIPELINE COMPLETE\n{sep}")
    print(f"Results saved to: {out_traj}")

if __name__ == "__main__":
    main()