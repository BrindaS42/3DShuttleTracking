"""
run_pipeline.py
═══════════════
Upgraded Orchestrator for MonoTrack.
Features:
- Batched GPU inference for Shuttle & Poses.
- Parallelized 3D trajectory reconstruction.
- Strict Player Pose Priors (No temporal chaining drift).
"""

import argparse
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from shot_classification import mark_hits_ui, fill_linear_shuttle_gaps
from config import VIDEO_FPS

def process_single_shot(shot_idx, start_frame, end_frame, hitter_side, 
                        global_shuttle_2d, P, poses, fps, n_total_frames, K, rvec, tvec):
    """Worker function to reconstruct a single shot in parallel."""
    from trajectory import reconstruct
    from pose_estimation import get_player_3d
    
    shot_shuttle = global_shuttle_2d[start_frame:end_frame]
    other_side = "far" if hitter_side == "near" else "near"
    
    # ── PASS CALIBRATION MATRICES ──
    hitter_3d = get_player_3d(poses, start_frame, hitter_side, K, rvec, tvec, use_floor=False)
    
    recv_idx = min(end_frame - 1, len(poses) - 1)
    receiver_3d = get_player_3d(poses, recv_idx, other_side, K, rvec, tvec, use_floor=False)
    
    res = reconstruct(shot_shuttle, P, hitter_3d, receiver_3d, fps, hitter_side)
    
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

def validate_and_interpolate_poses(poses, fps, K, rvec, tvec):
    """
    1. Temporarily projects to 3D to detect physically impossible jumps.
    2. Nullifies bad RAW 2D ankle positions.
    3. Interpolates missing 2D ankle positions strictly within the image plane.
    """
    from court_calibration import backproject_to_floor
    MAX_VELOCITY = 10.0  # m/s (Maximum human sprint speed)
    dist_threshold = MAX_VELOCITY / fps 

    for side in ['near', 'far']:
        last_valid_pos = None
        
        # --- STEP 1: Identify and Wipe Outliers in 2D Space ---
        for i, pf in enumerate(poses):
            player = getattr(pf, side)
            ankles = [px for px in [player.left_ankle_px, player.right_ankle_px] if px is not None]
            if not ankles: continue
                
            u_avg, v_avg = np.mean(ankles, axis=0)
            fp = backproject_to_floor(u_avg, v_avg, K, rvec, tvec, target_z=0.0)
            
            if fp is None:
                player.left_ankle_px, player.right_ankle_px = None, None
                continue
                
            curr_pos = np.array(fp[:2])
            
            # A: Out-of-Bounds Check (Hard glitches)
            if curr_pos[1] >= 13.4 or curr_pos[1] <= -1.0: 
                player.left_ankle_px, player.right_ankle_px = None, None
                continue

            # B: Velocity Check (Abrupt jumps)
            if last_valid_pos is not None:
                dist = np.linalg.norm(curr_pos - last_valid_pos)
                if dist > dist_threshold:
                    player.left_ankle_px, player.right_ankle_px = None, None
                    continue
            
            last_valid_pos = curr_pos

        # --- STEP 2: Interpolate the Raw 2D Ankle Pixels ---
        valid_idx, valid_lx, valid_ly, valid_rx, valid_ry = [], [], [], [], []
        
        for i, pf in enumerate(poses):
            player = getattr(pf, side)
            if player.left_ankle_px is not None and player.right_ankle_px is not None:
                valid_idx.append(i)
                valid_lx.append(player.left_ankle_px[0]); valid_ly.append(player.left_ankle_px[1])
                valid_rx.append(player.right_ankle_px[0]); valid_ry.append(player.right_ankle_px[1])

        if len(valid_idx) > 1:
            all_idx = np.arange(len(poses))
            ilx = np.interp(all_idx, valid_idx, valid_lx)
            ily = np.interp(all_idx, valid_idx, valid_ly)
            irx = np.interp(all_idx, valid_idx, valid_rx)
            iry = np.interp(all_idx, valid_idx, valid_ry)
            
            for i, pf in enumerate(poses):
                player = getattr(pf, side)
                if player.left_ankle_px is None or player.right_ankle_px is None:
                    player.left_ankle_px = [float(ilx[i]), float(ily[i])]
                    player.right_ankle_px = [float(irx[i]), float(iry[i])]
                    
    return poses

def main():
    ap = argparse.ArgumentParser(description="MonoTrack Full GPU Pipeline")
    ap.add_argument("--video",       default="test_assets/half_rally.mp4")
    ap.add_argument("--image",       default="test_assets/test_image.jpg", help="Frame for calibration")
    ap.add_argument("--tracknet",    default="tracknet_weights")
    ap.add_argument("--first_hitter",choices=["near", "far"], default="near")
    ap.add_argument("--calib_dir",   default="results/calib_out")
    ap.add_argument("--shuttle_dir", default="results/shuttle_out")
    ap.add_argument("--pose_dir",    default="results/pose_out")
    ap.add_argument("--traj_dir",    default="results/traj_out")
    ap.add_argument("--pose_batch_size", type=int, default=8)
    ap.add_argument("--traj_workers",    type=int, default=6)
    ap.add_argument("--preprocess",      action="store_true", help="Background-subtract video for TrackNet")
    ap.add_argument("--skip_calib",  action="store_true")
    ap.add_argument("--skip_shuttle",action="store_true")
    ap.add_argument("--skip_pose",   action="store_true")
    ap.add_argument("--only_traj",   action="store_true")

    args = ap.parse_args()
    if args.only_traj:
        args.skip_calib = args.skip_shuttle = args.skip_pose = True

    sep = "─" * 60

    out_traj = Path(args.traj_dir)
    out_traj.mkdir(parents=True, exist_ok=True)
    hits_cache = out_traj / "marked_hits.json"
    hit_frames = []
    
    if hits_cache.exists():
        if input(f"[?] Reuse existing hits from {hits_cache}? (y/n): ").lower() == 'y':
            with open(hits_cache, "r") as f: hit_frames = json.load(f)
    
    if not hit_frames:
        hit_frames = mark_hits_ui(args.video)
        with open(hits_cache, "w") as f: json.dump(hit_frames, f)

    cap = cv2.VideoCapture(args.video)
    fps = float(VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not args.skip_calib:
        print(f"\n{sep}\nMODULE 1 — Court Calibration\n{sep}")
        from court_calibration import calibrate, verify_and_save, save_calibration
        out = Path(args.calib_dir)
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ret, calib_frame = cap.read()
        cap.release()
        calib_img_path = str(out / "calib_source_frame.jpg")
        cv2.imwrite(calib_img_path, calib_frame)
        P, K, rvec, tvec, img_pts = calibrate(calib_img_path)
        verify_and_save(P, K, rvec, tvec, img_pts, calib_img_path, out)
        save_calibration(P, K, rvec, tvec, out)
    else:
        print("Module 1 skipped.")

    if not args.skip_shuttle:
        print(f"\n{sep}\nMODULE 2 — Shuttle Detection\n{sep}")
        from shuttle_detection import (
            preprocess_video, run_tracknet, parse_tracknet_csv, 
            clean_detections, detection_stats, save_shuttle, save_stats, render_debug_video
        )
        out = Path(args.shuttle_dir)
        track_input = args.video
        if args.preprocess: track_input = preprocess_video(args.video, str(out / "preprocessed.mp4"))

        csv_path = run_tracknet(track_input, args.tracknet, str(out), eval_mode="weight", batch_size=4)
        shuttle_2d = parse_tracknet_csv(csv_path, args.video)
        shuttle_2d = clean_detections(shuttle_2d, args.video, hit_frames=hit_frames)
        save_shuttle(shuttle_2d, out)
        save_stats(detection_stats(shuttle_2d), out)
        render_debug_video(args.video, shuttle_2d, str(out / "shuttle_detection.mp4"))
    else: print("Module 2 skipped.")

    if not args.skip_pose:
        print(f"\n{sep}\nMODULE 3 — Pose Estimation\n{sep}")
        from court_calibration import load_calibration
        from pose_estimation import load_pose_backend, estimate_poses_batched, save_poses, render_pose_debug
        
        out = Path(args.pose_dir)
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
        save_poses(poses, out)
        render_pose_debug(frames, poses, fps, str(out / "pose_debug.mp4"), K, rvec, tvec)
    else: print("Module 3 skipped.")

    print(f"\n{sep}\nMODULE 4 — Parallel 3D Trajectory Reconstruction\n{sep}")
    from court_calibration import load_calibration
    from shuttle_detection import load_shuttle
    from pose_estimation import load_poses
    from trajectory import render_annotated, save_trajectory, make_plots

    P, K, rvec, tvec = load_calibration(args.calib_dir)
    global_shuttle = load_shuttle(args.shuttle_dir)
    
    try: 
        global_poses = load_poses(args.pose_dir)
        print("Validating and interpolating raw 2D pose coordinates...")
        global_poses = validate_and_interpolate_poses(global_poses, fps, K, rvec, tvec)
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

    print(f"Executing {len(shot_intervals)} shots in parallel...\n")
    
    with ThreadPoolExecutor(max_workers=args.traj_workers) as executor:
        futures = {
            executor.submit(process_single_shot, idx, start, end, hitter, 
                            global_shuttle, P, global_poses, fps, n_total_frames, K, rvec, tvec): idx 
            for (idx, start, end, hitter) in shot_intervals
        }
        
        for future in as_completed(futures):
            idx, res = future.result()
            
            mean_e = res.get("mean_reproj_err", 0.0)
            quality = ("EXCELLENT" if mean_e < 2  else "GOOD" if mean_e < 5  else "ACCEPTABLE" if mean_e < 15 else "POOR")
            
            h3d = res.get("hitter_3d")
            r3d = res.get("receiver_3d")
            ss  = res.get("shuttle_start")
            se  = res.get("shuttle_end")
            bp0 = res.get("best_p0")
            
            h3d_str = f"X:{h3d[0]:5.2f}  Y:{h3d[1]:5.2f}  Z:{h3d[2]:5.2f}" if h3d is not None else "N/A"
            r3d_str = f"X:{r3d[0]:5.2f}  Y:{r3d[1]:5.2f}  Z:{r3d[2]:5.2f}" if r3d is not None else "N/A"
            ss_str  = f"X:{ss[0]:5.2f}  Y:{ss[1]:5.2f}  Z:{ss[2]:5.2f}" if ss is not None else "N/A"
            se_str  = f"X:{se[0]:5.2f}  Y:{se[1]:5.2f}  Z:{se[2]:5.2f}" if se is not None else "N/A"
            bp0_str = f"X:{bp0[0]:5.2f}  Y:{bp0[1]:5.2f}  Z:{bp0[2]:5.2f} | Vx:{bp0[3]:6.1f}  Vy:{bp0[4]:6.1f}  Vz:{bp0[5]:6.1f}" if bp0 is not None else "N/A"
            
            print(f"  ✓ Shot {idx+1:02d}/{len(shot_intervals)} | Mean reproj error: {mean_e:6.3f} px [{quality}]")
            print(f"      Hitter Pose (Start) : {h3d_str}")
            print(f"      Shuttle Start       : {ss_str}")
            print(f"      Winning Initial Guess -> {bp0_str}")
            print(f"      Receiver Pose (End) : {r3d_str}")
            print(f"      Shuttle End         : {se_str}\n")

            valid_mask = ~np.isnan(res["global_reproj"])
            master_traj_3d[valid_mask] = res["global_traj_3d"][valid_mask]
            master_traj_2d[valid_mask] = res["global_traj_2d"][valid_mask]
            master_reproj[valid_mask]  = res["global_reproj"][valid_mask]

            all_converged = all_converged and res.get("converged", True)
            if "v0" in res: v0_list.append(res["v0"])
            if "Cd" in res: cd_list.append(res["Cd"])
            if "x0" in res: x0_list.append(res["x0"])

    combined_result = {
        "traj_3d": master_traj_3d, "traj_2d_proj": master_traj_2d,
        "reproj_err": master_reproj,
        "mean_reproj_err": float(np.nanmean(master_reproj)) if np.any(~np.isnan(master_reproj)) else 0.0,
        "n_frames": n_total_frames, "n_valid": int(np.sum(~np.isnan(master_reproj))),
        "converged": all_converged,
        "v0": np.mean(v0_list, axis=0) if v0_list else np.zeros(3),
        "Cd": float(np.mean(cd_list)) if cd_list else 0.0,
        "x0": x0_list[0] if x0_list else np.zeros(3)
    }

    overall_mean = combined_result["mean_reproj_err"]
    overall_quality = ("EXCELLENT" if overall_mean < 2  else "GOOD" if overall_mean < 5  else "ACCEPTABLE" if overall_mean < 15 else "POOR")
    print(f"\n  => [OVERALL RALLY] Average pixel error: {overall_mean:.3f} px [{overall_quality}]")

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
    render_annotated(frames_for_vid, combined_result, global_shuttle, global_poses, fps, str(out_traj / "output_annotated.mp4"))

    print(f"\n{sep}\nPIPELINE COMPLETE\n{sep}")
    print(f"Results saved to: {out_traj}")

if __name__ == "__main__":
    main()