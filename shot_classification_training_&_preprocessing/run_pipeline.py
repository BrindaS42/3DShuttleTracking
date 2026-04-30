"""
run_pipeline.py
═══════════════
One-command orchestrator for all 4 MonoTrack modules.

Usage
─────
  # Run everything end-to-end
  python run_pipeline.py --video clip.mp4 --hitter near

  # Skip Module 1 (already calibrated)
  python run_pipeline.py --video clip.mp4 --hitter near --skip_calib

  # Skip Modules 1, 2, 3  (rerun optimiser only)
  python run_pipeline.py --video clip.mp4 --hitter near --only_traj

  # Run only Module 1
  python run_pipeline.py --image frame.jpg --only_calib

Module order & what each needs
───────────────────────────────
  M1  Court calibration   needs: broadcast frame (--image)
                          gives: calib_out/P.npy  K.npy  rvec.npy  tvec.npy

  M2  Shuttle detection   needs: --video  +  tracknet weights
                          gives: shuttle_out/shuttle_2d.npy

  M3  Pose estimation     needs: --video  +  calib_out/
                          gives: pose_out/poses.pkl

  M4  3-D trajectory      needs: --video  +  calib_out/  shuttle_out/  pose_out/
                          gives: traj_out/trajectory_3d.npy  +  video  +  plots
"""

import argparse
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="MonoTrack full pipeline")

    # Core inputs
    ap.add_argument("--video",       default="test_assets/test_video_1.mp4",
                    help="Input video (single shot or full rally)")
    ap.add_argument("--image",       default="test_assets/test_image.jpg",
                    help="Broadcast frame for calibration (Module 1)")
    ap.add_argument("--tracknet",    default="tracknet_weights",
                    help="Folder with predict.py + *.pt")

    # Player info
    ap.add_argument("--hitter",      default="near",
                    choices=["near", "far"])
    ap.add_argument("--hit_frame",   type=int, default=0)

    # Output directories
    ap.add_argument("--calib_dir",   default="results/calib_out")
    ap.add_argument("--shuttle_dir", default="results/shuttle_out")
    ap.add_argument("--pose_dir",    default="results/pose_out")
    ap.add_argument("--traj_dir",    default="results/traj_out")

    # Pose backend
    # ap.add_argument("--pose_model",   default="yolo26n-pose.pt",
    #                 help="Path to YOLOv26 pose model weights")
    ap.add_argument("--pose_backend", default="auto",
                    choices=["rtmpose", "mediapipe", "auto", "none"])

    # Module 2 — preprocessing
    ap.add_argument("--preprocess", action="store_true",
                    help="Background-subtract video before TrackNet. "
                         "Removes static ad banners at the source. "
                         "Recommended for broadcast footage.")

    # Skip flags
    ap.add_argument("--skip_calib",  action="store_true",
                    help="Skip Module 1 (use existing calib_dir)")
    ap.add_argument("--skip_shuttle",action="store_true",
                    help="Skip Module 2 (use existing shuttle_dir)")
    ap.add_argument("--skip_pose",   action="store_true",
                    help="Skip Module 3 (use existing pose_dir)")
    ap.add_argument("--only_calib",  action="store_true",
                    help="Run only Module 1 and exit")
    ap.add_argument("--only_traj",   action="store_true",
                    help="Run only Module 4 (all others already done)")

    args = ap.parse_args()

    # Apply --only_* shortcuts
    if args.only_traj:
        args.skip_calib   = True
        args.skip_shuttle = True
        args.skip_pose    = True

    sep = "─" * 60

    # ── Module 1 ──────────────────────────────────────────────────
    if not args.skip_calib:
        print(f"\n{sep}")
        print("MODULE 1 — Court Calibration")
        print(sep)
        from court_calibration import calibrate, verify_and_save, save_calibration
        out = Path(args.calib_dir)
        P, K, rvec, tvec, img_pts = calibrate(args.image)
        verify_and_save(P, K, rvec, tvec, img_pts, args.image, out)
        save_calibration(P, K, rvec, tvec, out)
        print("Module 1 done.\n")
    else:
        print("Module 1 skipped (using existing calibration).")

    if args.only_calib:
        print("--only_calib: stopping after Module 1.")
        return

    # ── Module 2 ──────────────────────────────────────────────────
    if not args.skip_shuttle:
        print(f"\n{sep}")
        print("MODULE 2 — Shuttle Detection")
        print(sep)
        from shuttle_detection import (
            preprocess_video,
            run_tracknet, parse_tracknet_csv, clean_detections,
            detection_stats, save_shuttle, save_stats, render_debug_video,
        )
        out = Path(args.shuttle_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Optional background subtraction before TrackNet
        tracknet_input = args.video
        if args.preprocess:
            print("  [preprocess] Background subtraction …")
            tracknet_input = preprocess_video(
                args.video, str(out / "preprocessed.mp4"))

        csv_path   = run_tracknet(tracknet_input, args.tracknet, str(out))
        shuttle_2d = parse_tracknet_csv(csv_path, args.video)
        render_debug_video(args.video, shuttle_2d, str(out / "shuttle_RAW_UNFILTERED.mp4"))
        shuttle_2d = clean_detections(shuttle_2d, args.video)
        stats      = detection_stats(shuttle_2d)
        print(f"Detection rate: {stats['detection_rate']:.1%}  "
              f"[{stats['quality'].split('(')[0].strip()}]")
        save_shuttle(shuttle_2d, out)
        save_stats(stats, out)
        render_debug_video(args.video, shuttle_2d,
                           str(out / "shuttle_detection.mp4"))
        print("Module 2 done.\n")
    else:
        print("Module 2 skipped.")

    # ── Module 3 ──────────────────────────────────────────────────
    if not args.skip_pose:
        print(f"\n{sep}")
        print("MODULE 3 — Pose Estimation")
        print(sep)
        import cv2
        from court_calibration import load_calibration
        from pose_estimation import (
            load_pose_backend, estimate_poses, save_poses, render_pose_debug,
        )
        out     = Path(args.pose_dir)
        P, K, rvec, tvec = load_calibration(args.calib_dir)
 
        cap = cv2.VideoCapture(args.video)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()
 
        det_model, pose_model, backend = load_pose_backend()
        poses = estimate_poses(frames, K, rvec, tvec,
                               det_model, pose_model)
        save_poses(poses, out)
        render_pose_debug(frames, poses, fps, str(out / "pose_debug.mp4"))
        print("Module 3 done.\n")
    else:
        print("Module 3 skipped.")

    # ── Module 4 ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("MODULE 4 — 3-D Trajectory Reconstruction")
    print(sep)
    import cv2
    from court_calibration import load_calibration
    from shuttle_detection import load_shuttle
    from pose_estimation   import load_poses, get_player_3d
    from trajectory import (
        reconstruct, save_trajectory, render_annotated,
        make_plots, write_report, 
    )
    from config import VIDEO_FPS

    out = Path(args.traj_dir)
    out.mkdir(parents=True, exist_ok=True)

    P, K, rvec, tvec = load_calibration(args.calib_dir)
    shuttle_2d       = load_shuttle(args.shuttle_dir)

    hitter_3d = receiver_3d = None
    try:
        poses       = load_poses(args.pose_dir)
        other_side  = "far" if args.hitter == "near" else "near"
        hitter_3d   = get_player_3d(poses, args.hit_frame, args.hitter, use_floor=False)
        receiver_3d = get_player_3d(poses, len(poses) - 1, other_side, use_floor=False)
    except FileNotFoundError:
        print("  poses.pkl not found — player priors disabled.")

    cap = cv2.VideoCapture(args.video)
    fps = float(VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()

    shuttle_2d = shuttle_2d[:len(frames)]

    result = reconstruct(shuttle_2d, P, hitter_3d, receiver_3d,
                         fps, args.hitter)

    save_trajectory(result, out)
    render_annotated(frames, result, shuttle_2d, fps,
                     str(out / "output_annotated.mp4"))
    make_plots(result, fps, str(out / "trajectory_plots.png"))
    write_report(result, fps, str(out / "accuracy_report.txt"))

    print(f"\n{'═'*60}")
    print("PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  Annotated video → {out}/output_annotated.mp4")
    print(f"  3-D trajectory  → {out}/trajectory_3d.npy")
    print(f"  Plots           → {out}/trajectory_plots.png")
    print(f"  Report          → {out}/accuracy_report.txt")
    print(f"  Mean reproj err → {result['mean_reproj_err']:.3f} px")


if __name__ == "__main__":
    main()