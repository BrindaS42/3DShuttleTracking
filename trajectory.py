"""
module4_trajectory.py
═════════════════════
3-D shuttle trajectory reconstruction  →  x(t), y(t), z(t) per frame

Physics model  (MonoTrack, Liu & Wang CVPR 2022, §4.5)
───────────────────────────────────────────────────────
  Shuttle modelled as a particle under gravity + quadratic air drag:

      d²x/dt² = g − Cd·‖v‖²·v          (Eq. 1)

  Optimiser minimises the full loss (Eq. 4):

      L = σ·Lr + ‖x(0)−xH‖² + ‖x(tR)−xR‖² + dOut²

  where:
    Lr    = mean squared reprojection error (2-D pixels, Eq. 3)
    xH/xR = hitter/receiver 3-D body positions from Module 3
    dOut  = distance shuttle lands outside court (0 if inside)
    σ     = 1/‖P‖² balances pixel-space vs world-space terms

  Parameters optimised: x0(3)  v0(3)  log(Cd)   — 7 parameters total.
  Solver: L-BFGS-B → Levenberg-Marquardt polish.

Usage
─────
  python module4_trajectory.py \\
      --video      clip.mp4 \\
      --calib_dir  calib_out \\
      --shuttle_dir shuttle_out \\
      --pose_dir   pose_out \\
      --hitter     near \\
      --hit_frame  0 \\
      --out_dir    traj_out

Outputs
───────
  trajectory_3d.npy     (N_frames, 3) float64  world-space X,Y,Z per frame
  output_annotated.mp4  video with XYZ overlay + amber projected trail
  trajectory_plots.png  3-D view / bird-eye / Z-time / reproj-error charts
  accuracy_report.txt   full numeric accuracy report
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAJ_OUT, HITTER_SIDE, HIT_FRAME, VIDEO_FPS,
    GRAVITY, MAX_SPEED_MS,
    COURT_W as W, COURT_L as L,
    NET_Y, NET_H, POST_H,
)
from court_calibration import load_calibration, project_to_pixel
from shuttle_detection import load_shuttle
from pose_estimation   import load_poses, get_player_3d


# ─────────────────────────────────────────────────────────────────────────────
#  PHYSICS: ODE + INTEGRATOR
# ─────────────────────────────────────────────────────────────────────────────
def _ode(t, state, Cd):
    """
    state = [x, y, z, vx, vy, vz]
    d²x/dt² = g − Cd·‖v‖²·v
    """
    vel  = state[3:]
    drag = -Cd * float(np.dot(vel, vel)) * vel
    return np.concatenate([vel, GRAVITY + drag])


def integrate(x0: np.ndarray, v0: np.ndarray,
              Cd: float, t_end: float, fps: float) -> np.ndarray:
    """
    Solve ODE from t=0 to t_end.
    Returns (N, 3) positions at equally-spaced frame times.
    Returns array of NaN if integration fails.
    """
    N  = max(2, int(round(t_end * fps)) + 1)
    ts = np.linspace(0.0, t_end, N)
    sol = solve_ivp(_ode, [0.0, t_end], np.r_[x0, v0],
                    args=(Cd,), t_eval=ts,
                    method="RK45", rtol=1e-6, atol=1e-8,
                    max_step=1.0 / fps)
    if not sol.success:
        return np.full((N, 3), np.nan)
    return sol.y[:3].T   # (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMISER (POSE-FREE WITH COURT CONSTRAINTS)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct(shuttle_2d: np.ndarray, P: np.ndarray, hitter_3d, receiver_3d,
                fps: float, hitter_side: str) -> dict:
    
    valid = ~np.any(np.isnan(shuttle_2d), axis=1)
    obs_2d_valid = shuttle_2d[valid]
    
    obs_t  = np.where(valid)[0] / fps         
    N_all  = len(shuttle_2d)
    all_t  = np.arange(N_all) / fps

    t_end = obs_t[-1]
    vy_sign = 1.0 if hitter_side == "near" else -1.0
    
    # CVPR SIGMA balances pixels vs world [cite: 272]
    # We use MEAN error to prevent the count of frames from overpowering the poses
    sigma = 1.0 / (np.linalg.norm(P) ** 2)

    # REFINED PRIORS 
    # Use 2.0m as the paper suggests, but we will treat it as a "soft" prior
    if hitter_3d is not None: hitter_3d[2] = 2.0 
    if receiver_3d is not None: receiver_3d[2] = 1.0 # Receivers are often lower

    # Change the bounds for log(Cd) to a more realistic range
    bnds = [
        (-1.0, W + 1.0), (-1.0, L + 1.0), (0.0, 5.0),  # x0, y0, z0
        (-45.0, 45.0), (2.0, 110.0) if vy_sign > 0 else (-110.0, -2.0), # vx, vy
        (-25.0, 45.0), 
        (np.log(0.05), np.log(0.45)) # NEW RANGE: 0.05 to 0.35 (Shuttle physics)
    ]

    def loss(p):
        x0, v0, Cd = p[:3], p[3:6], float(np.exp(p[6]))
        traj = integrate(x0, v0, Cd, t_end + 0.05, fps)
        if np.any(np.isnan(traj)): return 1e9

        traj_t = np.linspace(0.0, t_end + 0.05, len(traj))
        traj_obs = np.column_stack([np.interp(obs_t, traj_t, traj[:, i]) for i in range(3)])
        proj_obs = project_to_pixel(traj_obs, P)
        

        Lr = float(np.sum(np.linalg.norm(proj_obs - obs_2d_valid, axis=1)**2))
        
        total_loss = (sigma * 100.0) * Lr
        
        if hitter_3d is not None:
            total_loss += 0.5 * float(np.linalg.norm(traj[0] - hitter_3d)**2)
        if receiver_3d is not None:
            total_loss += 0.5 * float(np.linalg.norm(traj[-1] - receiver_3d)**2)

        dO = float(np.sum(np.maximum(0, -traj[:, 2])**2)) 
        total_loss += 50.0 * dO 

        return total_loss

    initial_guesses = [
        np.r_[W/2, 1.5,  2.2,  0.0, vy_sign*25.0,  5.0, np.log(0.15)], # Mid smash
        # np.r_[W/2, 1.5,  1.1,  0.0, vy_sign*15.0,  8.0, np.log(0.20)], # LOW SERVE/NET
        # np.r_[W/2, 1.0,  0.8,  0.0, vy_sign*10.0, 12.0, np.log(0.04)], # VERY LOW LIFT
        np.r_[W/2, 2.0,  2.8,  0.0, vy_sign*40.0, -5.0, np.log(0.06)], # Fast Drive
        np.r_[W/2, 1.5, 2.2, 0.0, vy_sign*40.0, 10.0, np.log(0.18)], # Long, high-drag smash
        # np.r_[W/2, 1.2, 1.1, 0.0, vy_sign*15.0, 8.0,  np.log(0.22)], # Low, high-drag serve
    ]
    
    best_res, best_loss = None, float('inf')
    t0  = time.time()
    for p0 in initial_guesses:
        res = minimize(loss, p0, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": 2000, "ftol": 1e-9}) 
        if res.fun < best_loss:
            best_loss, best_res = res.fun, res

    print(f"  [Optimizer] {time.time()-t0:.1f}s  best_loss={best_res.fun:.5f}")
    
    x0_opt, v0_opt, Cd_opt = best_res.x[:3], best_res.x[3:6], float(np.exp(best_res.x[6]))
    traj_final = integrate(x0_opt, v0_opt, Cd_opt, all_t[-1] + 0.1, fps)
    traj_t     = np.linspace(0.0, all_t[-1] + 0.1, len(traj_final))
    traj_all   = np.column_stack([np.interp(all_t, traj_t, traj_final[:, i]) for i in range(3)])

    proj_all = project_to_pixel(traj_all, P)
    reproj   = np.full(N_all, np.nan)
    reproj[valid] = np.linalg.norm(proj_all[valid] - shuttle_2d[valid], axis=1)

    return dict(
        x0=x0_opt, v0=v0_opt, Cd=Cd_opt,
        traj_3d=traj_all, traj_2d_proj=proj_all,
        reproj_err=reproj, mean_reproj_err=float(np.nanmean(reproj)),
        n_valid=int(valid.sum()), n_frames=N_all, converged=bool(best_res.success)
    )

# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATED VIDEO
# ─────────────────────────────────────────────────────────────────────────────
_TRAIL = 20        # frames of amber trail
_DETECTED_CLR  = (0, 255, 100)     # green  — TrackNet pixel
_PROJECTED_CLR = (0, 180, 255)     # amber  — 3-D projected back to 2-D


def render_annotated(frames: list, result: dict,
                     shuttle_2d: np.ndarray,
                     fps: float, out_path: str):
    """
    Draw on each frame:
      ● green circle   = raw TrackNet detection
      ● amber trail    = last _TRAIL projected 3-D points
      ● white text     = X, Y, Z coordinates in metres
    """
    traj_2d = result["traj_2d_proj"]
    traj_3d = result["traj_3d"]
    h, w    = frames[0].shape[:2]
    writer  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))

    for fi, frame in enumerate(frames):
        out = frame.copy()

        # Amber projected trail
        for j in range(max(0, fi - _TRAIL), fi):
            if j < len(traj_2d) and not np.any(np.isnan(traj_2d[j])):
                alpha = (j - max(0, fi - _TRAIL) + 1) / (_TRAIL + 1)
                c = (0, int(_PROJECTED_CLR[1] * alpha),
                        int(_PROJECTED_CLR[2] * alpha))
                cv2.circle(out, tuple(traj_2d[j].astype(int)),
                           3, c, -1, cv2.LINE_AA)

        # Current projected point (amber cross)
        if fi < len(traj_2d) and not np.any(np.isnan(traj_2d[fi])):
            pt = tuple(traj_2d[fi].astype(int))
            cv2.drawMarker(out, pt, _PROJECTED_CLR,
                           cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

        # Raw detection (green circle)
        if fi < len(shuttle_2d) and not np.any(np.isnan(shuttle_2d[fi])):
            cv2.circle(out, tuple(shuttle_2d[fi].astype(int)),
                       7, _DETECTED_CLR, 2, cv2.LINE_AA)

        # XYZ coordinate readout
        if fi < len(traj_3d) and not np.any(np.isnan(traj_3d[fi])):
            x, y, z = traj_3d[fi]
            txt = f"X:{x:5.2f}m  Y:{y:5.2f}m  Z:{z:5.2f}m"
            cv2.rectangle(out, (6, h - 36), (432, h - 8), (0, 0, 0), -1)
            cv2.putText(out, txt, (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
                        2, cv2.LINE_AA)

        # Frame counter
        cv2.putText(out, f"f{fi}", (w - 70, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200),
                    1, cv2.LINE_AA)
        writer.write(out)

    writer.release()
    print(f"Annotated video → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def make_plots(result: dict, fps: float, out_path: str):
    """4-panel figure: 3-D view | bird-eye | Z over time | reproj error."""
    traj     = result["traj_3d"]
    x, y, z  = traj[:, 0], traj[:, 1], traj[:, 2]
    reproj   = result["reproj_err"]
    N        = len(traj)
    time_arr = np.arange(N) / fps
    frames_a = np.arange(N)

    fig = plt.figure(figsize=(18, 5))

    # 3-D view
    ax3 = fig.add_subplot(141, projection="3d")
    ax3.plot(y, x, z, "b-", lw=2)
    ax3.scatter([y[0]], [x[0]], [z[0]], c="green", s=60, zorder=5,
                label="hit")
    ax3.scatter([y[-1]], [x[-1]], [z[-1]], c="red", s=60, zorder=5,
                label="end")
    # Court floor outline
    cy = [0, L, L, 0, 0];  cx_ = [0, 0, W, W, 0]
    ax3.plot(cy, cx_, [0]*5, "k--", lw=0.8, alpha=0.5)
    # Net
    ax3.plot([NET_Y, NET_Y], [0, W], [NET_H, NET_H],
             "gray", lw=1, alpha=0.6)
    ax3.set_xlabel("Y (m)"); ax3.set_ylabel("X (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title("3-D view");  ax3.legend(fontsize=7)

    # Bird-eye (X–Y plane, colour = height)
    ax2 = fig.add_subplot(142)
    sc  = ax2.scatter(y, x, c=z, cmap="plasma", s=6, vmin=0)
    plt.colorbar(sc, ax=ax2, label="Z (m)", shrink=0.8)
    ax2.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k--", lw=0.8)
    ax2.axvline(x=NET_Y, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Y (m)"); ax2.set_ylabel("X (m)")
    ax2.set_title("Bird-eye (colour = Z)"); ax2.set_aspect("equal")

    # Z over time
    ax_z = fig.add_subplot(143)
    ax_z.plot(time_arr, z, "b-", lw=2, label="Z estimated")
    ax_z.axhline(y=NET_H, color="orange", ls="--", lw=1,
                 label=f"Net {NET_H} m")
    ax_z.axhline(y=0,     color="brown",  ls="--", lw=0.8, label="Floor")
    ax_z.fill_between(time_arr, 0, z, where=(z > 0),
                      alpha=0.08, color="blue")
    ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Z (m)")
    ax_z.set_title("Shuttle height Z"); ax_z.legend(fontsize=8)
    ax_z.set_ylim(bottom=-0.2)

    # Reprojection error per frame
    ax_r = fig.add_subplot(144)
    valid_mask = ~np.isnan(reproj)
    ax_r.scatter(frames_a[valid_mask], reproj[valid_mask],
                 s=6, c="steelblue", label="per-frame error")
    ax_r.axhline(y=result["mean_reproj_err"],
                 color="red", ls="--", lw=1,
                 label=f"mean {result['mean_reproj_err']:.2f} px")
    ax_r.axhline(y=5,  color="orange", ls=":", lw=0.8)
    ax_r.axhline(y=15, color="gray",   ls=":", lw=0.8)
    ax_r.set_xlabel("Frame"); ax_r.set_ylabel("Reprojection error (px)")
    ax_r.set_title("Reprojection error"); ax_r.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def _net_clearance(traj: np.ndarray) -> float:
    """Height above net when shuttle crosses y = NET_Y."""
    y   = traj[:, 1]
    idx = int(np.argmin(np.abs(y - NET_Y)))
    return float(traj[idx, 2]) - NET_H


def write_report(result: dict, fps: float, out_path: str):
    reproj  = result["reproj_err"]
    valid_r = reproj[~np.isnan(reproj)]
    traj    = result["traj_3d"]
    speed   = float(np.linalg.norm(result["v0"]))
    mean_e  = result["mean_reproj_err"]
    quality = ("EXCELLENT" if mean_e < 2  else
               "GOOD"      if mean_e < 5  else
               "ACCEPTABLE" if mean_e < 15 else "POOR")

    lines = [
        "=" * 64,
        "Module 4 — 3-D Trajectory Accuracy Report",
        "=" * 64,
        "",
        f"  Frames total              : {result['n_frames']}",
        f"  Valid shuttle detections  : {result['n_valid']}  "
        f"({100*result['n_valid']/result['n_frames']:.1f}%)",
        f"  Optimiser converged       : {result['converged']}",
        "",
        "  Optimised physics parameters",
        f"    Drag coefficient Cd     : {result['Cd']:.6f}",
        f"    Initial position x0     : ({result['x0'][0]:.3f}, "
        f"{result['x0'][1]:.3f}, {result['x0'][2]:.3f}) m",
        f"    Initial speed ‖v0‖      : {speed:.2f} m/s  "
        f"({speed*3.6:.0f} kph)",
        "",
        "  Reprojection error (pixels)",
        f"    Mean                    : {mean_e:.3f}",
        f"    Median                  : {float(np.median(valid_r)):.3f}",
        f"    90th percentile         : {float(np.percentile(valid_r,90)):.3f}",
        f"    Max                     : {float(valid_r.max()):.3f}",
        "",
        "  Trajectory summary",
        f"    Peak height Z_max       : {float(traj[:,2].max()):.3f} m",
        f"    Min height  Z_min       : {float(traj[:,2].min()):.3f} m",
        f"    Net clearance           : {_net_clearance(traj):.3f} m "
        f"(positive = cleared)",
        f"    Landing position        : ({float(traj[-1,0]):.2f}, "
        f"{float(traj[-1,1]):.2f}) m",
        "",
        "  Quality scale",
        "    < 2 px  — EXCELLENT (analytics-grade)",
        "    2–5 px  — GOOD",
        "    5–15 px — ACCEPTABLE",
        "    > 15 px — POOR",
        "",
        f"  => Result: {quality}",
        "=" * 64,
    ]
    with open(out_path, "w", encoding="utf-8") as f:        
        f.write("\n".join(lines) + "\n")
    print(f"Report → {out_path}")
    print(f"\n  Mean reprojection error: {mean_e:.3f} px  [{quality}]")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def save_trajectory(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trajectory_3d.npy", result["traj_3d"])
    print(f"trajectory_3d.npy → {out_dir}")


def load_trajectory(traj_dir) -> np.ndarray:
    return np.load(Path(traj_dir) / "trajectory_3d.npy")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 4 — 3-D Trajectory")
    ap.add_argument("--video",        required=True)
    ap.add_argument("--calib_dir",    default="calib_out")
    ap.add_argument("--shuttle_dir",  default="shuttle_out")
    ap.add_argument("--pose_dir",     default="pose_out")
    ap.add_argument("--hitter",       default=HITTER_SIDE,
                    choices=["near", "far"])
    ap.add_argument("--hit_frame",    type=int, default=HIT_FRAME)
    ap.add_argument("--out_dir",      default=str(TRAJ_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("── [1/5] Calibration ────────────────────────────────────────")
    P, K, rvec, tvec = load_calibration(args.calib_dir)
    print(f"  P loaded  (‖P‖ = {np.linalg.norm(P):.1f})")

    print("\n── [2/5] Shuttle detections ─────────────────────────────────")
    shuttle_2d = load_shuttle(args.shuttle_dir)
    n_valid = int(np.sum(~np.any(np.isnan(shuttle_2d), axis=1)))
    print(f"  {n_valid}/{len(shuttle_2d)} valid detections")

    print("\n── [3/5] Player poses ───────────────────────────────────────")
    hitter_3d   = None
    receiver_3d = None
    try:
        poses        = load_poses(args.pose_dir)
        other_side   = "far" if args.hitter == "near" else "near"
        hitter_3d    = get_player_3d(poses, args.hit_frame, args.hitter, use_floor=False)
        receiver_3d  = get_player_3d(poses, len(poses) - 1, other_side)
        print(f"  Hitter   3D: "
              f"{np.round(hitter_3d,2) if hitter_3d is not None else 'N/A'}")
        print(f"  Receiver 3D: "
              f"{np.round(receiver_3d,2) if receiver_3d is not None else 'N/A'}")
    except FileNotFoundError:
        print("  poses.pkl not found — player priors disabled.")

    print("\n── [4/5] Loading video ──────────────────────────────────────")
    cap    = cv2.VideoCapture(args.video)
    fps    = float(VIDEO_FPS or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    print(f"  {len(frames)} frames @ {fps:.1f} fps")

    # Safety: trim shuttle array to actual video length
    shuttle_2d = shuttle_2d[:len(frames)]

    print("\n── [5/5] Physics optimiser ──────────────────────────────────")
    result = reconstruct(shuttle_2d, P, hitter_3d, receiver_3d,
                         fps, args.hitter)

    print("\n── Saving outputs ───────────────────────────────────────────")
    save_trajectory(result, out_dir)
    render_annotated(frames, result, shuttle_2d, fps,
                     str(out_dir / "output_annotated.mp4"))
    make_plots(result, fps, str(out_dir / "trajectory_plots.png"))
    write_report(result, fps, str(out_dir / "accuracy_report.txt"))

    print("\n── DONE ─────────────────────────────────────────────────────")
    print(f"  Annotated video    → {out_dir}/output_annotated.mp4")
    print(f"  3-D trajectory     → {out_dir}/trajectory_3d.npy")
    print(f"  Plots              → {out_dir}/trajectory_plots.png")
    print(f"  Accuracy report    → {out_dir}/accuracy_report.txt")