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
    vel  = state[3:]
    drag = -Cd * float(np.dot(vel, vel)) * vel
    return np.concatenate([vel, GRAVITY + drag])

def integrate(x0: np.ndarray, v0: np.ndarray,
              Cd: float, t_end: float, fps: float) -> np.ndarray:
    N  = max(2, int(round(t_end * fps)) + 1)
    ts = np.linspace(0.0, t_end, N)
    sol = solve_ivp(_ode, [0.0, t_end], np.r_[x0, v0],
                    args=(Cd,), t_eval=ts,
                    method="RK45", rtol=1e-6, atol=1e-8,
                    max_step=1.0 / fps)
    if not sol.success:
        return np.full((N, 3), np.nan)
    return sol.y[:3].T

# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMISER (STRICT POSE ANCHORS + HARD BOUNDARIES)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct(shuttle_2d: np.ndarray, P: np.ndarray, hitter_3d, receiver_3d,
                fps: float, hitter_side: str) -> dict:
    """
    Refined reconstruction for MonoTrack.
    Decouples simulation time from 2D detection availability to fix mid-court 
    truncation and high reprojection errors.
    """
    valid = ~np.any(np.isnan(shuttle_2d), axis=1)
    obs_2d_valid = shuttle_2d[valid]
    
    # ── PHYSICAL TIME VS OBSERVATION TIME ──
    # N_all represents the total frames in the hit-to-hit interval.
    # We must integrate for this entire time to reach the receiver[cite: 310].
    N_all = len(shuttle_2d)
    t_total = (N_all - 1) / fps 
    all_t = np.arange(N_all) / fps
    obs_t = np.where(valid)[0] / fps         

    vy_sign = 1.0 if hitter_side == "near" else -1.0
    sigma = 1.0 / (np.linalg.norm(P) ** 2) # Balancing term [cite: 272]

    # ── RELAXED SPATIAL BOUNDS ──
    # Start coordinates use player pose but allow a 2.0m search radius[cite: 295].
    start_x = hitter_3d[0] if hitter_3d is not None else W / 2.0
    start_y = hitter_3d[1] if hitter_3d is not None else (2.0 if hitter_side == "near" else L - 2.0)
    
    # Physics constraints based on professional play [cite: 275, 276]
    bnds = [
        (max(-1.0, start_x - 2.0), min(W + 1.0, start_x + 2.0)), # X
        (max(-1.0, start_y - 2.0), min(L + 1.0, start_y + 2.0)), # Y
        (0.0, 4.0),                                              # Z (allow for high clears)
        (-50.0, 50.0),                                           # Vx
        (2.0, 120.0) if vy_sign > 0 else (-120.0, -2.0),         # Vy (Max 120m/s [cite: 276])
        (-40.0, 40.0),                                           # Vz
        (np.log(0.02), np.log(0.50))                             # Cd (Drag coeff [cite: 247, 249])
    ]

    def loss(p):
        x0, v0, Cd = p[:3], p[3:6], float(np.exp(p[6]))
        
        # Integrate for the FULL duration of the shot
        traj = integrate(x0, v0, Cd, t_total, fps)
        if np.any(np.isnan(traj)) or len(traj) < N_all: return 1e9

        # ── Reprojection Loss (Lr) ──
        # Only calculated for frames where the shuttle was detected [cite: 256]
        traj_obs = np.column_stack([np.interp(obs_t, all_t, traj[:N_all, i]) for i in range(3)])
        proj_obs = project_to_pixel(traj_obs, P)
        Lr = float(np.sum(np.linalg.norm(proj_obs - obs_2d_valid, axis=1)**2))
        
        total_loss = (sigma * 100.0) * Lr
        
        # ── Anchor Priors (Eq. 4) ──
        # Soft constraints to pull trajectory toward hitter and receiver [cite: 265]
        if hitter_3d is not None:
            total_loss += float(np.linalg.norm(traj[0] - hitter_3d)**2)
            
        if receiver_3d is not None:
            total_loss += float(np.linalg.norm(traj[-1] - receiver_3d)**2)

        # Floor Penalty [cite: 263]
        dO = float(np.sum(np.maximum(0, -traj[:, 2])**2)) 
        total_loss += 10.0 * dO 

        return total_loss

    # ── DIVERSE INITIAL GUESSES ──
    initial_guesses = [
        np.r_[start_x, start_y, 2.5, 0.0, vy_sign*60.0, -5.0, np.log(0.12)], # Pro Smash
        np.r_[start_x, start_y, 1.8, 0.0, vy_sign*35.0, 12.0, np.log(0.18)], # Clear/Lob
        np.r_[start_x, start_y, 0.8, 0.0, vy_sign*15.0, 15.0, np.log(0.08)], # Net Lift
    ]
    
    best_res, best_loss, best_p0 = None, float('inf'), None
    for p0 in initial_guesses:
        res = minimize(loss, p0, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": 2500, "ftol": 1e-10}) 
        if res.fun < best_loss:
            best_loss, best_res, best_p0 = res.fun, res, p0
    
    x0_opt, v0_opt, Cd_opt = best_res.x[:3], best_res.x[3:6], float(np.exp(best_res.x[6]))
    
    # Final trajectory generation
    traj_final = integrate(x0_opt, v0_opt, Cd_opt, t_total, fps)
    proj_all = project_to_pixel(traj_final[:N_all], P)
    
    reproj = np.full(N_all, np.nan)
    reproj[valid] = np.linalg.norm(proj_all[valid] - shuttle_2d[valid], axis=1)

    return dict(
        x0=x0_opt, v0=v0_opt, Cd=Cd_opt,
        traj_3d=traj_final[:N_all], traj_2d_proj=proj_all,
        reproj_err=reproj, mean_reproj_err=float(np.nanmean(reproj)) if np.any(valid) else 0.0,
        n_valid=int(valid.sum()), n_frames=N_all, converged=bool(best_res.success),
        
        # --- NEW TRACKING EXPORTS ---
        hitter_3d=hitter_3d.copy() if hitter_3d is not None else None,
        receiver_3d=receiver_3d.copy() if receiver_3d is not None else None,
        shuttle_start=traj_final[0].copy() if len(traj_final) > 0 else None,
        shuttle_end=traj_final[N_all-1].copy() if len(traj_final) >= N_all else None,
        best_p0=best_p0.copy()
    )
# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATED VIDEO & PLOTS
# ─────────────────────────────────────────────────────────────────────────────
_TRAIL = 20        
_DETECTED_CLR  = (0, 255, 100)     
_PROJECTED_CLR = (0, 180, 255)     

def render_annotated(frames: list, result: dict, shuttle_2d: np.ndarray, poses: list, fps: float, out_path: str):
    traj_2d, traj_3d = result["traj_2d_proj"], result["traj_3d"]
    h, w    = frames[0].shape[:2]
    writer  = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for fi, frame in enumerate(frames):
        out = frame.copy()
        
        # --- DRAW FADING TRAILS FOR BOTH ARRAYS ---
        for j in range(max(0, fi - _TRAIL), fi):
            alpha = (j - max(0, fi - _TRAIL) + 1) / (_TRAIL + 1)
            
            # 1. Projected 3D Trail (Amber)
            if j < len(traj_2d) and not np.any(np.isnan(traj_2d[j])):
                c_proj = (0, int(_PROJECTED_CLR[1] * alpha), int(_PROJECTED_CLR[2] * alpha))
                cv2.circle(out, tuple(traj_2d[j].astype(int)), 3, c_proj, -1, cv2.LINE_AA)
                
            # 2. Raw 2D Detection Trail (Green)
            if j < len(shuttle_2d) and not np.any(np.isnan(shuttle_2d[j])):
                c_det = (0, int(_DETECTED_CLR[1] * alpha), int(_DETECTED_CLR[2] * alpha))
                cv2.circle(out, tuple(shuttle_2d[j].astype(int)), 3, c_det, -1, cv2.LINE_AA)
        # ------------------------------------------

        if fi < len(traj_2d) and not np.any(np.isnan(traj_2d[fi])):
            pt = tuple(traj_2d[fi].astype(int))
            cv2.drawMarker(out, pt, _PROJECTED_CLR, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            
        if fi < len(shuttle_2d) and not np.any(np.isnan(shuttle_2d[fi])):
            cv2.circle(out, tuple(shuttle_2d[fi].astype(int)), 7, _DETECTED_CLR, 2, cv2.LINE_AA)
            
        if poses and fi < len(poses):
            pf = poses[fi]
            near_txt = "Near Player : N/A"
            if pf.near and pf.near.floor_pos_3d is not None and not np.isnan(pf.near.floor_pos_3d[0]):
                nx, ny = pf.near.floor_pos_3d[0], pf.near.floor_pos_3d[1]
                near_txt = f"Near Player : X:{nx:5.2f}m Y:{ny:5.2f}m"
            
            far_txt = "Far Player  : N/A"
            if pf.far and pf.far.floor_pos_3d is not None and not np.isnan(pf.far.floor_pos_3d[0]):
                fx, fy = pf.far.floor_pos_3d[0], pf.far.floor_pos_3d[1]
                far_txt = f"Far Player  : X:{fx:5.2f}m Y:{fy:5.2f}m"
                
            cv2.rectangle(out, (6, 6), (320, 70), (0, 0, 0), -1)
            cv2.putText(out, near_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(out, far_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        if fi < len(traj_3d) and not np.any(np.isnan(traj_3d[fi])):
            x, y, z = traj_3d[fi]
            txt = f"Shuttle 3D  : X:{x:5.2f}m Y:{y:5.2f}m Z:{z:5.2f}m"
            cv2.rectangle(out, (6, h - 36), (460, h - 8), (0, 0, 0), -1)
            cv2.putText(out, txt, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            
        cv2.putText(out, f"f{fi}", (w - 70, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(out)
    writer.release()
    print(f"Annotated video → {out_path}")

def make_plots(result: dict, fps: float, out_path: str):
    traj, reproj = result["traj_3d"], result["reproj_err"]
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    N, time_arr, frames_a = len(traj), np.arange(len(traj)) / fps, np.arange(len(traj))
    fig = plt.figure(figsize=(18, 5))

    ax3 = fig.add_subplot(141, projection="3d")
    ax3.plot(y, x, z, "b-", lw=2)
    ax3.scatter([y[0]], [x[0]], [z[0]], c="green", s=60, zorder=5, label="hit")
    ax3.scatter([y[-1]], [x[-1]], [z[-1]], c="red", s=60, zorder=5, label="end")
    cy, cx_ = [0, L, L, 0, 0], [0, 0, W, W, 0]
    ax3.plot(cy, cx_, [0]*5, "k--", lw=0.8, alpha=0.5)
    ax3.plot([NET_Y, NET_Y], [0, W], [NET_H, NET_H], "gray", lw=1, alpha=0.6)
    ax3.set_xlabel("Y (m)"); ax3.set_ylabel("X (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title("3-D view"); ax3.legend(fontsize=7)

    ax2 = fig.add_subplot(142)
    sc = ax2.scatter(y, x, c=z, cmap="plasma", s=6, vmin=0)
    plt.colorbar(sc, ax=ax2, label="Z (m)", shrink=0.8)
    ax2.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k--", lw=0.8)
    ax2.axvline(x=NET_Y, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Y (m)"); ax2.set_ylabel("X (m)")
    ax2.set_title("Bird-eye (colour = Z)"); ax2.set_aspect("equal")

    ax_z = fig.add_subplot(143)
    ax_z.plot(time_arr, z, "b-", lw=2, label="Z estimated")
    ax_z.axhline(y=NET_H, color="orange", ls="--", lw=1, label=f"Net {NET_H} m")
    ax_z.axhline(y=0, color="brown", ls="--", lw=0.8, label="Floor")
    ax_z.fill_between(time_arr, 0, z, where=(z > 0), alpha=0.08, color="blue")
    ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Z (m)")
    ax_z.set_title("Shuttle height Z"); ax_z.legend(fontsize=8)
    ax_z.set_ylim(bottom=-0.2)

    ax_r = fig.add_subplot(144)
    valid_mask = ~np.isnan(reproj)
    ax_r.scatter(frames_a[valid_mask], reproj[valid_mask], s=6, c="steelblue", label="per-frame error")
    ax_r.axhline(y=result["mean_reproj_err"], color="red", ls="--", lw=1, label=f"mean {result['mean_reproj_err']:.2f} px")
    ax_r.axhline(y=5, color="orange", ls=":", lw=0.8)
    ax_r.axhline(y=15, color="gray", ls=":", lw=0.8)
    ax_r.set_xlabel("Frame"); ax_r.set_ylabel("Reprojection error (px)")
    ax_r.set_title("Reprojection error"); ax_r.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def _net_clearance(traj: np.ndarray) -> float:
    y, idx = traj[:, 1], int(np.argmin(np.abs(traj[:, 1] - NET_Y)))
    return float(traj[idx, 2]) - NET_H

def write_report(result: dict, fps: float, out_path: str):
    reproj = result["reproj_err"]
    valid_r, traj = reproj[~np.isnan(reproj)], result["traj_3d"]
    speed, mean_e = float(np.linalg.norm(result["v0"])), result["mean_reproj_err"]
    quality = "EXCELLENT" if mean_e < 2 else "GOOD" if mean_e < 5 else "ACCEPTABLE" if mean_e < 15 else "POOR"
    lines = [
        "=" * 64, "Module 4 — 3-D Trajectory Accuracy Report", "=" * 64, "",
        f"  Frames total              : {result['n_frames']}",
        f"  Valid shuttle detections  : {result['n_valid']} ({100*result['n_valid']/result['n_frames']:.1f}%)",
        f"  Optimiser converged       : {result['converged']}", "",
        "  Optimised physics parameters",
        f"    Drag coefficient Cd     : {result['Cd']:.6f}",
        f"    Initial position x0     : ({result['x0'][0]:.3f}, {result['x0'][1]:.3f}, {result['x0'][2]:.3f}) m",
        f"    Initial speed ‖v0‖      : {speed:.2f} m/s ({speed*3.6:.0f} kph)", "",
        "  Reprojection error (pixels)",
        f"    Mean                    : {mean_e:.3f}",
        f"    Median                  : {float(np.median(valid_r)):.3f}",
        f"    90th percentile         : {float(np.percentile(valid_r,90)):.3f}",
        f"    Max                     : {float(valid_r.max()):.3f}", "",
        "  Trajectory summary",
        f"    Peak height Z_max       : {float(traj[:,2].max()):.3f} m",
        f"    Min height  Z_min       : {float(traj[:,2].min()):.3f} m",
        f"    Net clearance           : {_net_clearance(traj):.3f} m (positive = cleared)",
        f"    Landing position        : ({float(traj[-1,0]):.2f}, {float(traj[-1,1]):.2f}) m", "",
        "  Quality scale",
        "    < 2 px  — EXCELLENT (analytics-grade)",
        "    2–5 px  — GOOD",
        "    5–15 px — ACCEPTABLE",
        "    > 15 px — POOR", "",
        f"  => Result: {quality}", "=" * 64,
    ]
    with open(out_path, "w", encoding="utf-8") as f:        
        f.write("\n".join(lines) + "\n")

def save_trajectory(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trajectory_3d.npy", result["traj_3d"])

def load_trajectory(traj_dir) -> np.ndarray:
    return np.load(Path(traj_dir) / "trajectory_3d.npy")

if __name__ == "__main__":
    pass # Managed by reconstruct.py now