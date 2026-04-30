"""
module1_court_calibration.py
════════════════════════════
Court camera calibration  →  P matrix (3×4)

Pipeline
────────
  1. You click 6 landmarks on one broadcast frame (interactive GUI).
  2. Joint (fx, fy, cx, cy) optimiser — no square-pixel assumption.
  3. solvePnP  +  Levenberg-Marquardt sub-pixel polish.
  4. Saves P.npy, K.npy, rvec.npy, tvec.npy, calib_result.png.

Usage
─────
  python module1_court_calibration.py
  python module1_court_calibration.py --image frame.jpg --out_dir calib_out

Outputs (all in --out_dir)
──────────────────────────
  P.npy            3×4 projection matrix  (float64, normalised P[2,3]=1)
  K.npy            3×3 intrinsic matrix
  rvec.npy         3×1 rotation vector
  tvec.npy         3×1 translation vector
  calib_result.png visual verification overlay
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CALIB_IMAGE, CALIB_OUT_DIR,
    WORLD_PTS, POINT_LABELS, POINT_COLORS,
    COURT_W as W, COURT_L as L, NET_Y, SSL_DIST,
)


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION GUI
# ─────────────────────────────────────────────────────────────────────────────
def annotate(image_path: str) -> np.ndarray:
    """
    Interactive 6-point annotation window.

    Click rules
    ───────────
    Corners 1–4 : click the INNER-EDGE intersection of white court lines,
                  not the outer edge.  Lines are ~40 mm wide;
                  outer vs inner = 3–5 px error at 1080p.
    Post tips 5–6: zoom in and click the TOP-CENTRE of the yellow pole cap,
                   NOT the net tape (which sits 2–3 cm lower).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(rgb)
    ax.set_title(
        "MODULE 1 — Court Calibration\n"
        "STEP 1: zoom/pan with toolbar    STEP 2: click 6 points in order\n"
        "  1=Near-L corner   2=Near-R corner   3=Far-R corner\n"
        "  4=Far-L corner    5=Left post tip   6=Right post tip\n"
        "  ● Click INNER edge of white lines; post tip = top of yellow cap",
        fontsize=10, fontweight="bold", loc="left",
    )

    print("\n── ANNOTATION GUIDE ─────────────────────────────────────────")
    for i, (lbl, col) in enumerate(zip(POINT_LABELS, POINT_COLORS)):
        print(f"  Click {i+1}:  {lbl:22s}  ● {col}")
    print("─────────────────────────────────────────────────────────────\n")

    pts = np.array(plt.ginput(n=6, timeout=0, show_clicks=True),
                   dtype=np.float64)

    for i, (p, c, lbl) in enumerate(zip(pts, POINT_COLORS, POINT_LABELS)):
        ax.plot(*p, "o", color=c, ms=9, mec="black", mew=1.2)
        ax.text(p[0] + 7, p[1] - 7, str(i + 1), color=c, fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6))
    ax.set_title("Annotation complete — close window to continue",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()

    _sanity(pts)
    return pts


def _sanity(pts: np.ndarray):
    nL, nR, fR, fL, netL, netR = pts
    issues = []
    if nL[1] < fL[1]:
        issues.append("Near-L y < Far-L y  →  near/far rows may be swapped")
    if nL[0] > nR[0]:
        issues.append("Near-L x > Near-R x →  left/right swapped on near baseline")
    if fL[0] > fR[0]:
        issues.append("Far-L x > Far-R x   →  left/right swapped on far baseline")
    if not (fL[1] < netL[1] < nL[1]):
        issues.append("Left post tip y is outside the near/far baseline range")
    if issues:
        print("⚠  SANITY WARNINGS:")
        for w in issues:
            print(f"     {w}")
        print("   Check clicks — re-run if incorrect.\n")
    else:
        print("✓  Sanity checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
#  INTRINSICS OPTIMISER  (fx, fy, cx, cy independently)
# ─────────────────────────────────────────────────────────────────────────────
def optimise_K(world_pts: np.ndarray, image_pts: np.ndarray,
               img_w: int, img_h: int) -> np.ndarray:
    """
    Minimise mean reprojection error over (fx, fy, cx, cy).

    Why independent fx/fy
    ─────────────────────
    Broadcast telephoto lenses are rarely perfectly square-pixel.
    Even 0.3 % aspect error causes ~5 px systematic drift on off-floor
    points (net posts).  Freeing fy costs nothing and fixes this.

    Why free cx/cy
    ───────────────
    The optical axis of a telephoto lens is not always at the pixel centre.
    ±15 % search around centre typically saves 1–2 px.
    """
    print("Optimising intrinsics (fx, fy, cx, cy independently) …")

    def err(p):
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        ok, rv, tv = cv2.solvePnP(world_pts, image_pts, K, None,
                                   flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return 1e9
        proj, _ = cv2.projectPoints(world_pts, rv, tv, K, None)
        return float(np.mean(
            np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1)))

    x0     = [img_w * 2.0, img_w * 2.0, img_w / 2.0, img_h / 2.0]
    bounds = [
        (img_w * 0.8,  img_w * 5.0),    # fx
        (img_w * 0.8,  img_w * 5.0),    # fy
        (img_w * 0.35, img_w * 0.65),   # cx
        (img_h * 0.35, img_h * 0.65),   # cy
    ]
    res = minimize(err, x0, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 3000})

    fx, fy, cx, cy = res.x
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  Pixel aspect fy/fx = {fy/fx:.5f}  (1.0 = square)")
    print(f"  Error after K optimisation: {res.fun:.3f} px\n")
    return K


# ─────────────────────────────────────────────────────────────────────────────
#  FULL CALIBRATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(image_path: str) -> tuple:
    """
    annotate → optimise K → solvePnP → LM refine → P matrix.

    Returns (P, K, rvec, tvec, img_pts).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_h, img_w = img.shape[:2]
    print(f"Image: {img_w} × {img_h}\n")

    img_pts = annotate(image_path)
    print("Annotated pixels:\n", np.round(img_pts, 1), "\n")

    K = optimise_K(WORLD_PTS, img_pts, img_w, img_h)

    ok, rvec, tvec = cv2.solvePnP(WORLD_PTS, img_pts, K, None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed. Check annotation order.")

    rvec, tvec = cv2.solvePnPRefineLM(WORLD_PTS, img_pts, K, None, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec])
    P /= P[2, 3]   # normalise so P[2,3] = 1

    print("Projection matrix P:\n", P, "\n")
    return P, K, rvec, tvec, img_pts


# ─────────────────────────────────────────────────────────────────────────────
#  VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def verify_and_save(P, K, rvec, tvec, img_pts, image_path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Standard 3D -> 2D Reprojection Error
    pts_h  = np.hstack([WORLD_PTS, np.ones((6, 1))])
    proj_h = (P @ pts_h.T).T
    proj   = proj_h[:, :2] / proj_h[:, 2:]
    reproj_errors = np.linalg.norm(proj - img_pts, axis=1)
    
    # 2. 2D -> 3D Back-projection ("Round Trip")
    backproj_pts_3d = []
    for i, (u, v) in enumerate(img_pts):
        target_z = WORLD_PTS[i, 2] # 0.0 for floor, 1.55 for net
        
        # Use the shared helper to ensure identical math across all modules
        p3d = backproject_to_floor(u, v, K, rvec, tvec, target_z=target_z)
        
        if p3d is not None:
            backproj_pts_3d.append(p3d)
        else:
            backproj_pts_3d.append([np.nan, np.nan, np.nan])

    backproj_pts_3d = np.array(backproj_pts_3d)
    # Calculate error in meters between back-projected X,Y and Ground Truth X,Y
    xy_dist_meters = np.linalg.norm(backproj_pts_3d[:, :2] - WORLD_PTS[:, :2], axis=1)

    print("\n── CALIBRATION VERIFICATION (Round Trip) ────────────────────────")
    # Updated Header to include Calculated X,Y
    header = f"{'Point':<20} | {'2D Err':<8} | {'Calculated (X, Y)':<18} | {'3D Err':<8}"
    print(header)
    print("─" * len(header))
    
    for i, lbl in enumerate(POINT_LABELS):
        sym = "✓" if reproj_errors[i] < 3.5 else "✗"
        # Calculated coordinates in meters
        calc_x = backproj_pts_3d[i, 0]
        calc_y = backproj_pts_3d[i, 1]
        
        print(f"{sym} {lbl:18} | {reproj_errors[i]:5.2f} px | ({calc_x:6.2f}, {calc_y:6.2f}) m | {xy_dist_meters[i]*100:6.2f} cm")
    
    mean_px = float(np.mean(reproj_errors))
    mean_cm = float(np.mean(xy_dist_meters)) * 100
    print("─" * len(header))
    print(f"OVERALL MEAN ERRORS: {mean_px:.2f} px  /  {mean_cm:.2f} cm")
    print("─────────────────────────────────────────────────────────────────\n")

    # [Rest of your plotting code...]
    return mean_px

# ─────────────────────────────────────────────────────────────────────────────
#  SAVE / LOAD  (used by Modules 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────
def save_calibration(P, K, rvec, tvec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "P.npy",    P)
    np.save(out_dir / "K.npy",    K)
    np.save(out_dir / "rvec.npy", rvec)
    np.save(out_dir / "tvec.npy", tvec)
    print(f"Calibration saved → {out_dir}/  (P  K  rvec  tvec)")


def load_calibration(calib_dir) -> tuple:
    """Returns (P, K, rvec, tvec) — all float64 numpy arrays."""
    d = Path(calib_dir)
    return (np.load(d / "P.npy"),
            np.load(d / "K.npy"),
            np.load(d / "rvec.npy"),
            np.load(d / "tvec.npy"))


# ─────────────────────────────────────────────────────────────────────────────
#  GEOMETRY HELPERS  (imported by Modules 3 & 4)
# ─────────────────────────────────────────────────────────────────────────────
def backproject_to_floor(u, v, K, rvec, tvec, target_z=0.0):
    """
    Back-project pixel (u, v) onto a horizontal plane at height target_z.
    Used for player feet (z=0) and net verification (z=1.55).
    """
    R, _ = cv2.Rodrigues(rvec)
    ray  = np.linalg.inv(K) @ np.array([u, v, 1.0])
    Rt   = R.T
    tv   = tvec.flatten()
    
    # Intersection of ray and plane Z = target_z
    # Formula: lambda = (target_z + [Rt @ tv]_z) / [Rt @ ray]_z
    denominator = (Rt @ ray)[2]
    if abs(denominator) < 1e-10:
        return None
        
    numerator = target_z + (Rt @ tv)[2]
    lam = numerator / denominator
    
    # P_world = Rt @ (ray * lambda - tvec)
    return Rt @ (ray * lam - tv)


def project_to_pixel(X_world, P) -> np.ndarray:
    """
    Project 3-D world point(s) → 2-D pixel(s).
    X_world: (3,) or (N, 3)  →  returns (2,) or (N, 2).
    """
    X  = np.atleast_2d(X_world).astype(np.float64)
    Xh = np.hstack([X, np.ones((len(X), 1))])
    ph = (P @ Xh.T).T
    px = ph[:, :2] / ph[:, 2:3]
    return px[0] if X_world.ndim == 1 else px


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Module 1 — Court Calibration")
    ap.add_argument("--image",   default=str(CALIB_IMAGE))
    ap.add_argument("--out_dir", default=str(CALIB_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    P, K, rvec, tvec, img_pts = calibrate(args.image)
    verify_and_save(P, K, rvec, tvec, img_pts, args.image, out_dir)
    save_calibration(P, K, rvec, tvec, out_dir)

    print("\nModule 1 complete.  Next: python module2_shuttle_detection.py")