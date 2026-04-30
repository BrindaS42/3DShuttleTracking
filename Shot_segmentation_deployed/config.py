# config.py  ── edit once, every module reads from here
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(".")

CALIB_IMAGE   = ROOT / "test_assets" /  "test_image.jpg"          # one clear broadcast frame for M1
CALIB_OUT_DIR = ROOT / "results" / "calib_out"          # M1 → P.npy K.npy rvec.npy tvec.npy

VIDEO_PATH    = ROOT / "test_assets" / "test_video.mp4"           # input video (shot clip or full rally)
TRACKNET_DIR  = ROOT / "tracknet_weights"   # folder with predict.py + *.pt
SHUTTLE_OUT   = ROOT / "results" / "shuttle_out"        # M2 → shuttle_2d.npy

POSE_OUT      = ROOT / "results" / "pose_out"           # M3 → poses.pkl
POSE_BACKEND  = "auto"                      # "rtmpose"|"mediapipe"|"auto"|"none"

TRAJ_OUT      = ROOT / "results" / "traj_out"           # M4 → trajectory_3d.npy + video + plots
HITTER_SIDE   = "near"                      # "near" | "far"
HIT_FRAME     = 0                           # frame index when hitter strikes
VIDEO_FPS     = None                        # None = read from file

# ── BWF court constants (metres, doubles) ────────────────────────────────────
COURT_W  = 6.7
COURT_L  = 13.4
NET_Y    = 6.7      # = COURT_L / 2
NET_H    = 1.524    # net centre height (BWF official)
POST_H   = 1.55     # net post tip height
SSL_DIST = 1.98     # short service line from baseline

# ── 6 annotation world points ────────────────────────────────────────────────
# Origin = near-left doubles corner  |  X = court width  |  Y = court length  |  Z = up
WORLD_PTS = np.array([
    [0.0,     0.0,     0.0 ],  # 0  Near-Left  corner
    [COURT_W, 0.0,     0.0 ],  # 1  Near-Right corner
    [COURT_W, COURT_L, 0.0 ],  # 2  Far-Right  corner
    [0.0,     COURT_L, 0.0 ],  # 3  Far-Left   corner
    [0.0,     NET_Y,   POST_H],# 4  Left  net post tip
    [COURT_W, NET_Y,   POST_H],# 5  Right net post tip
], dtype=np.float64)

POINT_LABELS = ["Near-L corner","Near-R corner","Far-R corner",
                "Far-L corner","Left post tip","Right post tip"]
POINT_COLORS = ["red","orange","limegreen","dodgerblue","magenta","cyan"]

# ── Physics ───────────────────────────────────────────────────────────────────
GRAVITY      = np.array([0.0, 0.0, -9.81])  # m/s²
MAX_SPEED_MS = 120.0                          # ≈ 432 kph

# ── Player geometry ───────────────────────────────────────────────────────────
PLAYER_HEIGHT = 1.75   # metres