"""
inference.py — Inference module for HD-A YOLO pipeline.
=========================================================
Contains:
  - assign_sides_pkl         : map near/far slots to court sides
  - detect_swing_actions_yolo: HD-A confidence stream (YOLOv8 classifier)
  - trajectory_smoothing     : HD-T Algorithm 1 (paper-faithful)
  - compute_hd_t             : hit moment detection from shuttle trajectory
  - sra / sra_per_rally_trimmed : Shot Refinement Algorithm (SRA)
  - apply_peak_nms / apply_temporal_nms : NMS helpers used in main_2
"""

import cv2
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from config import (
    WRIST_R, WRIST_L, ELBOW_R, ELBOW_L,
    SHOULD_R, SHOULD_L, PKL_ROOT,
    CROP_PAD, MIN_CROP_SIZE,
)
from data_loader import (
    load_pose_pkl, load_shuttle_npy, load_shot_frames,
    _reconstruct_frame_map_from_csvs,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_sides_pkl(poses: list) -> np.ndarray:
    """Assign court sides from pkl near/far slots.

    slot 0 (near) → side 2  (bottom court half)
    slot 1 (far)  → side 1  (top court half)
    """
    N     = len(poses)
    sides = np.zeros((N, 2), dtype=np.int32)
    sides[:, 0] = 2
    sides[:, 1] = 1
    return sides


# ══════════════════════════════════════════════════════════════════════════════
#  HD-A  (YOLOv8 swing classifier)
# ══════════════════════════════════════════════════════════════════════════════

def detect_swing_actions_yolo(poses: list,
                               frame_ids: list,
                               sides: np.ndarray,
                               trimmed_video_path: str,
                               clf_path: Path = Path('swing_classifier.pt'),
                               batch_size: int = 64,
                               swing_thr: float = 0.40) -> tuple:
    """HD-A using the trained YOLOv8 swing classifier.

    Crops near/far player bboxes from each trimmed frame, runs YOLOv8-cls in
    batches, and uses the raw swing-class probability as the per-player
    confidence stream fed into SRA.

    Args:
        swing_thr: frames whose swing probability exceeds this are emitted as
                   HD-A detections.  Unlike the heuristic approach this is a
                   fixed absolute threshold because the classifier output is
                   calibrated probability.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError('ultralytics not installed: pip install ultralytics')

    clf_path = Path(clf_path)
    if not clf_path.exists():
        raise FileNotFoundError(
            f'Swing classifier not found at {clf_path}. '
            'Run: python training.py')

    model = YOLO(str(clf_path))
    names = model.names
    swing_idx = next((k for k, v in names.items() if v == 'swing'), 1)

    N        = len(frame_ids)
    conf_raw = np.zeros((N, 2), dtype=np.float32)

    cap = cv2.VideoCapture(trimmed_video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open trimmed video: {trimmed_video_path}')
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    batch_items = []

    def _flush_batch():
        if not batch_items:
            return
        crops   = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, _, c in batch_items]
        results = model.predict(source=crops, verbose=False, imgsz=224)
        for (i, slot, _), res in zip(batch_items, results):
            probs = res.probs.data.cpu().numpy()
            conf_raw[i, slot] = float(probs[swing_idx])
        batch_items.clear()

    with tqdm(total=N, desc='  YOLO swing', leave=False) as pbar:
        for i, pf in enumerate(poses):
            ret, frame = cap.read()
            if not ret:
                break

            for slot, attr in enumerate(['near', 'far']):
                person = getattr(pf, attr, None)
                if person is None:
                    continue
                bb = getattr(person, 'bbox', None)
                if bb is None or len(bb) < 4:
                    continue
                x1 = max(0, int(bb[0]) - CROP_PAD)
                y1 = max(0, int(bb[1]) - CROP_PAD)
                x2 = min(w_vid, int(bb[2]) + CROP_PAD)
                y2 = min(h_vid, int(bb[3]) + CROP_PAD)
                if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                batch_items.append((i, slot, crop))

            if len(batch_items) >= batch_size:
                _flush_batch()

            pbar.update(1)

    _flush_batch()
    cap.release()

    hda_f, hda_p, hda_c = [], [], []
    for i in range(N):
        for slot in range(2):
            if conf_raw[i, slot] >= swing_thr:
                hda_f.append(frame_ids[i])
                hda_p.append(int(sides[i, slot]))
                hda_c.append(float(conf_raw[i, slot]))

    # Deduplicate: if both slots fire on the same frame keep the higher-conf one
    seen: dict = {}
    for f, p, c in zip(hda_f, hda_p, hda_c):
        if f not in seen or c > seen[f][1]:
            seen[f] = (p, c)
    hda_f = sorted(seen.keys())
    hda_p = [seen[f][0] for f in hda_f]
    hda_c = [seen[f][1] for f in hda_f]

    return hda_f, hda_p, hda_c


# ══════════════════════════════════════════════════════════════════════════════
#  HD-T  (trajectory-based hit detection)
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_smoothing(shuttle_2d: np.ndarray,
                          max_jump: int      = 100,
                          cw: int            = 7,
                          co: float          = 50.0,
                          iw: int            = 15,
                          it: float          = 5.0,
                          gap6: int          = 6,
                          return_debug: bool = False):
    """Three-step trajectory smoothing — Algorithm 1 of the paper."""
    N  = len(shuttle_2d)
    fn = np.arange(N, dtype=np.float64)

    # ── Step 1: Denoising ─────────────────────────────────────────────────────
    s1 = shuttle_2d.copy()
    for i in range(N):
        if np.any(np.isnan(s1[i])):
            continue
        bef = next((np.linalg.norm(s1[i] - s1[j])
                    for j in range(i - 1, max(i - 5, -1), -1)
                    if not np.any(np.isnan(s1[j]))), None)
        aft = next((np.linalg.norm(s1[i] - s1[j])
                    for j in range(i + 1, min(i + 5, N))
                    if not np.any(np.isnan(s1[j]))), None)
        if (bef is not None and bef >= max_jump) or \
           (aft is not None and aft >= max_jump):
            s1[i] = np.nan

    def _qd(pts, q):
        if len(pts) < 3:
            return np.inf
        try:
            return np.linalg.norm([
                np.polyval(np.polyfit(pts[:, 0], pts[:, 1], 2), q[0]) - q[1],
                np.polyval(np.polyfit(pts[:, 0], pts[:, 2], 2), q[0]) - q[2]])
        except Exception:
            return np.inf

    def _win(arr, idx, direction, size):
        pts = []
        for j in range(1, size + 1):
            target = idx + (direction * j)
            if 0 <= target < len(arr):
                val = arr[target]
                if val is not None and not np.any(np.isnan(val)):
                    pts.append(tuple(val))
        unique_pts = []
        for p in pts:
            if p not in unique_pts:
                unique_pts.append(p)
        if len(unique_pts) > 0:
            return np.array(unique_pts, dtype=np.float64)
        return np.empty((0, 2))

    # ── Step 2: Curve fitting ─────────────────────────────────────────────────
    s2 = s1.copy()
    for i in range(N):
        if np.any(np.isnan(s1[i])):
            continue
        q        = np.array([fn[i], s1[i, 0], s1[i, 1]])
        win_fwd  = _win(s1, i, +1, cw)
        win_back = _win(s1, i, -1, cw)
        fd = _qd(win_back, q) if len(win_back) >= 3 else np.inf
        bd = _qd(win_fwd, q)  if len(win_fwd)  >= 3 else np.inf
        if (np.isfinite(fd) and fd >= co) or (np.isfinite(bd) and bd >= co):
            s2[i] = np.nan

    # ── Step 3a: 15-frame window quadratic interpolation ─────────────────────
    s3 = s2.copy()
    for i in range(N):
        if not np.any(np.isnan(s3[i])):
            continue
        half = iw // 2
        wi   = [j for j in range(max(0, i - half), min(N, i + half + 1))
                if not np.any(np.isnan(s2[j]))]
        if len(wi) < 3:
            continue
        pts = np.array([[fn[j], s2[j, 0], s2[j, 1]] for j in wi])
        try:
            px = np.polyval(np.polyfit(pts[:, 0], pts[:, 1], 2), fn[i])
            py = np.polyval(np.polyfit(pts[:, 0], pts[:, 2], 2), fn[i])
            bp = pts[pts[:, 0] < fn[i]]
            ap = pts[pts[:, 0] > fn[i]]
            if len(bp) >= 3 and len(ap) >= 3:
                fd = np.linalg.norm([
                    np.polyval(np.polyfit(bp[:, 0], bp[:, 1], 2), fn[i]) - px,
                    np.polyval(np.polyfit(bp[:, 0], bp[:, 2], 2), fn[i]) - py])
                bd = np.linalg.norm([
                    np.polyval(np.polyfit(ap[:, 0], ap[:, 1], 2), fn[i]) - px,
                    np.polyval(np.polyfit(ap[:, 0], ap[:, 2], 2), fn[i]) - py])
                if fd <= it or bd <= it:
                    s3[i] = [px, py]
            else:
                s3[i] = [px, py]
        except Exception:
            continue

    # ── Step 3b: second-pass 6-frame adjacent gap fill ────────────────────────
    s4    = s3.copy()
    half6 = gap6 // 2
    for i in range(N):
        if not np.any(np.isnan(s4[i])):
            continue
        wi6 = ([j for j in range(max(0, i - half6), i)
                if not np.any(np.isnan(s3[j]))] +
               [j for j in range(i + 1, min(N, i + half6 + 1))
                if not np.any(np.isnan(s3[j]))])
        if len(wi6) < 3:
            continue
        pts = np.array([[fn[j], s3[j, 0], s3[j, 1]] for j in wi6])
        try:
            s4[i] = [
                np.polyval(np.polyfit(pts[:, 0], pts[:, 1], 2), fn[i]),
                np.polyval(np.polyfit(pts[:, 0], pts[:, 2], 2), fn[i]),
            ]
        except Exception:
            continue

    if return_debug:
        return s4, {k: int((~np.any(np.isnan(a), axis=1)).sum())
                    for k, a in [('original_valid', shuttle_2d),
                                 ('step1_valid',    s1),
                                 ('step2_valid',    s2),
                                 ('step3a_valid',   s3),
                                 ('final_valid',    s4)]}
    return s4


def compute_hd_t(shuttle_smooth: np.ndarray, frame_ids: list,
                 min_dist: int = 15, dir_thr: float = 90.0) -> tuple:
    """Detect hit moments from the smoothed shuttlecock trajectory (HD-T)."""
    xy    = shuttle_smooth[frame_ids].copy()
    valid = ~np.any(np.isnan(xy), axis=1)
    if valid.sum() < 10:
        return np.array([]), np.zeros(len(xy))
    xf   = xy.copy()
    nans = ~valid
    if nans.any() and (~nans).sum() > 1:
        idx = np.arange(len(xf))
        for c in range(2):
            xf[nans, c] = np.interp(idx[nans], idx[~nans], xf[~nans, c])
    y_min, _ = find_peaks(-xf[:, 1], distance=min_dist)
    dx = np.diff(xf[:, 0]); dy = np.diff(xf[:, 1])
    ang = np.zeros(len(xf))
    for i in range(len(dx) - 1):
        v1 = np.array([dx[i], dy[i]]); v2 = np.array([dx[i + 1], dy[i + 1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            ang[i + 1] = np.degrees(
                np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
    ang[~valid] = 0.
    dir_hits = np.where((ang > dir_thr) & valid)[0]
    all_hits = np.union1d(y_min, dir_hits)
    if len(all_hits) == 0:
        return np.array([]), ang
    filtered, i = [], 0
    while i < len(all_hits):
        cl = [all_hits[i]]
        while i + 1 < len(all_hits) and all_hits[i + 1] - all_hits[i] < min_dist:
            i += 1; cl.append(all_hits[i])
        filtered.append(max(cl, key=lambda f: ang[f]))
        i += 1
    return np.array([frame_ids[i] for i in filtered]), ang


# ══════════════════════════════════════════════════════════════════════════════
#  NMS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_peak_nms(f_list, p_list, c_list, dist: int = 18):
    """Refined NMS that specifically looks for local confidence maxima."""
    if not f_list:
        return [], [], []
    data = sorted(zip(f_list, p_list, c_list), key=lambda x: x[0])

    keep_f, keep_p, keep_c = [], [], []
    i = 0
    while i < len(data):
        cluster = [data[i]]
        j = i + 1
        while j < len(data) and (data[j][0] - data[i][0]) < dist:
            cluster.append(data[j])
            j += 1
        best = max(cluster, key=lambda x: x[2])
        keep_f.append(best[0]); keep_p.append(best[1]); keep_c.append(best[2])
        i = j

    return keep_f, keep_p, keep_c


def apply_temporal_nms(f_list, p_list, c_list, dist: int = 25):
    """Sliding-window temporal NMS — keeps highest-confidence peak per window."""
    if not f_list:
        return [], [], []
    combined = sorted(zip(f_list, p_list, c_list), key=lambda x: x[0])
    keep_f, keep_p, keep_c = [], [], []

    curr_f, curr_p, curr_c = combined[0]
    for i in range(1, len(combined)):
        f, p, c = combined[i]
        if f - curr_f < dist:
            if c > curr_c:
                curr_f, curr_p, curr_c = f, p, c
        else:
            keep_f.append(curr_f); keep_p.append(curr_p); keep_c.append(curr_c)
            curr_f, curr_p, curr_c = f, p, c
    keep_f.append(curr_f); keep_p.append(curr_p); keep_c.append(curr_c)
    return keep_f, keep_p, keep_c


# ══════════════════════════════════════════════════════════════════════════════
#  SRA  (Shot Refinement Algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def _build_segs(hda_f: list, hda_p: list, gap: int = 3) -> list:
    """Merge consecutive HD-A detections by the same player into segments."""
    if not hda_f:
        return []
    segs = []
    s, e, p = hda_f[0], hda_f[0], hda_p[0]
    for f, pp in zip(hda_f[1:], hda_p[1:]):
        if f - e <= gap and pp == p:
            e = f
        else:
            segs.append({'start': s, 'end': e, 'player': p})
            s, e, p = f, f, pp
    segs.append({'start': s, 'end': e, 'player': p})
    return segs


def _peak_frame(hda_f: list, hda_c: list, s: int, e: int) -> Optional[int]:
    """Frame with the highest HD-A confidence score in [s, e]."""
    bf, bc = None, -1.0
    for f, c in zip(hda_f, hda_c):
        if s <= f <= e and c > bc:
            bc, bf = c, f
    return bf


def sra(hdt_f: list, hda_f: list, hda_p: list, hda_c: list,
        total_frames: int) -> list:
    """Shot Refinement Algorithm — Algorithm 2 of the paper."""
    segs = _build_segs(hda_f, hda_p)
    hits = []

    for seg in segs:
        s, e, pl = seg['start'], seg['end'], seg['player']
        ins = [f for f in hdt_f if s <= f <= e]

        if pl is None:
            continue

        if len(ins) == 1:
            hits.append({'frame': int(ins[0]), 'player': int(pl)})

        elif len(ins) > 1:
            pf   = _peak_frame(hda_f, hda_c, s, e)
            best = min(ins, key=lambda f: abs(f - pf)) if pf is not None else ins[0]
            hits.append({'frame': int(best), 'player': int(pl)})

        else:
            pf = _peak_frame(hda_f, hda_c, s, e)
            if pf is not None:
                hits.append({'frame': int(pf), 'player': int(pl)})

    return sorted(hits, key=lambda x: x['frame'])
