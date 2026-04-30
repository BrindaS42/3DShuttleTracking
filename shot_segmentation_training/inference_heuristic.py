"""
inference.py — Inference module for HD-A Heuristic pipeline.
=============================================================
Contains:
  - assign_sides_pkl       : map near/far slots to court sides
  - detect_swing_actions_pkl : HD-A confidence stream (heuristic features)
  - trajectory_smoothing   : HD-T Algorithm 1 (paper-faithful)
  - compute_hd_t           : hit moment detection from shuttle trajectory
  - sra / sra_per_rally_trimmed : Shot Refinement Algorithm (SRA)
  - run_match              : per-match inference entry point
"""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from config import (
    WRIST_R, WRIST_L, ELBOW_R, ELBOW_L,
    SHOULD_R, SHOULD_L, L_HIP, R_HIP, PKL_ROOT,
)
from data_loader import (
    load_pose_pkl, load_shuttle_npy, load_shot_frames,
    _reconstruct_frame_map_from_csvs,
)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _kps_from_pf(pf, slot: int) -> np.ndarray:
    """Return (17, 2) keypoint array for slot 0=near, 1=far. Zeros if absent."""
    person = pf.near if slot == 0 else pf.far
    if person is None or not hasattr(person, 'keypoints') or person.keypoints is None:
        return np.zeros((17, 2), dtype=np.float64)
    kps = np.array(person.keypoints, dtype=np.float64)
    return kps[:17, :2] if kps.shape[0] >= 17 else np.zeros((17, 2), dtype=np.float64)


def _ankle_mid(person) -> np.ndarray:
    """Return ankle midpoint (2,) or zeros if unavailable."""
    la = (np.array(person.left_ankle_px,  dtype=np.float64)
          if getattr(person, 'left_ankle_px',  None) is not None else None)
    ra = (np.array(person.right_ankle_px, dtype=np.float64)
          if getattr(person, 'right_ankle_px', None) is not None else None)
    if la is not None and ra is not None:
        return (la + ra) / 2.0
    return la if la is not None else (ra if ra is not None else np.zeros(2))


def _bbox_area(person) -> float:
    """Return bounding-box area or 0."""
    bb = getattr(person, 'bbox', None)
    if bb is None or len(bb) < 4:
        return 0.0
    return max(0.0, float((bb[2] - bb[0]) * (bb[3] - bb[1])))


# ══════════════════════════════════════════════════════════════════════════════
#  SIDE ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_sides_pkl(poses: list) -> np.ndarray:
    """Assign court sides from pkl near/far slots.

    The YOLOv8 + court-polygon gate already splits players into near (lower
    half of frame, higher y) and far (upper half).  Map directly:
        slot 0 (near) → side 2  (bottom court half)
        slot 1 (far)  → side 1  (top court half)
    """
    N     = len(poses)
    sides = np.zeros((N, 2), dtype=np.int32)
    sides[:, 0] = 2   # near → bottom
    sides[:, 1] = 1   # far  → top
    return sides


# ══════════════════════════════════════════════════════════════════════════════
#  HD-A  (heuristic confidence stream from pkl pose features)
# ══════════════════════════════════════════════════════════════════════════════

def detect_swing_actions_pkl(poses: list,
                              frame_ids: list,
                              sides: np.ndarray,
                              smooth_sigma: float = 2.0,
                              thr_sigma:    float = 1.5) -> tuple:
    """HD-A confidence stream from pkl pose features.

    Four signals per player, each normalised to [0, 1]:
      1. Wrist speed accel  — frame-to-frame wrist velocity gradient
      2. Arm extension angle — shoulder→elbow→wrist angle / 180
      3. Ankle displacement  — frame-to-frame ankle midpoint delta (foot plant)
      4. Bbox area rate      — rate of change of player bounding-box area (lunge)

    Weighted sum → smoothed → threshold → hda_f / hda_p / hda_c.
    Output format identical to detect_swing_actions() so SRA is unchanged.
    """
    N = len(frame_ids)

    kps_arr   = np.zeros((N, 2, 17, 2), dtype=np.float64)
    ankle_arr = np.zeros((N, 2, 2),     dtype=np.float64)
    area_arr  = np.zeros((N, 2),        dtype=np.float64)
    conf_arr  = np.zeros((N, 2),        dtype=np.float64)

    for i, pf in enumerate(poses):
        for s, attr in enumerate(['near', 'far']):
            person = getattr(pf, attr, None)
            if person is None:
                continue
            kps_arr[i, s]   = _kps_from_pf(pf, s)
            ankle_arr[i, s] = _ankle_mid(person)
            area_arr[i, s]  = _bbox_area(person)
            conf_arr[i, s]  = float(getattr(person, 'confidence', 0.0))

    # ── Signal 1: wrist speed acceleration ───────────────────────────────────
    accel = np.zeros((N, 2))
    for s in range(2):
        wr_r  = kps_arr[:, s, WRIST_R]
        wr_l  = kps_arr[:, s, WRIST_L]
        speed = np.maximum(
            np.linalg.norm(np.diff(wr_r, axis=0, prepend=wr_r[:1]), axis=1),
            np.linalg.norm(np.diff(wr_l, axis=0, prepend=wr_l[:1]), axis=1),
        )
        accel[:, s] = np.abs(np.gradient(speed))

    # ── Signal 2: arm extension angle ────────────────────────────────────────
    ext = np.zeros((N, 2))
    for s in range(2):
        for i in range(N):
            sh = (kps_arr[i, s, SHOULD_R] if kps_arr[i, s, SHOULD_R, 0] > 0
                  else kps_arr[i, s, SHOULD_L])
            el = (kps_arr[i, s, ELBOW_R]  if kps_arr[i, s, ELBOW_R,  0] > 0
                  else kps_arr[i, s, ELBOW_L])
            wr = (kps_arr[i, s, WRIST_R]  if kps_arr[i, s, WRIST_R,  0] > 0
                  else kps_arr[i, s, WRIST_L])
            v1 = sh - el; v2 = wr - el
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-4 and n2 > 1e-4:
                ext[i, s] = np.degrees(
                    np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
    ext /= 180.0

    # ── Signal 3: ankle displacement (foot-plant signal) ─────────────────────
    ankle_disp = np.zeros((N, 2))
    for s in range(2):
        delta = np.linalg.norm(
            np.diff(ankle_arr[:, s], axis=0, prepend=ankle_arr[:1, s]), axis=1)
        ankle_disp[:, s] = delta

    # ── Signal 4: bbox area rate of change (lunge signal) ────────────────────
    area_rate = np.zeros((N, 2))
    for s in range(2):
        area_rate[:, s] = np.abs(np.gradient(area_arr[:, s]))

    # ── Smooth and normalise all signals ─────────────────────────────────────
    def _norm(arr):
        out = arr.copy()
        for s in range(2):
            mx = out[:, s].max()
            if mx > 1e-8:
                out[:, s] /= mx
        return out

    sm_accel = np.stack(
        [gaussian_filter1d(accel[:, s],      smooth_sigma) for s in range(2)], 1)
    sm_ext   = np.stack(
        [gaussian_filter1d(ext[:, s],        smooth_sigma) for s in range(2)], 1)
    sm_ankle = np.stack(
        [gaussian_filter1d(ankle_disp[:, s], smooth_sigma) for s in range(2)], 1)
    sm_area  = np.stack(
        [gaussian_filter1d(area_rate[:, s],  smooth_sigma) for s in range(2)], 1)

    sm_accel = _norm(sm_accel)
    sm_ext   = _norm(sm_ext)
    sm_ankle = _norm(sm_ankle)
    sm_area  = _norm(sm_area)

    # Weighted combination — wrist + arm dominate; ankle + bbox provide context
    conf_raw = (0.35 * sm_accel +
                0.35 * sm_ext   +
                0.20 * sm_ankle +
                0.10 * sm_area)

    # Scale by per-frame detection confidence
    conf_raw *= (0.5 + 0.5 * _norm(conf_arr))

    # Final per-player normalisation
    for s in range(2):
        rng = conf_raw[:, s].max() - conf_raw[:, s].min()
        if rng > 1e-8:
            conf_raw[:, s] = (conf_raw[:, s] - conf_raw[:, s].min()) / rng

    comb = conf_raw.max(axis=1)
    thr  = comb.mean() + thr_sigma * comb.std()

    hda_f, hda_p, hda_c = [], [], []
    for i in np.where(comb > thr)[0]:
        ps = int(np.argmax(conf_raw[i]))
        hda_f.append(frame_ids[i])
        hda_p.append(int(sides[i, ps]))
        hda_c.append(float(conf_raw[i, ps]))

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
    xf = xy.copy()
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


def sra_per_rally_trimmed(shot_df, hdt_f,
                           hda_f: list, hda_p: list, hda_c: list,
                           total_frames: int) -> list:
    """SRA variant for pkl / trimmed-index mode.

    Uses frame_num_t (trimmed indices) instead of frame_num (original) for
    the [rmin, rmax] rally windows so HD-A and HD-T frames (both in trimmed
    space) are correctly matched against GT annotations.
    """
    all_hits = []
    for rk, grp in shot_df.groupby('rally_key'):
        rv = grp['frame_num_t'].values
        if not len(rv):
            continue
        rmin, rmax = int(rv.min()), int(rv.max())
        hdt_r = [f for f in hdt_f if rmin <= f <= rmax]
        hda_r = [(f, p, c) for f, p, c in zip(hda_f, hda_p, hda_c)
                 if rmin <= f <= rmax]
        if not hda_r:
            continue
        ff, fp, fc = zip(*hda_r)
        for h in sra(hdt_r, list(ff), list(fp), list(fc), total_frames):
            h['rally'] = rk
            all_hits.append(h)
    return sorted(all_hits, key=lambda x: x['frame'])


# ══════════════════════════════════════════════════════════════════════════════
#  PER-MATCH INFERENCE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_match(match: dict, _inferencer_ref=None):
    """Run full HD-A (heuristic) + HD-T + SRA pipeline on one match.

    Args:
        match:            match dict from discover_pkl_matches()
        _inferencer_ref:  unused (kept for API compatibility with YOLO variant)

    Returns:
        hits      list[dict]   predicted shot hits  {frame, player, rally}
        gt_frames np.ndarray   ground-truth trimmed frame indices
        or (None, None) on failure.
    """
    name = match['name']

    # ── 1. Load pre-computed pose + shuttle ───────────────────────────────────
    poses      = load_pose_pkl(match)
    shuttle_2d = load_shuttle_npy(match)
    print(f'  shuttle_2d shape: {shuttle_2d.shape}'
          if shuttle_2d is not None else '  shuttle_2d: None')

    if poses is None or shuttle_2d is None:
        print(f'  [skip] no pkl/npy for {name[:55]}')
        return None, None

    # ── 2. Reconstruct frame map (CSVs only — no video probe) ─────────────────
    sorted_frames, orig_to_trimmed, expected_len = \
        _reconstruct_frame_map_from_csvs(match)

    n_poses, n_shuttle = len(poses), len(shuttle_2d)
    if n_poses != expected_len or n_shuttle != expected_len:
        print(f'  [skip] {name[:50]}: length mismatch — '
              f'frame_map={expected_len} poses={n_poses} shuttle={n_shuttle}')
        return None, None

    frame_ids = list(range(len(poses)))
    sides     = assign_sides_pkl(poses)

    # ── 3. HD-A  (heuristic pkl features) ────────────────────────────────────
    print(f'  [hda] using heuristic pkl features')
    hda_f, hda_p, hda_c = detect_swing_actions_pkl(poses, frame_ids, sides)

    # ── 4. HD-T  (shuttle trajectory) ────────────────────────────────────────
    if not isinstance(shuttle_2d, np.ndarray) or shuttle_2d.shape[1] != 2:
        print(f'  [skip] {name[:50]}: invalid shuttle_2d shape {shuttle_2d.shape}')
        return None, None

    hdt_f, _ = compute_hd_t(shuttle_2d, frame_ids)

    # ── 5. Ground-truth mapping (original → trimmed indices) ─────────────────
    _, shot_df = load_shot_frames(match['path'])
    shot_df = shot_df.copy()
    shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
    shot_df = shot_df.dropna(subset=['frame_num_t'])
    shot_df['frame_num_t'] = shot_df['frame_num_t'].astype(int)

    # ── 6. SRA ────────────────────────────────────────────────────────────────
    hits      = sra_per_rally_trimmed(shot_df, hdt_f, hda_f, hda_p, hda_c,
                                      len(poses))
    gt_frames = shot_df['frame_num_t'].values

    return hits, gt_frames
