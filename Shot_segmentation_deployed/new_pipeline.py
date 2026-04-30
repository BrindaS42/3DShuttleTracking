"""
pipeline.py — Badminton Shot Detection Pipeline
================================================
Run once; everything downloads and executes automatically.

    python pipeline.py

Requirements:
    pip install -r requirements.txt
    CUDA GPU must be available.

Fixes applied vs original:
  BUG1  workspace_name / workflow_id were bare Python identifiers → quoted strings
  BUG2  api_key was a bare Python identifier → quoted string  (ROTATE YOUR KEY)
  BUG3  InferenceHTTPClient constructed at module-level → lazy init inside function
  BUG4  TrackNetV3 CSV discovery used a fragile stem match → glob for *_ball.csv
  BUG5  Exception handler in download loop swallowed KeyboardInterrupt/SystemExit
  NOTE1 Added CUDA availability guard in build_inferencer()
  NOTE2 Empty test_results guard before build_metrics_df()
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import pickle
import os, sys, subprocess, shutil, glob, json, zipfile, csv, logging
from pathlib import Path
import random
from dataclasses import dataclass, field
from typing import Optional

# ── third-party (installed via requirements.txt) ──────────────────────────────
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
# from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════



# Set this to the folder that contains all your match sub-directories
MATCH_DB      = Path('SoloShuttlePose/ShuttleSet/ShuttleSet/set')
TRACKNET_DIR  = Path('TrackNetV3/TrackNetV3')           # predict.py lives here
TRACKNET_PT   = Path('TrackNetV3/TrackNetV3_ckpts/ckpts/TrackNet_best.pt')
INPAINTNET_PT = Path('TrackNetV3/TrackNetV3_ckpts/ckpts/InpaintNet_best.pt')

POSE_DIR     = Path('outputs/pose')
SHUTTLE_DIR  = Path('outputs/shuttle_matches')
RESULTS_DIR  = Path('outputs/results')
VIZ_DIR      = Path('outputs/viz')

# Root folder that contains per-match pose & shuttle outputs from the
# external estimation pipeline (poses_trimmed.pkl + shuttle_trimmed.npy)
PKL_ROOT     = Path('output files of pose & shuttle')
PKL_TRIM_BUF = 60   # must match the buffer used when building the pkl

# ── Swing classifier training config ─────────────────────────────────────
DATASET_DIR   = Path('dataset_swing')       # cropped images land here
SWING_CLF_PT  = Path('swing_classifier.pt') # final best.pt copied here
SWING_WINDOW  = 10    # ±frames around GT shot = positive crop
NEG_RATIO     = 3     # negative crops per positive crop
CROP_PAD      = 10    # pixel padding around bbox before crop
MIN_CROP_SIZE = 32    # skip crops smaller than this (px)

TRAIN_RATIO  = 0.8
RANDOM_SEED  = 42

# COCO keypoint indices
WRIST_R  = 10; WRIST_L  = 9
ELBOW_R  = 8;  ELBOW_L  = 7
SHOULD_R = 6;  SHOULD_L = 5
L_HIP    = 11; R_HIP    = 12
L_ANKLE  = 15; R_ANKLE  = 16

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — SETUP: clone repos, download weights, install deps
# ══════════════════════════════════════════════════════════════════════════════

def _run(cmd, **kwargs):
    print(f'  $ {cmd}')
    subprocess.run(cmd, shell=True, check=True, **kwargs)



def setup():
    print('\n' + '='*60)
    print('STEP 0 — Environment setup')
    print('='*60)

    # ── validate weights exist ────────────────────────────────────────────────
    if not TRACKNET_PT.exists() or not INPAINTNET_PT.exists():
        raise RuntimeError(
            f'TrackNet weights not found.\n'
            f'Expected:\n  {TRACKNET_PT}\n  {INPAINTNET_PT}\n'
            'Place the weight files at those paths and re-run.')
    else:
        print('[setup] TrackNetV3 weights found.')

    # ── validate match_db exists ──────────────────────────────────────────────
    if not MATCH_DB.exists():
        raise RuntimeError(
            f'match_db not found at {MATCH_DB}.\n'
            'Set MATCH_DB at the top of this file to the folder that contains '
            'your match sub-directories.')

    # ── install TrackNetV3 Python requirements ────────────────────────────────
    tnv3_req = Path('TrackNetV3/requirements.txt')
    if tnv3_req.exists():
        print('[setup] Installing TrackNetV3 requirements...')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(tnv3_req)],
            check=True)

    # ── install mmpose if missing ─────────────────────────────────────────────
    try:
        import mmpose  # noqa
        print('[setup] mmpose already installed.')
    except ImportError:
        print('\n[setup] Installing mmpose stack...')
        _run('pip install -q -U openmim')
        _run('mim install -q mmengine')
        _run('mim install -q "mmcv>=2.0.0"')
        _run('mim install -q "mmdet>=3.0.0"')
        _run('mim install -q "mmpose>=1.0.0"')

    for d in [POSE_DIR, SHUTTLE_DIR, RESULTS_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print('\n[setup] Done.\n')

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA DISCOVERY & TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def discover_matches(match_db: Path):
    matches = []
    for name in sorted(os.listdir(match_db)):
        p = match_db / name
        if not p.is_dir():
            continue
        videos = [f for f in os.listdir(p) if f.endswith('.mp4')]
        csvs   = [f for f in os.listdir(p) if f.startswith('set') and f.endswith('.csv')]
        if videos and csvs:
            pkl_available = (
                (PKL_ROOT / name / 'pose_out'    / 'poses_trimmed.pkl').exists() and
                (PKL_ROOT / name / 'shuttle_out' / 'shuttle_trimmed.npy').exists()
            )
            matches.append({
                'name':   name,
                'path':   str(p),
                'video':  str(p / videos[0]),
                'n_csvs': len(csvs),
                'has_pkl': pkl_available,
            })
    matches = [matches[0]] # remove later
    return matches

def train_test_split(matches, ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    # rng     = np.random.default_rng(seed)
    # idx     = rng.permutation(len(matches))
    # n_train = int(len(matches) * ratio)
    # return [matches[i] for i in idx[:n_train]], [matches[i] for i in idx[n_train:]]
    return matches, matches # remove later

def load_shot_frames(match_path: str):
    # ShuttleSet: one set*.csv per set, rally number in first column
    files = sorted(glob.glob(os.path.join(match_path, 'set*.csv')))
    if not files:
        raise FileNotFoundError(f'No set*.csv in {match_path}')
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        # ShuttleSet columns: rally, ball_round, time, frame_num, ...
        df.columns = range(len(df.columns))
        df = df.rename(columns={0: 'rally', 3: 'frame_num'})
        df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
        df = df.dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    # Unique rally key = (source_file, rally) since rally resets per set
    combined['rally_key'] = combined['source_file'] + '_r' + combined['rally'].astype(str)
    return combined['frame_num'].values, combined

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — POSE EXTRACTION  (RTMPose-m via MMPose)
# ══════════════════════════════════════════════════════════════════════════════

def build_inferencer():
    # NOTE1 FIX: guard against missing CUDA so the script fails fast with a
    # clear message rather than hanging or producing silent OOM errors.
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('[WARNING] CUDA not available — pose extraction will run on CPU and be slow.')
    except ImportError:
        device = 'cpu'
        print('[WARNING] torch not importable — defaulting to CPU.')

    from mmpose.apis import MMPoseInferencer
    return MMPoseInferencer(
        pose2d='rtmpose-m_8xb256-420e_coco-256x192',
        device=device)


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYER CANDIDATE FILTER  (referee / umpire removal)
# ══════════════════════════════════════════════════════════════════════════════
# Ported from module3_pose_estimation.py.  Referees sit near the net at frame
# edges with a small vertical keypoint span (~6-10% of frame height).  These
# two gates match the bbox filtering in estimate_poses_batched():
#   cx gate   — horizontal centre must be in the inner 80% of frame width
#   y_span    — vertical spread of confident keypoints > 12% of frame height
#               (standing player ~20-40%, seated referee ~6-10%)
#   near/far  — cy >= mid_y → near (slot 0), cy < mid_y → far (slot 1)
#               within each half we keep the detection with the largest y_span
# Thresholds are loose — crouching/lunging players must not be rejected.
# A last-resort confidence fallback fires only when spatial gates reject
# everything (e.g. both players in the same court half momentarily).

def _filter_player_candidates(preds: list, frame_w: int, frame_h: int) -> list:
    """Return up to 2 spatially plausible player detections, ordered [near, far].

    Thresholds are intentionally loose — the goal is only to screen out
    seated referees/umpires at the net, NOT to reject crouching or lunging
    players whose y_span can drop significantly during play.

    Gates:
      cx gate   — centre-x in inner 90% of width (refs are at frame edges)
      y_span    — > 5% of frame height (seated ref ~3-4%, crouching player ≥6%)
      min_conf  — at least 3 keypoints with score > 0.25
    """
    mid_y     = frame_h / 2.0
    cx_lo     = frame_w * 0.05   # very narrow edge exclusion only
    cx_hi     = frame_w * 0.95
    min_yspan = frame_h * 0.05   # just enough to reject seated refs

    near_cands, far_cands = [], []
    for pred in preds:
        kps    = np.array(pred['keypoints'])
        scores = np.array(pred['keypoint_scores'])
        mask   = scores > 0.25   # slightly lower than before to keep crouching detections
        if mask.sum() < 3:       # require at least 3 confident keypoints
            continue
        conf_kps = kps[mask]
        cx     = float(conf_kps[:, 0].mean())
        cy     = float(conf_kps[:, 1].mean())
        y_span = float(conf_kps[:, 1].max() - conf_kps[:, 1].min())

        if not (cx_lo < cx < cx_hi):
            continue   # literally at the frame edge — linesman / camera person
        if y_span < min_yspan:
            continue   # seated referee (very short vertical extent)

        if cy >= mid_y:
            near_cands.append((y_span, pred))
        else:
            far_cands.append((y_span, pred))

    result = []
    for cands in (near_cands, far_cands):
        if cands:
            result.append(max(cands, key=lambda t: t[0])[1])

    # Last-resort fallback: if the spatial filter removed everything
    # (e.g. players both in same half), fall back to top-2 by confidence
    # so we never return empty and silently drop an entire frame.
    if not result and preds:
        result = sorted(preds,
                        key=lambda p: float(np.mean(p['keypoint_scores'])),
                        reverse=True)[:2]

    return result[:2]


def extract_pose(inferencer, video_path: str, output_stem: str,
                 checkpoint_every: int = 1000,
                 frame_map: list = None):
    """Run pose extraction on video_path.

    If frame_map is provided (a list where frame_map[i] is the original
    frame number for trimmed frame i), frame IDs saved to the .npz will
    be remapped to original frame numbers so everything downstream stays
    consistent with shuttle_2d and ground-truth CSVs.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    kps_all, scores_all, fids_all = [], [], []

    def _save(path):
        np.savez(path,
                 keypoints  = np.array(kps_all,    dtype=np.float32),
                 scores     = np.array(scores_all, dtype=np.float32),
                 frame_ids  = np.array(fids_all,   dtype=np.int32),
                 fps        = np.array([fps]),
                 resolution = np.array([W, H]))

    latest  = f'{output_stem}_latest.npz'
    results = inferencer(video_path, show=False, batch_size=8)

    for fid, result in enumerate(tqdm(results, total=total,
                                      desc='  Pose', leave=False)):
        preds = result['predictions'][0]
        if not preds:
            continue
        # Spatial filter: removes referees/umpires before slot assignment
        top2 = _filter_player_candidates(preds, W, H)
        if not top2:
            continue
        fkps = np.zeros((2, 17, 2), dtype=np.float32)
        fsc  = np.zeros((2, 17),    dtype=np.float32)
        for i, p in enumerate(top2):
            fkps[i] = np.array(p['keypoints'],       dtype=np.float32)
            fsc[i]  = np.array(p['keypoint_scores'], dtype=np.float32)
        # Remap trimmed frame index → original frame index if a map was given
        orig_fid = frame_map[fid] if (frame_map is not None and fid < len(frame_map)) else fid
        kps_all.append(fkps); scores_all.append(fsc); fids_all.append(orig_fid)
        if fid > 0 and fid % checkpoint_every == 0:
            _save(latest)

    _save(f'{output_stem}.npz')

def run_pose_extraction(inferencer, matches):
    print('\n' + '='*60)
    print('STEP 2 — Pose extraction')
    print('='*60)
    for i, m in enumerate(matches, 1):
        out = str(POSE_DIR / (m['name'] + '_pose'))
        if Path(out + '.npz').exists():
            print(f'  [{i:2d}/{len(matches)}] skip (cached): {m["name"][:55]}')
            continue
        print(f'  [{i:2d}/{len(matches)}] {m["name"][:55]}')
        # Reuse the trimmed video (and its frame_map) already built by
        # run_shuttle_detection so we don't re-encode the video twice.
        tmp = SHUTTLE_DIR / m['name']
        tmp.mkdir(parents=True, exist_ok=True)
        trimmed_path, frame_map = _prepare_trimmed_video(m, tmp)
        extract_pose(inferencer, trimmed_path, out, frame_map=frame_map)
    print('Pose extraction complete.\n')

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2b — RALLY SUBSET TRIMMING
#  Trim each match video to only the frames covered by ground-truth rallies
#  (plus a small buffer).  This avoids feeding dead time / broadcast segments
#  to TrackNetV3 and MMPose, dramatically cutting RAM and run time.
# ══════════════════════════════════════════════════════════════════════════════

def create_rally_subset_video(video_path: str, gt_df: pd.DataFrame,
                              n_total: int, out_path: str,
                              logger: logging.Logger,
                              skip_extraction: bool = False):
    """Extract only exact rally frames (no buffer) from a full match video.

    Each rally_*.csv file is one rally (named rally_<set>-<rally>.csv).
    Buffer has been removed: frames outside annotated rally ranges are never
    written, so unannotated rallies between annotated ones don't bleed in.
    Returns sorted_frames: a list whose i-th element is the *original* frame
    number that corresponds to trimmed frame i.
    """
    logger.info("Calculating required frames for selected rallies...")
    required_frames = set()

    # Group by (set_file, rally) — rally number resets between sets so
    # both columns together form the unique key per rally.
    for (set_file, rally_num), group in gt_df.groupby(["set_file", "rally"]):
        first_hit = int(group["frame_num"].min())
        last_hit  = int(group["frame_num"].max())
        for f in range(first_hit, last_hit + 1):
            required_frames.add(f)

    sorted_frames = sorted(list(required_frames))

    # Bypasses the heavy FFmpeg decoding if the video already exists!
    if skip_extraction:
        logger.info(f"Skipping video slicing. Mapped {len(sorted_frames)} frames instantly.")
        return sorted_frames

    chunks = []
    if sorted_frames:
        start = sorted_frames[0]
        last  = sorted_frames[0]
        for f in sorted_frames[1:]:
            if f == last + 1:
                last = f
            else:
                chunks.append((start, last))
                start = f
                last  = f
        chunks.append((start, last))

    logger.info(f"Total required frames: {len(sorted_frames)} across {len(chunks)} segments.")
    logger.info("Slicing video (Fast Chunk-Seek Mode)...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for i, (start_f, end_f) in enumerate(chunks):
        logger.info(f"  -> Extracting segment {i+1}/{len(chunks)} "
                    f"(Frames {start_f} to {end_f})...")

        seek_target = max(0, start_f - 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
        curr_frame = seek_target

        while curr_frame < start_f:
            cap.read()
            curr_frame += 1

        bad_consecutive = 0
        while curr_frame <= end_f:
            ret, frame = cap.read()

            # Progress log so you know it isn't frozen
            if curr_frame % 100 == 0:
                print(f"     ... processing frame {curr_frame}/{end_f} ...",
                      end='\r')
            if not ret:
                bad_consecutive += 1
                if bad_consecutive > 5:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 1)
                    bad_consecutive = 0
            else:
                bad_consecutive = 0
                out.write(frame)

            curr_frame += 1
        print("")  # clear progress line

    cap.release()
    out.release()
    logger.info(f"Temporary trimmed video created at {out_path}")
    return sorted_frames


def _prepare_trimmed_video(match: dict, tmp_dir: Path) -> tuple:
    """Return (trimmed_video_path, frame_map) for a match, using cache if available.

    frame_map[i] = original frame number for trimmed frame i.
    If the trimmed video and frame map already exist on disk they are reused
    without re-encoding — only the frame map is loaded (fast path).
    """
    trimmed_path = str(tmp_dir / 'trimmed.mp4')
    map_path     = tmp_dir / 'frame_map.npy'
    logger       = logging.getLogger('pipeline')

    # ── load ground-truth set CSV files for this match (ShuttleSet format) ───
    rally_files = sorted(glob.glob(os.path.join(match['path'], 'set*.csv')))
    if not rally_files:
        raise FileNotFoundError(f'No set*.csv in {match["path"]}')

    dfs = []
    for rf in rally_files:
        df = pd.read_csv(rf, header=None)
        df.columns = range(len(df.columns))
        df = df.rename(columns={0: 'rally', 3: 'frame_num'})
        df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
        df = df.dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['set_file'] = os.path.basename(rf)
        dfs.append(df)
    gt_df = pd.concat(dfs, ignore_index=True)

    # ── probe original frame count ────────────────────────────────────────────
    cap     = cv2.VideoCapture(match['video'])
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── fast path: both files already exist ──────────────────────────────────
    if Path(trimmed_path).exists() and map_path.exists():
        logger.info(f'[trim] cache hit — loading frame_map for {match["name"][:55]}')
        frame_map = np.load(str(map_path)).tolist()
        return trimmed_path, frame_map

    # ── slow path: encode trimmed video ──────────────────────────────────────
    print(f'  [trim] building trimmed video for {match["name"][:55]} ...')
    frame_map = create_rally_subset_video(
        video_path      = match['video'],
        gt_df           = gt_df,
        n_total         = n_total,
        out_path        = trimmed_path,
        logger          = logger,
        skip_extraction = False,
    )
    np.save(str(map_path), np.array(frame_map, dtype=np.int32))
    pct = 100.0 * len(frame_map) / n_total if n_total else 0
    print(f'  [trim] {len(frame_map)}/{n_total} frames kept ({pct:.1f}%) '
          f'→ {trimmed_path}')
    return trimmed_path, frame_map


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — SHUTTLE DETECTION  (TrackNetV3 + cleaning)
# ══════════════════════════════════════════════════════════════════════════════

def _run_tracknet_subprocess(video_path: str, out_dir: Path) -> Path:
    # BUG4 FIX: use glob to find the actual *_ball.csv output instead of
    # constructing a path from the video stem (TrackNetV3 may append suffixes).
    existing = glob.glob(str(out_dir / '*_ball.csv'))
    if existing:
        return Path(existing[0])

    video_stem   = Path(video_path).stem
    expected_csv = out_dir / f'{video_stem}_ball.csv'

    cmd = [
        sys.executable, str(TRACKNET_DIR / 'predict.py'),
        '--video_file',      str(Path(video_path).resolve()),
        '--tracknet_file',   str(TRACKNET_PT.resolve()),
        '--inpaintnet_file', str(INPAINTNET_PT.resolve()),
        '--save_dir',        str(out_dir.resolve()),
        '--eval_mode',       'weight',
        '--batch_size',      '4',
        '--max_sample_num',  '50',   # prevents OOM on median image step
        '--large_video',
    ]
    # Pass a per-frame FFmpeg read timeout (microseconds) so that a corrupted
    # NAL unit that causes FFmpeg to hang will be abandoned after 2 s instead
    # of blocking until the global stream timeout kills the whole process.
    # OPENCV_FFMPEG_OPTIONS is read by OpenCV's FFmpeg back-end for every
    # VideoCapture opened in the child process — no TrackNetV3 source changes
    # needed.  The value is a space-separated list of key=value pairs that map
    # directly to AVFormatContext / AVIOContext options.
    #   timeout        — max microseconds to wait for a single read (libavformat)
    #   stimeout       — same but for RTSP/TCP streams; harmless for local files
    subprocess_env = os.environ.copy()
    subprocess_env['OPENCV_FFMPEG_OPTIONS'] = 'timeout=2000000 stimeout=2000000'
    try:
        subprocess.run(cmd, check=True, env=subprocess_env)
    except subprocess.CalledProcessError as exc:
        # A corrupted frame can cause FFmpeg to abort predict.py with a non-zero
        # exit code.  Log the failure and continue — _parse_csv will fill the
        # missing frames with np.nan naturally since those rows won't appear in
        # any partial output CSV that was written before the crash.
        print(f'    [WARN] TrackNetV3 exited with code {exc.returncode} '
              f'for {Path(video_path).name} — will use partial CSV if available.')

    # TrackNetV3 sometimes writes the CSV next to the input video instead of
    # save_dir (path-joining bug on some versions).  Intercept and move it.
    rogue_csv = Path(video_path).parent / f'{video_stem}_ball.csv'
    if not expected_csv.exists() and rogue_csv.exists():
        shutil.move(str(rogue_csv), str(expected_csv))
        print(f'    [TrackNet] intercepted CSV from video folder → {expected_csv}')

    # Final glob as belt-and-braces (handles version-specific suffix variants)
    found = glob.glob(str(out_dir / '*_ball.csv'))
    if not found:
        raise FileNotFoundError(
            f'No *_ball.csv found in {out_dir} after running predict.py')
    return Path(found[0])

def _parse_csv(csv_path: Path, video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    pts   = np.full((N, 2), np.nan, dtype=np.float64)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        col    = {h.strip().lower(): h.strip() for h in (reader.fieldnames or [])}
        fc     = col.get('frame',      col.get('frame_id', 'Frame'))
        vc     = col.get('visibility', col.get('vis',      'Visibility'))
        xc     = col.get('x',          col.get('ball_x',   'X'))
        yc     = col.get('y',          col.get('ball_y',   'Y'))
        for row in reader:
            try:
                fi, vis = int(row[fc]), int(row[vc])
                x,  y   = float(row[xc]), float(row[yc])
            except (KeyError, ValueError):
                continue
            if 0 <= fi < N and vis == 1 and not (x == 0 and y == 0):
                pts[fi] = [x, y]
    return pts

def run_shuttle_detection(matches):
    print('='*60)
    print('STEP 3 — Shuttle detection')
    print('='*60)
    for i, m in enumerate(matches, 1):
        npy = SHUTTLE_DIR / (m['name'] + '_shuttle.npy')
        if npy.exists():
            print(f'  [{i:2d}/{len(matches)}] skip (cached): {m["name"][:55]}')
            continue
        print(f'  [{i:2d}/{len(matches)}] {m["name"][:55]}')
        tmp = SHUTTLE_DIR / m['name']
        tmp.mkdir(parents=True, exist_ok=True)
        # ── trim to rally frames only before running TrackNetV3 ───────────
        trimmed_path, frame_map = _prepare_trimmed_video(m, tmp)

        # Run TrackNetV3 on the smaller trimmed video
        csv_path = _run_tracknet_subprocess(trimmed_path, tmp)

        # _parse_csv returns an array sized to the trimmed video frame count.
        # Remap each valid trimmed-frame index → original frame index so that
        # shuttle_2d is always indexed by original frame number (everything
        # downstream depends on this).
        shuttle_trimmed = _parse_csv(csv_path, trimmed_path)

        cap_orig = cv2.VideoCapture(m['video'])
        n_orig   = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_orig.release()
        shuttle_2d    = np.full((n_orig, 2), np.nan, dtype=np.float64)
        frame_map_arr = np.array(frame_map, dtype=np.int32)
        for trimmed_idx, orig_idx in enumerate(frame_map_arr):
            if trimmed_idx < len(shuttle_trimmed) and \
               not np.any(np.isnan(shuttle_trimmed[trimmed_idx])):
                shuttle_2d[orig_idx] = shuttle_trimmed[trimmed_idx]

        np.save(npy, shuttle_2d)
        valid = int(np.sum(~np.any(np.isnan(shuttle_2d), axis=1)))
        print(f'    saved {m["name"]}_shuttle.npy  '
              f'({valid}/{len(shuttle_2d)} frames valid)')
    print('Shuttle detection complete.\n')

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — COURT DETECTION & AGGREGATION  (Roboflow + median homography)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Keypoint:
    x: float; y: float; confidence: float; class_id: int; class_name: str

@dataclass
class CourtCorner:
    detection_id: str; bbox_x: float; bbox_y: float
    confidence: float; keypoints: list

@dataclass
class NetLine:
    detection_id: str; confidence: float; top: Keypoint; bottom: Keypoint

@dataclass
class FrameDetection:
    frame_num: int; image_width: int; image_height: int
    corners: list; nets: list

@dataclass
class AggregatedCourt:
    corners_median: dict
    net_top_median: tuple
    net_bottom_median: tuple
    net_y_median: float
    homography: object
    n_frames_used: int
    corner_stability: dict
    raw_frames: list = field(default_factory=list)


# def _make_rf_client() -> InferenceHTTPClient:
#     """BUG3 FIX: construct the Roboflow client lazily (not at module import time).

#     Building the client at module scope caused the whole script to crash on
#     import if the API was unreachable or the key was wrong — before setup()
#     even ran. Now each call to run_court_detection() gets a fresh client.
#     """
#     return InferenceHTTPClient(
#         api_url='https://serverless.roboflow.com',
#         api_key=ROBOFLOW_API_KEY)   # BUG2 FIX: uses the string constant above


# def _detect_court_corners(video_path: str, match_path: str):
#     """Sample one frame from the midpoint of the first rally and run
#     Roboflow court detection on it.  One API call per match; the resulting
#     homography is reused for the entire match.
#     """
#     # BUG3 FIX: client created here, not at module scope
#     client = _make_rf_client()

#     # Find the midpoint frame of the first rally in set1.csv
#     set_files = sorted(glob.glob(os.path.join(match_path, 'set*.csv')))
#     if not set_files:
#         raise FileNotFoundError(f'No set*.csv in {match_path}')
#     first_set = pd.read_csv(set_files[0], header=None)
#     first_set.columns = range(len(first_set.columns))
#     first_set = first_set.rename(columns={0: 'rally', 3: 'frame_num'})
#     first_set['frame_num'] = pd.to_numeric(first_set['frame_num'], errors='coerce')
#     first_set = first_set.dropna(subset=['frame_num'])
#     # Use midpoint of first rally (rally==1) only
#     first_rally = first_set[first_set['rally'] == first_set['rally'].iloc[0]]
#     fn = int((first_rally['frame_num'].min() + first_rally['frame_num'].max()) // 2)
#     print(f'    [court] sampling frame {fn} (midpoint of first rally)')

#     cap = cv2.VideoCapture(video_path)
#     cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
#     cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
#     results = []
#     try:
#         ok, frame = cap.read()
#     except Exception as e:
#         print(f'    [WARN] court frame {fn} read failed ({e}), skipping')
#         cap.release()
#         return results
#     if not ok:
#         print(f'    [WARN] court frame {fn} returned ok=False, skipping')
#         cap.release()
#         return results
#     cap.release()

#     tmp = f'/tmp/court_{fn}.jpg'
#     cv2.imwrite(tmp, frame)
#     try:
#         r = client.run_workflow(
#             workspace_name=ROBOFLOW_WORKSPACE,   # BUG1 FIX: was bare identifier
#             workflow_id=ROBOFLOW_WORKFLOW,        # BUG1 FIX: was bare identifier
#             images={'image': tmp}, use_cache=True)
#         results.append({'frame': fn, 'result': r})
#     except Exception as e:
#         print(f'    [WARN] court detection failed for frame {fn}: {e}')
#     return results

def _parse_frame(fn, result):
    if not result or not result[0]:
        return None
    d  = result[0]
    cp = d.get('court_predictions', {})
    np_ = d.get('net_predictions', {})
    if not cp or not np_:
        return None
    iw = cp['image']['width']; ih = cp['image']['height']
    corners = []
    for pred in cp.get('predictions', []):
        kps = [Keypoint(kp['x'], kp['y'], kp['confidence'],
                        kp['class_id'], kp['class'])
               for kp in pred.get('keypoints', [])]
        corners.append(CourtCorner(pred['detection_id'], pred['x'], pred['y'],
                                   pred['confidence'], kps))
    nets = []
    for pred in np_.get('predictions', []):
        kps = pred.get('keypoints', [])
        if len(kps) >= 2:
            t = Keypoint(kps[0]['x'], kps[0]['y'], kps[0]['confidence'],
                         kps[0]['class_id'], kps[0]['class'])
            b = Keypoint(kps[1]['x'], kps[1]['y'], kps[1]['confidence'],
                         kps[1]['class_id'], kps[1]['class'])
            nets.append(NetLine(pred['detection_id'], pred['confidence'], t, b))
    return FrameDetection(fn, iw, ih, corners, nets)

def _centroid(c):
    return (float(np.mean([kp.x for kp in c.keypoints])),
            float(np.mean([kp.y for kp in c.keypoints])))

def _match_corners(frames, n=4):
    ref = next((f for f in frames if len(f.corners) == n),
               max(frames, key=lambda f: len(f.corners)))
    rc  = [_centroid(c) for c in ref.corners]
    trk = {i: [rc[i]] for i in range(len(rc))}
    for fr in frames:
        if fr.frame_num == ref.frame_num:
            continue
        for c in fr.corners:
            cx, cy = _centroid(c)
            trk[int(np.argmin([np.hypot(cx-rx, cy-ry) for rx, ry in rc]))].append((cx, cy))
    return trk

_COURT_W = 610.0; _COURT_H = 1340.0

def _compute_homography(corners_median):
    pts = list(corners_median.values())
    if len(pts) != 4:
        return None
    arr = np.array(pts, dtype=np.float32)
    cy  = np.mean(arr[:, 1])
    top = arr[arr[:, 1] < cy];  bot = arr[arr[:, 1] >= cy]
    src = np.vstack([top[np.argsort(top[:, 0])],
                     bot[np.argsort(bot[:, 0])[::-1]]]).astype(np.float32)
    dst = np.array([[0,0],[_COURT_W,0],[_COURT_W,_COURT_H],[0,_COURT_H]],
                   dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

# def _aggregate(raw_results) -> AggregatedCourt:
#     frames = []
#     for item in raw_results:
#         f = _parse_frame(item['frame'], item['result'])
#         if f is not None:
#             frames.append(f)
#     if not frames:
#         raise ValueError('No valid court frames.')
#     trk = _match_corners(frames)
#     cm, cs = {}, {}
#     for idx, pos in trk.items():
#         xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
#         cm[f'corner_{idx}'] = (float(np.median(xs)), float(np.median(ys)))
#         cs[f'corner_{idx}'] = round(float(np.mean([np.std(xs), np.std(ys)])), 2)

#     def _med(pts):
#         if not pts: return (0., 0.)
#         return (float(np.median([p[0] for p in pts])),
#                 float(np.median([p[1] for p in pts])))

#     nt = _med([(n.top.x,    n.top.y)    for f in frames for n in f.nets])
#     nb = _med([(n.bottom.x, n.bottom.y) for f in frames for n in f.nets])
#     ny = float(np.median([nt[1], nb[1]]))
#     H  = _compute_homography(cm)
#     return AggregatedCourt(cm, nt, nb, ny, H, len(frames), cs, frames)

# def run_court_detection(match) -> AggregatedCourt:
#     raw = _detect_court_corners(match['video'], match['path'])
#     return _aggregate(raw)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — PLAYER SIDE ASSIGNMENT  (perspective-transform ankles)
# ══════════════════════════════════════════════════════════════════════════════

def _project_ankles(pose_frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project both players' ankle midpoints through homography H."""
    out = np.full((2, 2), np.nan, dtype=np.float64)
    for p in range(2):
        la = pose_frame[p, L_ANKLE].astype(np.float64)
        ra = pose_frame[p, R_ANKLE].astype(np.float64)
        if np.all(la == 0) and np.all(ra == 0):
            continue
        elif np.all(la == 0):
            mid = ra
        elif np.all(ra == 0):
            mid = la
        else:
            mid = (la + ra) / 2.0
        pt  = np.array([mid[0], mid[1], 1.0])
        dst = H @ pt
        if abs(dst[2]) > 1e-8:
            out[p] = dst[:2] / dst[2]
    return out


def assign_sides(pose: np.ndarray, frame_ids: list,
                 court: AggregatedCourt) -> np.ndarray:
    """Assign each player to court side 1 (top) or 2 (bottom) per frame."""
    H             = court.homography
    N             = len(frame_ids)
    sides         = np.zeros((N, 2), dtype=np.int32)
    net_y_court   = _COURT_H / 2.0   # 670 cm in court coordinates

    for i in range(N):
        for p in range(2):
            assigned = False
            if H is not None:
                ankles = _project_ankles(pose[i], H)
                if not np.any(np.isnan(ankles[p])):
                    sides[i, p] = 1 if ankles[p, 1] < net_y_court else 2
                    assigned = True
            if not assigned:
                hip_y = float(np.mean(pose[i, p, [L_HIP, R_HIP], 1]))
                sides[i, p] = 1 if hip_y < court.net_y_median else 2

    return sides

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5b — HD-A  (geometry-based swing confidence stream)
# ══════════════════════════════════════════════════════════════════════════════

def _arm_extension_angle(pose_seq: np.ndarray, player: int) -> np.ndarray:
    """Shoulder→elbow→wrist angle for each frame (degrees, NaN on failure)."""
    N   = pose_seq.shape[0]
    ang = np.full(N, np.nan)
    for i in range(N):
        sh = (pose_seq[i, player, SHOULD_R]
              if pose_seq[i, player, SHOULD_R, 0] > 0
              else pose_seq[i, player, SHOULD_L]).astype(np.float64)
        el = (pose_seq[i, player, ELBOW_R]
              if pose_seq[i, player, ELBOW_R, 0] > 0
              else pose_seq[i, player, ELBOW_L]).astype(np.float64)
        wr = (pose_seq[i, player, WRIST_R]
              if pose_seq[i, player, WRIST_R, 0] > 0
              else pose_seq[i, player, WRIST_L]).astype(np.float64)
        v1 = sh - el; v2 = wr - el
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-4 and n2 > 1e-4:
            ang[i] = np.degrees(
                np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
    return ang


def detect_swing_actions(pose: np.ndarray, frame_ids: list,
                         sides: np.ndarray,
                         smooth_sigma: float = 2.0,
                         thr_sigma: float    = 1.5) -> tuple:
    """Produce HD-A confidence stream from pose geometry."""
    N = len(frame_ids)

    accel = np.zeros((N, 2))
    for p in range(2):
        speed = np.maximum(
            np.linalg.norm(
                np.diff(pose[:, p, WRIST_R], axis=0,
                        prepend=pose[:1, p, WRIST_R]), axis=1),
            np.linalg.norm(
                np.diff(pose[:, p, WRIST_L], axis=0,
                        prepend=pose[:1, p, WRIST_L]), axis=1),
        )
        accel[:, p] = np.abs(np.gradient(speed))

    ext = np.zeros((N, 2))
    for p in range(2):
        ang = _arm_extension_angle(pose, p)
        ext[:, p] = np.where(np.isnan(ang), 0.0, ang) / 180.0

    sm_accel = np.stack([gaussian_filter1d(accel[:, p], sigma=smooth_sigma)
                         for p in range(2)], axis=1)
    sm_ext   = np.stack([gaussian_filter1d(ext[:, p],   sigma=smooth_sigma)
                         for p in range(2)], axis=1)

    a_max = sm_accel.max()
    if a_max > 1e-8:
        sm_accel = sm_accel / a_max

    conf_raw = 0.5 * sm_accel + 0.5 * sm_ext   # (N, 2)
    for p in range(2):
        rng = conf_raw[:, p].max() - conf_raw[:, p].min()
        if rng > 1e-8:
            conf_raw[:, p] = (conf_raw[:, p] - conf_raw[:, p].min()) / rng

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
#  STEP 6 — HD-T  (paper-faithful trajectory smoothing)
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

    def _win(arr, c, d, sz):
        pts, j = [], c + d
        while len(pts) < sz and 0 <= j < N:
            if not np.any(np.isnan(arr[j])):
                pts.append([fn[j], arr[j, 0], arr[j, 1]])
            j += d
        return np.array(pts) if pts else np.empty((0, 3))

    # ── Step 2: Curve fitting ─────────────────────────────────────────────────
    s2 = s1.copy()
    for i in range(N):
        if np.any(np.isnan(s1[i])):
            continue
        q  = np.array([fn[i], s1[i, 0], s1[i, 1]])
        fd = _qd(_win(s1, i, -1, cw), q) if len(_win(s1, i, -1, cw)) >= 3 else np.inf
        bd = _qd(_win(s1, i, +1, cw), q) if len(_win(s1, i, +1, cw)) >= 3 else np.inf
        if (np.isfinite(fd) and fd >= co) or (np.isfinite(bd) and bd >= co):
            s2[i] = np.nan

    # ── Step 3a: 15-frame window quadratic interpolation ──────────────────────
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
    s4     = s3.copy()
    half6  = gap6 // 2
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
#  STEP 7 — SRA  (all 5 cases correctly implemented)
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
            # Case 2: HD-T(s) in None segment → false positive, discard
            # Case 5: no HD-T in None segment → true negative, skip
            continue

        if len(ins) == 1:
            # Case 1: exactly one HD-T in player segment → correct detection
            hits.append({'frame': int(ins[0]), 'player': int(pl)})

        elif len(ins) > 1:
            # Case 3: multiple HD-T → closest to peak-confidence frame
            pf = _peak_frame(hda_f, hda_c, s, e)
            best = min(ins, key=lambda f: abs(f - pf)) if pf is not None else ins[0]
            hits.append({'frame': int(best), 'player': int(pl)})

        else:
            # Case 4: no HD-T in player segment → peak-confidence frame
            pf = _peak_frame(hda_f, hda_c, s, e)
            if pf is not None:
                hits.append({'frame': int(pf), 'player': int(pl)})

    return sorted(hits, key=lambda x: x['frame'])


def sra_per_rally(shot_df: pd.DataFrame, hdt_f,
                  hda_f: list, hda_p: list, hda_c: list,
                  total_frames: int) -> list:
    all_hits = []
    # ShuttleSet: group by rally_key = (source_file, rally) since rally
    # numbers reset between sets
    for rk, grp in shot_df.groupby('rally_key'):
        rv = grp['frame_num'].values
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



def sra_per_rally_trimmed(shot_df: pd.DataFrame, hdt_f,
                           hda_f: list, hda_p: list, hda_c: list,
                           total_frames: int) -> list:
    """sra_per_rally variant for pkl/trimmed-index mode.

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
#  STEP 8 — EVALUATION  (temporal IoU metric matching the paper)
# ══════════════════════════════════════════════════════════════════════════════

def _hits_to_intervals(hit_frames: list) -> list:
    """Convert sorted hit frame numbers to [start, end] shot intervals."""
    intervals = []
    sf = sorted(hit_frames)
    for i in range(len(sf) - 1):
        intervals.append((sf[i], sf[i + 1] - 1))
    return intervals


def _t_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def evaluate_tiou(pred_frames: list, gt_frames: list,
                  iou_thresholds: tuple = (0.5, 0.85, 0.95)) -> dict:
    """Precision / recall / F1 at each t-IoU threshold."""
    pred_ivs = _hits_to_intervals(pred_frames)
    gt_ivs   = _hits_to_intervals(gt_frames)
    results  = {}

    for thr in iou_thresholds:
        matched_gt = set()
        tp = 0
        for ps, pe in pred_ivs:
            best_iou, best_j = 0.0, -1
            for j, (gs, ge) in enumerate(gt_ivs):
                if j in matched_gt:
                    continue
                iou = _t_iou(ps, pe, gs, ge)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= thr and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
        pr = tp / len(pred_ivs) if pred_ivs else 0.0
        rc = tp / len(gt_ivs)   if gt_ivs   else 0.0
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        results[thr] = {'precision': pr, 'recall': rc, 'f1': f1}

    return results


def evaluate(pred, gt, tol=10):
    """Frame-tolerance evaluate — kept for quick per-match console logging."""
    tp, matched = 0, set()
    for p in pred:
        for g in gt:
            if abs(p - g) <= tol and g not in matched:
                tp += 1; matched.add(g); break
    pr = tp / len(pred) if pred else 0.
    rc = tp / len(gt)   if len(gt) else 0.
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    return pr, rc, f1

# ══════════════════════════════════════════════════════════════════════════════
#  PER-MATCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  PKL / NP POSE+SHUTTLE LOADERS  (external estimation pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def _pkl_paths(match: dict):
    """Return (pkl_path, npy_path) for a match, or (None, None) if absent."""
    base = PKL_ROOT / match['name']
    pkl  = base / 'pose_out'  / 'poses_trimmed.pkl'
    npy  = base / 'shuttle_out' / 'shuttle_trimmed.npy'
    return (pkl, npy) if pkl.exists() and npy.exists() else (None, None)


def _reconstruct_frame_map(match: dict):
    """Reconstruct trimmed→original and original→trimmed frame maps.

    Replicates the trimming logic from the external pose estimation pipeline
    exactly (buffer=PKL_TRIM_BUF, sort by set_file/rally/frame_num, contiguous
    cap.grab() segments).  No video I/O — pure pandas/set arithmetic.

    Returns:
        sorted_frames   list[int]   sorted_frames[trimmed_idx] = original_frame
        orig_to_trimmed dict[int,int]
    """
    files = sorted(glob.glob(os.path.join(match['path'], 'set*.csv')))
    if not files:
        raise FileNotFoundError(f'No set*.csv in {match["path"]}')

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        df.columns = range(len(df.columns))
        df = df.rename(columns={0: 'rally', 3: 'frame_num'})
        df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
        df = df.dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['set_file']  = os.path.basename(f)
        dfs.append(df)
    gt_df = pd.concat(dfs, ignore_index=True)
    gt_df = gt_df.sort_values(['set_file', 'rally', 'frame_num']).reset_index(drop=True)

    # Probe original video length
    cap    = cv2.VideoCapture(match['video'])
    n_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Build sorted_frames exactly as the trimming pipeline does
    required: set = set()
    for (sf, rv), grp in gt_df.groupby(['set_file', 'rally']):
        first_hit = int(grp['frame_num'].min())
        last_hit  = int(grp['frame_num'].max())
        for f in range(max(0, first_hit - PKL_TRIM_BUF),
                       min(n_orig, last_hit + PKL_TRIM_BUF + 1)):
            required.add(f)

    sorted_frames   = sorted(required)
    orig_to_trimmed = {orig: t for t, orig in enumerate(sorted_frames)}
    return sorted_frames, orig_to_trimmed



def _reconstruct_frame_map_from_csvs(match: dict):
    """Reconstruct trimmed→original frame maps using only the set CSVs.

    Unlike _reconstruct_frame_map(), this version never opens the video.
    n_orig is derived from max(frame_num) + 1 across all set CSVs, which is
    sufficient for clamping the buffer range.

    Also returns the expected trimmed length so callers can validate it
    against the actual pkl / npy lengths before proceeding.

    Returns:
        sorted_frames    list[int]    sorted_frames[trimmed_idx] = original_frame
        orig_to_trimmed  dict[int,int]
        expected_len     int          == len(sorted_frames)
    """
    files = sorted(glob.glob(os.path.join(match['path'], 'set*.csv')))
    if not files:
        raise FileNotFoundError(f'No set*.csv in {match["path"]}')

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        df.columns = range(len(df.columns))
        df = df.rename(columns={0: 'rally', 3: 'frame_num'})
        df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
        df = df.dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['set_file']  = os.path.basename(f)
        dfs.append(df)
    gt_df = pd.concat(dfs, ignore_index=True)
    gt_df = gt_df.sort_values(['set_file', 'rally', 'frame_num']).reset_index(drop=True)

    # Upper bound from CSVs — pad by buffer so the last rally's trailing
    # buffer is never clamped short (the pkl used actual video length which
    # is always >= max(frame_num) + PKL_TRIM_BUF).
    n_orig = int(gt_df['frame_num'].max()) + PKL_TRIM_BUF + 1

    required: set = set()
    for (sf, rv), grp in gt_df.groupby(['set_file', 'rally']):
        first_hit = int(grp['frame_num'].min())
        last_hit  = int(grp['frame_num'].max())
        for f in range(max(0, first_hit - PKL_TRIM_BUF),
                       min(n_orig, last_hit + PKL_TRIM_BUF + 1)):
            required.add(f)

    sorted_frames   = sorted(required)
    orig_to_trimmed = {orig: t for t, orig in enumerate(sorted_frames)}
    return sorted_frames, orig_to_trimmed, len(sorted_frames)


def load_pose_pkl(match: dict):
    """Load poses_trimmed.pkl → list[PoseFrame], dense (one per trimmed frame)."""
    import pickle
    pkl, _ = _pkl_paths(match)
    if pkl is None:
        return None
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_shuttle_npy(match: dict) -> np.ndarray:
    """Load shuttle_trimmed.npy → (N_trimmed, 2) float64 array."""
    _, npy = _pkl_paths(match)
    if npy is None:
        return None
    return np.load(str(npy)).astype(np.float64)


# ── Feature extraction helpers ────────────────────────────────────────────────

def _kps_from_pf(pf, slot: int) -> np.ndarray:
    """Return (17, 2) keypoint array for slot 0=near, 1=far. Zeros if absent."""
    person = pf.near if slot == 0 else pf.far
    if person is None or not hasattr(person, 'keypoints') or person.keypoints is None:
        return np.zeros((17, 2), dtype=np.float64)
    kps = np.array(person.keypoints, dtype=np.float64)
    return kps[:17, :2] if kps.shape[0] >= 17 else np.zeros((17, 2), dtype=np.float64)


def _ankle_mid(person) -> np.ndarray:
    """Return ankle midpoint (2,) or zeros if unavailable."""
    la = np.array(person.left_ankle_px,  dtype=np.float64)          if getattr(person, 'left_ankle_px',  None) is not None else None
    ra = np.array(person.right_ankle_px, dtype=np.float64)          if getattr(person, 'right_ankle_px', None) is not None else None
    if la is not None and ra is not None:
        return (la + ra) / 2.0
    return la if la is not None else (ra if ra is not None else np.zeros(2))


def _bbox_area(person) -> float:
    """Return bounding-box area or 0."""
    bb = getattr(person, 'bbox', None)
    if bb is None or len(bb) < 4:
        return 0.0
    return max(0.0, float((bb[2] - bb[0]) * (bb[3] - bb[1])))


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


def detect_swing_actions_pkl(poses: list,
                              frame_ids: list,
                              sides: np.ndarray,
                              smooth_sigma: float = 2.0,
                              thr_sigma:    float = 1.5) -> tuple:
    """HD-A confidence stream from pkl pose features.

    Four signals per player, each normalised to [0,1]:
      1. Wrist speed accel  — frame-to-frame wrist velocity gradient
      2. Arm extension angle — shoulder→elbow→wrist angle / 180
      3. Ankle displacement  — frame-to-frame ankle midpoint delta (foot plant)
      4. Bbox area rate      — rate of change of player bounding-box area (lunge)

    Weighted sum → smoothed → threshold → hda_f / hda_p / hda_c.
    Output format identical to detect_swing_actions() so SRA is unchanged.
    """
    N = len(frame_ids)

    # Pre-extract arrays  (N, 2, 17, 2), ankle midpoints (N, 2, 2), bbox areas
    kps_arr    = np.zeros((N, 2, 17, 2), dtype=np.float64)
    ankle_arr  = np.zeros((N, 2, 2),     dtype=np.float64)
    area_arr   = np.zeros((N, 2),        dtype=np.float64)
    conf_arr   = np.zeros((N, 2),        dtype=np.float64)

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
        wr_r = kps_arr[:, s, WRIST_R]   # (N, 2)
        wr_l = kps_arr[:, s, WRIST_L]
        speed = np.maximum(
            np.linalg.norm(np.diff(wr_r, axis=0, prepend=wr_r[:1]), axis=1),
            np.linalg.norm(np.diff(wr_l, axis=0, prepend=wr_l[:1]), axis=1),
        )
        accel[:, s] = np.abs(np.gradient(speed))

    # ── Signal 2: arm extension angle ────────────────────────────────────────
    ext = np.zeros((N, 2))
    for s in range(2):
        for i in range(N):
            sh = kps_arr[i, s, SHOULD_R] if kps_arr[i, s, SHOULD_R, 0] > 0                  else kps_arr[i, s, SHOULD_L]
            el = kps_arr[i, s, ELBOW_R]  if kps_arr[i, s, ELBOW_R,  0] > 0                  else kps_arr[i, s, ELBOW_L]
            wr = kps_arr[i, s, WRIST_R]  if kps_arr[i, s, WRIST_R,  0] > 0                  else kps_arr[i, s, WRIST_L]
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

    # ── Smooth all signals ────────────────────────────────────────────────────
    def _norm(arr):
        """Normalise (N,2) array column-wise to [0,1]."""
        out = arr.copy()
        for s in range(2):
            mx = out[:, s].max()
            if mx > 1e-8:
                out[:, s] /= mx
        return out

    sm_accel  = np.stack([gaussian_filter1d(accel[:, s],      smooth_sigma) for s in range(2)], 1)
    sm_ext    = np.stack([gaussian_filter1d(ext[:, s],        smooth_sigma) for s in range(2)], 1)
    sm_ankle  = np.stack([gaussian_filter1d(ankle_disp[:, s], smooth_sigma) for s in range(2)], 1)
    sm_area   = np.stack([gaussian_filter1d(area_rate[:, s],  smooth_sigma) for s in range(2)], 1)

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



def detect_swing_actions_yolo(poses: list,
                               frame_ids: list,
                               sides: np.ndarray,
                               trimmed_video_path: str,
                               clf_path: Path = "swing_classifier.pt",
                               batch_size: int = 64,
                               swing_thr: float = 0.40) -> tuple:
    """HD-A using the trained YOLOv8 swing classifier.

    For each trimmed frame crops the near and far player bboxes (from the
    pkl), runs the YOLOv8-cls model in batches, and uses the raw swing-class
    probability as the per-player confidence stream fed into SRA — replacing
    the wrist-accel heuristic entirely.

    swing_thr: frames whose swing probability exceeds this are emitted as
    HD-A detections.  Unlike the heuristic approach (mean + k*std threshold)
    this is a fixed absolute threshold because the classifier output is
    calibrated probability.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError('ultralytics not installed: pip install ultralytics')

    if not Path(clf_path).exists():
        raise FileNotFoundError(
            f'Swing classifier not found at {clf_path}. '
            'Run: python pipeline.py train')

    model = YOLO(str(clf_path))

    # Determine which class index corresponds to "swing"
    # YOLOv8-cls sorts class names alphabetically:
    #   0 → not_swing,  1 → swing
    names = model.names   # {0: 'not_swing', 1: 'swing'} or vice-versa
    swing_idx = next((k for k, v in names.items() if v == 'swing'), 1)

    N = len(frame_ids)
    # conf_raw[i, s] = swing probability for frame i, slot s
    conf_raw = np.zeros((N, 2), dtype=np.float32)

    cap = cv2.VideoCapture(trimmed_video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open trimmed video: {trimmed_video_path}')
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Collect (frame_idx, slot, crop_image) in batches
    batch_items  = []   # list of (i, slot, crop_bgr)
    frame_cursor = 0

    def _flush_batch():
        if not batch_items:
            return
        crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, _, c in batch_items]
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

    _flush_batch()   # flush any remaining crops
    cap.release()

    # Emit HD-A detections for frames above the swing threshold
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


def discover_pkl_matches() -> list:
    """Scan PKL_ROOT for matches that have both poses_trimmed.pkl and
    shuttle_trimmed.npy, then verify a matching GT directory exists in
    MATCH_DB with at least one set*.csv.  Logs every skip with its reason.

    Returns list of match dicts compatible with the rest of the pipeline.
    """
    if not PKL_ROOT.exists():
        print(f'[discover_pkl] PKL_ROOT not found: {PKL_ROOT}')
        return []

    matches = []
    skipped = []

    for name in sorted(os.listdir(PKL_ROOT)):
        base = PKL_ROOT / name
        if not base.is_dir():
            continue

        pkl = base / 'pose_out'    / 'poses_trimmed.pkl'
        npy = base / 'shuttle_out' / 'shuttle_trimmed.npy'

        if not pkl.exists():
            skipped.append(f'{name[:60]}: missing poses_trimmed.pkl')
            continue
        if not npy.exists():
            skipped.append(f'{name[:60]}: missing shuttle_trimmed.npy')
            continue

        gt_dir = MATCH_DB / name
        if not gt_dir.exists():
            skipped.append(f'{name[:60]}: no matching dir in MATCH_DB')
            continue

        csvs = sorted(glob.glob(str(gt_dir / 'set*.csv')))
        if not csvs:
            skipped.append(f'{name[:60]}: no set*.csv in MATCH_DB dir')
            continue

        # Best-effort original video path — only needed if fallback code runs
        videos = [f for f in os.listdir(gt_dir) if f.endswith('.mp4')]
        video_path = str(gt_dir / videos[0]) if videos else ''

        matches.append({
            'name':     name,
            'path':     str(gt_dir),
            'video':    video_path,
            'n_csvs':   len(csvs),
            'has_pkl':  True,
            'shuttle_path': str(npy),
            'pose_path':   str(pkl)
        })

    print(f'[discover_pkl] {len(matches)} match(es) ready, '
          f'{len(skipped)} skipped.')
    for s in skipped:
        print(f'  [skip] {s}')

    return matches


def run_match(match, inferencer_ref):
    name = match['name']

    # ── Load pkl/npy outputs from external estimation pipeline ──────────
    poses      = load_pose_pkl(match)
    shuttle_2d = load_shuttle_npy(match)
    if poses is None or shuttle_2d is None:
        print(f'  [skip] no pkl/npy for {name[:55]}')
        return None, None

    if True:

        # Reconstruct frame_map from CSVs only (no video probe)
        sorted_frames, orig_to_trimmed, expected_len = \
            _reconstruct_frame_map_from_csvs(match)

        # Validate lengths before doing any work
        n_poses   = len(poses)
        n_shuttle = len(shuttle_2d)
        if n_poses != expected_len or n_shuttle != expected_len:
            print(f'  [skip] {name[:50]}: length mismatch — '
                  f'frame_map={expected_len} poses={n_poses} '
                  f'shuttle={n_shuttle}')
            return None, None

        frame_ids  = list(range(len(poses)))   # dense: 0..N_trimmed-1
        sides      = assign_sides_pkl(poses)

        # Use trained YOLO swing classifier if available, else fall back
        # to the heuristic pkl feature extractor.
        trimmed_video = str(PKL_ROOT / name / 'trimmed.mp4')
        if Path("swing_classifier.pt").exists() and Path(trimmed_video).exists():
            print(f'  [hda] using YOLO swing classifier')
            hda_f, hda_p, hda_c = detect_swing_actions_yolo(
                poses, frame_ids, sides, trimmed_video)
        else:
            print(f'  [hda] using heuristic pkl features '
                  f'(train classifier with: python pipeline.py train)')
            hda_f, hda_p, hda_c = detect_swing_actions_pkl(
                poses, frame_ids, sides)

        shuttle_smooth = trajectory_smoothing(shuttle_2d)
        hdt_f, _       = compute_hd_t(shuttle_smooth, frame_ids)

        # Convert GT frame numbers (original) → trimmed indices
        shot_frames_orig, shot_df = load_shot_frames(match['path'])
        shot_df = shot_df.copy()
        shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
        shot_df = shot_df.dropna(subset=['frame_num_t'])
        shot_df['frame_num_t'] = shot_df['frame_num_t'].astype(int)

        hits = sra_per_rally_trimmed(
            shot_df, hdt_f, hda_f, hda_p, hda_c, len(poses))
        gt_frames = shot_df['frame_num_t'].values

    # Visualization calls removed for pkl-only pipeline.
    # dump_pose_video(match) and dump_shuttle_video(match) are
    # still defined below and can be called manually if needed.

    return hits, gt_frames


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# COCO-17 skeleton pairs
_VIZ_COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
_VIZ_PLAYER_COLORS  = [(0, 200, 255), (255, 100, 0)]   # amber, blue (BGR)
_VIZ_PLAYER_LABELS  = ['P0', 'P1']
_VIZ_TRAIL_LEN      = 15
_VIZ_SHUTTLE_COLOR  = (0, 255, 255)


def _viz_draw_pose_frame(frame, keypoints, scores, color, label,
                          score_thr=0.3):
    """Draw full COCO skeleton + keypoints + label for one player (in-place)."""
    H, W = frame.shape[:2]
    kpts = keypoints  # (17, 2)

    for p1, p2 in _VIZ_COCO_PAIRS:
        if scores[p1] < score_thr or scores[p2] < score_thr:
            continue
        x1, y1 = int(kpts[p1, 0]), int(kpts[p1, 1])
        x2, y2 = int(kpts[p2, 0]), int(kpts[p2, 1])
        if x1 <= 0 or y1 <= 0 or x2 <= 0 or y2 <= 0:
            continue
        if not (0 <= x1 < W and 0 <= y1 < H and
                0 <= x2 < W and 0 <= y2 < H):
            continue
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    valid_kpts = []
    for j in range(17):
        if scores[j] < score_thr:
            continue
        x, y = int(kpts[j, 0]), int(kpts[j, 1])
        if x <= 0 or y <= 0 or not (0 <= x < W and 0 <= y < H):
            continue
        cv2.circle(frame, (x, y), 4, color,          -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255),  1, cv2.LINE_AA)
        valid_kpts.append((x, y))

    if valid_kpts:
        min_y  = min(k[1] for k in valid_kpts)
        avg_x  = int(sum(k[0] for k in valid_kpts) / len(valid_kpts))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx = max(0, avg_x - tw // 2)
        by = max(th + 8, min_y - 8)
        cv2.rectangle(frame,
                      (bx - 4, by - th - 6), (bx + tw + 4, by + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _viz_draw_shuttle_trail(frame, shuttle_2d, orig_frame,
                             trail_len=_VIZ_TRAIL_LEN):
    """Draw a fading shuttlecock trail ending at orig_frame (in-place)."""
    for past in range(max(0, orig_frame - trail_len + 1), orig_frame + 1):
        if past >= len(shuttle_2d):
            continue
        px, py = shuttle_2d[past]
        if np.isnan(px) or np.isnan(py):
            continue
        age       = orig_frame - past
        intensity = int(255 * (1.0 - age / trail_len))
        radius    = max(2, 6 - age // 3)
        cv2.circle(frame, (int(px), int(py)), radius,
                   (0, intensity, 255), -1, cv2.LINE_AA)


def _viz_build_lookup(frame_ids, frame_map):
    """Build dict: trimmed_frame_index -> pose array row index."""
    if frame_map is not None:
        orig_to_trimmed = {int(orig): t for t, orig in enumerate(frame_map)}
    else:
        orig_to_trimmed = {int(fid): int(fid) for fid in frame_ids}
    result = {}
    for pose_idx, orig_fid in enumerate(frame_ids):
        t = orig_to_trimmed.get(int(orig_fid))
        if t is not None:
            result[t] = pose_idx
    return result


def dump_pose_video(match: dict, score_thr: float = 0.3) -> None:
    """Render pose skeleton overlay on the trimmed video for one match."""
    name        = match['name']
    pose_path   = POSE_DIR    / (name + '_pose.npz')
    trimmed_path = SHUTTLE_DIR / name / 'trimmed.mp4'
    map_path    = SHUTTLE_DIR / name / 'frame_map.npy'
    out_path    = VIZ_DIR / (name + '_pose.mp4')

    if not pose_path.exists():
        print(f'  [viz-pose] skip — no pose cache for {name[:55]}')
        return
    if not trimmed_path.exists():
        print(f'  [viz-pose] skip — no trimmed video for {name[:55]}')
        return
    if out_path.exists():
        print(f'  [viz-pose] skip (cached): {name[:55]}')
        return

    data      = np.load(pose_path)
    keypoints = data['keypoints']
    scores    = data['scores']
    frame_ids = data['frame_ids']
    frame_map = np.load(str(map_path)).astype(np.int32) if map_path.exists() else None
    t2p       = _viz_build_lookup(frame_ids, frame_map)

    cap   = cv2.VideoCapture(str(trimmed_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    print(f'  [viz-pose] rendering {name[:55]} ...')
    for t in tqdm(range(total), desc='    pose', leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        pose_idx = t2p.get(t)
        if pose_idx is not None:
            kps = keypoints[pose_idx]
            scs = scores[pose_idx]
            for player in range(2):
                if np.all(kps[player] == 0):
                    continue
                _viz_draw_pose_frame(frame, kps[player], scs[player],
                                     _VIZ_PLAYER_COLORS[player],
                                     _VIZ_PLAYER_LABELS[player],
                                     score_thr)
        cv2.putText(frame, f'f:{t}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'f:{t}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f'  [viz-pose] saved -> {out_path}')


def dump_shuttle_video(match: dict) -> None:
    """Render shuttlecock trail overlay on the trimmed video for one match."""
    name         = match['name']
    shuttle_npy  = SHUTTLE_DIR / (name + '_shuttle.npy')
    trimmed_path = SHUTTLE_DIR / name / 'trimmed.mp4'
    map_path     = SHUTTLE_DIR / name / 'frame_map.npy'
    out_path     = VIZ_DIR / (name + '_shuttle.mp4')

    if not shuttle_npy.exists():
        print(f'  [viz-shuttle] skip — no shuttle cache for {name[:55]}')
        return
    if not trimmed_path.exists():
        print(f'  [viz-shuttle] skip — no trimmed video for {name[:55]}')
        return
    if not map_path.exists():
        print(f'  [viz-shuttle] skip — no frame_map for {name[:55]}')
        return
    if out_path.exists():
        print(f'  [viz-shuttle] skip (cached): {name[:55]}')
        return

    shuttle_2d = np.load(str(shuttle_npy))
    frame_map  = np.load(str(map_path)).astype(np.int32)

    cap   = cv2.VideoCapture(str(trimmed_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    print(f'  [viz-shuttle] rendering {name[:55]} ...')
    for t in tqdm(range(total), desc='    shuttle', leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if t < len(frame_map):
            orig_f = int(frame_map[t])
            _viz_draw_shuttle_trail(frame, shuttle_2d, orig_f)
            if orig_f < len(shuttle_2d):
                px, py = shuttle_2d[orig_f]
                if not (np.isnan(px) or np.isnan(py)):
                    cv2.circle(frame, (int(px), int(py)), 8,
                               _VIZ_SHUTTLE_COLOR, -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(px), int(py)), 9,
                               (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'f:{t}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'f:{t}', (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f'  [viz-shuttle] saved -> {out_path}')

# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_df(results: dict) -> pd.DataFrame:
    """Build metrics DataFrame using t-IoU at 0.5 / 0.85 / 0.95."""
    rows = []
    iou_thresholds = (0.5, 0.85, 0.95)
    for name, res in results.items():
        pred = [h['frame'] for h in res['hits']]
        gt   = list(res['gt'])
        tiou = evaluate_tiou(pred, gt, iou_thresholds)
        for thr in iou_thresholds:
            m = tiou[thr]
            rows.append({
                'match':     name,
                'iou_thr':   thr,
                'precision': m['precision'],
                'recall':    m['recall'],
                'f1':        m['f1'],
                'n_pred':    len(_hits_to_intervals(pred)),
                'n_gt':      len(_hits_to_intervals(gt)),
            })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str):
    print(f'\n{label} metrics (mean across matches, t-IoU thresholds):')
    print(df.groupby('iou_thr')[['precision', 'recall', 'f1']]
            .mean().round(3).to_string())


def plot_metrics(df: pd.DataFrame, path: Path):
    thresholds = [0.5, 0.85, 0.95]
    summary    = df.groupby('iou_thr')[['precision', 'recall', 'f1']].mean()
    fig, axes  = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    colors     = ['#4C72B0', '#DD8452', '#55A868']
    for ax, metric in zip(axes, ['precision', 'recall', 'f1']):
        vals = [summary.loc[t, metric] for t in thresholds]
        bars = ax.bar([f'IoU>={t}' for t in thresholds], vals,
                      color=colors, width=0.5)
        ax.set_ylim(0, 1); ax.set_title(metric.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    fig.suptitle('Test set — SRA performance by t-IoU threshold', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f'Plot saved -> {path}')

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover matches with pkl + npy + GT CSVs ──────────────────────────
    print('='*60)
    print('STEP 1 — Discover pkl matches')
    print('='*60)
    all_matches = discover_pkl_matches()
    if not all_matches:
        print('[ERROR] No valid matches found. Exiting.')
        return
    print(f'\n{len(all_matches)} match(es) will be evaluated:')
    for m in all_matches:
        print(f'  {m["name"][:70]}')

    # ── 2. Run HD-A / HD-T / SRA on every match ───────────────────────────────
    print('\n' + '='*60)
    print('STEP 2 — HD-A / HD-T / SRA  (pkl features)')
    print('='*60)
    results = {}
    for m in tqdm(all_matches, desc='Matches'):
        print(f'\n  {m["name"][:65]}')
        try:
            hits, gt = run_match(m, None)
        except Exception as exc:
            print(f'  [ERROR] {m["name"][:55]}: {type(exc).__name__}: {exc}')
            continue
        if hits is None:
            continue
        results[m['name']] = {'hits': hits, 'gt': gt}
        p, r, f = evaluate([h['frame'] for h in hits], list(gt), tol=5)
        print(f'  tol=±5  P:{p:.3f} R:{r:.3f} F1:{f:.3f}  '
              f'pred:{len(hits)} gt:{len(gt)}')

    if not results:
        print('[WARNING] No results produced. Check pkl/npy/CSV alignment.')
        return

    # ── 3. Metrics + reporting ────────────────────────────────────────────────
    print('\n' + '='*60)
    print('STEP 3 — Evaluation & reporting')
    print('='*60)
    df = build_metrics_df(results)
    df.to_csv(RESULTS_DIR / 'metrics.csv', index=False)
    print_summary(df, 'All matches')
    plot_metrics(df, RESULTS_DIR / 'metrics.png')

    print('\nPer-match breakdown (IoU ≥ 0.5):')
    piv = (df[df.iou_thr == 0.5]
           [['match', 'precision', 'recall', 'f1', 'n_pred', 'n_gt']]
           .sort_values('f1', ascending=False)
           .reset_index(drop=True))
    piv['match'] = piv['match'].str[:55]
    print(piv.to_string(index=False))

    with open(RESULTS_DIR / 'predictions.json', 'w') as f:
        json.dump(
            {n: [{'frame': int(h['frame']), 'player': int(h['player']),
                  'rally': str(h.get('rally', ''))} for h in r['hits']]
             for n, r in results.items()}, f, indent=2)
    print(f'\nPredictions → {RESULTS_DIR}/predictions.json')
    print(f'Metrics     → {RESULTS_DIR}/metrics.csv')
    print(f'Plot        → {RESULTS_DIR}/metrics.png')
    print('\nPipeline complete.')


def main_2():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_matches = discover_pkl_matches()
    if not all_matches: return

    results = {}
    print('\n' + '='*60)
    print('PRECISION-LOCKED SRA: Dual-Stage Peak Suppression')
    print('='*60)

    for m in tqdm(all_matches, desc='Matches'):
        try:
            poses = load_pose_pkl(m)
            shuttle_2d = load_shuttle_npy(m)
            if poses is None or shuttle_2d is None: continue

            # 1. Trajectory (HD-T) - Keep min_dist at 15 to prevent micro-peaks
            shuttle_smooth = trajectory_smoothing(shuttle_2d)
            hdt_f, _ = compute_hd_t(shuttle_smooth, list(range(len(poses))), min_dist=15)

            # 2. YOLO (HD-A) - Stage 1: Aggressive Local Maxima
            trimmed_video = str(PKL_ROOT / m['name'] / 'trimmed.mp4')
            raw_f, raw_p, raw_c = detect_swing_actions_yolo(poses, list(range(len(poses))), assign_sides_pkl(poses), trimmed_video)
            
            hda_f, hda_p, hda_c = [], [], []
            data = sorted(zip(raw_f, raw_p, raw_c), key=lambda x: x[0])
            i = 0
            while i < len(data):
                # Increase window to 25 frames (nearly 0.5s) to swallow follow-through noise
                cluster = [data[i]]
                j = i + 1
                while j < len(data) and (data[j][0] - data[i][0]) < 20: # 25 earlier
                    cluster.append(data[j])
                    j += 1
                best = max(cluster, key=lambda x: x[2])
                # Only keep YOLO hits with confidence > 0.5 to prune weak detections early
                if best[2] > 0.5:
                    hda_f.append(best[0]); hda_p.append(best[1]); hda_c.append(best[2])
                i = j

            # 3. Load GT
            sorted_frames, orig_to_trimmed, _ = _reconstruct_frame_map_from_csvs(m)
            _, shot_df = load_shot_frames(m['path'])
            shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
            gt_frames = shot_df['frame_num_t'].dropna().astype(int).values

            # 4. Stage 2: Confirmed Intersection Only
            hits = []
            used_hdt = set()
            for f_y, p_y, c_y in zip(hda_f, hda_p, hda_c):
                # Window: Impact usually happens slightly before the YOLO peak
                possible_hdt = [f for f in hdt_f if (f_y - 15) <= f <= (f_y + 5) and f not in used_hdt]
                
                if possible_hdt:
                    best_hdt = min(possible_hdt, key=lambda f: abs(f - (f_y - 3)))
                    hits.append({'frame': int(best_hdt), 'player': int(p_y), 'conf': c_y})
                    used_hdt.add(best_hdt)
                elif c_y > 0.90: #
                    hits.append({'frame': int(f_y - 2), 'player': int(p_y), 'conf': c_y})

            # 5. Final Temporal De-Duplication
            hits = sorted(hits, key=lambda x: x['conf'], reverse=True)
            final_hits = []
            for h in hits:
                # No two hits can be within 15 frames (0.25s)
                if not any(abs(h['frame'] - fh['frame']) < 15 for fh in final_hits):
                    final_hits.append(h)

            final_hits = sorted(final_hits, key=lambda x: x['frame'])
            results[m['name']] = {'hits': final_hits, 'gt': gt_frames}
            p, r, f = evaluate([h['frame'] for h in final_hits], list(gt_frames), tol=10)
            print(f'  {m["name"][:35]}... P:{p:.3f} R:{r:.3f} F1:{f:.3f} | Pred:{len(final_hits)} GT:{len(gt_frames)}')

        except Exception as exc:
            logging.error(f"Error: {exc}")
            continue


        # 6. Multi-Tolerance Evaluation
        pred_frames = [h['frame'] for h in final_hits]
        gt_list = list(gt_frames)
            
            # Calculate for all three levels
        p5, r5, f5 = evaluate(pred_frames, gt_list, tol=5)
        p10, r10, f10 = evaluate(pred_frames, gt_list, tol=10)
        p15, r15, f15 = evaluate(pred_frames, gt_list, tol=15)

        print(f"\nMatch: {m['name'][:40]}...")
        print(f"  [TOL ±5]  P: {p5:.3f} R: {r5:.3f} F1: {f5:.3f}")
        print(f"  [TOL ±10] P: {p10:.3f} R: {r10:.3f} F1: {p10:.3f}") # Often the "Sweet Spot"
        print(f"  [TOL ±15] P: {p15:.3f} R: {r15:.3f} F1: {f15:.3f}")
        print(f"  Stats: Pred: {len(final_hits)} | GT: {len(gt_list)}")
            
        # Save the one you care about most to results
        results[m['name']] = {'hits': final_hits, 'gt': gt_frames}

def apply_peak_nms(f_list, p_list, c_list, dist=18):
    """Refined NMS that specifically looks for local confidence maxima."""
    if not f_list: return [], [], []
    data = sorted(zip(f_list, p_list, c_list), key=lambda x: x[0])
    
    keep_f, keep_p, keep_c = [], [], []
    i = 0
    while i < len(data):
        cluster = [data[i]]
        j = i + 1
        while j < len(data) and (data[j][0] - data[i][0]) < dist:
            cluster.append(data[j])
            j += 1
        
        # Pick the absolute peak of the cluster
        best = max(cluster, key=lambda x: x[2])
        keep_f.append(best[0]); keep_p.append(best[1]); keep_c.append(best[2])
        i = j
    return keep_f, keep_p, keep_c

def apply_temporal_nms(f_list, p_list, c_list, dist=25):
    if not f_list: return [], [], []
    combined = sorted(zip(f_list, p_list, c_list), key=lambda x: x[0])
    keep_f, keep_p, keep_c = [], [], []
    
    curr_f, curr_p, curr_c = combined[0]
    for i in range(1, len(combined)):
        f, p, c = combined[i]
        if f - curr_f < dist:
            if c > curr_c: curr_f, curr_p, curr_c = f, p, c
        else:
            keep_f.append(curr_f); keep_p.append(curr_p); keep_c.append(curr_c)
            curr_f, curr_p, curr_c = f, p, c
    keep_f.append(curr_f); keep_p.append(curr_p); keep_c.append(curr_c)
    return keep_f, keep_p, keep_c

def main_improved():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_matches = discover_pkl_matches()
    if not all_matches: return

    results = {}
    print('\n' + '='*60)
    print('STEP 2 — STRICT HD-T/YOLO INTERSECTION (Precision Mode)')
    print('='*60)

    for m in tqdm(all_matches, desc='Matches'):
        try:
            # 1. Load Data
            poses = load_pose_pkl(m)
            shuttle_2d = load_shuttle_npy(m)
            if poses is None or shuttle_2d is None: continue

            # 2. Setup mapping
            sorted_frames, orig_to_trimmed, expected_len = _reconstruct_frame_map_from_csvs(m)
            frame_ids = list(range(len(poses)))
            sides = assign_sides_pkl(poses)

            # 3. IMPROVED HD-T (The Filter)
            # We use a higher min_dist (25 frames ~ 0.4s) to prevent double-hits
            shuttle_smooth = trajectory_smoothing(shuttle_2d)
            hdt_f, _ = compute_hd_t(shuttle_smooth, frame_ids, min_dist=25)

            # 4. YOLO HD-A (The Candidate Source)
            trimmed_video = str(PKL_ROOT / m['name'] / 'trimmed.mp4')
            hda_f, hda_p, hda_c = detect_swing_actions_yolo(poses, frame_ids, sides, trimmed_video)

            # 5. Load/Map Ground Truth
            _, shot_df = load_shot_frames(m['path'])
            shot_df = shot_df.copy()
            shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
            shot_df = shot_df.dropna(subset=['frame_num_t']).copy()
            gt_frames = shot_df['frame_num_t'].astype(int).values

            # 6. CUSTOM STRICT SRA (Algorithm 2 - Modified)
            # We only keep hits where HD-T occurs WITHIN a YOLO swing segment.
            segs = _build_segs(hda_f, hda_p) # Group YOLO detections into swing windows
            hits = []
            for seg in segs:
                s, e, pl = seg['start'], seg['end'], seg['player']
                if pl is None: continue
                
                # Check for Trajectory hits inside this YOLO window
                ins = [f for f in hdt_f if s <= f <= e]
                if ins:
                    # If multiple HD-T, pick the one closest to the YOLO peak
                    pf = _peak_frame(hda_f, hda_c, s, e)
                    best = min(ins, key=lambda f: abs(f - pf)) if pf is not None else ins[0]
                    hits.append({'frame': int(best), 'player': int(pl)})
                
                # CRITICAL CHANGE: We REMOVE the "else: hits.append(pf)" Case 4.
                # This stops the 6000+ false positives from YOLO.

            results[m['name']] = {'hits': hits, 'gt': gt_frames}
            
            # 7. Evaluate
            p, r, f = evaluate([h['frame'] for h in hits], list(gt_frames), tol=5)
            print(f'  {m["name"][:35]}... P:{p:.3f} R:{r:.3f} F1:{f:.3f} | Pred:{len(hits)} GT:{len(gt_frames)}')

        except Exception as exc:
            logging.error(f"Error on {m['name']}: {exc}")
            continue

    # 3. Final Export
    df = build_metrics_df(results)
    df.to_csv(RESULTS_DIR / 'strict_metrics.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
#  SWING CLASSIFIER TRAINING  (Steps 1-3)
# ══════════════════════════════════════════════════════════════════════════════

def _trim_match_video(match: dict) -> Path:
    """Trim one match video to rally frames (buffer=PKL_TRIM_BUF).

    Uses the fast cap.grab() skip approach — reads only frames that belong
    to the required set, skips all others without decoding.

    Returns path to the trimmed video (already existing or newly created).
    """
    out_path = PKL_ROOT / match['name'] / 'trimmed.mp4'
    if out_path.exists():
        print(f'  [trim] cached: {match["name"][:60]}')
        return out_path

    # Derive required frames from CSVs only
    sorted_frames, _, _ = _reconstruct_frame_map_from_csvs(match)
    if not sorted_frames:
        raise RuntimeError(f'No frames derived from CSVs for {match["name"]}')

    fast_lookup = set(sorted_frames)
    max_frame   = sorted_frames[-1]

    cap = cv2.VideoCapture(match['video'])
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {match["video"]}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    written = 0
    for curr in tqdm(range(max_frame + 1),
                     desc=f'  trim {match["name"][:45]}', leave=False):
        if curr in fast_lookup:
            ret, frame = cap.read()
            if ret:
                writer.write(frame)
                written += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr + 1)
        else:
            ret = cap.grab()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr + 1)

    cap.release()
    writer.release()
    print(f'  [trim] {written}/{len(sorted_frames)} frames written '
          f'→ {out_path}')
    return out_path

def _build_swing_dataset(matches: list) -> dict:
    import pickle as _pickle
    import gc

    pos_dir = DATASET_DIR / 'swing'
    neg_dir = DATASET_DIR / 'not_swing'
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for m in matches:
        name = m['name']
        print(f'\n  [dataset] {name[:65]}')

        # ── load pkl ──────────────────────────────────────────────────────────
        pkl_path = PKL_ROOT / name / 'pose_out' / 'poses_trimmed.pkl'
        if not pkl_path.exists():
            print(f'    [skip] poses_trimmed.pkl not found')
            continue
        with open(pkl_path, 'rb') as f:
            poses = _pickle.load(f)

        # ── load trimmed video ────────────────────────────────────────────────
        trimmed_path = PKL_ROOT / name / 'trimmed.mp4'
        if not trimmed_path.exists():
            print(f'    [skip] trimmed.mp4 not found — run trim step first')
            continue

        # ── reconstruct frame map ─────────────────────────────────────────────
        sorted_frames, orig_to_trimmed, expected_len = \
            _reconstruct_frame_map_from_csvs(m)

        if len(poses) != expected_len:
            print(f'    [skip] length mismatch: poses={len(poses)} '
                  f'expected={expected_len}')
            continue

        # ── build swing set (trimmed indices) ─────────────────────────────────
        _, shot_df = load_shot_frames(m['path'])
        shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
        shot_df = shot_df.dropna(subset=['frame_num_t'])
        shot_df['frame_num_t'] = shot_df['frame_num_t'].astype(int)

        swing_set: set = set()
        for t in shot_df['frame_num_t']:
            for delta in range(-SWING_WINDOW, SWING_WINDOW + 1):
                idx = int(t) + delta
                if 0 <= idx < len(poses):
                    swing_set.add(idx)

        # ── probe video dimensions ────────────────────────────────────────────
        cap = cv2.VideoCapture(str(trimmed_path))
        if not cap.isOpened():
            print(f'    [skip] cannot open trimmed video')
            continue
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # ── PASS 1: write positives immediately; collect negative metadata ─────
        # Storing only (tidx, slot) integers for negatives — no images in RAM.
        pos_count    = 0
        neg_pool_meta = []   # list of (tidx, slot)

        for tidx in tqdm(range(len(poses)),
                         desc=f'    pass1', leave=False):
            ret, frame = cap.read()
            if not ret:
                print(f'    [warn] video ended early at trimmed frame {tidx}')
                break

            pf = poses[tidx]
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

                if tidx in swing_set:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    slot_name = 'near' if slot == 0 else 'far'
                    fname = f'{name}_t{tidx:06d}_{slot_name}.jpg'
                    cv2.imwrite(str(pos_dir / fname), crop)
                    pos_count += 1
                else:
                    # Store only metadata — bbox coords so pass 2 can re-crop
                    neg_pool_meta.append((tidx, slot, x1, y1, x2, y2))

        cap.release()

        # ── sample negative metadata before pass 2 ────────────────────────────
        n_neg_target  = min(len(neg_pool_meta), NEG_RATIO * pos_count)
        neg_sampled_meta = (random.sample(neg_pool_meta, n_neg_target)
                            if n_neg_target > 0 else [])

        # Build a lookup: tidx → list of (slot, x1, y1, x2, y2)
        # so pass 2 can skip frames not in the sample entirely.
        neg_by_tidx: dict = {}
        for tidx, slot, x1, y1, x2, y2 in neg_sampled_meta:
            neg_by_tidx.setdefault(tidx, []).append((slot, x1, y1, x2, y2))

        neg_frames_needed = set(neg_by_tidx.keys())
        neg_count = 0

        # ── PASS 2: re-open video; decode only sampled negative frames ─────────
        if neg_frames_needed:
            cap2 = cv2.VideoCapture(str(trimmed_path))
            if not cap2.isOpened():
                print(f'    [warn] cannot re-open trimmed video for pass 2')
            else:
                for tidx in tqdm(range(len(poses)),
                                 desc=f'    pass2', leave=False):
                    if tidx in neg_frames_needed:
                        ret, frame = cap2.read()
                        if not ret:
                            break
                        for slot, x1, y1, x2, y2 in neg_by_tidx[tidx]:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            slot_name = 'near' if slot == 0 else 'far'
                            fname = f'{name}_t{tidx:06d}_{slot_name}.jpg'
                            cv2.imwrite(str(neg_dir / fname), crop)
                            neg_count += 1
                    else:
                        # Advance without decoding
                        ret = cap2.grab()
                        if not ret:
                            break

                cap2.release()

        summary[name] = {
            'n_pos':         pos_count,
            'n_neg_sampled': neg_count,
            'n_neg_pool':    len(neg_pool_meta),
        }
        print(f'    pos={pos_count}  '
              f'neg_sampled={neg_count}  '
              f'neg_pool={len(neg_pool_meta)}')

        # ── free memory before next match ─────────────────────────────────────
        del poses, neg_pool_meta, neg_sampled_meta, neg_by_tidx
        gc.collect()

    total_pos = sum(v['n_pos']         for v in summary.values())
    total_neg = sum(v['n_neg_sampled'] for v in summary.values())
    print(f'\n[dataset] Total: {total_pos} positives, {total_neg} negatives')
    print(f'[dataset] Saved to {DATASET_DIR}')
    return summary

def _train_yolo_classifier() -> Path:
    """Fine-tune YOLOv8-cls on the built dataset and return best.pt path."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            'ultralytics not installed. Run: pip install ultralytics')

    total_pos = len(list((DATASET_DIR / 'swing').glob('*.jpg')))
    total_neg = len(list((DATASET_DIR / 'not_swing').glob('*.jpg')))
    if total_pos == 0:
        raise RuntimeError(
            f'No positive crops found in {DATASET_DIR / "swing"}. '
            'Run the dataset build step first.')

    print(f'\n[train] {total_pos} swing + {total_neg} not_swing crops')
    print('[train] Starting YOLOv8-cls training...')

    model = YOLO('yolov8s-cls.pt')   # downloads pretrained weights if absent
    results = model.train(
        data        = str(DATASET_DIR),
        epochs      = 30,
        imgsz       = 224,
        batch       = 64,
        project     = 'runs/swing_classifier',
        name        = 'train',
        exist_ok    = True,
        verbose     = True,
    )

    best = Path('runs/swing_classifier/train/weights/best.pt')
    if not best.exists():
        raise RuntimeError(f'Training finished but best.pt not found at {best}')

    shutil.copy(str(best), str(SWING_CLF_PT))
    print(f'\n[train] Best weights saved → {SWING_CLF_PT}')
    return SWING_CLF_PT


def _smoke_test_classifier(clf_path: Path, n: int = 10) -> None:
    """Run the trained classifier on n random crops from each class."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[smoke] ultralytics not installed — skipping smoke test')
        return

    model = YOLO(str(clf_path))
    print(f'\n[smoke] Testing {clf_path} ...')

    for label, folder in [('swing', DATASET_DIR / 'swing'),
                           ('not_swing', DATASET_DIR / 'not_swing')]:
        imgs = list(folder.glob('*.jpg'))
        if not imgs:
            print(f'  [{label}] no images found')
            continue
        sample = random.sample(imgs, min(n, len(imgs)))
        results = model.predict(source=[str(p) for p in sample],
                                verbose=False)
        scores = []
        for r in results:
            # index 0 = 'not_swing', index 1 = 'swing' (alphabetical)
            probs = r.probs.data.cpu().numpy()
            swing_conf = float(probs[1]) if len(probs) > 1 else float(probs[0])
            scores.append(swing_conf)
        mean_conf = sum(scores) / len(scores)
        print(f'  [{label}] mean swing confidence over {len(scores)} crops: '
              f'{mean_conf:.3f}')


def train_swing_classifier() -> None:
    """Entry point: trim → dataset → train → smoke test."""
    import random as _random
    _random.seed(42)

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s')

    # ── Discover matches ──────────────────────────────────────────────────────
    print('=' * 60)
    print('SWING CLASSIFIER — Step 1: Discover matches')
    print('=' * 60)
    matches = discover_pkl_matches()
    if not matches:
        print('[ERROR] No valid matches found.')
        return

    # ── Step 1: Trim videos ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 1: Trim videos')
    print('=' * 60)
    for m in matches:
        if not m['video']:
            print(f'  [skip] no video path for {m["name"][:60]}')
            continue
        try:
            _trim_match_video(m)
        except Exception as exc:
            print(f'  [ERROR] {m["name"][:55]}: {exc}')

    # ── Step 2: Build dataset ─────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 2: Build dataset')
    print('=' * 60)
    _build_swing_dataset(matches)

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 3: Train YOLOv8-cls')
    print('=' * 60)
    clf_path = _train_yolo_classifier()

    # ── Step 4: Smoke test ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 4: Smoke test')
    print('=' * 60)
    _smoke_test_classifier(clf_path)

    print('\nSwing classifier training complete.')
    print(f'Weights → {SWING_CLF_PT}')

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == 'train':
    #     train_swing_classifier()
    # else:
    main_2()