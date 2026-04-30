"""
data_loader.py — Data loading module for HD-A pipelines.
=========================================================
Loads pre-computed pose and shuttle data from disk:
  - shots.pkl      → ground-truth shot annotations
  - shuttle.npy    → shuttlecock trajectory array  (N_trimmed, 2)

Also provides match discovery, GT CSV parsing, and frame-map reconstruction.
"""

import os
import glob
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import cv2

from config import MATCH_DB, PKL_ROOT, PKL_TRIM_BUF


# ══════════════════════════════════════════════════════════════════════════════
#  MATCH DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

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

        videos = [f for f in os.listdir(gt_dir) if f.endswith('.mp4')]
        video_path = str(gt_dir / videos[0]) if videos else ''

        matches.append({
            'name':         name,
            'path':         str(gt_dir),
            'video':        video_path,
            'n_csvs':       len(csvs),
            'has_pkl':      True,
            'shuttle_path': str(npy),
            'pose_path':    str(pkl),
        })

    print(f'[discover_pkl] {len(matches)} match(es) ready, '
          f'{len(skipped)} skipped.')
    for s in skipped:
        print(f'  [skip] {s}')

    return matches


# ══════════════════════════════════════════════════════════════════════════════
#  CORE DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _pkl_paths(match: dict):
    """Return (pkl_path, npy_path) for a match, or (None, None) if absent."""
    base = PKL_ROOT / match['name']
    pkl  = base / 'pose_out'    / 'poses_trimmed.pkl'
    npy  = base / 'shuttle_out' / 'shuttle_trimmed.npy'
    return (pkl, npy) if pkl.exists() and npy.exists() else (None, None)


def load_pose_pkl(match: dict):
    """Load poses_trimmed.pkl → list[PoseFrame], dense (one per trimmed frame).

    Returns None if the file is missing.
    """
    pkl, _ = _pkl_paths(match)
    if pkl is None:
        print(f'  [load_pose_pkl] pkl not found for {match["name"][:55]}')
        return None
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def load_shuttle_npy(match: dict) -> Optional[np.ndarray]:
    """Load shuttle_trimmed.npy → (N_trimmed, 2) float64 array.

    Returns None if the file is missing.
    """
    _, npy = _pkl_paths(match)
    if npy is None:
        print(f'  [load_shuttle_npy] npy not found for {match["name"][:55]}')
        return None
    return np.load(str(npy)).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND-TRUTH CSV PARSING
# ══════════════════════════════════════════════════════════════════════════════

def load_shot_frames(match_path: str):
    """Parse all set*.csv files in match_path (ShuttleSet format).

    ShuttleSet columns: rally, ball_round, time, frame_num, ...

    Returns:
        frame_nums   np.ndarray[int]   all shot frame numbers
        shot_df      pd.DataFrame      full annotation table with rally_key
    """
    files = sorted(glob.glob(os.path.join(match_path, 'set*.csv')))
    if not files:
        raise FileNotFoundError(f'No set*.csv in {match_path}')

    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        df.columns = range(len(df.columns))
        df = df.rename(columns={0: 'rally', 3: 'frame_num'})
        df['frame_num'] = pd.to_numeric(df['frame_num'], errors='coerce')
        df = df.dropna(subset=['frame_num'])
        df['frame_num'] = df['frame_num'].astype(int)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['rally_key'] = (combined['source_file'] + '_r'
                             + combined['rally'].astype(str))
    return combined['frame_num'].values, combined


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME MAP RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_frame_map_from_csvs(match: dict):
    """Reconstruct trimmed→original frame maps using only the set CSVs.

    Unlike the video-based version this never opens the video file.
    n_orig is derived from max(frame_num) + PKL_TRIM_BUF + 1 across all set
    CSVs, which is sufficient for clamping the buffer range.

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
    gt_df = gt_df.sort_values(
        ['set_file', 'rally', 'frame_num']).reset_index(drop=True)

    n_orig = int(gt_df['frame_num'].max()) + PKL_TRIM_BUF + 1

    required: set = set()
    for (sf, rv), grp in gt_df.groupby(['set_file', 'rally']):
        first_hit = int(grp['frame_num'].min())
        last_hit  = int(grp['frame_num'].max())
        for frm in range(max(0, first_hit - PKL_TRIM_BUF),
                         min(n_orig, last_hit + PKL_TRIM_BUF + 1)):
            required.add(frm)

    sorted_frames   = sorted(required)
    orig_to_trimmed = {orig: t for t, orig in enumerate(sorted_frames)}
    return sorted_frames, orig_to_trimmed, len(sorted_frames)
