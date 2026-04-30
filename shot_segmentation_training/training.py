"""
training.py — Swing Classifier Training Module (YOLO pipeline).
================================================================
Steps:
  1. Trim match videos to rally frames only
  2. Build positive / negative crop dataset from trimmed videos + pkl poses
  3. Fine-tune YOLOv8-cls on the dataset
  4. Smoke-test the trained classifier

Usage:
    python training.py
"""

import gc
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import (
    PKL_ROOT, DATASET_DIR, SWING_CLF_PT,
    SWING_WINDOW, NEG_RATIO, CROP_PAD, MIN_CROP_SIZE, PKL_TRIM_BUF,
)
from data_loader import (
    discover_pkl_matches, load_shot_frames, _reconstruct_frame_map_from_csvs,
)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — VIDEO TRIMMING
# ══════════════════════════════════════════════════════════════════════════════

def _trim_match_video(match: dict) -> Path:
    """Trim one match video to rally frames only (buffer = PKL_TRIM_BUF).

    Uses the fast cap.grab() skip approach — reads only frames that belong
    to the required set, skips all others without decoding.

    Returns path to the trimmed video (already existing or newly created).
    """
    out_path = PKL_ROOT / match['name'] / 'trimmed.mp4'
    if out_path.exists():
        print(f'  [trim] cached: {match["name"][:60]}')
        return out_path

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
    print(f'  [trim] {written}/{len(sorted_frames)} frames written → {out_path}')
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DATASET BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_swing_dataset(matches: list) -> dict:
    """Build positive (swing) and negative (not_swing) crop datasets.

    Two-pass approach per match:
      Pass 1: decode every frame; write positives immediately; collect
              negative (frame_idx, slot, bbox) metadata only — no images in RAM.
      Pass 2: re-open video; decode only the sampled negative frames.
    """
    import pickle as _pickle

    pos_dir = DATASET_DIR / 'swing'
    neg_dir = DATASET_DIR / 'not_swing'
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for m in matches:
        name = m['name']
        print(f'\n  [dataset] {name[:65]}')

        pkl_path = PKL_ROOT / name / 'pose_out' / 'poses_trimmed.pkl'
        if not pkl_path.exists():
            print(f'    [skip] poses_trimmed.pkl not found')
            continue
        with open(pkl_path, 'rb') as f:
            poses = _pickle.load(f)

        trimmed_path = PKL_ROOT / name / 'trimmed.mp4'
        if not trimmed_path.exists():
            print(f'    [skip] trimmed.mp4 not found — run trim step first')
            continue

        sorted_frames, orig_to_trimmed, expected_len = \
            _reconstruct_frame_map_from_csvs(m)

        if len(poses) != expected_len:
            print(f'    [skip] length mismatch: poses={len(poses)} '
                  f'expected={expected_len}')
            continue

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

        cap = cv2.VideoCapture(str(trimmed_path))
        if not cap.isOpened():
            print(f'    [skip] cannot open trimmed video')
            continue
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # ── PASS 1 ────────────────────────────────────────────────────────────
        pos_count     = 0
        neg_pool_meta = []

        for tidx in tqdm(range(len(poses)), desc='    pass1', leave=False):
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
                    neg_pool_meta.append((tidx, slot, x1, y1, x2, y2))

        cap.release()

        # ── PASS 2: sample negatives and re-decode only those frames ──────────
        n_neg_target     = min(len(neg_pool_meta), NEG_RATIO * pos_count)
        neg_sampled_meta = (random.sample(neg_pool_meta, n_neg_target)
                            if n_neg_target > 0 else [])

        neg_by_tidx: dict = {}
        for tidx, slot, x1, y1, x2, y2 in neg_sampled_meta:
            neg_by_tidx.setdefault(tidx, []).append((slot, x1, y1, x2, y2))

        neg_frames_needed = set(neg_by_tidx.keys())
        neg_count = 0

        if neg_frames_needed:
            cap2 = cv2.VideoCapture(str(trimmed_path))
            if not cap2.isOpened():
                print(f'    [warn] cannot re-open trimmed video for pass 2')
            else:
                for tidx in tqdm(range(len(poses)), desc='    pass2', leave=False):
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

        del poses, neg_pool_meta, neg_sampled_meta, neg_by_tidx
        gc.collect()

    total_pos = sum(v['n_pos']         for v in summary.values())
    total_neg = sum(v['n_neg_sampled'] for v in summary.values())
    print(f'\n[dataset] Total: {total_pos} positives, {total_neg} negatives')
    print(f'[dataset] Saved to {DATASET_DIR}')
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — YOLO TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _train_yolo_classifier() -> Path:
    """Fine-tune YOLOv8-cls on the built dataset and return best.pt path."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError('ultralytics not installed: pip install ultralytics')

    total_pos = len(list((DATASET_DIR / 'swing').glob('*.jpg')))
    total_neg = len(list((DATASET_DIR / 'not_swing').glob('*.jpg')))
    if total_pos == 0:
        raise RuntimeError(
            f'No positive crops found in {DATASET_DIR / "swing"}. '
            'Run the dataset build step first.')

    print(f'\n[train] {total_pos} swing + {total_neg} not_swing crops')
    print('[train] Starting YOLOv8-cls training...')

    model   = YOLO('yolov8s-cls.pt')
    results = model.train(
        data     = str(DATASET_DIR),
        epochs   = 30,
        imgsz    = 224,
        batch    = 64,
        project  = 'runs/swing_classifier',
        name     = 'train',
        exist_ok = True,
        verbose  = True,
    )

    best = Path('runs/swing_classifier/train/weights/best.pt')
    if not best.exists():
        raise RuntimeError(f'Training finished but best.pt not found at {best}')

    shutil.copy(str(best), str(SWING_CLF_PT))
    print(f'\n[train] Best weights saved → {SWING_CLF_PT}')
    return SWING_CLF_PT


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test_classifier(clf_path: Path, n: int = 10) -> None:
    """Run the trained classifier on n random crops from each class."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print('[smoke] ultralytics not installed — skipping smoke test')
        return

    model = YOLO(str(clf_path))
    print(f'\n[smoke] Testing {clf_path} ...')

    for label, folder in [('swing',     DATASET_DIR / 'swing'),
                           ('not_swing', DATASET_DIR / 'not_swing')]:
        imgs = list(folder.glob('*.jpg'))
        if not imgs:
            print(f'  [{label}] no images found')
            continue
        sample  = random.sample(imgs, min(n, len(imgs)))
        results = model.predict(source=[str(p) for p in sample], verbose=False)
        scores  = []
        for r in results:
            probs      = r.probs.data.cpu().numpy()
            swing_conf = float(probs[1]) if len(probs) > 1 else float(probs[0])
            scores.append(swing_conf)
        mean_conf = sum(scores) / len(scores)
        print(f'  [{label}] mean swing confidence over {len(scores)} crops: '
              f'{mean_conf:.3f}')


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_swing_classifier() -> None:
    """Full training pipeline: discover → trim → dataset → train → smoke test."""
    random.seed(42)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    # ── Step 1: Discover matches ──────────────────────────────────────────────
    print('=' * 60)
    print('SWING CLASSIFIER — Step 1: Discover matches')
    print('=' * 60)
    matches = discover_pkl_matches()
    if not matches:
        print('[ERROR] No valid matches found.')
        return

    # ── Step 2: Trim videos ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 2: Trim videos')
    print('=' * 60)
    for m in matches:
        if not m['video']:
            print(f'  [skip] no video path for {m["name"][:60]}')
            continue
        try:
            _trim_match_video(m)
        except Exception as exc:
            print(f'  [ERROR] {m["name"][:55]}: {exc}')

    # ── Step 3: Build dataset ─────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 3: Build dataset')
    print('=' * 60)
    _build_swing_dataset(matches)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 4: Train YOLOv8-cls')
    print('=' * 60)
    clf_path = _train_yolo_classifier()

    # ── Step 5: Smoke test ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('SWING CLASSIFIER — Step 5: Smoke test')
    print('=' * 60)
    _smoke_test_classifier(clf_path)

    print('\nSwing classifier training complete.')
    print(f'Weights → {SWING_CLF_PT}')


if __name__ == '__main__':
    train_swing_classifier()
