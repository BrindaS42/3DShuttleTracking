"""
main_2.py — HD-A YOLO Pipeline
================================
Precision-locked SRA with Dual-Stage Peak Suppression.
Uses the trained YOLOv8 swing classifier for HD-A combined with HD-T
trajectory detection and strict HD-T ∩ YOLO intersection logic.

Usage:
    python main_2.py          # run inference (requires swing_classifier.pt)
    python training.py        # train the classifier first
"""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import RESULTS_DIR, PKL_ROOT
from data_loader import (
    discover_pkl_matches, load_pose_pkl, load_shuttle_npy,
    load_shot_frames, _reconstruct_frame_map_from_csvs,
)
from inference_yolo import (
    assign_sides_pkl,
    detect_swing_actions_yolo,
    compute_hd_t,
    _build_segs, _peak_frame,
)
from evaluation import (
    evaluate, build_metrics_df, print_summary,
    plot_metrics, save_predictions,
)


def main_2():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover matches ───────────────────────────────────────────────────
    all_matches = discover_pkl_matches()
    if not all_matches:
        print('[ERROR] No valid matches found. Exiting.')
        return

    results = {}
    print('\n' + '=' * 60)
    print('PRECISION-LOCKED SRA: Dual-Stage Peak Suppression')
    print('=' * 60)

    for m in tqdm(all_matches, desc='Matches'):
        try:
            # ── 2. Load data ──────────────────────────────────────────────────
            poses      = load_pose_pkl(m)
            shuttle_2d = load_shuttle_npy(m)
            print(f'  shuttle_2d shape: {shuttle_2d.shape}'
                  if shuttle_2d is not None else '  shuttle_2d: None')
            if poses is None or shuttle_2d is None:
                continue

            n_nan = np.isnan(shuttle_2d).sum()
            print(f'  shuttle_2d has {n_nan} NaN values out of {shuttle_2d.size} total')

            if not isinstance(shuttle_2d, np.ndarray) or shuttle_2d.shape[1] != 2:
                print(f'  [skip] {m["name"][:50]}: invalid shuttle_2d shape '
                      f'{shuttle_2d.shape}')
                continue

            # ── 3. HD-T ───────────────────────────────────────────────────────
            hdt_f, _ = compute_hd_t(shuttle_2d, list(range(len(poses))),
                                    min_dist=15)

            # ── 4. YOLO HD-A — Stage 1: Aggressive Local Maxima ──────────────
            trimmed_video = str(PKL_ROOT / m['name'] / 'trimmed.mp4')
            raw_f, raw_p, raw_c = detect_swing_actions_yolo(
                poses, list(range(len(poses))),
                assign_sides_pkl(poses), trimmed_video)

            hda_f, hda_p, hda_c = [], [], []
            data = sorted(zip(raw_f, raw_p, raw_c), key=lambda x: x[0])
            i = 0
            while i < len(data):
                cluster = [data[i]]
                j = i + 1
                while j < len(data) and (data[j][0] - data[i][0]) < 20:
                    cluster.append(data[j])
                    j += 1
                best = max(cluster, key=lambda x: x[2])
                if best[2] > 0.5:
                    hda_f.append(best[0])
                    hda_p.append(best[1])
                    hda_c.append(best[2])
                i = j

            # ── 5. Load and map ground truth ──────────────────────────────────
            sorted_frames, orig_to_trimmed, _ = _reconstruct_frame_map_from_csvs(m)
            _, shot_df = load_shot_frames(m['path'])
            shot_df['frame_num_t'] = shot_df['frame_num'].map(orig_to_trimmed)
            gt_frames = shot_df['frame_num_t'].dropna().astype(int).values

            # ── 6. Stage 2: Confirmed Intersection Only ───────────────────────
            hits     = []
            used_hdt = set()
            for f_y, p_y, c_y in zip(hda_f, hda_p, hda_c):
                possible_hdt = [f for f in hdt_f
                                if (f_y - 15) <= f <= (f_y + 5)
                                and f not in used_hdt]
                if possible_hdt:
                    best_hdt = min(possible_hdt, key=lambda f: abs(f - (f_y - 3)))
                    hits.append({'frame': int(best_hdt), 'player': int(p_y),
                                 'conf': c_y})
                    used_hdt.add(best_hdt)
                elif c_y > 0.90:
                    hits.append({'frame': int(f_y - 2), 'player': int(p_y),
                                 'conf': c_y})

            # ── 7. Final temporal de-duplication ──────────────────────────────
            hits = sorted(hits, key=lambda x: x['conf'], reverse=True)
            final_hits = []
            for h in hits:
                if not any(abs(h['frame'] - fh['frame']) < 15
                           for fh in final_hits):
                    final_hits.append(h)
            final_hits = sorted(final_hits, key=lambda x: x['frame'])

            results[m['name']] = {'hits': final_hits, 'gt': gt_frames}

            # ── 8. Multi-tolerance evaluation ─────────────────────────────────
            pred_frames = [h['frame'] for h in final_hits]
            gt_list     = list(gt_frames)
            p5,  r5,  f5  = evaluate(pred_frames, gt_list, tol=5)
            p10, r10, f10 = evaluate(pred_frames, gt_list, tol=10)
            p15, r15, f15 = evaluate(pred_frames, gt_list, tol=15)

            print(f"\nMatch: {m['name'][:40]}...")
            print(f"  [TOL ±5]  P: {p5:.3f} R: {r5:.3f} F1: {f5:.3f}")
            print(f"  [TOL ±10] P: {p10:.3f} R: {r10:.3f} F1: {f10:.3f}")
            print(f"  [TOL ±15] P: {p15:.3f} R: {r15:.3f} F1: {f15:.3f}")
            print(f"  Stats: Pred: {len(final_hits)} | GT: {len(gt_list)}")

        except Exception as exc:
            logging.error(f"Error on {m['name']}: {exc}")
            continue

    if not results:
        print('[WARNING] No results produced. Check pkl/npy/CSV alignment.')
        return

    # ── 9. Metrics + reporting ─────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 3 — Evaluation & reporting')
    print('=' * 60)
    df = build_metrics_df(results)
    df.to_csv(RESULTS_DIR / 'metrics_yolo.csv', index=False)
    print_summary(df, 'All matches')
    plot_metrics(df, RESULTS_DIR / 'metrics_yolo.png')

    print('\nPer-match breakdown (IoU ≥ 0.5):')
    piv = (df[df.iou_thr == 0.5]
           [['match', 'precision', 'recall', 'f1', 'n_pred', 'n_gt']]
           .sort_values('f1', ascending=False)
           .reset_index(drop=True))
    piv['match'] = piv['match'].str[:55]
    print(piv.to_string(index=False))

    save_predictions(results, RESULTS_DIR / 'predictions_yolo.json')
    print(f'Metrics → {RESULTS_DIR}/metrics_yolo.csv')
    print(f'Plot    → {RESULTS_DIR}/metrics_yolo.png')
    print('\nPipeline complete.')


if __name__ == '__main__':
    main_2()
