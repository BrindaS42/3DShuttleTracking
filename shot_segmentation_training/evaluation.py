"""
evaluation.py — Evaluation and reporting utilities.
====================================================
Shared between heuristic and YOLO pipelines.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
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


def evaluate(pred: list, gt: list, tol: int = 10) -> tuple:
    """Frame-tolerance evaluate — quick per-match console logging."""
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


def print_summary(df: pd.DataFrame, label: str = 'All matches'):
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


def save_predictions(results: dict, path: Path):
    with open(path, 'w') as f:
        json.dump(
            {n: [{'frame': int(h['frame']), 'player': int(h['player']),
                  'rally': str(h.get('rally', ''))} for h in r['hits']]
             for n, r in results.items()}, f, indent=2)
    print(f'Predictions → {path}')
