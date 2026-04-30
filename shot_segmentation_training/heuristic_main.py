"""
main.py — HD-A Heuristic Pipeline
===================================
Runs the full HD-A (heuristic) + HD-T + SRA inference pipeline on all
discovered matches and reports t-IoU metrics.

Usage:
    python main.py
"""

import logging
from tqdm import tqdm

from config import RESULTS_DIR
from data_loader import discover_pkl_matches
from inference_heuristic import run_match
from evaluation import (
    evaluate, build_metrics_df, print_summary,
    plot_metrics, save_predictions,
)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover matches with pkl + npy + GT CSVs ──────────────────────────
    print('=' * 60)
    print('STEP 1 — Discover pkl matches')
    print('=' * 60)
    all_matches = discover_pkl_matches()
    if not all_matches:
        print('[ERROR] No valid matches found. Exiting.')
        return
    print(f'\n{len(all_matches)} match(es) will be evaluated:')
    for m in all_matches:
        print(f'  {m["name"][:70]}')

    # ── 2. Run HD-A / HD-T / SRA on every match ───────────────────────────────
    print('\n' + '=' * 60)
    print('STEP 2 — HD-A (heuristic) / HD-T / SRA')
    print('=' * 60)
    results = {}
    for m in tqdm(all_matches, desc='Matches'):
        print(f'\n  {m["name"][:65]}')
        try:
            hits, gt = run_match(m)
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
    print('\n' + '=' * 60)
    print('STEP 3 — Evaluation & reporting')
    print('=' * 60)
    df = build_metrics_df(results)
    df.to_csv(RESULTS_DIR / 'metrics_heuristic.csv', index=False)
    print_summary(df, 'All matches')
    plot_metrics(df, RESULTS_DIR / 'metrics_heuristic.png')

    print('\nPer-match breakdown (IoU ≥ 0.5):')
    piv = (df[df.iou_thr == 0.5]
           [['match', 'precision', 'recall', 'f1', 'n_pred', 'n_gt']]
           .sort_values('f1', ascending=False)
           .reset_index(drop=True))
    piv['match'] = piv['match'].str[:55]
    print(piv.to_string(index=False))

    save_predictions(results, RESULTS_DIR / 'predictions_heuristic.json')
    print(f'Metrics → {RESULTS_DIR}/metrics_heuristic.csv')
    print(f'Plot    → {RESULTS_DIR}/metrics_heuristic.png')
    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
