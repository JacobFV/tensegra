"""Aggregate Stage10 cells without changing selection or gate thresholds."""
import argparse
import json
from pathlib import Path

ARMS = ('ledger_only', 'raw_randomized', 'raw_correlated')
SEEDS = (20, 21, 22)
ORIGINAL = {'clean', 'reorder', 'duplicate', 'long_duplicate', 'contradiction', 'retract', 'partial', 'empty'}

def summarize(root):
    runs, rows, decisions = [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            folder = root / f'{arm}-{seed}'
            if not (folder / 'manifest.json').exists():
                decisions.append(dict(arm=arm, seed=seed, status='not_run'))
                continue
            manifest = json.loads((folder / 'manifest.json').read_text())
            cells = json.loads((folder / 'metrics.json').read_text())
            runs.append(manifest)
            rows.extend(cells)
            for scope in ('original', 'expanded', 'additional_n32'):
                expected_conditions = ORIGINAL if scope == 'original' else set(manifest['config']['conditions'])
                expected_sizes = (32,) if scope == 'additional_n32' else (8, 16)
                selected = [r for r in cells if r['split'] == 'validation' and r['candidates'] in expected_sizes
                            and r['condition'] in expected_conditions]
                found = {(r['candidates'], r['condition']) for r in selected}
                expected = {(n, c) for n in expected_sizes for c in expected_conditions}
                failed = []
                for r in selected:
                    reasons = []
                    if r['count'] < 512:
                        reasons.append('small_cell')
                    if r['final_support_accuracy'] <= (.98 if r['candidates'] == 8 else .95):
                        reasons.append('final_support_accuracy')
                    if r['mean_l1'] >= .05:
                        reasons.append('posterior_l1')
                    if r['mean_impossible'] >= .01:
                        reasons.append('impossible_mass')
                    if reasons:
                        failed.append(dict(condition=r['condition'], candidates=r['candidates'], reasons=reasons,
                                           final_support_accuracy=r['final_support_accuracy'], mean_l1=r['mean_l1'],
                                           mean_impossible=r['mean_impossible']))
                decisions.append(dict(arm=arm, seed=seed, scope=scope, cells=len(selected),
                                      missing=sorted(expected-found), failed=failed,
                                      status='passed_restricted' if found == expected and not failed else 'failed'))
    return dict(runs=len(runs), cells=len(rows), episode_evaluations=sum(r['count'] for r in rows),
                optimizer_presentations=sum(r['optimizer_presentations'] for r in runs),
                summed_run_seconds=sum(r['total_seconds'] for r in runs),
                peak_cuda_bytes=max((r['cuda_peak_bytes'] for r in runs), default=0),
                peak_rss_kib=max((r['rss_peak_kib'] for r in runs), default=0),
                composition_allowed=False, supervision_withdrawal_allowed=False, decisions=decisions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('research/results/stage10/beliefs/main'))
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'decisions'}, indent=2))
