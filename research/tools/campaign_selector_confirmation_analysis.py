"""Paired A06 summaries; shared event bootstrap across fixed initialization seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root, output):
    seeds = [601, 602, 603]
    modes = ['soft', 'hard', 'context', 'none']
    summary = {'seeds': seeds, 'cells': {}, 'paired_effects': {},
               'uncertainty': 'Event bootstrap shares indices across fixed seeds; seed variability reported separately.'}
    raws = {}
    for mode in modes:
        manifests = [json.loads((root / str(seed) / mode / 'eval-01000.json').read_text()) for seed in seeds]
        raws[mode] = [np.load(root / str(seed) / mode / 'eval-01000.npz') for seed in seeds]
        summary['cells'][mode] = [dict(condition=rows[0]['condition'],
            task=[r['task'] for r in rows],
            suffix=[r['suffix_value_trajectory'] for r in rows],
            path=[r['exact_pointer_path'] for r in rows],
            selected_mass=[r['selected_mass'] for r in rows],
            forward_seconds=[r['forward_seconds'] for r in rows])
            for rows in zip(*(m['rows'] for m in manifests))]
    rng = np.random.default_rng(709)
    def compare(name, left_mode, left_cell, right_mode, right_cell):
        a = np.stack([r[f'c{left_cell}_task'].astype(float) for r in raws[left_mode]])
        b = np.stack([r[f'c{right_cell}_task'].astype(float) for r in raws[right_mode]])
        difference = a - b
        boot = [difference[:, rng.integers(a.shape[1], size=a.shape[1])].mean() for _ in range(2000)]
        summary['paired_effects'][name] = dict(per_seed=difference.mean(1).tolist(),
            mean=float(difference.mean()), event_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
            correct_to_wrong=((b == 1) & (a == 0)).sum(1).tolist(),
            wrong_to_correct=((b == 0) & (a == 1)).sum(1).tolist())
    # Frozen config cell indexes: joint8, content16joint18, both16joint21.
    for mode in modes:
        compare(mode+'_content16_minus_original',mode,18,mode,8)
    compare('context_both16_minus_original','context',21,'context',8)
    compare('soft_content16_minus_gather_content16','soft',18,'hard',18)
    compare('soft_content16_minus_context_both16','soft',18,'context',21)
    output.write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();main(args.root,args.output)
