"""Paired identity counterfactual diagnostics from archived probabilities only."""
import argparse
import gzip
import json
import random
from pathlib import Path


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def read(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def analyze(root):
    rows, failures = [], []
    for folder in sorted(root.iterdir()):
        if not (folder / 'manifest.json').exists():
            continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        inventory = set(json.loads((folder / 'training-id-inventory.json').read_text()))
        for n in manifest['config']['eval_candidates']:
            for split in manifest['config']['split_seeds']:
                clean_path = folder / f'raw-{split}-{n}-clean.json.gz'
                clean = read(clean_path)
                for condition in ('id_rename', 'id_permute_seen'):
                    changed_path = folder / f'raw-{split}-{n}-{condition}.json.gz'
                    changed = read(changed_path)
                    if clean['target'] != changed['target']:
                        raise ValueError('counterfactual changed semantic targets')
                    deltas, final_deltas, per_episode = [], [], []
                    final_changes = 0
                    outcomes = {name: 0 for name in ('correct_correct', 'correct_wrong', 'wrong_correct', 'wrong_wrong')}
                    for index, (a, b, targets) in enumerate(zip(clean['posterior'], changed['posterior'], clean['target'])):
                        frame_deltas = [max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(a, b)]
                        delta = max(frame_deltas)
                        deltas.append(delta)
                        final_deltas.append(frame_deltas[-1])
                        first = max(range(len(a[-1])), key=a[-1].__getitem__)
                        second = max(range(len(b[-1])), key=b[-1].__getitem__)
                        final_changes += first != second
                        first_correct = targets[-1][first] > 0
                        second_correct = targets[-1][second] > 0
                        outcomes[('correct' if first_correct else 'wrong') + '_' + ('correct' if second_correct else 'wrong')] += 1
                        handles = random.Random(changed['event_seed'] + index + 654).sample(
                            range(256, 65536) if condition == 'id_rename' else range(4), 4)
                        per_episode.append(dict(index=index, max_probability_delta=delta,
                                                final_probability_delta=frame_deltas[-1],
                                                unseen_renamed_handles=sum(i not in inventory for i in handles),
                                                final_argmax_changed=first != second,
                                                clean_final_correct=first_correct, renamed_final_correct=second_correct))
                        if delta > .1:
                            frame = frame_deltas.index(delta)
                            failures.append(dict(run=folder.name, split=split, candidates=n, condition=condition,
                                                 episode=index, event_seed=changed['event_seed'], frame=frame,
                                                 probability_delta=delta, clean=a[frame], renamed=b[frame], target=targets[frame],
                                                 renamed_handles=handles, unseen_handles=[i for i in handles if i not in inventory]))
                    coverage = []
                    for unseen in (False, True):
                        values = [e['max_probability_delta'] for e in per_episode
                                  if bool(e['unseen_renamed_handles']) == unseen]
                        coverage.append(dict(any_unseen_renamed_handle=unseen, count=len(values),
                                             max_delta=max(values) if values else None,
                                             mean_delta=sum(values) / len(values) if values else None))
                    rows.append(dict(run=folder.name, split=split, candidates=n, condition=condition,
                                     count=len(deltas), targets_identical=True, max=max(deltas), p95=quantile(deltas, .95),
                                     p99=quantile(deltas, .99), above_01=sum(x > .01 for x in deltas),
                                     above_05=sum(x > .05 for x in deltas), above_1=sum(x > .1 for x in deltas),
                                     final_max=max(final_deltas), final_argmax_changes=final_changes,
                                     final_outcomes=outcomes, coverage=coverage, episodes=per_episode))
    failures.sort(key=lambda x: x['probability_delta'], reverse=True)
    by_run = {}
    for example in failures:
        group = by_run.setdefault(example['run'], [])
        if len(group) < 3:
            group.append(example)
    return dict(kind='archived_counterfactual_probability_diagnosis_not_new_inference', rows=rows,
                above_1_examples_total=len(failures), largest_examples=failures[:24], largest_examples_per_run=by_run)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('research/results/stage10/beliefs/main'))
    args = parser.parse_args()
    result = analyze(args.root)
    with gzip.open(args.root / 'invariance-tails.json.gz', 'wt') as handle:
        json.dump(result, handle, separators=(',', ':'))
    print(json.dumps({'paired_cells': len(result['rows']), 'large_delta_examples': result['above_1_examples_total']}))
