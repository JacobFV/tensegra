"""Summarize frozen semantic exposure curves without importing training frameworks.

Usage: python .development/stage8-semantic-summarize.py RESULT_DIR...
No thresholds or model choices are selected here.
"""
import argparse
import json
import statistics
from pathlib import Path


def flatten(row, dataset, calibration):
    out = []
    for surface, metrics in row.get(calibration, {}).items():
        groups = {'all': metrics, **metrics.get('by_lesson', {})}
        for lesson, group in groups.items():
            out.append(dict(
                corpus=dataset, arm=row['arm'], seed=row['seed'],
                presentations=row['optimizer_presentations'],
                actual_unique_graphs=row['actual_unique_graphs_seen'],
                decoder=calibration, surface=surface, lesson=lesson,
                examples=group['examples'],
                node_type=group['node_type_accuracy'],
                identity_copy=group['identity_copy_accuracy'],
                entity_equivalence=group['entity_equivalence'],
                exact_graph=group['semantic_equivalence'],
                node_f1=group['node']['f1'],
                typed_edge_f1=group['typed_edge']['f1'],
                ordered_edge_f1=group['ordered_edge']['f1']))
    return out


def summarize(paths):
    rows, manifests, completeness = [], [], []
    for path in paths:
        manifest=json.loads((path/'manifest.json').read_text())
        curves=[json.loads(line) for line in (path/'curves.jsonl').read_text().splitlines()]
        config=manifest['config']; dataset=config['train_count']
        expected={(arm,seed,exposure) for arm in config['arms'] for seed in config['seeds']
                  for exposure in config['eval_presentations']}
        observed={(r['arm'],r['seed'],r['optimizer_presentations']) for r in curves if r['arm']!='frequency'}
        if len(observed)!=len([r for r in curves if r['arm']!='frequency']):
            raise ValueError(f'duplicate checkpoint records: {path}')
        completeness.append(dict(corpus=dataset,complete=observed==expected,
                                 missing=sorted(expected-observed),unexpected=sorted(observed-expected),
                                 finished_runs=len(manifest['runs'])))
        manifests.append(dict(corpus=dataset,heldout_digest=manifest['heldout_digest'],
                              vocabulary_sha256=manifest['vocabulary_sha256'],source_sha256=manifest['source_sha256'],
                              runs=manifest['runs']))
        for row in curves:
            for decoder in ('evaluation','calibrated_evaluation'):
                rows.extend(flatten(row,dataset,decoder))
    for field in ('heldout_digest','vocabulary_sha256','source_sha256'):
        if len({json.dumps(m[field],sort_keys=True) for m in manifests})!=1:
            raise ValueError(f'mismatched {field}')
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('corpus','arm','presentations','decoder','surface','lesson'))
        groups.setdefault(key,[]).append(row)
    aggregate=[]
    for key,group in sorted(groups.items()):
        item=dict(zip(('corpus','arm','presentations','decoder','surface','lesson'),key))
        item['seeds']=[r['seed'] for r in group]
        for metric in ('node_type','identity_copy','entity_equivalence','exact_graph','node_f1','typed_edge_f1','ordered_edge_f1'):
            values=[r[metric] for r in group]
            item[metric]=dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                              sample_std=statistics.stdev(values) if len(values)>1 else None)
        aggregate.append(item)
    return dict(completeness=completeness,manifests=manifests,per_seed=rows,aggregate=aggregate,
                interpretation='Seed ranges and sample standard deviations; no threshold selection or significance claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directories',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.directories),indent=2))
