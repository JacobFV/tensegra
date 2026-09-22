"""Artifact-only Stage 4 analysis; no torch dependency or experiment selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics

from .grounding_analysis import describe, validate_finite

NODES = (16, 32, 64, 128)
DEPTHS = (4, 8, 16, 32, 64)


def pearson(x, y):
    """Undefined correlations (constant features/outcomes) are null, not zero."""
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx, dy = [v-mx for v in x], [v-my for v in y]
    denom = math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy))/denom if denom else None


def trajectory_summary(bitstrings):
    """Initial grounding followed by each post-hop grounding; no duplicated states."""
    if not bitstrings or len({len(s) for s in bitstrings}) != 1:
        raise ValueError('expected equal-length nonempty trajectories')
    if any(not s or set(s)-{'0', '1'} for s in bitstrings):
        raise ValueError('trajectories must be binary strings')
    total, length = len(bitstrings), len(bitstrings[0])
    marginals = [sum(s[i] == '1' for s in bitstrings)/total for i in range(length)]
    counts = {pair: sum(s[i:i+2] == pair for s in bitstrings for i in range(length-1))
              for pair in ('00', '01', '10', '11')}
    at_risk = counts['00']+counts['01']
    complete = sum('0' not in s for s in bitstrings)/total
    return {'examples': total, 'state_count': length, 'complete_path': complete,
            'state_accuracy': marginals, 'independent_marginal_product': math.prod(marginals),
            'first_error_survival': [sum('0' not in s[:i+1] for s in bitstrings)/total for i in range(length)],
            'transition_counts': counts, 'error_transition_denominator': at_risk,
            'error_persistence': counts['00']/at_risk if at_risk else None,
            'error_recovery': counts['01']/at_risk if at_risk else None,
            'reconverged_after_error': sum('0' in s and s[-1] == '1' for s in bitstrings)}


def stability_gate(observations, seeds=(0, 1, 2), threshold=.95):
    """All declared seeds must pass both criteria; never average away a failure."""
    by_seed = {o['seed']: o for o in observations}
    if len(by_seed) != len(observations) or set(by_seed) != set(seeds):
        raise ValueError('gate requires exactly all declared seeds')
    return {'threshold': threshold, 'passed': all(o['task_accuracy'] >= threshold and
            o['exact_path_completion'] >= threshold for o in observations),
            'per_seed': [dict(by_seed[s], passed=by_seed[s]['task_accuracy'] >= threshold and
                        by_seed[s]['exact_path_completion'] >= threshold) for s in seeds]}


def feature_associations(examples, features):
    failures = [float('0' in e['canonical_correct']) for e in examples]
    output = {}
    for feature in features:
        values = [e[feature] for e in examples]
        output[feature] = {'pearson_with_path_failure': pearson(values, failures),
                          'complete': describe([v for v, f in zip(values, failures) if not f]) if any(not f for f in failures) else None,
                          'failed': describe([v for v, f in zip(values, failures) if f]) if any(failures) else None}
    return output


CONTRASTS = [('cosine_mixed', 'stage3_soft4'),
             ('cosine_attention_full', 'cosine_mixed'),
             ('cosine_attention_identity', 'cosine_attention_full'),
             ('cosine_pointer_identity', 'cosine_attention_identity'),
             ('cosine_attention_identity', 'dot_attention_identity'),
             ('cosine_attention_identity', 'graph_input_keyed'),
             ('cosine_pointer_identity', 'graph_input_keyed'),
             ('cosine_attention_identity', 'known')]
for update in ('attention', 'pointer'):
    for suffix in ('aux001', 'aux01', 'aux1', 'ground1', 'nullcycle'):
        CONTRASTS.append((f'random_cosine_{update}_{suffix}', f'random_cosine_{update}'))
    CONTRASTS.append((f'random_cosine_{update}_aux1', f'random_cosine_{update}_ground1'))


def canonical_examples(evaluation):
    step = evaluation['per_step']
    pre, post = step['pre_correct'], step['post_correct']
    columns = evaluation['per_example']
    examples = []
    for i, (before, after) in enumerate(zip(pre, post)):
        if len(before) != evaluation['depth'] or len(after) != evaluation['depth']:
            raise ValueError('trajectory length differs from depth')
        row = {key: values[i] for key, values in columns.items()}
        row['canonical_correct'] = evaluation['canonical_correct'][i] if 'canonical_correct' in evaluation else ''.join('1' if x in (True, 1, '1') else '0' for x in [before[0], *after])
        if not isinstance(row['canonical_correct'], str):
            row['canonical_correct'] = ''.join('1' if x else '0' for x in row['canonical_correct'])
        if len(row['canonical_correct']) != evaluation['depth'] + 1 or set(row['canonical_correct'])-{'0', '1'}:
            raise ValueError('invalid canonical trajectory')
        examples.append(row)
    if len(examples) != evaluation['examples']:
        raise ValueError('trajectory count differs from examples')
    return examples


def summarize(rows, config=None):
    if not rows:
        raise ValueError('no experiment rows')
    config = config or {}
    seeds = config.get('seeds', [0, 1, 2])
    variants = config.get('variants', sorted({r['variant'] for r in rows}))
    grid = {(n, d) for n in config.get('eval_sizes', NODES) for d in config.get('eval_depths', DEPTHS)}
    expected = {(v, s) for v in variants for s in seeds}
    found = {(r['variant'], r['seed']) for r in rows}
    if found != expected or len(found) != len(rows):
        raise ValueError('incomplete or duplicate variant/seed grid')
    sources = {json.dumps(r['source'], sort_keys=True) for r in rows}
    hashes = {r['config_hash'] for r in rows}
    if len(sources) != 1 or len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError('mixed or missing source/config provenance')
    groups, runs, initial = defaultdict(dict), {}, defaultdict(dict)
    for row in rows:
        validate_finite(row)
        v, seed = row['variant'], row['seed']
        runs[v, seed] = row
        if not row['training']['schedule_hash'] or not row['training']['initial_parameter_hashes']:
            raise ValueError('missing schedule/initialization provenance')
        if config.get('steps', row['training']['steps']) != row['training']['steps']:
            raise ValueError('training budget mismatch')
        cells = [(e['nodes'], e['depth']) for e in row['evaluations']]
        if len(cells) != len(grid) or set(cells) != grid:
            raise ValueError('incomplete or duplicate depth/size grid')
        for e in row['evaluations']:
            if not isinstance(e['task_accuracy'], (int, float)) or not 0 <= e['task_accuracy'] <= 1:
                raise ValueError('invalid task accuracy')
            if not e['data_hash']:
                raise ValueError('missing data hash')
            groups[v, e['nodes'], e['depth']][seed] = e
        for e in row.get('initial_evaluations', []):
            initial[v, e['nodes'], e['depth']][seed] = e
    # All variants receive identical data and schedule for each paired seed.
    for seed in seeds:
        if len({runs[v, seed]['training']['schedule_hash'] for v in variants}) != 1:
            raise ValueError('unpaired training schedule')
        for n, d in grid:
            if len({groups[v, n, d][seed]['data_hash'] for v in variants}) != 1:
                raise ValueError('unpaired evaluation data')
    aggregates, trajectories, associations = [], [], []
    for (v, n, d), observations in sorted(groups.items()):
        observations = [observations[s] for s in seeds]
        metric_names = sorted(set.intersection(*(set(e['metrics']) for e in observations)))
        metrics = {'task_accuracy': describe(e['task_accuracy'] for e in observations)}
        for name in metric_names:
            values = [e['metrics'][name] for e in observations]
            metrics[name] = describe(values) if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values) else None
        examples = [x for e in observations for x in canonical_examples(e)]
        paths = trajectory_summary([x['canonical_correct'] for x in examples])
        for e in observations:
            measured = trajectory_summary([x['canonical_correct'] for x in canonical_examples(e)])['complete_path']
            reported = e['metrics'].get('canonical_complete_path', measured)
            if not math.isclose(measured, reported, abs_tol=1e-6):
                raise ValueError('canonical completion disagrees with raw trajectories')
        per_seed_paths = [trajectory_summary([x['canonical_correct'] for x in canonical_examples(e)])['complete_path'] for e in observations]
        metrics['exact_path_completion'] = describe(per_seed_paths)
        aggregates.append(dict(variant=v, nodes=n, depth=d, metrics=metrics,
                               per_seed=[dict(seed=s, task_accuracy=e['task_accuracy'], exact_path_completion=p) for s, e, p in zip(seeds, observations, per_seed_paths)]))
        trajectories.append(dict(variant=v, nodes=n, depth=d, **paths))
        features = sorted(k for k in examples[0] if (k.startswith('mean_') and any(t in k for t in ('entropy', 'margin', 'norm', 'drift', 'interference', 'override', 'proposal', 'strength'))) or k == 'distinct_gold_nodes')
        associations.append(dict(variant=v, nodes=n, depth=d, associations=feature_associations(examples, features)))
    gates = []
    for v in variants:
        corner = next((a for a in aggregates if a['variant'] == v and a['nodes'] == 128 and a['depth'] == 64), None)
        if corner:
            gates.append(dict(variant=v, **stability_gate(corner['per_seed'], seeds)))
    comparisons = []
    for treatment, control in CONTRASTS:
        if treatment not in variants or control not in variants:
            continue
        for n, d in sorted(grid):
            changes, initialization = [], []
            for s in seeds:
                a, b = groups[treatment, n, d][s], groups[control, n, d][s]
                ah, bh = runs[treatment, s]['training']['initial_parameter_hashes'], runs[control, s]['training']['initial_parameter_hashes']
                common = ah.keys() & bh.keys()
                initialization.append(dict(seed=s, common_parameter_count=len(common),
                    differing_common_parameters=sorted(k for k in common if ah[k] != bh[k]),
                    treatment_only=sorted(ah.keys()-bh.keys()), control_only=sorted(bh.keys()-ah.keys())))
                changes.append(dict(seed=s, difference=a['task_accuracy']-b['task_accuracy']))
            comparisons.append(dict(treatment=treatment, control=control, nodes=n, depth=d,
                task_difference=describe(c['difference'] for c in changes), per_seed=changes,
                initialization=initialization, prior_differs='pointer' in treatment))
    initialization = []
    for (v, n, d), obs in sorted(initial.items()):
        if set(obs) != set(seeds):
            raise ValueError('incomplete initial evaluation seeds')
        final = groups[v, n, d]
        if any(obs[s]['data_hash'] != final[s]['data_hash'] for s in seeds):
            raise ValueError('unpaired initial/final data')
        initialization.append(dict(variant=v, nodes=n, depth=d,
            initial_task=describe(obs[s]['task_accuracy'] for s in seeds),
            final_task=describe(final[s]['task_accuracy'] for s in seeds),
            initial_path=describe(trajectory_summary([x['canonical_correct'] for x in canonical_examples(obs[s])])['complete_path'] for s in seeds),
            final_path=describe(trajectory_summary([x['canonical_correct'] for x in canonical_examples(final[s])])['complete_path'] for s in seeds)))
    return dict(schema_version=1, config_hash=next(iter(hashes)), source=json.loads(next(iter(sources))),
        seeds=seeds, variants=variants, aggregates=aggregates, trajectories=trajectories,
        associations=associations, gates=gates, paired_contrasts=comparisons, initialization=initialization,
        resources=[dict(variant=r['variant'], seed=r['seed'], **r['resources']) for r in rows])


def _fmt(stats):
    if stats is None:
        return 'N/A'
    return f"{stats['mean']:.4f} ± {stats['sd']:.4f}" if stats['sd'] is not None else f"{stats['mean']:.4f} (one seed)"


def render_report(result):
    lines = ['# Stage 4 artifact analysis', '',
        'Task accuracy is primary; complete grounding trajectories are a separate mechanism diagnostic. Values are mean ± sample SD across seeds. This is not a confidence interval.', '',
        'The stability gate requires task accuracy AND complete trajectory accuracy >=0.95 at D64/N128 in EACH declared seed. It is not a checkpoint selector and does not automatically launch adaptive-strength experiments.', '',
        '| Variant | Stability gate |', '|---|---|']
    lines += [f"| {g['variant']} | {'PASS' if g['passed'] else 'FAIL'} |" for g in result['gates']]
    if not result['gates']:
        lines.append('Gate unavailable: the D64/N128 cell was not evaluated.')
    lines += ['', '## Initialization versus final checkpoint', '',
        '| Variant | N | D | Initial task | Final task | Initial complete path | Final complete path |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['initialization']:
        lines.append(f"| {r['variant']} | {r['nodes']} | {r['depth']} | {_fmt(r['initial_task'])} | {_fmt(r['final_task'])} | {_fmt(r['initial_path'])} | {_fmt(r['final_path'])} |")
    lines += ['', '## Full task and trajectory matrix', '',
        '| Variant | N | D | Task | Complete path | Seed task/path pairs |', '|---|---:|---:|---:|---:|---|']
    for a in result['aggregates']:
        values = '; '.join(f"{s['seed']}: {s['task_accuracy']:.4f}/{s['exact_path_completion']:.4f}" for s in a['per_seed'])
        lines.append(f"| {a['variant']} | {a['nodes']} | {a['depth']} | {_fmt(a['metrics']['task_accuracy'])} | {_fmt(a['metrics']['exact_path_completion'])} | {values} |")
    lines += ['', '## Paired intervention contrasts', '',
        'Differences are treatment minus control on matched evaluation examples and training schedules. The summary records all differing initial parameter hashes; architecture/initialization differences are not silently called parameter matched. Pointer writes have a stronger programmed-transition prior than attention writes.', '',
        '| Treatment | Control | N | D | Task difference |', '|---|---|---:|---:|---:|']
    for c in result['paired_contrasts']:
        lines.append(f"| {c['treatment']} | {c['control']} | {c['nodes']} | {c['depth']} | {_fmt(c['task_difference'])} |")
    lines += ['', '## Failure persistence and survival', '',
        'The canonical sequence contains initial grounding once followed by post-hop grounding at every hop. Error persistence/recovery divide pooled transition counts by the number of transitions starting wrong. The marginal-product reference multiplies accuracy at these same nonduplicated states. It is descriptive: trajectory heterogeneity, revisits and reconvergence can explain divergence; it does not prove an attractor or test an iid error model.', '',
        '| Variant | N | D | Complete | Marginal product | Error→error / eligible | Error→correct / eligible |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t in result['trajectories']:
        c = t['transition_counts']
        lines.append(f"| {t['variant']} | {t['nodes']} | {t['depth']} | {t['complete_path']:.4f} | {t['independent_marginal_product']:.6g} | {c['00']}/{t['error_transition_denominator']} | {c['01']}/{t['error_transition_denominator']} |")
    lines += ['', '## Within-cell diagnostic associations', '',
        'Pearson correlations below use per-example feature means and complete-path failure, within each variant × depth × size cell (pooled paired seeds). Features include observations after errors, so associations are not causal predictors. Constant features or outcomes have undefined (N/A) correlations. summary.json includes complete/failed feature means and sample SD, survival curves and full diagnostics.', '',
        '| Variant | N | D | Feature | Correlation with failure | Complete mean ± SD | Failed mean ± SD |',
        '|---|---:|---:|---|---:|---:|---:|']
    for a in result['associations']:
        for feature, info in a['associations'].items():
            value = info['pearson_with_path_failure']
            value = 'N/A' if value is None else f'{value:.4f}'
            lines.append(f"| {a['variant']} | {a['nodes']} | {a['depth']} | {feature} | {value} | {_fmt(info['complete'])} | {_fmt(info['failed'])} |")
    lines += ['', '## Provenance', '', '```json', json.dumps({k: result[k] for k in ('source', 'config_hash', 'artifact') if k in result}, indent=2), '```', '']
    return '\n'.join(lines)


def plot_heatmaps(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    for variant in result['variants']:
        cells = [a for a in result['aggregates'] if a['variant'] == variant]
        sizes, depths = sorted({a['nodes'] for a in cells}), sorted({a['depth'] for a in cells})
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        for ax, metric, title in zip(axes, ('task_accuracy', 'exact_path_completion'), ('Task accuracy', 'Complete trajectory')):
            lookup = {(a['nodes'], a['depth']): a['metrics'][metric]['mean'] for a in cells}
            data = [[lookup[n, d] for n in sizes] for d in depths]
            im = ax.imshow(data, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_yticks(range(len(depths)), depths)
            ax.set_xlabel('Runtime graph node count N')
            ax.set_ylabel('Traversal depth D')
            ax.set_title(title)
            for i, row in enumerate(data):
                for j, value in enumerate(row):
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='black' if value > .6 else 'white')
        fig.suptitle(f'{variant} — mean of {len(result["seeds"])} seeds')
        fig.colorbar(im, ax=axes, label='Accuracy / fraction complete', shrink=.8)
        for suffix in ('png', 'svg'):
            fig.savefig(output/f'{variant}-matrix.{suffix}', dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    config_path = args.config or args.metrics.parent/'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    opener = gzip.open if args.metrics.suffix == '.gz' else open
    with opener(args.metrics, 'rt') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = summarize(rows, config)
    result['artifact'] = dict(metrics_file=args.metrics.name, metrics_sha256=hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_analysis_sha256=hashlib.sha256(Path(__file__).with_name('grounding_analysis.py').read_bytes()).hexdigest(),
        config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'report.md').write_text(render_report(result))
    if not args.no_plots:
        plot_heatmaps(result, args.output)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file() and p.name != 'provenance.json'}
    (args.output/'provenance.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    return result


if __name__ == '__main__':
    main()
