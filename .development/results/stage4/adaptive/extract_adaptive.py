"""Audit paired adaptive-versus-fixed structural strength from frozen artifacts."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

metrics, output = map(Path, sys.argv[1:])
with gzip.open(metrics, 'rt') as stream:
    runs = {(r['variant'], r['seed']): r for r in map(json.loads, stream)}
assert set(runs) == {(v, s) for v in ('pointer_fixed4', 'pointer_adaptive4') for s in (0, 1, 2)}
comparisons, audit = [], []
for seed in (0, 1, 2):
    fixed, adaptive = [runs[v, seed] for v in ('pointer_fixed4', 'pointer_adaptive4')]
    assert fixed['source'] == adaptive['source']
    assert fixed['config_hash'] == adaptive['config_hash']
    assert fixed['training']['schedule_hash'] == adaptive['training']['schedule_hash']
    assert fixed['training']['initial_parameter_hashes'] == adaptive['training']['initial_parameter_hashes']
    audit.append(dict(seed=seed, initial_parameters_equal=True, schedule_hash=fixed['training']['schedule_hash']))
    left = {(e['nodes'], e['depth']): e for e in adaptive['evaluations']}
    right = {(e['nodes'], e['depth']): e for e in fixed['evaluations']}
    assert set(left) == set(right) and len(left) == 20
    for n, d in sorted(left):
        a, b = left[n, d], right[n, d]
        assert a['data_hash'] == b['data_hash']
        comparisons.append(dict(seed=seed, nodes=n, depth=d, data_hash=a['data_hash'],
            task_difference=a['task_accuracy']-b['task_accuracy'],
            path_difference=a['metrics']['canonical_complete_path']-b['metrics']['canonical_complete_path'],
            fixed_task=b['task_accuracy'], adaptive_task=a['task_accuracy'],
            fixed_path=b['metrics']['canonical_complete_path'], adaptive_path=a['metrics']['canonical_complete_path']))
aggregates = []
for n, d in sorted({(r['nodes'], r['depth']) for r in comparisons}):
    rows = [r for r in comparisons if r['nodes'] == n and r['depth'] == d]
    values = {k: dict(mean=statistics.fmean(r[k] for r in rows), sd=statistics.stdev(r[k] for r in rows))
              for k in ('task_difference', 'path_difference', 'fixed_task', 'adaptive_task', 'fixed_path', 'adaptive_path')}
    aggregates.append(dict(nodes=n, depth=d, metrics=values))
result = dict(interpretation='Adaptive minus fixed; paired seeds, initialization and data; sample SD is not a confidence interval.',
    source=fixed['source'], config_hash=fixed['config_hash'],
    metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
    extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    audit=audit, per_seed=comparisons, aggregates=aggregates)
(output/'adaptive-paired.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
            if p.is_file() and p.name != 'provenance.json'}
(output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
