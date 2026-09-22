"""Independent streaming Stage 4 artifact audit (stdlib only; one run in memory)."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'nonfinite number'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def audit(path, config, partial=False):
    seen, references, gate, initial = set(), {}, {}, {}
    source = None
    config_hash = None
    cells = {(n, d) for n in config['eval_sizes'] for d in config['eval_depths']}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            finite(row)
            pair = row['variant'], row['seed']
            assert pair not in seen, 'duplicate run'
            seen.add(pair)
            assert row['training']['steps'] == config['steps']
            assert [c['step'] for c in row['training']['curve']] == config['checkpoints']
            assert not row['source']['dirty']
            source = source or row['source']
            config_hash = config_hash or row['config_hash']
            assert row['source'] == source and row['config_hash'] == config_hash
            assert config_hash == hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            seed = row['seed']
            ordinary = {k: v for k, v in row['training']['initial_parameter_hashes'].items()
                        if not k.startswith('grounders.') and k != 'strengths'}
            current = (row['training']['schedule_hash'], ordinary)
            assert references.setdefault(('train', seed), current) == current, 'unpaired ordinary parameters/schedule'
            assert len(row['evaluations']) == len(cells)
            assert {(e['nodes'], e['depth']) for e in row['evaluations']} == cells
            initial_cells = {(e['nodes'], e['depth']): e for e in row['initial_evaluations']}
            assert (128, 64) in initial_cells, 'missing initialization corner'
            for e in row['evaluations']:
                n, d = e['nodes'], e['depth']
                assert e['examples'] == config['eval_examples']
                assert references.setdefault((seed, n, d), e['data_hash']) == e['data_hash'], 'unpaired evaluation'
                canonical = [[bit in (True, 1, '1') for bit in sequence] for sequence in e['canonical_correct']]
                assert len(canonical) == e['examples'] and all(len(x) == d+1 for x in canonical)
                computed = sum(all(x) for x in canonical)/len(canonical)
                assert math.isclose(computed, e['metrics']['canonical_complete_path'], abs_tol=1e-7)
                counts = e['counts']
                wrong = sum(not x for seq in canonical for x in seq[:-1])
                persist = sum(not x and not y for seq in canonical for x, y in zip(seq, seq[1:]))
                recover = sum(not x and y for seq in canonical for x, y in zip(seq, seq[1:]))
                assert (wrong, persist, recover) == (counts['error_transition_total'], counts['error_to_error'], counts['error_to_correct'])
                assert math.isclose(sum(e['per_example']['task_correct'])/e['examples'], e['task_accuracy'])
                if (n, d) in initial_cells:
                    before = initial_cells[n, d]
                    assert before['data_hash'] == e['data_hash'], 'unpaired initialization'
                if (n, d) == (128, 64):
                    gate.setdefault(pair[0], []).append(dict(seed=seed, task=e['task_accuracy'], path=computed,
                        passed=e['task_accuracy'] >= .95 and computed >= .95))
                    initial[str(pair)] = dict(task=initial_cells[n,d]['task_accuracy'], path=initial_cells[n,d]['metrics']['canonical_complete_path'])
    expected = {(v,s) for v in config['variants'] for s in config['seeds']}
    assert seen <= expected
    if not partial:
        assert seen == expected, 'incomplete coverage'
    return dict(runs=len(seen), expected=len(expected), complete=seen == expected, source=source,
        config_hash=config_hash, gates={v: dict(passed=len(rows)==len(config['seeds']) and all(r['passed'] for r in rows), per_seed=sorted(rows,key=lambda r:r['seed'])) for v,rows in gate.items()}, initial_corner=initial)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    config=json.loads((args.metrics.parent/'config.json').read_text())
    print(json.dumps(audit(args.metrics,config,args.partial),indent=2))
