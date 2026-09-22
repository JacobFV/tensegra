#!/usr/bin/env python3
"""Independent streaming Stage 5 counts/provenance/checkpoint audit.

No model inference or full JSONL materialization. Torch is imported only when
checkpoint verification is requested; checkpoint loads use weights_only=True.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

RATES = {
    'binding_accuracy': ('binding_correct', 'steps'),
    'primitive_accuracy': ('primitive_correct', 'steps'),
    'result_accuracy': ('execution_correct', 'defined_examples'),
    'execution_accuracy': ('execution_correct', 'defined_examples'),
    'task_accuracy': ('task_correct', 'defined_examples'),
    'complete_trajectory_accuracy': ('complete_trajectory', 'defined_examples'),
    'task_given_binding_correct': ('task_given_binding_correct', 'all_binding_correct'),
    'execution_given_lowering_correct': ('execution_given_lowering_correct', 'all_lowering_correct'),
    'task_given_execution_correct': ('task_given_execution_correct', 'execution_correct'),
    'oracle_lifting_accuracy': ('oracle_lift_correct', 'defined_examples'),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def finite(value):
    if isinstance(value, float): require(math.isfinite(value), 'nonfinite metric')
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)


def audit_cell(cell, examples, exact):
    c, p = cell['counts'], cell['per_example']
    require(c['examples'] == examples, 'example count mismatch')
    require(all(v is None or isinstance(v, int) and v >= 0 for v in c.values()), 'invalid count')
    require(c['defined_examples'] + c['undefined_examples'] == examples, 'defined population mismatch')
    for key in ('task', 'execution', 'binding', 'primitive', 'lowering', 'confidence', 'runtime_node_count', 'clause_count'):
        require(len(p[key]) == examples, f'per-example length {key}')
    require(sum(p['clause_count']) == c['steps'], 'step count mismatch')
    for key, count in [('binding','all_binding_correct'), ('primitive','all_primitive_correct'), ('lowering','all_lowering_correct')]:
        require(sum(p[key]) == c[count], f'per-example count mismatch {count}')
    require(p['lowering'] == [a and b for a,b in zip(p['binding'],p['primitive'])], 'lowering intersection mismatch')
    defined = c['defined_examples']
    if defined == examples:
        for key, count in [('task','task_correct'), ('execution','execution_correct')]:
            require(sum(p[key]) == c[count], f'per-example count mismatch {count}')
        for count, left, right in [('task_given_binding_correct','task','binding'),
                                  ('execution_given_lowering_correct','execution','lowering'),
                                  ('task_given_execution_correct','task','execution')]:
            require(sum(a and b for a,b in zip(p[left],p[right])) == c[count], f'intersection mismatch {count}')
    else:
        require(defined == 0 and c['task_correct'] == c['execution_correct'] == 0, 'undefined target rewarded')
        for key in ('task_accuracy','execution_accuracy','result_accuracy','task_given_binding_correct',
                    'execution_given_lowering_correct','task_given_execution_correct','oracle_lifting_accuracy'):
            require(cell[key] is None, f'undefined conditional {key}')
    for name,(num,den) in RATES.items():
        value = cell[name]
        if value is None: continue
        require(c[num] is not None and c[den] > 0 and c[num] <= c[den], f'bad denominator {name}')
        require(math.isclose(value,c[num]/c[den],abs_tol=1e-12), f'inconsistent rate {name}')
    if not exact:
        for key in ('execution_accuracy','complete_trajectory_accuracy','oracle_lifting_accuracy','execution_given_lowering_correct'):
            require(cell[key] is None, f'neural runtime metric mislabeled {key}')
    for grouping in ('family_breakdown','style_breakdown'):
        parts = cell[grouping].values()
        require(sum(x['examples'] for x in parts) == examples, f'{grouping} population')
        require(sum(x['defined_examples'] for x in parts) == defined, f'{grouping} defined population')
        require(sum(x['task_correct'] for x in parts) == c['task_correct'], f'{grouping} task count')
    for risk in cell['confidence']:
        require(risk['accepted'] == risk['invoked'], 'gate invocation mismatch')
        require(0 <= risk['accepted_wrong'] <= risk['accepted'], 'invalid actual gate error')
        require(0 <= risk['correct'] <= min(risk['answered'],risk['defined_examples']), 'undefined confidence success')
        require(risk['answered'] <= examples and risk['deferred'] <= examples and risk['rejected'] <= examples, 'invalid gate populations')
        require(risk['hypothetical_gate_wrong'] <= risk['hypothetical_gate_count'], 'invalid hypothetical gate error')
    return 1


def checkpoint_audit(row, directory):
    import torch
    filename = row['checkpoint']
    require(Path(filename).name == filename, 'unsafe checkpoint filename')
    path = directory / filename
    saved = torch.load(path, map_location='cpu', weights_only=True)
    require((saved['variant'],saved['seed']) == (row['variant'],row['seed']), 'checkpoint identity mismatch')
    require(fingerprint(saved['config']) == row['config_hash'], 'checkpoint config mismatch')
    require(saved['source'] == row['source'], 'checkpoint source mismatch')
    hashes = {}
    for name,value in saved['state_dict'].items():
        value = value.detach().cpu().contiguous()
        require(bool(torch.isfinite(value).all()), f'nonfinite checkpoint {name}')
        hashes[name] = hashlib.sha256(str((list(value.shape),str(value.dtype))).encode()+value.numpy().tobytes()).hexdigest()
    require(fingerprint(hashes) == row['training']['final_state_hash'], 'checkpoint state hash mismatch')
    return dict(filename=filename, sha256=file_hash(path), state_hash=fingerprint(hashes))


def audit(metrics, config_path, checkpoints=None, source_root=None, expected_commit=None):
    config = json.loads(config_path.read_text())
    expected = {(v,s) for v in config['variants'] for s in config['seeds']}
    conditions = {f'n{n}_d{d}' for n in config['eval_sizes'] for d in config['eval_depths']}
    if config['extra_evaluations']: conditions.update(('renamed','paraphrase','noise_medium','noise_high','ambiguous','invalid','permuted','wrong_graph'))
    seen, records, pairing, source, cell_count = set(), [], {}, None, 0
    opener = gzip.open if str(metrics).endswith('.gz') else open
    with opener(metrics,'rt') as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line); finite(row)
            key = row['variant'],row['seed']
            require(key in expected and key not in seen, f'unexpected/duplicate run {key}'); seen.add(key)
            require(row['config_hash'] == fingerprint(config), 'config hash mismatch')
            if source is None: source = row['source']
            require(row['source'] == source and not source['dirty'], 'mixed/dirty source')
            if expected_commit: require(source['commit'] == expected_commit, 'unexpected source commit')
            training = row['training']
            require(training['steps'] == config['steps'] and len(training['losses']) == config['steps'], 'training coverage')
            require(training['initial_state_hash'] == fingerprint(training['initial_parameter_hashes']), 'initial hash inconsistent')
            require([x['step'] for x in training['curve']] == config['checkpoints'], 'checkpoint curve coverage')
            require({x['condition'] for x in row['evaluations']} == conditions and len(row['evaluations']) == len(conditions), 'evaluation coverage')
            identity = (training['schedule_hash'], training['initial_state_hash'])
            old = pairing.setdefault(('training',row['seed']),identity)
            require(old == identity, 'unpaired training/initialization')
            exact = row['variant_options']['mode'] == 'exact'
            cells = [('final',x) for x in row['evaluations']]
            cells += [('initial',x) for x in row['initial_evaluations']]
            cells += [(f'curve:{x["step"]}',x['diagnostics']) for x in training['curve']]
            for phase,cell in cells:
                cell_count += audit_cell(cell,config['eval_examples'],exact)
                pkey = (phase,cell['condition'],row['seed'])
                require(pairing.setdefault(pkey,cell['data_hash']) == cell['data_hash'], 'unpaired evaluation data')
            record = dict(variant=row['variant'],seed=row['seed'],final_state_hash=training['final_state_hash'])
            if checkpoints: record['checkpoint'] = checkpoint_audit(row,checkpoints)
            records.append(record)
    require(seen == expected, f'incomplete run coverage: missing {sorted(expected-seen)}')
    if checkpoints:
        require({p.name for p in checkpoints.glob('*.pt')} == {x['checkpoint']['filename'] for x in records}, 'checkpoint inventory mismatch')
    if source_root:
        for name,digest in source['files'].items(): require(file_hash(source_root/name)==digest, f'source differs {name}')
    return dict(schema_version=1, passed=True, runs=len(records), cells=cell_count,
                final_conditions=len(conditions), metrics_sha256=file_hash(metrics), config_sha256=file_hash(config_path),
                source=source, checkpoint_verification=checkpoints is not None,
                paired_initialization_schedule_data=True, records=records,
                caveats=['Undefined task outcomes excluded; no success inferred from placeholder values.',
                         'Neural trajectory audits are external exact executions, not neural execution trajectories.',
                         'Training timing includes diagnostic work where explicitly labeled.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--expected-commit')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.metrics,args.config,args.checkpoints,args.source_root,args.expected_commit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,runs=result['runs'],cells=result['cells'],output=str(args.output))))


if __name__ == '__main__': main()
