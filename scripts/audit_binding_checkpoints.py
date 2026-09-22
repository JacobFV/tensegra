#!/usr/bin/env python3
"""Audit Stage 4 saved parameters against completed metrics, without model inference.

Run where the checkpoint files reside. Metrics are streamed one row at a time;
checkpoint loading explicitly disallows arbitrary pickle code (weights_only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((list(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(row, directory):
    filename = row.get('checkpoint')
    if not filename or Path(filename).name != filename:
        raise ValueError('checkpoint must be a plain filename')
    path = Path(directory) / filename
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['variant'], checkpoint['seed']) != (row['variant'], row['seed']):
        raise ValueError(f'{filename}: variant/seed mismatch')
    if fingerprint(checkpoint['config']) != row['config_hash']:
        raise ValueError(f'{filename}: configuration mismatch')
    if checkpoint['source'] != row['source']:
        raise ValueError(f'{filename}: source provenance mismatch')
    state = checkpoint['state_dict']
    state_hash = fingerprint({name: tensor_hash(value) for name, value in state.items()})
    if state_hash != row['training']['final_state_hash']:
        raise ValueError(f'{filename}: final state hash mismatch')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f'{filename}: nonfinite saved parameters')
    parameters = {}
    for name, value in state.items():
        if name.endswith(('query_null_logit', 'key_null_logit')) or 'temperature' in name or name == 'strengths':
            parameters[name] = value.tolist()
            if name.endswith('log_temperature'):
                parameters[name.replace('log_temperature', 'effective_temperature')] = value.clamp(-12., 12.).exp().tolist()
        elif name.startswith('grounders.') and 'projection' in name and name.endswith('weight'):
            parameters[name] = dict(frobenius_norm=float(value.norm()),
                                    row_norms=value.norm(dim=-1).tolist(),
                                    minimum=float(value.min()), maximum=float(value.max()))
    options = row.get('variant_options', {})
    return dict(variant=row['variant'], seed=row['seed'], checkpoint=filename,
                checkpoint_sha256=file_hash(path), final_state_hash=state_hash,
                initial_identity_prior=options.get('identity_init', True),
                initial_null_logit=options.get('null_init', .65),
                initial_strength=options.get('strength', checkpoint['config'].get('strength', 4.)),
                auxiliary_weights=row.get('loss_weights'), parameters=parameters)


def audit(metrics, checkpoints):
    records, seen = [], set()
    with Path(metrics).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['variant'], row['seed'])
            if key in seen:
                raise ValueError(f'duplicate metrics run {key}')
            seen.add(key)
            records.append(audit_checkpoint(row, checkpoints))
    if not records:
        raise ValueError('metrics contain no completed runs')
    actual = {path.name for path in Path(checkpoints).glob('*.pt')}
    expected = {record['checkpoint'] for record in records}
    if actual != expected:
        raise ValueError(f'checkpoint inventory differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    return dict(schema_version=1, runs=len(records), metrics_sha256=file_hash(Path(metrics)),
                all_state_hashes_match=True, records=records,
                interpretation='Parameter changes are descriptive. Cold-null0 and cold-null0.65 comparisons are distinct; an aligned initialization is not evidence of learned binding.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--checkpoints', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.metrics, args.checkpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(runs=result['runs'], all_state_hashes_match=True, output=str(args.output))))


if __name__ == '__main__':
    main()
