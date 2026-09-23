"""Frozen Stage10 three-arm identity-invariance study; no runtime composition."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from .belief_state import collate, loss
from .belief_contracts import contract_episodes, empty_ledger_frames
from .belief_contracts_study import metrics
from .belief_invariance import InvariantBeliefModel, make_training_episodes


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_digest(model):
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        fingerprint.update(name.encode())
        fingerprint.update(str((tensor.shape, tensor.dtype)).encode())
        fingerprint.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()

def on_device(batch, device):
    return {group: {key: value.to(device) for key, value in fields.items()}
            for group, fields in batch.items()}

def evaluate(model, config, out, seed, count=None, write_raw=True):
    rows = []
    count = config['eval_count'] if count is None else count
    for n in config['eval_candidates']:
        for split, namespace in config['split_seeds'].items():
            clean_predictions = None
            for condition in config['conditions']:
                episodes = contract_episodes(count, namespace + seed, n, condition)
                predictions, targets, empty = [], [], []
                for offset in range(0, count, config['chunk']):
                    batch = on_device(collate(episodes[offset:offset + config['chunk']]), config['device'])
                    with torch.no_grad():
                        p = model(batch['public'])['logits'].softmax(-1)
                    predictions.extend(p.cpu().tolist())
                    targets.extend(batch['targets']['posterior'].cpu().tolist())
                    empty.extend(empty_ledger_frames(batch['public']).cpu().tolist())
                row = metrics(predictions, targets)
                p = torch.tensor(predictions)
                q = torch.tensor(targets)
                row['ordinary_impossible_mass'] = float((p[..., :-1] * (q[..., :-1] == 0)).sum(-1).mean())
                row['null_signed_error'] = float((p[..., -1] - q[..., -1]).mean())
                row['null_probability'] = float(p[..., -1].mean())
                row['mean_supported_probability_error'] = float(((p - q).abs() * (q > 0)).sum(-1).mean())
                if condition == 'clean':
                    clean_predictions = p
                if condition in ('id_rename', 'id_permute_seen'):
                    if clean_predictions is None:
                        raise ValueError('clean must precede ID controls')
                    delta = (p - clean_predictions).abs()
                    row['bijection_max_probability_delta'] = float(delta.max())
                    row['bijection_mean_l1'] = float(delta.sum(-1).mean())
                row.update(arm=config['arm'], seed=seed, split=split, candidates=n, condition=condition)
                row['passed'] = (count >= 512 and row['final_support_accuracy'] > (.98 if n == 8 else .95)
                                 and row['mean_l1'] < .05 and row['mean_impossible'] < .01)
                if write_raw:
                    name = f'raw-{split}-{n}-{condition}.json.gz'
                    raw = dict(posterior=predictions, target=targets, empty_ledger=empty,
                               event_seed=namespace + seed, candidates=n, condition=condition,
                               event_sha256=digest(episodes))
                    with gzip.open(out / name, 'wt', compresslevel=1) as handle:
                        json.dump(raw, handle, separators=(',', ':'))
                    row.update(raw=name, raw_sha256=file_digest(out / name))
                rows.append(row)
    return rows

def run(config, out):
    torch.set_num_threads(config.get('threads', 2))
    torch.manual_seed(config['seed'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = InvariantBeliefModel(config['arm'], width=1024, inner=2048).to(config['device'])
    torch.save(model.state_dict(), out / 'initial.pt')
    sources = [Path(__file__).with_name(name) for name in
               ('belief_invariance_study.py', 'belief_invariance.py', 'belief_contracts.py',
                'belief_contracts_study.py', 'belief_state.py')]
    manifest = dict(config=config, config_sha256=digest(config), source_sha256={p.name: file_digest(p) for p in sources},
                    initial_checkpoint_sha256=file_digest(out / 'initial.pt'), initial_tensor_sha256=tensor_digest(model),
                    parameters=sum(p.numel() for p in model.parameters()), workspace_width=1024, ff_width=2048,
                    inactive_id_weights=16384 if config['arm'] == 'ledger_only' else 0,
                    composition_allowed=False, device_name=torch.cuda.get_device_name() if config['device'] == 'cuda' else 'cpu')
    start = time.perf_counter()
    probe_config = {**config, 'eval_candidates': [8], 'conditions': ['clean', 'id_rename'],
                    'split_seeds': {'development_curve': config['curve_namespace']}}
    curve = []
    semantic_stream = hashlib.sha256()
    public_stream = hashlib.sha256()
    unique_constructions = set()
    observed_ids = set()
    bit_ones = [0] * 16
    id_assignments = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', .0003))
    schedule = ('clean', 'partial', 'contradiction', 'retract', 'duplicate', 'reorder')
    for step in range(config['steps'] + 1):
        if step in (0, config['steps']) or step % config.get('log_every', 200) == 0:
            curve.append(dict(step=step, seconds=time.perf_counter() - start,
                              cells=evaluate(model, probe_config, out, config['seed'], count=64, write_raw=False)))
            print(json.dumps({'step': step, 'arm': config['arm'], 'seconds': curve[-1]['seconds'],
                              'clean': curve[-1]['cells'][0]['final_support_accuracy'],
                              'renamed': curve[-1]['cells'][1]['final_support_accuracy']}), flush=True)
        if step == config['steps']:
            break
        episodes = make_training_episodes(config['batch'], config['train_namespace'] + step,
                                          config['arm'], schedule[step % len(schedule)])
        for episode in episodes:
            construction = {'keys': episode['keys'], 'records': sorted(episode['records'])}
            unique_constructions.add(digest(construction))
            content = {**construction, 'events': [event[1:] for event in episode['events']]}
            semantic_stream.update(digest(content).encode())
            public_stream.update(digest({**construction, 'events': episode['events']}).encode())
            handles = {event[0] for event in episode['events'] if event[0] >= 0}
            observed_ids.update(handles)
            id_assignments += len(handles)
            for identity in handles:
                for bit in range(16):
                    bit_ones[bit] += (identity >> bit) & 1
        batch = on_device(collate(episodes), config['device'])
        optimizer.zero_grad()
        terms = loss(model(batch['public']), batch)
        total = terms['posterior'] + terms['compatibility']
        if not torch.isfinite(total):
            raise RuntimeError('nonfinite acquisition: stop instead of changing recipe')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with (out / 'loss.jsonl').open('a') as handle:
            handle.write(json.dumps({'step': step, 'episode_seed': config['train_namespace'] + step,
                                     'condition': schedule[step % len(schedule)],
                                     **{key: float(value.detach()) for key, value in terms.items()}}) + '\n')
    manifest['training_and_curve_seconds'] = time.perf_counter() - start
    manifest['optimizer_presentations'] = config['steps'] * config['batch']
    manifest['unique_training_generation_keys'] = config['steps'] * config['batch']
    manifest['unique_public_constructions'] = len(unique_constructions)
    manifest['construction_identity'] = 'Exact nonce-key and candidate-record content, candidate-order invariant; not alpha-equivalence'
    manifest['semantic_training_stream_sha256'] = semantic_stream.hexdigest()
    manifest['public_training_stream_sha256'] = public_stream.hexdigest()
    manifest['training_id_coverage'] = dict(unique=len(observed_ids), assignments=id_assignments,
                                            minimum=min(observed_ids) if observed_ids else None,
                                            maximum=max(observed_ids) if observed_ids else None,
                                            bit_one_counts=bit_ones)
    (out / 'curve.json').write_text(json.dumps(curve))
    torch.save(model.state_dict(), out / 'model.pt')
    manifest['checkpoint_sha256'] = file_digest(out / 'model.pt')
    manifest['final_tensor_sha256'] = tensor_digest(model)
    evaluation_start = time.perf_counter()
    rows = evaluate(model, config, out, config['seed'])
    (out / 'metrics.json').write_text(json.dumps(rows))
    manifest.update(evaluation_seconds=time.perf_counter() - evaluation_start,
                    total_seconds=time.perf_counter() - start,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated() if config['device'] == 'cuda' else None,
                    rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.out)
