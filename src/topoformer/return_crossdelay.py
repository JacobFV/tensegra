"""Frozen 1024-wide return readout transfer, with auditable cross-delay fits."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .return_memory import ReturnMemoryModel, FIELDS
from .return_memory_study import move
from .retention_data import make_batch
from .return_diagnostics_probe import capture, fit_ridge, predict_ridge, metrics

DELAYS = (0, 1, 2, 4, 8, 16, 32)


def tensor_hash(tree):
    digest = hashlib.sha256()
    def visit(value, name):
        if isinstance(value, dict):
            for key in sorted(value): visit(value[key], name + '/' + key)
        else:
            value = value.detach().contiguous().cpu()
            digest.update(name.encode())
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    visit(tree, '')
    return digest.hexdigest()


def pooled_indices(size, delays, rows, seed=0):
    """Label-blind balanced cyclic selection; retain all event identities."""
    if rows > size * len(delays) or rows < size:
        raise ValueError('Pool must cover all events without duplicate states')
    quotient, remainder = divmod(rows, len(delays))
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    selected = []
    for index, delay in enumerate(delays):
        count = quotient + int(index < remainder)
        events = permutation.roll(index * (size // len(delays)))[:count]
        selected.extend((delay, int(event)) for event in events)
    if len({event for _, event in selected}) != size:
        raise ValueError('Cyclic selection did not preserve all event identities')
    if len(set(selected)) != rows:
        raise ValueError('Duplicate delay/event rows in pooled selection')
    return selected


def save_cache(path, payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    blob = buffer.getvalue()
    path.write_bytes(gzip.compress(blob, compresslevel=1, mtime=0))
    return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                uncompressed_sha256=hashlib.sha256(blob).hexdigest(),
                bytes=path.stat().st_size, uncompressed_bytes=len(blob))


def feature_batch(model, seed, size, distractors, delays, batch_size, device):
    """Generate once, then capture every delay on exactly the same events."""
    batch = make_batch(seed, size, distractors=distractors)
    chunks = {delay: [] for delay in delays}
    predictions = {delay: {field: [] for field in FIELDS} for delay in delays}
    with torch.no_grad():
        for start in range(0, size, batch_size):
            def section(tree):
                return {key: section(value) if isinstance(value, dict)
                        else value[start:start + batch_size] for key, value in tree.items()}
            public = move(section(batch['public']), device)
            states = []
            hook = model.norm.register_forward_pre_hook(lambda module, args: states.append(args[0].detach().clone()))
            try:
                snapshots, _ = capture(model, public, 'factorized', 'persistent', tuple(delays))
            finally:
                hook.remove()
            if len(states) != len(delays) or list(delays) != sorted(delays):
                raise ValueError('Unexpected historical capture boundary order')
            for delay, state in zip(delays, states):
                chunks[delay].append(snapshots[f'workspace_{delay}'].cpu())
                decoded = model.decode(state, public)
                for field in FIELDS:
                    predictions[delay][field].append(decoded[field].argmax(-1).cpu())
    return dict(features={delay: torch.cat(parts) for delay, parts in chunks.items()},
                labels=batch['targets']['value'], targets={field: batch['targets'][field] for field in FIELDS},
                original_predictions={delay: {field: torch.cat(parts) for field, parts in fields.items()}
                                      for delay, fields in predictions.items()},
                event_row_hashes=[tensor_hash({key: value[index] for key, value in batch['public']['event'].items()})
                                  for index in range(size)],
                data_seed=seed, event_indices=list(range(size)),
                event_sha256=tensor_hash(batch['public']['event']),
                public_sha256=tensor_hash(batch['public']), distractors=distractors)


def select_features(batch, delays, rows, device):
    pairs = pooled_indices(len(batch['labels']), delays, rows)
    x = torch.stack([batch['features'][delay][index] for delay, index in pairs]).to(device)
    y = batch['labels'][torch.tensor([index for _, index in pairs])].to(device)
    return x, y, pairs


def validation_score(cache, fit, delays, distractors, device):
    count = 0
    total = 0
    cells = []
    for distractor in distractors:
        batch = cache[f'calibration/{distractor}']
        for delay in delays:
            pred = predict_ridge(batch['features'][delay].to(device), fit).argmax(-1).cpu()
            score = int((pred == batch['labels']).sum())
            cells.append(dict(delay=delay, distractors=distractor, correct=score, total=len(pred)))
            count += score
            total += len(pred)
    return count, total, cells



def prediction_record(seed, head, split, delay, distractor, batch, scalar):
    predictions = dict(batch['original_predictions'][delay], value=scalar)
    matches = {field: predictions[field] == batch['targets'][field] for field in FIELDS}
    counts = {field: dict(correct=int(correct.sum()), total=len(correct))
              for field, correct in matches.items()}
    counts['joint'] = dict(correct=int(torch.stack(list(matches.values())).all(0).sum()), total=len(scalar))
    counts['nonvalue_joint'] = dict(correct=int(torch.stack([matches[field] for field in FIELDS[1:]]).all(0).sum()), total=len(scalar))
    required = batch['targets']['argument1'] != 6
    counts['argument1_required'] = dict(correct=int((matches['argument1'] & required).sum()), total=int(required.sum()))
    return dict(seed=seed, head=head, split=split, target_delay=delay, distractors=distractor,
                event_sha256=batch['event_sha256'],
                predictions={field: value.tolist() for field, value in predictions.items()},
                targets={field: value.tolist() for field, value in batch['targets'].items()},
                counts=counts, metrics=metrics(scalar, batch['labels']))


def run(config, output):
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = config['device']
    seeds = [config['data'][split]['seed'] for split in ('train', 'calibration', 'test')]
    if len(set(seeds)) != 3:
        raise ValueError('Data partitions share seeds')
    original = json.loads((Path(config['checkpoint_dir']) / 'manifest.json').read_text())
    source = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('return_crossdelay.py', 'return_diagnostics_probe.py', 'return_memory.py',
                           'return_memory_study.py', 'retention_data.py', 'thinking.py')}
    manifest = dict(config=config, source=source,
                    config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     device=torch.cuda.get_device_name(),
                                     total_device_bytes=torch.cuda.get_device_properties(0).total_memory), runs=[])
    for seed in config['backbone_seeds']:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        path = Path(config['checkpoint_dir']) / f'{seed}-backbone.pt'
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        historical = next(row for row in original['runs'] if row['seed'] == seed)
        if checkpoint_hash != historical['checkpoint_hashes'][path.name]:
            raise ValueError('Frozen checkpoint hash mismatch')
        model = ReturnMemoryModel(width=1024, encoding='factorized').to(device).eval()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        cache = {}
        for split in ('train', 'calibration', 'test'):
            spec = config['data'][split]
            for distractor in ([2] if split == 'train' else config['eval_distractors']):
                cache[f'{split}/{distractor}'] = feature_batch(
                    model, spec['seed'], spec['size'], distractor, config['delays'],
                    config['batch_size'], device)
        hashes = [cache[f'{split}/2']['event_sha256'] for split in ('train', 'calibration', 'test')]
        if len(set(hashes)) != 3:
            raise ValueError('Data partition events overlap')
        row_sets = [set(cache[f'{split}/2']['event_row_hashes']) for split in ('train', 'calibration', 'test')]
        if any(left & right for i, left in enumerate(row_sets) for right in row_sets[i+1:]):
            raise ValueError('Individual events overlap across data partitions')
        for split in ('calibration', 'test'):
            for distractor in config['eval_distractors']:
                batch = cache[f'{split}/{distractor}']
                if batch['event_sha256'] != cache[f'{split}/2']['event_sha256']:
                    raise ValueError('Distractor conditions changed underlying events')
                torch.testing.assert_close(batch['labels'], cache[f'{split}/2']['labels'])
                if batch['event_row_hashes'] != cache[f'{split}/2']['event_row_hashes']:
                    raise ValueError('Individual event ordering differs across distractor conditions')
        cache_meta = save_cache(output / f'{seed}-features.pt.gz', cache)
        heads = [(f'delay_{delay}', [delay], config['data']['train']['size'])
                 for delay in config['source_delays']]
        heads.extend([(name, delays, config['pooled_rows'])
                      for name, delays in config.get('pools', {}).items()])
        fit_records = []
        prediction_path = output / f'{seed}-predictions.jsonl.gz'
        with gzip.open(prediction_path, 'wt') as stream:
            # The unchanged classifier shares the exact normalized workspace inputs.
            for split in ('calibration', 'test'):
                for distractor in config['eval_distractors']:
                    batch = cache[f'{split}/{distractor}']
                    for delay in config['delays']:
                        pred = batch['original_predictions'][delay]['value']
                        stream.write(json.dumps(prediction_record(seed, 'original', split, delay, distractor, batch, pred)) + '\n')
            for name, fit_delays, row_count in heads:
                x, y, pairs = select_features(cache['train/2'], fit_delays, row_count, device)
                candidates = []
                for alpha in config['ridge_grid']:
                    fit = fit_ridge(x, F.one_hot(y, 33).float(), alpha)
                    correct, total, cells = validation_score(cache, fit, fit_delays,
                                                              config['eval_distractors'], device)
                    candidates.append((correct, alpha, fit, total, cells))
                # Stable first maximum implements the prespecified grid-index tie rule.
                chosen = max(candidates, key=lambda item: item[0])
                _, alpha, fit, _, _ = chosen
                fit_path = output / f'{seed}-{name}-fit.pt'
                torch.save(tuple(value.cpu() for value in fit), fit_path)
                record = dict(head=name, fit_delays=fit_delays, rows=row_count,
                    unique_events=len({index for _, index in pairs}),
                    per_delay_rows={str(delay): sum(d == delay for d, _ in pairs) for delay in fit_delays},
                    selection_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                    coefficients=int(fit[2].numel()), input_dimensions=1024,
                    label_support=sorted(y.unique().tolist()), chosen_alpha=alpha,
                    validation_grid=[dict(alpha=a, correct=c, total=n, cells=cells)
                                     for c, a, _, n, cells in candidates],
                    fit_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest())
                fit_records.append(record)
                for split in ('train', 'calibration', 'test'):
                    for distractor in ([2] if split == 'train' else config['eval_distractors']):
                        batch = cache[f'{split}/{distractor}']
                        for delay in config['delays']:
                            pred = predict_ridge(batch['features'][delay].to(device), fit).argmax(-1).cpu()
                            record = prediction_record(seed, name, split, delay, distractor, batch, pred)
                            record['fit_delays'] = fit_delays
                            stream.write(json.dumps(record) + '\n')
        meta = dict(seed=seed, checkpoint_sha256=checkpoint_hash, width=1024,
                    backbone_parameters=parameters, memory_tokens=6, memory_coordinates=6144,
                    backbone_optimizer_presentations=0, ridge_fit_records=fit_records, feature_cache=cache_meta,
                    event_hashes={key: value['event_sha256'] for key, value in cache.items()},
                    public_hashes={key: value['public_sha256'] for key, value in cache.items()},
                    prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                    seconds=time.monotonic() - started,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        manifest['runs'].append(meta)
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(seed, meta['seconds'], flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output)
