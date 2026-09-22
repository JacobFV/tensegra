"""Paired, bounded synthetic studies of programmable attention geometry."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch

from .data import dynamics_step, trajectories, windows
from .graphs import corrupt_graph
from .study_data import deterministic_future, graph_quality, make_system, supply_graph
from .study_model import StudyPredictor


@dataclass
class StudyConfig:
    suites: list[str] = field(default_factory=lambda: ['efficiency', 'efficiency_identity', 'corruption', 'transfer', 'heterogeneous', 'learned'])
    domains: list[str] = field(default_factory=lambda: ['sparse', 'robot'])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    counts: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256])
    checkpoints: list[int] = field(default_factory=lambda: [0, 25, 50, 100, 200, 300, 600])
    modes: list[str] | None = None
    nodes: int = 12
    history: int = 4
    width: int = 32
    heads: int = 4
    layers: int = 2
    batch_size: int = 32
    learning_rate: float = .001
    train_count: int = 128
    validation_count: int = 16
    test_count: int = 16
    steps: int = 40
    horizon: int = 10
    noise: float = .01
    burn_in: int = 32
    degree: int = 3
    transfer_train_graphs: int = 16
    transfer_validation_graphs: int = 4
    transfer_test_graphs: int = 8
    transfer_test_sizes: list[int] = field(default_factory=lambda: [12, 32, 64, 128])
    mixed_sizes: list[int] = field(default_factory=lambda: [9, 12, 16])
    eval_batch_size: int = 32
    eval_windows: int = 256
    threads: int = 2
    wall_seconds: float = 7200

    def __post_init__(self):
        valid_suites = {'efficiency', 'efficiency_identity', 'corruption', 'transfer', 'heterogeneous', 'learned'}
        valid_modes = {'none', 'soft1', 'soft4', 'hard', 'permuted1', 'permuted4', 'graph_input', 'learned', 'typed'}
        if not self.suites or len(set(self.suites)) != len(self.suites) or set(self.suites) - valid_suites:
            raise ValueError('unknown or empty suites')
        if not self.domains or len(set(self.domains)) != len(self.domains) or set(self.domains) - {'sparse', 'robot'}:
            raise ValueError('unknown or empty domains')
        if self.modes is not None and (not self.modes or len(set(self.modes)) != len(self.modes) or set(self.modes) - valid_modes):
            raise ValueError('unknown, repeated or empty modes')
        if self.modes and 'typed' in self.modes and set(self.suites) != {'heterogeneous'}:
            raise ValueError('typed mode is restricted to the signed heterogeneous suite')
        for name in ('nodes', 'history', 'width', 'heads', 'layers', 'batch_size', 'train_count',
                     'validation_count', 'test_count', 'steps', 'horizon', 'degree',
                     'transfer_train_graphs', 'transfer_validation_graphs', 'transfer_test_graphs',
                     'eval_batch_size', 'eval_windows', 'threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('counts', 'mixed_sizes', 'transfer_test_sizes'):
            values = getattr(self, name)
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
                raise ValueError(f'{name} must contain positive integers')
            if values != sorted(set(values)):
                raise ValueError(f'{name} must be distinct and increasing')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError('seeds must be distinct nonnegative integers')
        if any(isinstance(s, bool) or not isinstance(s, int) for s in self.checkpoints):
            raise ValueError('checkpoints must be integers')
        if self.checkpoints != sorted(set(self.checkpoints)) or not self.checkpoints or self.checkpoints[0] != 0 or self.checkpoints[-1] <= 0:
            raise ValueError('checkpoints must increase from zero to a positive training budget')
        if self.width % self.heads or self.steps < self.history + self.horizon:
            raise ValueError('width must be divisible by heads and series must fit history plus horizon')
        if self.eval_batch_size > 64 or self.threads > 2:
            raise ValueError('evaluation batch <=64 and threads <=2 are required')
        if isinstance(self.burn_in, bool) or not isinstance(self.burn_in, int) or self.burn_in < 0:
            raise ValueError('burn_in must be a nonnegative integer')
        for name in ('learning_rate', 'wall_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(self.noise, bool) or not isinstance(self.noise, (int, float)) or not math.isfinite(self.noise) or self.noise < 0:
            raise ValueError('noise must be finite and nonnegative')


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(bytes(value.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def first_crossing(curve, threshold):
    return next((point['step'] for point in curve if point['normalized_mse'] <= threshold), None)


def _seed(seed, split, graph=0):
    return 1_000_000 * (seed + 1) + 10_000 * split + graph


def _record(config, domain, mechanism, nodes, seed, split, index, count):
    graph_seed = _seed(seed, split, index)
    system = make_system(domain, nodes, graph_seed, mechanism=mechanism, degree=config.degree)
    trajectory_seed = graph_seed + 100_000_000
    series = trajectories(system, count=count, steps=config.steps, seed=trajectory_seed,
                          noise=config.noise, burn_in=config.burn_in)
    return {'system': system, 'series': series, 'graph_seed': graph_seed,
            'trajectory_seed': trajectory_seed, 'graph_hash': tensor_hash(system.read_graph),
            'weights_hash': tensor_hash(system.weights), 'nodes': nodes}


def _dataset(config, domain, mechanism, seed, train_sizes=None):
    """Graph splits are disjoint for transfer; fixed-process suites split trajectories."""
    if train_sizes is None:
        count = max(config.counts) if mechanism == 'uniform' else config.train_count
        count = max(count, config.train_count)
        train = [_record(config, domain, mechanism, config.nodes, seed, 1, 0, count)]
        validation, test = [], []
        for split, count, destination in ((2, config.validation_count, validation), (3, config.test_count, test)):
            item = dict(train[0])
            item['trajectory_seed'] = _seed(seed, split) + 100_000_000
            item['series'] = trajectories(item['system'], count=count, steps=config.steps,
                                           seed=item['trajectory_seed'], noise=config.noise,
                                           burn_in=config.burn_in)
            destination.append(item)
    else:
        train = [_record(config, domain, mechanism, train_sizes[i % len(train_sizes)], seed,
                         1, i, config.train_count) for i in range(config.transfer_train_graphs)]
        validation = [_record(config, domain, mechanism, train_sizes[i % len(train_sizes)], seed,
                              2, i, config.validation_count) for i in range(config.transfer_validation_graphs)]
        test = [_record(config, domain, mechanism, nodes, seed, 3, size_index * 100 + i,
                        config.test_count)
                for size_index, nodes in enumerate(config.transfer_test_sizes)
                for i in range(config.transfer_test_graphs)]
        hashes = [{record['graph_hash'] for record in split} for split in (train, validation, test)]
        if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('graph identity collision between transfer splits')
    normalization_count = min(config.counts) if train_sizes is None else config.train_count
    values = torch.cat([item['series'][:normalization_count].flatten() for item in train])
    normalization = {'mean': values.mean().item(), 'std': values.std(unbiased=False).clamp_min(1e-8).item()}
    return train, validation, test, normalization


def _prepared(records, normalization, history, count=None, eval_windows=None):
    result = []
    mean, std = normalization['mean'], normalization['std']
    for item in records:
        series = item['series'] if count is None else item['series'][:count]
        x, y = windows((series - mean) / std, history)
        if eval_windows is not None and len(x) > eval_windows:
            # Equally spaced trajectory-local windows: deterministic and identical across models.
            indices = torch.linspace(0, len(x) - 1, eval_windows).long()
            x, y = x[indices], y[indices]
        result.append({**item, 'x': x, 'y': y, 'used_trajectories': len(series)})
    return result


def _graph(record, mode, corruption, fraction, seed):
    graph = supply_graph(record['system'].read_graph, corruption, fraction,
                         seed=seed + record['graph_seed'])
    if mode.startswith('permuted'):
        graph = corrupt_graph(graph, mode='permuted', seed=seed + record['graph_seed'] + 99)
    return graph


def _model(config, mode, seed, common=None, node_identity=False):
    variant = 'soft' if mode.startswith(('soft', 'permuted')) else mode
    strength = 1.0 if mode.endswith('1') else 4.0
    torch.manual_seed(seed)
    model = StudyPredictor(config.history, config.width, config.heads, config.layers,
                           variant=variant, strength=strength,
                           node_count=config.nodes if node_identity else None)
    if common is not None:
        current = model.state_dict()
        current.update({key: value for key, value in common.items()
                        if key in current and current[key].shape == value.shape})
        model.load_state_dict(current)
    return model


def _schedule(config, train, seed):
    generator = torch.Generator().manual_seed(seed)
    graphs = torch.randint(len(train), (config.checkpoints[-1],), generator=generator)
    indices = torch.stack([torch.randint(len(train[g]['x']), (config.batch_size,), generator=generator)
                           for g in graphs])
    return graphs, indices


def _prediction(model, x, graph, weights, batch_size):
    return torch.cat([model(chunk, graph, weights) for chunk in x.split(batch_size)])


def _validation(model, records, graphs, normalization, config):
    """Macro-average graphs; each graph loss averages evaluated windows and nodes."""
    losses, oracle, zero = [], [], []
    mean, std = normalization['mean'], normalization['std']
    with torch.no_grad():
        for item, graph in zip(records, graphs):
            prediction = _prediction(model, item['x'], graph, item['system'].weights, config.eval_batch_size)
            losses.append(torch.mean((prediction - item['y']) ** 2).item())
            expected = (dynamics_step(item['system'], item['x'][..., -1] * std + mean) - mean) / std
            oracle.append(torch.mean((expected - item['y']) ** 2).item())
            zero.append(torch.mean((item['y'] + mean / std) ** 2).item())
    return {'normalized_mse': sum(losses) / len(losses),
            'oracle_normalized_mse': sum(oracle) / len(oracle),
            'zero_normalized_mse': sum(zero) / len(zero)}


def _rollout(model, history, graph, weights, horizon, batch_size):
    output = []
    for chunk in history.split(batch_size):
        current, predictions = chunk, []
        for _ in range(horizon):
            prediction = model(current, graph, weights)
            predictions.append(prediction)
            current = torch.cat((current[..., 1:], prediction.unsqueeze(-1)), dim=-1)
        output.append(torch.stack(predictions, dim=1))
    return torch.cat(output)


def _evaluate(model, record, graph, normalization, config):
    mean, std = normalization['mean'], normalization['std']
    with torch.no_grad():
        predicted = _prediction(model, record['x'], graph, record['system'].weights, config.eval_batch_size)
        target = record['y'] * std + mean
        predicted = predicted * std + mean
        oracle = dynamics_step(record['system'], record['x'][..., -1] * std + mean)
        history = record['series'][:, :config.history].transpose(1, 2)
        deterministic = deterministic_future(record['system'], history, config.horizon)
        stochastic = record['series'][:, config.history:config.history + config.horizon]
        rollout = _rollout(model, (history - mean) / std, graph, record['system'].weights,
                           config.horizon, config.eval_batch_size) * std + mean
        metrics = {
            'one_step_mse': torch.mean((predicted - target) ** 2).item(),
            'oracle_one_step_mse': torch.mean((oracle - target) ** 2).item(),
            'zero_one_step_mse': torch.mean(target ** 2).item(),
            'stochastic_rollout_mse': torch.mean((rollout - stochastic) ** 2).item(),
            'deterministic_rollout_mse': torch.mean((rollout - deterministic) ** 2).item(),
            'oracle_stochastic_rollout_mse': torch.mean((deterministic - stochastic) ** 2).item(),
            'oracle_deterministic_rollout_mse': 0.0,
            'zero_stochastic_rollout_mse': torch.mean(stochastic ** 2).item(),
            'zero_deterministic_rollout_mse': torch.mean(deterministic ** 2).item(),
        }
    metrics.update({key.replace('_mse', '_normalized_mse'): value / std ** 2
                    for key, value in list(metrics.items())})
    return {**metrics, 'nodes': record['nodes'], 'graph_seed': record['graph_seed'],
            'graph_hash': record['graph_hash'], 'supplied_graph_hash': tensor_hash(graph),
            'trajectory_seed': record['trajectory_seed'], 'windows': len(record['x']),
            'rollout_trajectories': len(history), 'graph_quality': graph_quality(record['system'].read_graph, graph)}


def _corruptions():
    return [('clean', 0.0)] + [(kind, fraction) for kind in ('drop', 'add')
                              for fraction in (.1, .25, .5)] + [('mixed', .25)]


def _cases(config):
    core = ['none', 'soft1', 'soft4', 'hard', 'permuted1', 'permuted4']
    for suite in config.suites:
        domains = ['sparse'] if suite == 'transfer' else config.domains
        for domain in domains:
            for seed in config.seeds:
                base = {'suite': suite, 'domain': domain, 'seed': seed, 'mechanism': 'uniform',
                        'train_sizes': None, 'corruption': 'clean', 'fraction': 0.0,
                        'train_count': config.train_count}
                if suite in ('efficiency', 'efficiency_identity'):
                    modes = core if suite == 'efficiency' else ['none', 'soft4', 'hard']
                    for count in config.counts:
                        yield {**base, 'train_count': count, 'modes': config.modes or modes}
                elif suite == 'corruption':
                    for corruption, fraction in _corruptions():
                        yield {**base, 'corruption': corruption, 'fraction': fraction,
                               'modes': config.modes or ((['none'] if corruption == 'clean' else []) + ['soft1', 'soft4', 'hard'])}
                elif suite == 'transfer':
                    for sizes in ([config.nodes], config.mixed_sizes):
                        yield {**base, 'train_sizes': sizes,
                               'modes': config.modes or ['none', 'soft4', 'hard', 'permuted4', 'graph_input']}
                elif suite == 'heterogeneous':
                    yield {**base, 'mechanism': 'signed',
                           'modes': config.modes or ['none', 'soft1', 'soft4', 'hard', 'permuted4', 'graph_input', 'typed']}
                elif suite == 'learned':
                    yield {**base, 'modes': config.modes or ['none', 'soft4', 'hard', 'learned']}


def _split_provenance(train, validation, test):
    return {name: [{key: item[key] for key in ('nodes', 'graph_seed', 'trajectory_seed', 'graph_hash', 'weights_hash')}
                   for item in records]
            for name, records in [('train', train), ('validation', validation), ('test', test)]}


def _train(config, case, mode, train, validation, normalization, common, deadline):
    model = _model(config, mode, _seed(case['seed'], 7), common,
                   node_identity=case['suite'] == 'efficiency_identity')
    initial = state_hash(model.state_dict())
    shared = state_hash({key: model.state_dict()[key] for key in common})
    graph_indices, indices = _schedule(config, train, _seed(case['seed'], 8))
    train_graphs = [_graph(item, mode, case['corruption'], case['fraction'], case['seed']) for item in train]
    validation_graphs = [_graph(item, mode, case['corruption'], case['fraction'], case['seed']) for item in validation]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    curve, last_step = [], 0
    started = time.monotonic()

    def checkpoint(step, loss=None):
        model.eval()
        point = _validation(model, validation, validation_graphs, normalization, config)
        point.update(step=step, train_loss=loss, elapsed_seconds=time.monotonic() - started,
                     coefficients=model.structure_coefficients(), model_state_hash=state_hash(model.state_dict()))
        curve.append(point)
        model.train()

    checkpoint(0)
    for step, (graph_index, batch) in enumerate(zip(graph_indices, indices), 1):
        if time.monotonic() >= deadline:
            break
        item = train[graph_index]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(item['x'][batch], train_graphs[graph_index], item['system'].weights)
        loss = torch.mean((prediction - item['y'][batch]) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss: {case}, {mode}, step {step}')
        loss.backward()
        optimizer.step()
        last_step = step
        if step in config.checkpoints:
            checkpoint(step, loss.item())
    if curve[-1]['step'] != last_step:
        checkpoint(last_step, loss.item())
    first = curve[0]
    thresholds = {str(percent): first['oracle_normalized_mse'] + percent / 100 *
                  (first['zero_normalized_mse'] - first['oracle_normalized_mse']) for percent in (10, 25, 50)}
    if first['zero_normalized_mse'] <= first['oracle_normalized_mse']:
        thresholds = {key: None for key in thresholds}
    metadata = {'final_state_hash': state_hash(model.state_dict()), 'initialization_hash': initial, 'shared_initialization_hash': shared,
                'schedule_hash': tensor_hash(torch.cat((graph_indices[:, None], indices), dim=1)),
                'schedule_seed': _seed(case['seed'], 8), 'completed_steps': last_step,
                'optimizer_examples': last_step * config.batch_size,
                'validation_curve': curve, 'thresholds': thresholds,
                'threshold_steps': {key: first_crossing(curve, threshold) if threshold is not None else None for key, threshold in thresholds.items()},
                'parameters': sum(p.numel() for p in model.parameters()),
                'training_seconds': time.monotonic() - started,
                'training_graph_quality': [graph_quality(item['system'].read_graph, graph)
                                           for item, graph in zip(train, train_graphs)]}
    return model, metadata


def _provenance():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'source_revision': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'python': platform.python_version(), 'torch': torch.__version__,
            'platform': platform.platform(), 'device': 'cpu',
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def _write_summary(output, summary):
    temporary = output / 'summary.tmp'
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    temporary.replace(output / 'summary.json')


def run_study(config, output):
    """Run paired cases; write each completed/partial row before starting the next."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing nonempty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    started = time.monotonic()
    deadline = started + config.wall_seconds
    summary = {'schema_version': 1, 'config': asdict(config), 'provenance': _provenance(),
               'status': 'running', 'runs': []}
    _write_summary(output, summary)
    cached_key, dataset = None, None
    for case in _cases(config):
        if time.monotonic() >= deadline:
            break
        key = (case['domain'], case['mechanism'], case['seed'], tuple(case['train_sizes'] or []))
        if key != cached_key:
            dataset = _dataset(config, case['domain'], case['mechanism'], case['seed'], case['train_sizes'])
            cached_key = key
        raw_train, raw_validation, raw_test, normalization = dataset
        train = _prepared(raw_train, normalization, config.history, case['train_count'])
        validation = _prepared(raw_validation, normalization, config.history, eval_windows=config.eval_windows)
        test = _prepared(raw_test, normalization, config.history, eval_windows=config.eval_windows)
        common = _model(config, 'none', _seed(case['seed'], 7),
                        node_identity=case['suite'] == 'efficiency_identity').state_dict()
        for mode in case['modes']:
            if time.monotonic() >= deadline:
                break
            run_started = time.monotonic()
            print(json.dumps({'event': 'start', **case, 'mode': mode}), flush=True)
            model, metadata = _train(config, case, mode, train, validation, normalization, common, deadline)
            model.eval()
            evaluations = []
            conditions = [(case['corruption'], case['fraction'], 'test')]
            if case['suite'] == 'corruption' and case['corruption'] == 'clean':
                conditions += [(kind, fraction, 'runtime_corruption') for kind, fraction in _corruptions()[1:]]
            for corruption, fraction, split in conditions:
                for item in test:
                    if time.monotonic() >= deadline:
                        break
                    graph = _graph(item, mode, corruption, fraction, case['seed'])
                    evaluations.append({**_evaluate(model, item, graph, normalization, config),
                                        'split': split, 'corruption': corruption, 'fraction': fraction})
            complete = metadata['completed_steps'] == config.checkpoints[-1] and len(evaluations) == len(test) * len(conditions)
            row = {**{key: value for key, value in case.items() if key != 'modes'}, 'mode': mode,
                   'status': 'complete' if complete else 'partial',
                   'node_identity': case['suite'] == 'efficiency_identity', **metadata,
                   'normalization': normalization, 'normalization_trajectories_per_graph': min(config.counts) if case['train_sizes'] is None else config.train_count,
                   'information': 'signed_edge_weights' if mode == 'typed' else 'none' if mode == 'none' else 'adjacency',
                   'evaluations': evaluations,
                   'graph_splits': _split_provenance(train, validation, test),
                   'training_trajectories': sum(item['used_trajectories'] for item in train),
                   'training_windows': sum(len(item['x']) for item in train),
                   'elapsed_seconds': time.monotonic() - run_started,
                   'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            with (output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
            summary['runs'].append(row)
            summary['elapsed_seconds'] = time.monotonic() - started
            _write_summary(output, summary)
            print(json.dumps({'event': 'complete', 'suite': case['suite'], 'domain': case['domain'],
                              'seed': case['seed'], 'mode': mode, 'status': row['status'],
                              'completed_steps': row['completed_steps'],
                              'validation_mse': row['validation_curve'][-1]['normalized_mse'],
                              'elapsed_seconds': row['elapsed_seconds']}), flush=True)
    planned_runs = sum(len(case['modes']) for case in _cases(config))
    summary['planned_runs'] = planned_runs
    summary['status'] = 'complete' if len(summary['runs']) == planned_runs and all(row['status'] == 'complete' for row in summary['runs']) else 'partial'
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _write_summary(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--suite', choices=['efficiency', 'efficiency_identity', 'corruption', 'transfer', 'heterogeneous', 'learned'])
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text())
    if args.suite:
        values['suites'] = [args.suite]
    summary = run_study(StudyConfig(**values), args.output)
    print(json.dumps({'status': summary['status'], 'runs': len(summary['runs']),
                      'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
