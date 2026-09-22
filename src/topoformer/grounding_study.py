"""Reproducible task-only training and diagnostics for latent graph grounding.

Gold paths are accessed only in evaluation. The model receives immutable entity
memories, a start key, relation instructions and the currently supplied graph.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time

import torch
from torch.nn import functional as F


VARIANTS = {
    'soft': {}, 'soft_strength4': {'strength': 4.}, 'known': {'mode': 'known'}, 'frozen': {'mode': 'frozen'},
    'permuted': {'mode': 'permuted'}, 'none': {'mode': 'none'},
    'graph_input': {'mode': 'graph_input'}, 'hard': {'mode': 'hard'},
    'random_init': {'grounding_init': 'random'},
    'learned_temperature': {'learned_temperature': True},
    'shared_strength': {'shared_strength': True},
    'untyped': {'typed': False}, 'period4': {'projection_period': 4},
}


@dataclass
class GroundingStudyConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    steps: int = 400
    checkpoints: list[int] = field(default_factory=lambda: [0, 25, 50, 100, 200, 400])
    batch_size: int = 32
    eval_examples: int = 128
    eval_batch_size: int = 32
    nodes: int = 16
    key_dim: int = 16
    classes: int = 8
    relations: int = 3
    distractors: int = 4
    width: int = 32
    heads: int = 4
    max_train_depth: int = 4
    eval_depths: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    eval_sizes: list[int] = field(default_factory=lambda: [16, 32, 64])
    corruptions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    eval_distractors: list[int] = field(default_factory=lambda: [0, 16])
    learning_rate: float = 0.001
    temperature: float = 0.05
    strength: float = 8.0
    threads: int = 2
    wall_seconds: float = 7200

    def __post_init__(self):
        for name in ('steps', 'batch_size', 'eval_examples', 'eval_batch_size', 'nodes',
                     'key_dim', 'classes', 'relations', 'width', 'heads', 'max_train_depth', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads or self.key_dim > self.width:
            raise ValueError('width must fit keys and divide into heads')
        if self.threads > 2 or self.eval_batch_size > 64:
            raise ValueError('resource bounds require <=2 threads and <=64 evaluation batch')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('seeds must be distinct nonnegative integers')
        if not self.variants or len(set(self.variants)) != len(self.variants) or set(self.variants) - VARIANTS.keys():
            raise ValueError('unknown, repeated or empty variants')
        if self.checkpoints != sorted(set(self.checkpoints)) or not self.checkpoints or self.checkpoints[0] != 0 or self.checkpoints[-1] != self.steps:
            raise ValueError('checkpoints must increase from zero to steps')
        for name in ('eval_depths', 'eval_sizes'):
            if not getattr(self, name) or any(type(v) is not int or v <= 0 for v in getattr(self, name)):
                raise ValueError(f'{name} must contain positive integers')
        if type(self.distractors) is not int or self.distractors < 0 or any(type(v) is not int or v < 0 for v in self.eval_distractors):
            raise ValueError('distractors must be nonnegative integers')
        if any(not 0 <= v <= 1 for v in self.corruptions):
            raise ValueError('corruptions must lie in [0,1]')
        for name in ('learning_rate', 'temperature', 'wall_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.strength):
            raise ValueError('strength must be finite')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def conditions(config):
    """One-factor evaluation grid; no accidental depth × size Cartesian claim."""
    base = dict(depth=config.max_train_depth, nodes=config.nodes,
                distractors=config.distractors, composition='train', corruption=0.0)
    result = []
    def add(label, **changes):
        condition = {**base, **changes}
        if not any(x['settings'] == condition for x in result):
            result.append({'name': label, 'settings': condition})
    for depth in config.eval_depths:
        add(f'depth_{depth}', depth=depth)
    for nodes in config.eval_sizes:
        add(f'size_{nodes}', nodes=nodes)
    add('heldout_composition', composition='heldout')
    if max(config.eval_depths) >= 32 and max(config.eval_sizes) >= 64:
        add('deep_large', depth=32, nodes=64)
    for corruption in config.corruptions:
        add(f'corruption_{corruption:g}', corruption=corruption)
    for distractors in config.eval_distractors:
        add(f'distractors_{distractors}', distractors=distractors)
    return result


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((list(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()).hexdigest()


def state_hash(model):
    return fingerprint({k: tensor_hash(v) for k, v in model.state_dict().items()})


def source_identity():
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip()
    return {'commit': revision, 'dirty': bool(dirty), 'files': {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / 'src' / 'topoformer').glob('*.py'))}}


def make_data(config, seed, *, depth, nodes=None, distractors=None, composition='train', corruption=0., batch_size=None):
    from .traversal_data import make_batch
    return make_batch(batch_size=batch_size or config.batch_size, nodes=nodes or config.nodes,
                      depth=depth, relations=config.relations, key_dim=config.key_dim,
                      classes=config.classes, distractors=config.distractors if distractors is None else distractors,
                      seed=seed, composition=composition, corruption=corruption)


def batch_hash(batch):
    return fingerprint({key: tensor_hash(value) for key, value in batch.items() if isinstance(value, torch.Tensor)})


def model_inputs(batch):
    """Enforce diagnostic separation, even if a later model begins reading extra keys."""
    return {key: batch[key] for key in ('entity_keys', 'adjacency', 'node_ids', 'token_keys',
                                      'token_values', 'start_keys', 'relations')}


def build_model(config, variant, seed):
    from .traversal_model import TraversalTransformer
    options = dict(VARIANTS[variant])
    mode = options.pop('mode', 'soft')
    options['identity_init'] = options.pop('grounding_init', 'identity') == 'identity'
    torch.manual_seed(seed)
    model = TraversalTransformer(key_dim=config.key_dim, width=config.width, heads=config.heads,
                                 classes=config.classes, relations=config.relations,
                                 temperature=config.temperature, strength=options.pop('strength', config.strength), **options)
    return model, mode


@torch.no_grad()
def evaluate(model, mode, config, condition, seed):
    from .traversal_data import oracle_traverse
    from .traversal_oracle import exact_attention_traverse
    model.eval()
    settings = condition['settings']
    depth = settings['depth']
    sums = {key: 0. for key in ('task_accuracy', 'grounding_accuracy', 'grounding_entropy',
            'pre_step_next_node_mass', 'structural_next_node_mass', 'relation_attention_mass', 'clean_next_attention_mass', 'exact_path_completion', 'exact_attention_path_completion', 'exact_pre_step_grounding', 'null_mass',
            'key_grounding_accuracy', 'distractor_null_accuracy', 'oracle_task_accuracy',
            'mean_distinct_path_nodes', 'oracle_exact_path_completion', 'attention_oracle_task_accuracy', 'attention_oracle_exact_path_completion')}
    step_sums = [{key: 0. for key in ('grounding_accuracy', 'grounding_entropy', 'pre_step_next_node_mass',
                                       'relation_attention_mass', 'clean_next_attention_mass', 'null_mass')} for _ in range(depth)]
    hashes, example = [], []
    started = time.perf_counter()
    for offset in range(0, config.eval_examples, config.eval_batch_size):
        count = min(config.eval_batch_size, config.eval_examples - offset)
        batch = make_data(config, seed + offset, batch_size=count, **settings)
        hashes.append(batch_hash(batch))
        sums['mean_distinct_path_nodes'] += sum(len(set(path)) for path in batch['path_nodes'].tolist())
        logits, diagnostic = model(model_inputs(batch), mode=mode, return_diagnostics=True)
        if len(diagnostic) != depth:
            raise ValueError('model diagnostics must contain exactly one entry per relation instruction')
        sums['task_accuracy'] += (logits.argmax(-1) == batch['targets']).sum().item()
        exact = torch.ones(count, dtype=torch.bool)
        exact_attention = torch.ones(count, dtype=torch.bool)
        exact_post = torch.ones(count, dtype=torch.bool)
        for step, record in enumerate(diagnostic):
            pq, pk = record['pq'][:, 0], record['pk']
            gold = batch['path_nodes'][:, step]
            nxt = batch['path_nodes'][:, step + 1]
            correct = pq.argmax(-1) == gold
            exact &= correct
            entropy = -(pq * pq.clamp_min(1e-12).log()).sum(-1)
            next_mass = pq.gather(-1, nxt[:, None]).squeeze(-1)
            attention = record['attention'].mean(1)[:, 0]
            next_attention = (attention * (batch['token_nodes'] == nxt[:, None])).sum(-1)
            attended_node = batch['token_nodes'].gather(1, attention.argmax(-1)[:, None]).squeeze(-1)
            exact_attention &= attended_node == nxt
            exact_post &= record['pq_after'][:, 0].argmax(-1) == nxt
            selected_adjacency = batch['adjacency'][torch.arange(count), batch['relations'][:, step]]
            supplied_next = selected_adjacency[torch.arange(count), gold].argmax(-1)
            pushed = torch.bmm(pq[:, None, :-1], selected_adjacency)[:, 0]
            structural_next_mass = pushed.gather(-1, nxt[:, None]).squeeze(-1)
            relation_mass = (attention * (batch['token_nodes'] == supplied_next[:, None])).sum(-1)
            values = dict(grounding_accuracy=correct.float(), grounding_entropy=entropy,
                          pre_step_next_node_mass=next_mass, structural_next_node_mass=structural_next_mass, relation_attention_mass=relation_mass, clean_next_attention_mass=next_attention, null_mass=pq[:, -1])
            for key, value in values.items():
                step_sums[step][key] += value.sum().item()
                sums[key] += value.sum().item() / depth
            token_nodes = batch['token_nodes']
            entity_mask = token_nodes >= 0
            entity_correct = (pk.argmax(-1) == token_nodes) & entity_mask
            sums['key_grounding_accuracy'] += (entity_correct.sum(-1) / entity_mask.sum(-1)).sum().item() / depth
            null_mask = ~entity_mask
            if null_mask.any():
                null_correct = (pk.argmax(-1) == pq.shape[-1] - 1) & null_mask
                sums['distractor_null_accuracy'] += (null_correct.sum(-1) / null_mask.sum(-1).clamp_min(1)).sum().item() / depth
            if offset == 0:
                example.append(dict(step=step, gold_node=int(gold[0]), next_node=int(nxt[0]),
                                    grounded_node=int(pq[0].argmax()), max_probability=float(pq[0].max()),
                                    entropy=float(entropy[0]), next_attention_mass=float(next_attention[0]),
                                    probabilities=pq[0].tolist()))
        sums['exact_pre_step_grounding'] += exact.sum().item()
        sums['exact_attention_path_completion'] += exact_attention.sum().item()
        sums['exact_path_completion'] += (exact & exact_post).sum().item()
        attention_oracle = exact_attention_traverse(model_inputs(batch), classes=config.classes)
        sums['attention_oracle_task_accuracy'] += (attention_oracle['predictions'] == batch['targets']).sum().item()
        sums['attention_oracle_exact_path_completion'] += (attention_oracle['path_nodes'] == batch['path_nodes']).all(-1).sum().item()
        oracle = oracle_traverse(batch['adjacency'], batch['start_nodes'], batch['relations'])
        destination = oracle[:, -1]
        destination_mask = batch['token_nodes'] == destination[:, None]
        oracle_answer = batch['token_values'].gather(1, destination_mask.long().argmax(-1)[:, None]).squeeze(-1)
        sums['oracle_task_accuracy'] += (oracle_answer == batch['targets']).sum().item()
        sums['oracle_exact_path_completion'] += (oracle == batch['path_nodes']).all(-1).sum().item()
    result = {key: value / config.eval_examples for key, value in sums.items()}
    if settings['distractors'] == 0:
        result['distractor_null_accuracy'] = None
    return dict(condition=condition['name'], **settings, examples=config.eval_examples,
                data_hash=fingerprint(hashes), **result,
                step_diagnostics=[dict(step=i, **{k: v / config.eval_examples for k, v in row.items()})
                                  for i, row in enumerate(step_sums)],
                example=example, seconds=time.perf_counter() - started)


def train_run(config, variant, seed, source):
    model, mode = build_model(config, variant, seed)
    initial_hashes = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    curve, schedule = [], []
    started = time.perf_counter()
    validation = {'name': 'validation', 'settings': dict(depth=config.max_train_depth, nodes=config.nodes,
                  distractors=config.distractors, composition='train', corruption=0.)}
    train_seconds = 0.
    for step in range(config.steps + 1):
        if step in config.checkpoints:
            score = evaluate(model, mode, config, validation, 20_000_000 + seed * 100_000)
            curve.append(dict(step=step, accuracy=score['task_accuracy'], grounding_accuracy=score['grounding_accuracy'],
                              exact_path_completion=score['exact_path_completion'], seconds=time.perf_counter()-started))
        if step == config.steps:
            break
        if time.perf_counter() - started > config.wall_seconds:
            raise TimeoutError(f'run {variant}/{seed} exceeded wall_seconds before completing training')
        batch_seed = 1_000_000 + seed * 100_000 + step
        depth = 1 + step % config.max_train_depth
        batch = make_data(config, batch_seed, depth=depth)
        schedule.append({'seed': batch_seed, 'depth': depth, 'data_hash': batch_hash(batch)})
        model.train()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(model_inputs(batch), mode=mode)
        loss = F.cross_entropy(logits, batch['targets'])
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite training loss in {variant}/{seed} step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        train_seconds += time.perf_counter() - tick
    evaluations = [evaluate(model, mode, config, condition, 40_000_000 + seed * 100_000)
                   for i, condition in enumerate(conditions(config))]
    temperatures = {name: float(module.temperature.detach()) for name, module in model.named_modules()
                    if hasattr(module, 'temperature') and isinstance(module.temperature, torch.Tensor) and module.temperature.numel() == 1}
    return dict(schema_version=1, variant=variant, seed=seed, config_hash=fingerprint(asdict(config)), source=source,
                training=dict(steps=config.steps, examples=config.steps * config.batch_size,
                              curve=curve, schedule_hash=fingerprint(schedule), initial_parameter_hashes=initial_hashes,
                              initial_state_hash=fingerprint(initial_hashes), final_state_hash=state_hash(model)),
                model=dict(parameters=sum(p.numel() for p in model.parameters()),
                           strengths=model.structure_coefficients(), temperatures=temperatures),
                resources=dict(seconds=time.perf_counter()-started, train_seconds=train_seconds,
                               peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                               device='cpu', threads=config.threads, torch_version=torch.__version__),
                evaluations=evaluations)


def run(config, output):
    torch.set_num_threads(config.threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = fingerprint(asdict(config))
    source = source_identity()
    config_path = output / 'config.json'
    if config_path.exists() and fingerprint(json.loads(config_path.read_text())) != config_hash:
        raise ValueError('output directory contains a different experiment configuration')
    config_path.write_text(json.dumps(asdict(config), indent=2) + '\n')
    path = output / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(row['config_hash'] != config_hash or row['source']['files'] != source['files'] for row in rows):
        raise ValueError('cannot resume metrics from a different configuration or source code')
    completed = {(row['variant'], row['seed']) for row in rows}
    if len(completed) != len(rows):
        raise ValueError('duplicate run records in metrics artifact')
    for seed in config.seeds:
        for variant in config.variants:
            if (variant, seed) in completed:
                continue
            print(json.dumps({'event': 'start', 'variant': variant, 'seed': seed}), flush=True)
            row = train_run(config, variant, seed, source)
            with path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
            rows.append(row)
            print(json.dumps({'event': 'complete', 'variant': variant, 'seed': seed,
                              'seconds': row['resources']['seconds'],
                              'validation_accuracy': row['training']['curve'][-1]['accuracy']}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(GroundingStudyConfig(**json.loads(Path(args.config).read_text())), args.output)


if __name__ == '__main__':
    main()
