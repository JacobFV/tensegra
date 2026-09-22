"""Stage 4: paired experiments on identity stability, separate from Stage 1–3."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .grounding_study import (GroundingStudyConfig, batch_hash, fingerprint, make_data,
                              model_inputs, source_identity, state_hash, tensor_hash)


VARIANTS = {
    'stage3_soft4': {'strength': 4.},
    'stage3_soft8': {'strength': 8.},
    'known': {'mode': 'known', 'strength': 8.},
    'graph_input_keyed': {'mode': 'graph_input', 'content_identity_bias': 8., 'strength': 16.},
    'cosine_mixed': {'matcher': 'cosine'},
    'cosine_attention_full': {'matcher': 'cosine', 'identity_update': 'attention'},
    'cosine_attention_identity': {'matcher': 'cosine', 'identity_update': 'attention', 'identity_only': True},
    'cosine_pointer_identity': {'matcher': 'cosine', 'identity_update': 'pointer', 'identity_only': True},
    'dot_attention_identity': {'identity_update': 'attention', 'identity_only': True},
}
for update in ('mixed', 'attention', 'pointer'):
    VARIANTS[f'random_cosine_{update}'] = dict(matcher='cosine', identity_update=update,
        identity_only=update != 'mixed', identity_init=False, null_init=0.)
VARIANTS['random_cosine_attention_nullprior'] = {
    **VARIANTS['random_cosine_attention'], 'null_init': .65}
for update in ('attention', 'pointer'):
    for name, weights in {'aux001': (.01, .1, .1), 'aux01': (.1, .1, .1),
                          'aux1': (1., .1, .1), 'ground1': (1., 0., 0.),
                          'nullcycle': (0., .1, .1)}.items():
        VARIANTS[f'random_cosine_{update}_{name}'] = {
            **VARIANTS[f'random_cosine_{update}'], 'loss_weights': list(weights)}


@dataclass
class BindingStudyConfig(GroundingStudyConfig):
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    eval_depths: list[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    eval_sizes: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    eval_batch_size: int = 16
    strength: float = 4.
    initial_matrix: bool = False
    save_checkpoints: bool = True
    corruptions: list[float] = field(default_factory=list)
    eval_distractors: list[int] = field(default_factory=list)
    include_extra_evaluations: bool = False

    def __post_init__(self):
        # Reuse resource/tensor validation without changing the Stage 3 registry.
        own = self.variants
        self.variants = ['soft']
        try:
            super().__post_init__()
        finally:
            self.variants = own
        if not own or len(set(own)) != len(own) or set(own) - VARIANTS.keys():
            raise ValueError('unknown, repeated or empty binding variants')
        if self.eval_batch_size > 16:
            raise ValueError('binding evaluation batch must be <=16')
        if self.width < self.key_dim + self.classes:
            raise ValueError('width must fit identity and value coordinates')


def conditions(config):
    return [dict(name=f'n{nodes}_d{depth}', settings=dict(nodes=nodes, depth=depth,
                distractors=config.distractors, composition='train', corruption=0.))
            for nodes in config.eval_sizes for depth in config.eval_depths]


def build_model(config, variant, seed):
    from .binding_model import BindingTransformer
    options = dict(matcher='dot', identity_update='mixed', identity_only=False, null_init=.65)
    options.update(VARIANTS[variant])
    mode = options.pop('mode', 'soft')
    weights = options.pop('loss_weights', (0., 0., 0.))
    torch.manual_seed(seed)
    model = BindingTransformer(key_dim=config.key_dim, width=config.width, heads=config.heads,
        classes=config.classes, relations=config.relations, temperature=config.temperature,
        strength=options.pop('strength', config.strength), **options)
    return model, mode, weights


@torch.no_grad()
def evaluate(model, mode, config, condition, seed):
    from .binding_metrics import binding_diagnostics, summarize_binding
    model.eval()
    records, hashes = [], []
    started = time.perf_counter()
    correct = 0
    for offset in range(0, config.eval_examples, config.eval_batch_size):
        count = min(config.eval_batch_size, config.eval_examples - offset)
        batch = make_data(config, seed + offset, batch_size=count, **condition['settings'])
        hashes.append(batch_hash(batch))
        logits, diags = model(model_inputs(batch), mode=mode, return_diagnostics=True)
        correct += int((logits.argmax(-1) == batch['targets']).sum())
        record = binding_diagnostics(diags, batch)
        record['per_example']['task_correct'] = logits.argmax(-1) == batch['targets']
        records.append(record)
    result = summarize_binding(records)
    # Compact raw example features only; aggregate statistics retain full precision.
    result['per_example'] = {name: [float(f'{v:.6g}') if isinstance(v, float) else v for v in values]
                             for name, values in result['per_example'].items()}
    return dict(condition=condition['name'], **condition['settings'], examples=config.eval_examples,
                data_hash=fingerprint(hashes), task_accuracy=correct / config.eval_examples,
                **result, seconds=time.perf_counter() - started)


def train_run(config, variant, seed, source, checkpoint_dir=None):
    from .binding_metrics import binding_losses
    model, mode, weights = build_model(config, variant, seed)
    initial_hashes = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    curve, schedule, losses = [], [], []
    started = time.perf_counter()
    validation = dict(name='validation', settings=dict(depth=config.max_train_depth, nodes=config.nodes,
                      distractors=config.distractors, composition='train', corruption=0.))
    grid = conditions(config)
    initial_conditions = grid if config.initial_matrix else [validation, max(grid, key=lambda c: c['settings']['nodes'] * c['settings']['depth'])]
    initial_evaluations = [evaluate(model, mode, config, c, 40_000_000 + seed * 100_000) for c in initial_conditions]
    train_seconds = 0.
    for step in range(config.steps + 1):
        if step in config.checkpoints:
            score = evaluate(model, mode, config, validation, 20_000_000 + seed * 100_000)
            curve.append(dict(step=step, accuracy=score['task_accuracy'], diagnostics=score,
                              losses=losses[-1] if losses else None, seconds=time.perf_counter()-started))
        if step == config.steps:
            break
        if time.perf_counter() - started > config.wall_seconds:
            raise TimeoutError(f'run {variant}/{seed} exceeded wall_seconds')
        batch_seed = 1_000_000 + seed * 100_000 + step
        depth = 1 + step % config.max_train_depth
        batch = make_data(config, batch_seed, depth=depth)
        schedule.append(dict(seed=batch_seed, depth=depth, data_hash=batch_hash(batch)))
        model.train()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits, diagnostic = model(model_inputs(batch), mode=mode, return_diagnostics=True)
        components = binding_losses(diagnostic, batch)
        task_loss = F.cross_entropy(logits, batch['targets'])
        loss = task_loss + sum(weight * components[name] for weight, name in zip(weights, ('ground', 'null', 'cycle')))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss in {variant}/{seed}/{step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        train_seconds += time.perf_counter() - tick
        losses.append(dict(step=step+1, task=float(task_loss.detach()), total=float(loss.detach()),
                           **{name: float(value.detach()) for name, value in components.items()}))
    evaluations = [evaluate(model, mode, config, c, 40_000_000 + seed * 100_000) for c in grid]
    benchmark_batch = model_inputs(make_data(config, 60_000_000 + seed, depth=config.max_train_depth,
                                             batch_size=config.eval_batch_size))
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(benchmark_batch, mode=mode)
        tick = time.perf_counter()
        for _ in range(10):
            model(benchmark_batch, mode=mode)
        inference_seconds = (time.perf_counter() - tick) / 10
    checkpoint = None
    if config.save_checkpoints and checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = f'{variant}-seed{seed}.pt'
        torch.save({'state_dict': model.state_dict(), 'config': asdict(config), 'variant': variant,
                    'seed': seed, 'source': source}, checkpoint_dir / checkpoint)
    return dict(schema_version=4, variant=variant, seed=seed, config_hash=fingerprint(asdict(config)), source=source,
        variant_options=VARIANTS[variant], loss_weights=dict(zip(('ground', 'null', 'cycle'), weights)),
        training=dict(steps=config.steps, examples=config.steps*config.batch_size, curve=curve, losses=losses,
                      schedule_hash=fingerprint(schedule), initial_parameter_hashes=initial_hashes,
                      initial_state_hash=fingerprint(initial_hashes), final_state_hash=state_hash(model)),
        model=dict(parameters=sum(p.numel() for p in model.parameters()), strengths=model.structure_coefficients()),
        resources=dict(seconds=time.perf_counter()-started, train_seconds=train_seconds,
                       peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       device='cpu', threads=config.threads, torch_version=torch.__version__,
                       inference_batch_seconds=inference_seconds, inference_batch_size=config.eval_batch_size,
                       inference_depth=config.max_train_depth, inference_repeats=10),
        checkpoint=checkpoint, initial_evaluations=initial_evaluations, evaluations=evaluations)


def run(config, output):
    torch.set_num_threads(config.threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash, source = fingerprint(asdict(config)), source_identity()
    config_path = output / 'config.json'
    if config_path.exists() and fingerprint(json.loads(config_path.read_text())) != config_hash:
        raise ValueError('output directory has a different configuration')
    config_path.write_text(json.dumps(asdict(config), indent=2)+'\n')
    path = output / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(row['config_hash'] != config_hash or row['source']['files'] != source['files'] for row in rows):
        raise ValueError('cannot resume from different configuration or source')
    completed = {(row['variant'], row['seed']) for row in rows}
    expected = {(variant, seed) for variant in config.variants for seed in config.seeds}
    if len(completed) != len(rows) or not completed <= expected:
        raise ValueError('duplicate or unexpected run records')
    for seed in config.seeds:
        for variant in config.variants:
            if (variant, seed) in completed:
                continue
            print(json.dumps(dict(event='start', variant=variant, seed=seed)), flush=True)
            row = train_run(config, variant, seed, source, output/'checkpoints')
            with path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            rows.append(row)
            print(json.dumps(dict(event='complete', variant=variant, seed=seed,
                                  seconds=row['resources']['seconds'],
                                  validation_accuracy=row['training']['curve'][-1]['accuracy'])), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(BindingStudyConfig(**json.loads(Path(args.config).read_text())), args.output)


if __name__ == '__main__':
    main()
