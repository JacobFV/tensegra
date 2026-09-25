"""Stage7 independent evidence timing. No runtime/proposal execution dependency.

Public certified keyed shares are strong supplied representational priors. A
current public snapshot retains all arrived evidence; this tests sufficient-
evidence timing, not memory acquisition. Private labels never enter the actor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from .thinking import ThinkingConfig, ThinkingModel

HORIZON = 8
FEATURES = 13
CONTROLS = ('learned', 'minimum', 'fixed', 'oracle', 'soft_timing_bias', 'learned_soft_bias')


def generate_episodes(n: int, seed: int, split: str, *, missing_share_seed: int | None = None):
    """Optional missing-share seed supports causally paired hidden-value controls."""
    if n < 1 or split not in ('train', 'validation', 'test'):
        raise ValueError('positive n and train/validation/test split required')
    rng = random.Random(seed)
    hidden_rng = random.Random(seed + 1000003 if missing_share_seed is None else missing_share_seed)
    offset = {'train': 0, 'validation': 32, 'test': 64}[split]
    arrivals = (2, 4, 6) if split == 'train' else (3, 5, 7)
    context = torch.zeros(n, HORIZON, 1, FEATURES)
    memory = torch.zeros(n, HORIZON, 4, FEATURES)
    mask = torch.ones(n, HORIZON, 4, dtype=torch.bool)
    answers, stops, rejects, names = [], [], [], []
    for i in range(n):
        name = offset + rng.randrange(32)
        names.append(name)
        key = [float((name >> b) & 1) for b in range(7)]
        other = [float(((name ^ 1) >> b) & 1) for b in range(7)]
        reject = i % 4 == 3
        arrival = HORIZON if reject else arrivals[(i // 4 * 3 + i % 4) % 3]
        a, b = rng.randrange(2), hidden_rng.randrange(2)
        # role bits 7:9; source certification 9; value one-hot 10:12.
        def row(k, role, certified, value):
            return torch.tensor(k + [float(role == 0), float(role == 1),
                                    float(certified), float(value == 0), float(value == 1), 0.])
        rows = torch.stack((row(key, 0, True, a), row(key, 1, True, b),
                            row(other, 1, False, rng.randrange(2)), row(other, 1, True, rng.randrange(2))))
        order = list(range(4))
        rng.shuffle(order)
        missing = order.index(1)
        context[i, :, 0, :7] = torch.tensor(key)
        context[i, -1, 0, -1] = 1.  # public deadline, including solvable episodes
        for t in range(HORIZON):
            memory[i, t] = rows[order]
            if reject or t + 1 < arrival:
                memory[i, t, missing] = row(other, 1, True, rng.randrange(2))
                memory[i, t, order.index(2), :7] = torch.tensor(key)
        answers.append(2 if reject else a ^ b)
        stops.append(arrival)
        rejects.append(reject)
    return dict(public=dict(context=context, memory=memory, mask=mask),
                answer=torch.tensor(answers), arrival=torch.tensor(stops),
                reject=torch.tensor(rejects), names=names)


def rule_predictions(public):
    """Exact public-evidence XOR reference; no learned model or runtime calls.

    This supplied rule is an observability/task upper bound, not a parameter-
    matched learned control. Observation steps are not neural microsteps.
    """
    records = []
    for i in range(public['context'].shape[0]):
        for t in range(HORIZON):
            key = public['context'][i, t, 0, :7]
            rows = public['memory'][i, t]
            valid = ((rows[:, :7] == key).all(-1) & (rows[:, 9] == 1)
                     & public['mask'][i, t])
            values = [rows[valid & (rows[:, 7+role] == 1), 11] for role in (0, 1)]
            if all(len(v) == 1 for v in values):
                prediction = int(values[0].item()) ^ int(values[1].item())
            elif public['context'][i, t, 0, -1] == 1:
                prediction = 2
            else:
                continue
            records.append(dict(index=i, stop=t+1, prediction=prediction, model_updates=0))
            break
    return records


def rule_summary(data):
    records = rule_predictions(data['public'])
    n = len(data['answer'])
    correct = sum(r['prediction'] == int(data['answer'][r['index']]) for r in records)
    exact = sum(r['stop'] == int(data['arrival'][r['index']]) for r in records)
    reject = [r for r in records if data['reject'][r['index']]]
    return dict(n=n, task_accuracy=correct/n, halt_exact_accuracy=exact/n,
        reject_n=len(reject), reject_correct=sum(r['prediction'] == 2 for r in reject),
        mean_observation_steps=sum(r['stop'] for r in records)/n,
        neural_updates=0, parameter_matched=False, records=records)


def make_model(width=16):
    return ThinkingModel(ThinkingConfig(feature_dim=FEATURES, width=width,
        heads=2, structural_heads=1, workspace_rows=4, candidates=1,
        output_classes=3, min_microsteps=1, max_microsteps=HORIZON))


def _step(model, state, public, t, indices=None):
    if indices is None:
        indices = slice(None)
    return model.step(state, public['context'][indices, t], public['memory'][indices, t],
                      memory_mask=public['mask'][indices, t], microstep=t+1)


def losses(model, data):
    public = data['public']
    state = model.initialize({'context': public['context'][:, 0]})
    task_terms, halt_terms = [], []
    for t in range(HORIZON):
        out = _step(model, state, public, t)
        state = out['workspace']
        ready = data['arrival'] <= t+1
        halt_terms.append(F.binary_cross_entropy_with_logits(out['emit_logits'], ready.float()))
        if ready.any():
            task_terms.append(F.cross_entropy(out['output_logits'][ready], data['answer'][ready]))
    return dict(task=torch.stack(task_terms).mean(), halt=torch.stack(halt_terms).mean())


@torch.no_grad()
def evaluate(model, data, control='learned'):
    if control not in CONTROLS:
        raise ValueError('unknown timing control')
    model.eval()
    public = data['public']
    active = torch.arange(len(data['answer']))
    state = model.initialize({'context': public['context'][:, 0]})
    records = []
    for t in range(HORIZON):
        out = _step(model, state, public, t, active)
        if control == 'minimum':
            stop = torch.ones(len(active), dtype=torch.bool)
        elif control == 'fixed':
            stop = torch.full((len(active),), t+1 == HORIZON)
        elif control == 'oracle':
            stop = data['arrival'][active] == t+1
        elif control == 'soft_timing_bias':
            # Same frozen model and observations; discard only learned halt logit.
            bias = -.5 * F.softplus(torch.tensor(2.-(t+1))) + .05 * F.softplus(torch.tensor((t+1)-6.))
            stop = torch.full((len(active),), bool(bias >= 0))
        elif control == 'learned_soft_bias':
            stop = out['emit']
        else:
            stop = out['emit_logits'] >= 0
        stop = stop | (t+1 == HORIZON)
        predicted = out['output_logits'].argmax(-1)
        for local in stop.nonzero().flatten().tolist():
            i = int(active[local])
            arrival, target, prediction = int(data['arrival'][i]), int(data['answer'][i]), int(predicted[local])
            records.append(dict(index=i, name=data['names'][i], arrival=arrival, stop=t+1,
                target=target, prediction=prediction, reject=bool(data['reject'][i]),
                correct=prediction == target, exact=t+1 == arrival, premature=t+1 < arrival,
                extra=max(0, t+1-arrival), updates=t+1, forced_deadline=bool(t+1 == HORIZON)))
        active, state = active[~stop], out['workspace'][~stop]
        if not len(active):
            break
    return sorted(records, key=lambda x: x['index'])


def summarize(records):
    n = len(records)
    if not n:
        raise ValueError('empty evaluation')
    mean = lambda key: sum(r[key] for r in records)/n
    arrival_accuracy = {}
    histograms = {}
    for arrival in sorted({r['arrival'] for r in records}):
        subset = [r for r in records if r['arrival'] == arrival]
        histograms[str(arrival)] = dict(Counter(str(r['stop']) for r in subset))
        if any(not r['reject'] for r in subset):
            arrival_accuracy[str(arrival)] = sum(r['exact'] and r['correct'] for r in subset)/len(subset)
    rejects = [r for r in records if r['reject']]
    return dict(n=n, task_accuracy=mean('correct'), halt_exact_accuracy=mean('exact'),
        stop_reject_accuracy=sum(r['exact'] and r['correct'] for r in records)/n,
        premature_rate=mean('premature'), extra_steps=mean('extra'), mean_updates=mean('updates'),
        reject_n=len(rejects), reject_accuracy=(sum(r['correct'] and r['exact'] for r in rejects)/len(rejects) if rejects else None),
        arrival_accuracy=arrival_accuracy, arrival_n=dict(Counter(str(r['arrival']) for r in records)),
        microsteps_by_arrival=histograms,
        cost_scores={str(c): mean('correct')-c*mean('updates') for c in (0., .01, .05, .1)})


def gate(seed_metrics):
    checks = []
    for metrics in seed_metrics:
        learned, minimum = metrics['learned'], metrics['minimum']
        checks.append(dict(stop_reject=learned['stop_reject_accuracy'] > .95,
            premature=learned['premature_rate'] < .01,
            rejection=learned.get('reject_accuracy') is not None and learned['reject_accuracy'] > .95,
            variable_arrivals=sum(v > .95 for v in learned['arrival_accuracy'].values()) >= 3,
            task_advantage=learned['task_accuracy']-minimum['task_accuracy'] > .20))
    passed = bool(checks) and all(all(c.values()) for c in checks)
    return dict(passed=passed, validation_seed_checks=checks, composition_allowed=False, halting_competence=passed,
                reason='all validation seeds must pass; no dependent recomposition on failure')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, output):
    """Explicit runner invocation authorizes training; importing never trains."""
    torch.set_num_threads(min(2, int(config.get('threads', 2))))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(config, indent=2, sort_keys=True))
    manifest = dict(config_sha256=_hash(output/'config.json'), torch_version=torch.__version__,
        source_sha256={p.name: _hash(p) for p in (Path(__file__), Path(__file__).with_name('thinking.py'))},
        controls=list(CONTROLS), public_priors='certified keyed shares, persistent current evidence, observable deadline',
        main_budget_frozen=bool(config.get('main_budget_frozen', False)), seeds=[])
    validation_metrics = []
    for seed in config['seeds']:
        start = time.monotonic()
        torch.manual_seed(seed)
        model = make_model(config.get('width', 16))
        seed_dir = output/f'seed-{seed}'
        seed_dir.mkdir()
        datasets = {split: generate_episodes(config['train_examples'] if split == 'train' else config['eval_examples'],
                    seed*1000+{'train': 11, 'validation': 22, 'test': 33}[split], split)
                    for split in ('train', 'validation', 'test')}
        torch.save(datasets, seed_dir/'data.pt')
        torch.save(model.state_dict(), seed_dir/'initial.pt')
        curves = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        generator = torch.Generator().manual_seed(seed+917)
        for step in range(config['steps']+1):
            if step % config['eval_every'] == 0 or step == config['steps']:
                metrics = {}
                for split in (('validation', 'test') if manifest['main_budget_frozen'] and step == config['steps'] else ('validation',)):
                    metrics[split] = {}
                    (seed_dir/f'{split}-exact-rule-{step}.json').write_text(json.dumps(rule_summary(datasets[split])))
                    for control in CONTROLS:
                        records = evaluate(model, datasets[split], control)
                        metrics[split][control] = summarize(records)
                        (seed_dir/f'{split}-{control}-{step}.json').write_text(json.dumps(records))
                curves.append(dict(step=step, examples_seen=step*config['batch_size'], metrics=metrics))
            if step == config['steps']:
                break
            indices = torch.randint(config['train_examples'], (config['batch_size'],), generator=generator)
            full = datasets['train']
            batch = dict(public={k: v[indices] for k,v in full['public'].items()},
                         answer=full['answer'][indices], arrival=full['arrival'][indices])
            model.train()
            optimizer.zero_grad()
            terms = losses(model, batch)
            sum(terms.values()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            with (seed_dir/'losses.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step+1, **{k: float(v.detach()) for k,v in terms.items()}))+'\n')
        validation_metrics.append(curves[-1]['metrics']['validation'])
        torch.save(model.state_dict(), seed_dir/'checkpoint.pt')
        (seed_dir/'curves.json').write_text(json.dumps(curves, indent=2))
        manifest['seeds'].append(dict(seed=seed, parameters=sum(p.numel() for p in model.parameters()),
            elapsed_seconds=time.monotonic()-start, examples_seen=config['steps']*config['batch_size'],
            optimizer_steps=config['steps'], hashes={p.name: _hash(p) for p in seed_dir.glob('*.pt')}))
    result = gate(validation_metrics)
    result['scope'] = 'main' if manifest['main_budget_frozen'] else 'acquisition_only_not_main_claim'
    (output/'gate.json').write_text(json.dumps(result, indent=2))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text()), args.output), indent=2))


if __name__ == '__main__':
    main()
