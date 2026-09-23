"""Standalone P01 learned dispatch acquisition; no return-consumer composition."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import torch
from .campaign_composition import make_lowering_batch, model_inputs, make_model, public_schema_error
from .interface_proposals import proposal_loss


def private_labels(labels):
    primitive = torch.tensor([op for op, _ in labels])
    return dict(primitive=primitive, targets=torch.tensor([p for _, p in labels]), binary=primitive != 3)


def canonical_prediction(out, public):
    op = out['primitive'].argmax(-1)
    raw = out['pointers'].argmax(-1)
    pointers = raw.clone()
    # Public null identity, predicted arity only. No labels accepted here.
    null = public['keys'].abs().sum(-1).argmin(-1)
    pointers[:, 2] = torch.where(op == 3, null, pointers[:, 2])
    return op, raw, pointers


def score(out, public, labels):
    op, raw, canonical = canonical_prediction(out, public)
    target = labels['targets']; truth = labels['primitive']; binary = truth != 3
    equal = canonical == target; learned = raw == target
    full = (op == truth) & equal.all(-1)
    noncomm = (truth == 1) | (truth == 4)
    rows = torch.arange(len(op)); types = public['operand_types']
    destination_valid = (types[rows, canonical[:, 0]] == -1) & (public['keys'][rows, canonical[:, 0]].abs().sum(-1) != 0)
    operand0_valid = types[rows, canonical[:, 1]] >= 0
    operand1_valid = (op == 3) | (types[rows, canonical[:, 2]] >= 0)
    schema_valid = destination_valid & operand0_valid & operand1_valid
    def count(mask, selected=None):
        selected = torch.ones_like(mask) if selected is None else selected
        return dict(correct=int((mask & selected).sum()), total=int(selected.sum()))
    return dict(full_semantic=count(full), predicted_public_schema_valid=count(schema_valid), primitive=count(op == truth), destination=count(learned[:, 0]),
                operand0=count(learned[:, 1]), required_operand1=count(learned[:, 2], binary),
                raw_null_pointer_diagnostic=count(learned[:, 2], ~binary),
                canonical_null_schema=count(equal[:, 2], ~binary),
                noncommutative_order=count(learned[:, 1:].all(-1), noncomm))


def controlled_rows(rows, kind, seed):
    """Interventions use only observable rows; targets are rederived separately."""
    result = copy.deepcopy(rows); rng = random.Random(seed)
    for row in result:
        if kind == 'record_order':
            perm = torch.tensor([2, 0, 3, 1])
            for k in ('instruction_destinations', 'instruction_arguments', 'instruction_cues'):
                row[k] = row[k][perm]
        elif kind == 'inventory_order':
            order = list(range(11)); rng.shuffle(order)
            row['keys'] = row['keys'][order]
            for k in ('names', 'values', 'roles'): row[k] = tuple(row[k][i] for i in order)
        elif kind == 'fresh_names':
            row['names'] = tuple(f'control_{seed}_{rng.getrandbits(96):024x}' for _ in row['names'])
        elif kind == 'reverse_roles':
            binary = row['instruction_cues'].argmax(-1) != 3
            row['instruction_arguments'][binary] = row['instruction_arguments'][binary][:, [1, 0]]
        elif kind == 'unrelated_instructions':
            for j, destination in enumerate(row['instruction_destinations']):
                if torch.equal(destination, row['query_destination']): continue
                op = rng.randrange(5)
                ids = rng.sample([i for i, role in enumerate(row['roles']) if role == 'operand'], 2)
                if op == 3: ids[1] = row['roles'].index('null')
                row['instruction_cues'][j] = torch.nn.functional.one_hot(torch.tensor(op), 5).float()
                row['instruction_arguments'][j] = row['keys'][ids]
        else: raise ValueError(kind)
    return result


def labels_from_public(rows):
    """Privileged exact-address evaluator only. Never called by actor/runtime."""
    labels = []
    for row in rows:
        assert public_schema_error(row) is None
        record = next(i for i, d in enumerate(row['instruction_destinations']) if torch.equal(d, row['query_destination']))
        roles = torch.cat((row['query_destination'][None], row['instruction_arguments'][record]))
        pointers = tuple(next(i for i, key in enumerate(row['keys']) if torch.equal(key, role)) for role in roles)
        labels.append((int(row['instruction_cues'][record].argmax()), pointers))
    return private_labels(labels)


def sliced(tree, idx, device):
    return {k: v[idx].to(device) for k, v in tree.items()}


def evaluate(model, public, labels, device, chunk):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels['primitive']), chunk):
            outputs.append({k: v.cpu() for k, v in model(sliced(public, slice(start, start + chunk), device)).items()})
    out = {k: torch.cat([p[k] for p in outputs]) for k in outputs[0]}
    return score(out, public, labels), out


def tensor_digest(public):
    digest = hashlib.sha256()
    for key, value in sorted(public.items()):
        value = value.contiguous().cpu()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run(config, output, device):
    if config.get('hidden') != 1024 or config.get('key_dim') != 32:
        raise ValueError('P01 experiment requires hidden1024/key_dim32')
    if config['checkpoints'] != sorted(set(config['checkpoints'])) or config['checkpoints'][0] != 0 or config['checkpoints'][-1] != config['updates']:
        raise ValueError('checkpoint schedule must include0 and final update in ascending order')
    if len({spec['seed'] for spec in config['data'].values()}) != 3:
        raise ValueError('data populations must have distinct seeds')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); torch.set_num_threads(config.get('cpu_threads', 4))
    torch.manual_seed(config['seed'])
    datasets = {}; manifests = {}
    for split in ('train', 'calibration', 'validation'):
        spec = config['data'][split]
        data = make_lowering_batch(spec['seed'], spec['count'])
        public = model_inputs(data['public']); labels = private_labels(data['labels'])
        datasets[split] = (public, labels, data['public'])
        manifests[split] = dict(**spec, public_tensor_sha256=tensor_digest(public))
        # Original return values and targets are discarded before all forwards.
        del data
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'] + 99000000)
    curve = []; best = -1; selected_step = None
    train_public, train_labels, _ = datasets['train']
    training_started = time.monotonic()
    for step in range(config['updates'] + 1):
        if step in config['checkpoints']:
            metrics, logits = evaluate(model, *datasets['calibration'][:2], device, config['batch_size'])
            train_metrics, _ = evaluate(model, sliced(train_public, slice(0, 1024), 'cpu'), sliced(train_labels, slice(0, 1024), 'cpu'), device, config['batch_size'])
            curve.append(dict(step=step, calibration=metrics, train_prefix=train_metrics))
            torch.save(dict(logits=logits, labels=datasets['calibration'][1]), output / f'calibration-{step}.pt')
            count = metrics['full_semantic']['correct']
            # Ascending steps and strict comparison implement earlier tie break.
            if count > best:
                best = count; selected_step = step
                torch.save(model.state_dict(), output / 'selected.pt')
            (output / 'curve.json').write_text(json.dumps(curve, indent=2))
            print(json.dumps(curve[-1]), flush=True)
        if step == config['updates']: break
        model.train()
        idx = torch.randint(len(train_labels['primitive']), (config['batch_size'],), generator=generator)
        actor_public = sliced(train_public, idx, device)
        private = sliced(train_labels, idx, device)
        optimizer.zero_grad(set_to_none=True)
        losses = proposal_loss(model(actor_public), private)
        sum(losses.values()).backward()
        optimizer.step()
    if device.startswith('cuda'): torch.cuda.synchronize()
    fit_seconds = time.monotonic() - training_started
    model.load_state_dict(torch.load(output / 'selected.pt', map_location=device, weights_only=True))
    validation_public, validation_labels, rows = datasets['validation']
    metrics, logits = evaluate(model, validation_public, validation_labels, device, config['batch_size'])
    torch.save(dict(logits=logits, labels=validation_labels, public=validation_public), output / 'validation.pt')
    controls = {}
    for i, kind in enumerate(config['controls']):
        altered = controlled_rows(rows, kind, config['control_seed'] + i)
        public = model_inputs(altered); supplied_labels = labels_from_public(altered)
        measured, out = evaluate(model, public, supplied_labels, device, config['batch_size'])
        # Retain labels in original inventory coordinates only for unpermuted controls.
        changed = (supplied_labels['primitive'] != validation_labels['primitive']) | (supplied_labels['targets'] != validation_labels['targets']).any(-1)
        controls[kind] = dict(supplied=measured, original=None if kind == 'inventory_order' else score(out, public, validation_labels),
                              changed_proposal_support=None if kind == 'inventory_order' else int(changed.sum()))
        torch.save(dict(logits=out, labels=supplied_labels, original_labels=validation_labels,
                        original_labels_comparable=(kind != 'inventory_order'), public=public), output / f'control-{kind}.pt')
    source_names = ('campaign_composition.py', 'campaign_composition_acquire.py', 'interface_proposals.py', 'retention_data.py', 'thinking_runtime.py')
    summary = dict(config=config, device=device, data=manifests, selected_step=selected_step, validation=metrics, controls=controls,
                   curve=curve, fit_seconds=fit_seconds, elapsed_seconds=time.monotonic() - started,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                   source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
                   config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   claim='single-development-seed typed dispatch only; no composition or promotion')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, args.device)
