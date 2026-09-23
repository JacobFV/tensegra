"""Reproducible independent Stage7 A/B and tiny arithmetic-delta diagnostics.

No execution, progressive evidence, halting, or interface coupling occurs here.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import random
import time
import torch
from torch import nn
from .interface_proposals import (PRIMITIVES, ProposalModel, make_proposals, collate_proposals,
                                  proposal_loss, proposal_metrics, gate_a, gate_a_matrix, arithmetic_delta)
from .interface_readiness import (Calibrator, make_readiness, select_threshold,
                                  readiness_metrics, global_scores, gate_b)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_hash(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _evaluate(model, data, control='complete'):
    batch = collate_proposals(data, control)
    with torch.no_grad():
        output = model(batch)
    return proposal_metrics(output, batch), {
        'primitive': output['primitive'].argmax(-1).tolist(),
        'pointers': output['pointers'].argmax(-1).tolist(),
        'target_primitive': batch['primitive'].tolist(), 'target_pointers': batch['targets'].tolist()}


def _readiness(config, seed, output):
    datasets = {name: make_readiness(config['readiness_groups'], seed=seed*10000+i+400)
                for i, name in enumerate(('calibration', 'validation', 'test'))}
    model = Calibrator()
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('readiness_lr', .03))
    records = datasets['calibration']
    x = torch.tensor([r['factors'] for r in records])
    y = torch.tensor([float(r['correct']) for r in records])
    curves = []
    for step in range(config['readiness_steps']+1):
        probability = model(x)
        loss = nn.functional.binary_cross_entropy(probability.clamp(1e-6, 1-1e-6), y)
        if step % config['eval_every'] == 0 or step == config['readiness_steps']:
            curves.append({'step': step, 'calibration_bce': float(loss.detach())})
        if step < config['readiness_steps']:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    scores = {}
    with torch.no_grad():
        for name, rows in datasets.items():
            scores[name] = model(torch.tensor([r['factors'] for r in rows])).tolist()
    val = datasets['validation']
    truth = [r['correct'] for r in val]
    threshold = select_threshold(scores['validation'], truth)
    global_threshold = select_threshold(global_scores(val, scores['validation']), truth)
    product_threshold = select_threshold([r['product_score'] for r in val], truth)
    result = {'threshold_source': 'validation', 'threshold': threshold, 'global_threshold': global_threshold,
              'product_threshold': product_threshold, 'curves': curves, 'initial_hash': initial,
              'checkpoint_hash': state_hash(model), 'data_hashes': {k: digest(v) for k, v in datasets.items()},
              'supplied_supports_not_learned_hypotheses': True,
              'generator_limitation':'Synthetic calibration law has sharp probability jump at latent .9985; product is not posterior.',
              'correlation_matrix_calibration': torch.corrcoef(x.T).tolist()}
    for name in (('validation','test') if config.get('main_budget_frozen',False) else ('validation',)):
        rows, score = datasets[name], scores[name]
        labels = [r['correct'] for r in rows]
        local = readiness_metrics(score, labels, threshold)
        global_result = readiness_metrics(global_scores(rows, score), labels, global_threshold)
        mixed_indices = [i for i,r in enumerate(rows) if r['group_kind'] != 'zero_ready']
        gs = global_scores(rows, score)
        mixed_local = readiness_metrics([score[i] for i in mixed_indices], [labels[i] for i in mixed_indices], threshold)
        mixed_global = readiness_metrics([gs[i] for i in mixed_indices], [labels[i] for i in mixed_indices], global_threshold)
        local['local_advantage'] = mixed_local['recall']-mixed_global['recall']
        by_kind = {}
        for kind in sorted({r['kind'] for r in rows}):
            indices = [i for i,r in enumerate(rows) if r['kind'] == kind]
            by_kind[kind] = readiness_metrics([score[i] for i in indices], [labels[i] for i in indices], threshold)
        result[name] = {'local': local, 'global': global_result, 'mixed_local':mixed_local, 'mixed_global':mixed_global, 'by_kind': by_kind,
                        'product': readiness_metrics([r['product_score'] for r in rows], labels, product_threshold),
                        'gate_b': gate_b(local), 'records': [dict(r, calibrated=s) for r,s in zip(rows, score)]}
    torch.save(model.state_dict(), output/f'seed{seed}-readiness.pt')
    return result


def _delta(config, seed, output):
    """Supplied operation/operands -> small bounded insert-value classifier.

Removal identities are supplied, not learned; only inserted arithmetic value is
predicted. This deliberately does not claim general rewriting or execution.
"""
    rng = random.Random(seed+70000)
    examples = []
    for op in PRIMITIVES:
        for a in range(-3, 4):
            for b in (range(-3,4) if op != 'neg' else (0,)):
                examples.append({'operation': op, 'left': a, 'right': b, 'delta': arithmetic_delta(op,a,b)})
    rng.shuffle(examples)
    train, test = examples[:int(.8*len(examples))], examples[int(.8*len(examples)):]
    def batch(rows):
        x = torch.tensor([[float(op == r['operation']) for op in PRIMITIVES]+[r['left']/3,r['right']/3] for r in rows])
        y = torch.tensor([r['delta']['insert_value']+9 for r in rows])
        return x,y
    x,y = batch(train)
    model = nn.Sequential(nn.Linear(7, config['hidden']), nn.Tanh(), nn.Linear(config['hidden'],19))
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', .01))
    curves = []
    for step in range(config['delta_steps']+1):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits,y)
        if step % config['eval_every'] == 0 or step == config['delta_steps']:
            curves.append({'step': step, 'insert_value_loss': float(loss.detach()),
                           'train_accuracy': float((logits.argmax(-1)==y).float().mean())})
        if step < config['delta_steps']:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    tx,ty = batch(test)
    with torch.no_grad():
        predictions = model(tx).argmax(-1)
    torch.save(model.state_dict(), output/f'seed{seed}-delta.pt')
    return {'supplied_operation_and_operands': True, 'removals_supplied_not_predicted': True,
            'curves': curves, 'heldout_insert_value_accuracy': float((predictions==ty).float().mean()),
            'heldout_partition':'test' if config.get('main_budget_frozen',False) else 'validation',
            'test_count': len(test), 'raw': [dict(r, predicted_insert=int(p)-9) for r,p in zip(test,predictions)],
            'data_hashes': {'train':digest(train), 'test':digest(test)},
            'initial_hash':initial, 'checkpoint_hash':state_hash(model)}


def run(config, output):
    config = dict(config)
    for k,v in {'seeds':[0], 'steps':20, 'readiness_steps':20, 'delta_steps':20,
                'train_size':128, 'eval_size':128, 'readiness_groups':32, 'batch_size':32,
                'hidden':32, 'eval_every':10}.items():
        config.setdefault(k,v)
    if config['eval_every'] < 1 or any(config[k] < 0 for k in ('steps','readiness_steps','delta_steps')):
        raise ValueError('invalid training budgets')
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__), Path(__file__).with_name('interface_proposals.py'), Path(__file__).with_name('interface_readiness.py'))}
    manifest = {'config':config, 'config_hash':digest(config), 'source_hashes':source,
                'torch_version':torch.__version__, 'threads':2,
                'priors': 'Supplied typed instruction records and destination query; learned retrieval and ordered binding. Names/values are metadata-only invariance axes. OOD has larger instruction sets and two-derived-input wiring. No natural-language parsing or execution claim.',
                'readiness_scope':'Generator-supplied correlated support features; not inferred posterior or learned candidate construction.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    gates = []
    validation_matrix = {}
    started = time.monotonic()
    for seed in config['seeds']:
        torch.manual_seed(seed)
        rng = random.Random(seed)
        train = make_proposals(config['train_size'],seed=seed*10000)
        datasets = {f'{split}_{partition}':make_proposals(config['eval_size'],seed=seed*10000+100+i*2+j,split=split)
                    for i,split in enumerate(('iid','ood')) for j,partition in enumerate(('validation','test'))}
        model = ProposalModel(hidden=config['hidden'])
        initial = state_hash(model)
        optimizer = torch.optim.Adam(model.parameters(),lr=config.get('lr',.01))
        curves = []
        for step in range(config['steps']+1):
            if step % config['eval_every'] == 0 or step == config['steps']:
                curves.append({'step':step, 'train':_evaluate(model,train)[0], 'validation':{k:_evaluate(model,v)[0] for k,v in datasets.items() if k.endswith('validation')},
                               'losses':{k:float(v.detach()) for k,v in proposal_loss(model(collate_proposals(train)),collate_proposals(train)).items()}})
            if step < config['steps']:
                data = rng.choices(train,k=config['batch_size'])
                batch = collate_proposals(data)
                losses = proposal_loss(model(batch),batch)
                optimizer.zero_grad(); sum(losses.values()).backward(); optimizer.step()
        raw = {'curves':curves, 'data_hashes':{k:digest([asdict(x) for x in v]) for k,v in dict(train=train,**datasets).items()},
               'initial_hash':initial, 'checkpoint_hash':state_hash(model), 'optimizer_examples':config['steps']*config['batch_size'],
               'train_cardinality':len(train), 'evaluations':{}}
        for name,data in datasets.items():
            if name.endswith('test') and not config.get('main_budget_frozen',False):
                continue
            raw['evaluations'][name] = {}
            for control in ('complete','missing','permuted','reverse'):
                metrics,predictions = _evaluate(model,data,control)
                raw['evaluations'][name][control] = {'metrics':metrics, 'raw':predictions}
        passed = gate_a([raw['evaluations'][k]['complete']['metrics'] for k in ('iid_validation','ood_validation')])
        raw['typed_instruction_gate_a_validation'] = passed
        validation_matrix[seed] = {k:raw['evaluations'][k]['complete']['metrics'] for k in ('iid_validation','ood_validation')}
        raw['readiness'] = _readiness(config,seed,output)
        raw['delta'] = _delta(config,seed,output)
        torch.save(model.state_dict(),output/f'seed{seed}-proposal.pt')
        (output/f'seed{seed}-raw.json').write_text(json.dumps(raw,indent=2))
        gates.append({'seed':seed,'restricted_a':passed,'b_validation':raw['readiness']['validation']['gate_b'],
                      'b_test':raw['readiness'].get('test',{}).get('gate_b',False)})
    summary = {'seeds':gates, 'gate_a_all_validation_seeds':gate_a_matrix(validation_matrix, config['seeds']),
               'gate_b_all_test_seeds':bool(gates) and all(g['b_test'] for g in gates),
               'composition_authorized':False, 'progressive_authorized':False,
               'scope_limitation':'Typed instruction selection and binding only; progressive evidence and runtime composition untested.',
               'elapsed_seconds':time.monotonic()-started}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text()),args.output),indent=2))

if __name__ == '__main__':
    main()
