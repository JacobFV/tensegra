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


def run_progressive(validation_matrix, expected_seeds, output, *, steps, seed=0, train_size=128, eval_size=512):
    """Explicit separate A2 run; refuses incomplete or failed frozen Gate A.

Caller must authorize/freeze its budget. The finite candidate set and public
likelihood observations are supplied priors, not learned proposal discovery.
"""
    if not gate_a_matrix(validation_matrix, expected_seeds):
        raise ValueError('Gate A must pass every expected seed and validation condition')
    from .interface_proposals import JointPosteriorModel, joint_evidence
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    train = [joint_evidence(seed=seed*10000+i) for i in range(train_size)]
    validation = [joint_evidence(seed=seed*10000+2000+i) for i in range(eval_size)]
    frames = torch.stack([r['frames'] for r in train])
    targets = torch.stack([r['private_posterior'] for r in train])
    model = JointPosteriorModel()
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    curves = []
    for step in range(steps+1):
        belief = model(frames)
        loss = -(targets*belief.clamp_min(1e-8).log()).sum(-1).mean()
        if step % 50 == 0 or step == steps:
            curves.append({'step':step,'joint_cross_entropy':float(loss.detach())})
        if step < steps:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        beliefs = model(torch.stack([r['frames'] for r in validation]))
    truth = torch.stack([r['private_posterior'] for r in validation])
    result = {'scope':'Supplied finite joint hypotheses and likelihood evidence; no runtime composition',
              'curves':curves,'mean_absolute_posterior_error':float((beliefs-truth).abs().mean()),
              'private_targets':truth.tolist(),'beliefs':beliefs.tolist(),'initial_hash':initial,
              'checkpoint_hash':state_hash(model),'gate_matrix_hash':digest(validation_matrix),
              'config':dict(steps=steps,seed=seed,train_size=train_size,eval_size=eval_size),
              'public_frames_hash':digest(frames.tolist())}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),output/'progressive.pt')
    (output/'progressive.json').write_text(json.dumps(result,indent=2))
    return result


def proposal_confidence_records(model, examples, *, condition, split):
    """Confidence from frozen learned proposals, current/past public evidence only."""
    batch = collate_proposals(examples,condition)
    past = collate_proposals(examples,'missing')
    with torch.no_grad():
        current_out, past_out = model(batch),model(past)
        operation_support, operation = current_out['primitive'].softmax(-1).max(-1)
        pointer_support,pointers = current_out['pointers'].softmax(-1).max(-1)
        old_operation = past_out['primitive'].argmax(-1)
        old_pointers = past_out['pointers'].argmax(-1)
    records = []
    for i,example in enumerate(examples):
        op,ptr = int(operation[i]),pointers[i].tolist()
        unary = op == PRIMITIVES.index('neg')
        boolean_keys = {example.keys[k] for k,t in enumerate(batch['register_types'][i,:len(example.keys)]) if int(t) == 1}
        used = ptr[1:2] if unary else ptr[1:]
        schema = all(example.keys[k] not in boolean_keys for k in used)
        schema = schema and ((example.keys[ptr[0]] in boolean_keys) == (op == PRIMITIVES.index('compare')))
        full = op == PRIMITIVES.index(example.primitive) and ptr[:2] == list(example.targets[:2]) and (example.primitive == 'neg' or ptr[2] == example.targets[2])
        matches = [op == int(old_operation[i])] + [ptr[j] == int(old_pointers[i,j]) for j in range(2 if unary else 3)]
        stability = sum(matches)/len(matches)
        factors = [float(operation_support[i]),float(pointer_support[i,0]),float(pointer_support[i,1]),
                   1. if unary else float(pointer_support[i,2]),float(schema),stability]
        records.append({'condition':condition,'split':split,'index':i,'factors':factors,
                        'primitive':op,'pointers':ptr,'target_primitive':PRIMITIVES.index(example.primitive),
                        'target_pointers':list(example.targets),'schema':bool(schema),'full_correct':bool(full),
                        'correct':bool(full and schema),'product_score':float(torch.tensor(factors).prod())})
    return records


def run_proposal_calibration(checkpoint_directory, output, config):
    """Fresh calibration of actual learned candidate errors; no runtime action.

Executable means private full ordered task correctness AND exact schema validity.
A group comprises independent examples in deterministic shuffled groups of four.
"""
    checkpoint_directory,output = Path(checkpoint_directory),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    base = json.loads((checkpoint_directory/'manifest.json').read_text())
    torch.set_num_threads(2)
    result = {'config':config,'config_hash':digest(config),'base_manifest_hash':digest(base),
              'factor_order':['primitive','destination','operand1','operand2','schema','stability'],
              'stability_history':'Past missing operation/argument evidence; current condition public evidence. No future observations.',
              'executable_definition':'Full correct ordered proposal and exact schema valid',
              'public_type_prior':'Exact immutable register type table supplied in every evidence frame; schema uses only this public table.',
              'scope_limitation':'Calibration on prespecified missing/permuted mixture; independent examples grouped4, not competing hypotheses in one world.',
              'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('interface_readiness.py'),Path(__file__).with_name('interface_proposals.py'))},
              'seeds':{}}
    for seed in config['seeds']:
        torch.manual_seed(seed+810000)
        model = ProposalModel(hidden=base['config']['hidden'])
        checkpoint = checkpoint_directory/f'seed{seed}-proposal.pt'
        model.load_state_dict(torch.load(checkpoint,weights_only=True))
        model.eval()
        datasets = {}
        for part_index,part in enumerate(('calibration','validation','test')):
            rows = []
            for split_index,split in enumerate(('iid','ood')):
                data = make_proposals(config['examples'],seed=810000+seed*10000+part_index*100+split_index,split=split)
                for condition in ('complete','missing','permuted'):
                    rows.extend(proposal_confidence_records(model,data,condition=condition,split=split))
            random.Random(seed*100+part_index).shuffle(rows)
            for index,row in enumerate(rows): row['group'] = index//4
            groups = {}
            for row in rows: groups.setdefault(row['group'],[]).append(row)
            for row in rows:
                n = sum(r['correct'] for r in groups[row['group']])
                row['group_kind'] = 'zero_ready' if n == 0 else 'multiple_ready' if n>1 else 'mixed'
            datasets[part] = rows
        calibrator = Calibrator(factor_count=6,schema_index=4)
        initial = state_hash(calibrator)
        optimizer = torch.optim.Adam(calibrator.parameters(),lr=config.get('lr',.03))
        x = torch.tensor([r['factors'] for r in datasets['calibration']])
        y = torch.tensor([float(r['correct']) for r in datasets['calibration']])
        curves = []
        for step in range(config['steps']+1):
            probs = calibrator(x)
            loss = nn.functional.binary_cross_entropy(probs.clamp(1e-6,1-1e-6),y)
            if step % 100 == 0 or step == config['steps']:
                curves.append({'step':step,'calibration_bce':float(loss.detach())})
            if step < config['steps']:
                optimizer.zero_grad();loss.backward();optimizer.step()
        scores = {}
        with torch.no_grad():
            for part,rows in datasets.items(): scores[part] = calibrator(torch.tensor([r['factors'] for r in rows])).tolist()
        val,vs = datasets['validation'],scores['validation']
        threshold = select_threshold(vs,[r['correct'] for r in val])
        global_threshold = select_threshold(global_scores(val,vs),[r['correct'] for r in val])
        sr = {'threshold':threshold,'global_threshold':global_threshold,'threshold_source':'validation','curves':curves,
              'proposal_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'data_hashes':{k:digest(v) for k,v in datasets.items()},'initial_hash':initial,'checkpoint_hash':state_hash(calibrator),
              'correlation_matrix':torch.corrcoef(x.T).tolist()}
        for part in ('validation','test'):
            rows,score = datasets[part],scores[part]
            truth = [r['correct'] for r in rows]
            gs = global_scores(rows,score)
            local = readiness_metrics(score,truth,threshold)
            global_result = readiness_metrics(gs,truth,global_threshold)
            mixed = [i for i,r in enumerate(rows) if 0 < sum(t['correct'] for t in rows[(i//4)*4:(i//4)*4+4]) < 4]
            ml = readiness_metrics([score[i] for i in mixed],[truth[i] for i in mixed],threshold)
            mg = readiness_metrics([gs[i] for i in mixed],[truth[i] for i in mixed],global_threshold)
            local['local_advantage'] = ml['recall']-mg['recall']
            per_condition = {}
            for split in ('iid','ood'):
                for condition in ('complete','missing','permuted'):
                    indices = [i for i,r in enumerate(rows) if r['split']==split and r['condition']==condition]
                    per_condition[f'{split}_{condition}'] = readiness_metrics([score[i] for i in indices],[truth[i] for i in indices],threshold)
            sr[part] = {'local':local,'global':global_result,'mixed_local':ml,'mixed_global':mg,
                        'per_condition':per_condition,'gate_b':gate_b(local),
                        'records':[dict(r,calibrated=s) for r,s in zip(rows,score)]}
        torch.save(calibrator.state_dict(),output/f'seed{seed}-calibrator.pt')
        (output/f'seed{seed}-actual-confidence.json').write_text(json.dumps(sr,indent=2))
        result['seeds'][seed] = {'validation':sr['validation']['local'],'test':sr['test']['local'],'gate_b_test':sr['test']['gate_b']}
    result['all_seed_gate_b'] = all(r['gate_b_test'] for r in result['seeds'].values()) and bool(result['seeds'])
    result['runtime_composition_authorized'] = False
    (output/'actual-confidence-summary.json').write_text(json.dumps(result,indent=2))
    return result


def run_instruction_progressive(checkpoint_directory, output, config):
    """Gate-A-dependent public-record posterior acquisition; no runtime coupling."""
    from .interface_proposals import ProgressiveInstructionModel, progressive_instruction_episode
    checkpoint_directory,output = Path(checkpoint_directory),Path(output)
    base = json.loads((checkpoint_directory/'manifest.json').read_text())
    if config['seeds'] != base['config']['seeds'] or not base['config'].get('main_budget_frozen'):
        raise ValueError('Gate A requires the complete frozen main seed matrix')
    matrix = {}
    for seed in config['seeds']:
        raw = json.loads((checkpoint_directory/f'seed{seed}-raw.json').read_text())
        matrix[seed] = {k:raw['evaluations'][k]['complete']['metrics'] for k in ('iid_validation','ood_validation')}
    if not gate_a_matrix(matrix,config['seeds']):
        raise ValueError('Gate A failed')
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    manifest = {'config':config,'config_hash':digest(config),'gate_matrix':matrix,
                'gate_matrix_hash':digest(matrix),'base_manifest_hash':digest(base),
                'scope':'Supplied finite joint hypothesis set with unordered operand pair; public typed instruction op/order fields revealed over5frames. Existing4workspace blocks update persistent candidate states. No execution or composition.',
                'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('interface_proposals.py'),Path(__file__).with_name('thinking.py'))}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summaries = []
    started = time.monotonic()
    for seed in config['seeds']:
        torch.manual_seed(seed+910000)
        rng = random.Random(seed)
        datasets = {name:[progressive_instruction_episode(seed=910000+seed*10000+offset+i,records=records) for i in range(count)]
                    for name,offset,records,count in [('train',0,4,config['train_size']),('iid_validation',2000,4,config['eval_size']),
                    ('ood_validation',3000,6,config['eval_size']),('iid_test',4000,4,config['eval_size']),('ood_test',5000,6,config['eval_size'])]}
        def tensorize(rows):
            return tuple(torch.stack([r[k] for r in rows]) for k in ('frames','hypotheses','private_posterior'))
        batches = {k:tensorize(v) for k,v in datasets.items()}
        def evaluate(model,name):
            frames,hypotheses,target = batches[name]
            with torch.no_grad(): belief = model(frames,hypotheses)
            valid = target[:,-1,-1] == 0
            final = belief[:,-1].argmax(-1) == target[:,-1].argmax(-1)
            support = (target > 0).float()
            impossible = (belief*(1-support)).sum(-1)
            return {'count':len(target),'final_joint_accuracy':float(final.float().mean()),
                    'executable_count':int(valid.sum()),'no_executable_count':int((~valid).sum()),
                    'executable_joint_accuracy':float(final[valid].float().mean()) if valid.any() else None,
                    'reject_accuracy':float(final[~valid].float().mean()) if (~valid).any() else None,
                    'mean_absolute_posterior_error':float((belief-target).abs().mean()),
                    'impossible_mass_by_frame':impossible.mean(0).tolist(),
                    'posterior_entropy_by_frame':(-(belief*belief.clamp_min(1e-8).log()).sum(-1)).mean(0).tolist()},belief.tolist(),target.tolist()
        model = ProgressiveInstructionModel(width=config['width'])
        initial = state_hash(model)
        optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])
        curves = []
        for step in range(config['steps']+1):
            if step % config['eval_every'] == 0 or step == config['steps']:
                curves.append({'step':step,'train':evaluate(model,'train')[0],
                               'validation':{k:evaluate(model,k)[0] for k in ('iid_validation','ood_validation')}})
            if step < config['steps']:
                indices = rng.choices(range(config['train_size']),k=config['batch_size'])
                frames,hypotheses,target = (x[indices] for x in batches['train'])
                belief = model(frames,hypotheses)
                loss = -(target*belief.clamp_min(1e-8).log()).sum(-1).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if step % config['eval_every'] == 0: curves[-1]['joint_cross_entropy'] = float(loss.detach())
        final = {}
        for name in ('iid_validation','ood_validation','iid_test','ood_test'):
            metrics,belief,target = evaluate(model,name)
            final[name] = {'metrics':metrics,'beliefs':belief,'private_posterior':target}
        raw = {'curves':curves,'final':final,'initial_hash':initial,'checkpoint_hash':state_hash(model),
               'data_hashes':{name:digest({'frames':b[0].tolist(),'hypotheses':b[1].tolist(),'targets':b[2].tolist()}) for name,b in batches.items()},
               'optimizer_examples':config['steps']*config['batch_size'],'train_cardinality':config['train_size']}
        torch.save(model.state_dict(),output/f'seed{seed}-progressive.pt')
        (output/f'seed{seed}-progressive.json').write_text(json.dumps(raw,indent=2))
        summaries.append({'seed':seed,'final':{k:v['metrics'] for k,v in final.items()}})
    result = {'seeds':summaries,'runtime_composition_authorized':False,'elapsed_seconds':time.monotonic()-started}
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result
