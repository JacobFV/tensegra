"""C01 scheduled bounded composition and equally observed neural controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from .campaign_composition import make_lowering_batch, model_inputs
from .campaign_composition_acquire import private_labels, controlled_rows, labels_from_public, sliced, tensor_digest
from .campaign_composition_models import NeuralOperandBaseline, ContextualBaseline
from .campaign_composition_runtime import frozen_interfaces, load_checked, propose, build_returns, consume, manipulate, exact_copy, interfaces_state_hashes
from .campaign_returns_use import answer
from .interface_proposals import proposal_loss


def counts(prediction, target, mask=None):
    mask = torch.ones_like(target, dtype=torch.bool) if mask is None else mask
    return dict(correct=int(((prediction == target) & mask).sum()), total=int(mask.sum()))


def paired_metrics(prediction, original, supplied):
    valid = supplied >= 0; changed = valid & (original != supplied)
    return dict(original=counts(prediction, original), supplied=counts(prediction, supplied, valid),
                changed_original=counts(prediction, original, changed), changed_supplied=counts(prediction, supplied, changed),
                refused=int((prediction < 0).sum()), supplied_absent=int((~valid).sum()))


def baseline_evaluate(model, public, labels, device, chunk):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels['task']), chunk):
            outputs.append({k: v.cpu() for k, v in model(sliced(public, slice(start, start+chunk), device)).items()})
    out = {k: torch.cat([part[k] for part in outputs]) for k in outputs[0]}
    return dict(answer=counts(out['answer'].argmax(-1), labels['task']), value=counts(out['value'].argmax(-1), labels['value']), groups=stratified(out['answer'].argmax(-1), labels, public['query'])), out


def stratified(prediction, labels, query):
    groups = {}
    for name in ('type', 'value', 'primitive'):
        if name not in labels: continue
        for value in labels[name].unique().tolist():
            groups[f'{name}/{value}'] = counts(prediction, labels['task'], labels[name] == value)
    gap = (labels['value'].float()-16)/2 - query[:, 0]
    for value in (0., .5, 1.):
        groups[f'absolute_query_gap/{value}'] = counts(prediction, labels['task'], gap.abs() == value)
    return groups


def get_data(spec, distractors=2):
    data = make_lowering_batch(spec['seed'], spec['count'], distractors)
    labels = dict(private_labels(data['labels']), task=data['reference']['targets']['task'], value=data['reference']['targets']['value'], type=data['reference']['targets']['type'])
    return data, model_inputs(data['public']), labels


def train_baseline(config, phase, output, device):
    start = time.monotonic(); datasets = {}; manifest = {}
    for split, spec in config['data'].items():
        data, public, labels = get_data(spec)
        datasets[split] = (public, labels, data['public'])
        manifest[split] = dict(**spec, public_sha256=tensor_digest(public))
        del data  # no source return event enters baseline forward
    best = -1; best_lr = None; best_step = None; curves = []; parameter_count = None
    for lr in config['learning_rates']:
        torch.manual_seed(config['seed'])
        model = (ContextualBaseline() if phase == 'n2' else NeuralOperandBaseline()).to(device)
        if phase != 'n2':
            model.lowerer.load_state_dict(load_checked(config['interfaces']['lowerer'], config['interfaces']['lowerer_sha256'], device))
            if phase == 'n1_frozen': model.lowerer.requires_grad_(False)
        parameter_count = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
        generator = torch.Generator().manual_seed(config['sample_seed'])
        public, labels, _ = datasets['train']; visits = torch.zeros(len(labels['task']), dtype=torch.long)
        for step in range(config['updates']+1):
            if step in config['checkpoints']:
                metrics, logits = baseline_evaluate(model, *datasets['calibration'][:2], device, config['batch_size'])
                record = dict(lr=lr, step=step, calibration=metrics, optimizer_presentations=int(visits.sum()), unique_events=int((visits > 0).sum()))
                curves.append(record); print(json.dumps(record), flush=True)
                torch.save(dict(logits=logits, labels=datasets['calibration'][1]), output/f'calibration-{lr}-{step}.pt')
                # Exact declared order: accuracy, earlier endpoint, first registered LR.
                candidate = (metrics['answer']['correct'], -step, -config['learning_rates'].index(lr))
                incumbent = (best, -best_step, -config['learning_rates'].index(best_lr)) if best_lr is not None else (-1, 0, 0)
                if candidate > incumbent:
                    best = metrics['answer']['correct']; best_lr = lr; best_step = step
                    torch.save(model.state_dict(), output/'selected.pt')
                (output/'curve.json').write_text(json.dumps(curves, indent=2))
            if step == config['updates']: break
            model.train(); idx = torch.randint(len(labels['task']), (config['batch_size'],), generator=generator)
            visits += torch.bincount(idx, minlength=len(visits))
            targets = sliced(labels, idx, device); pred = model(sliced(public, idx, device))
            loss = nn.functional.cross_entropy(pred['answer'], targets['task']) + nn.functional.cross_entropy(pred['value'], targets['value'])
            loss = loss + sum(proposal_loss(pred, targets).values())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        torch.save(visits, output/f'visitation-{lr}.pt')
    model.load_state_dict(torch.load(output/'selected.pt', map_location=device, weights_only=True))
    metrics, out = baseline_evaluate(model, *datasets['validation'][:2], device, config['batch_size'])
    torch.save(dict(logits=out, labels=datasets['validation'][1], public=datasets['validation'][0]), output/'validation.pt')
    control_metrics = {}
    for i, kind in enumerate(config['public_controls']):
        altered = controlled_rows(datasets['validation'][2], kind, config['control_seed']+i)
        oracle = labels_from_public(altered)
        bundle = build_returns(altered, oracle['primitive'], oracle['targets'])
        if len(bundle['indices']) != len(altered): raise RuntimeError('declared public control unexpectedly leaves legal task range')
        event = bundle['public']['event']; query = bundle['public']['query']
        labels = dict(oracle, task=answer(event['values'][:, 0], query), value=(event['values'][:, 0]*2+16).long())
        supplied_metrics, logits = baseline_evaluate(model, model_inputs(altered), labels, device, config['batch_size'])
        control_metrics[kind] = dict(supplied=supplied_metrics, paired=paired_metrics(logits['answer'].argmax(-1), datasets['validation'][1]['task'], labels['task']))
        torch.save(dict(logits=logits, labels=labels, original_labels=datasets['validation'][1], public=model_inputs(altered)), output/f'control-{kind}.pt')
    return dict(phase=phase, data=manifest, selected_step=best_step, selected_lr=best_lr, validation=metrics, controls=control_metrics,
                curves=curves, parameter_count=parameter_count, trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                selected_checkpoint_sha256=hashlib.sha256((output/'selected.pt').read_bytes()).hexdigest(), elapsed_seconds=time.monotonic()-start)


def hybrid(config, output, device):
    start = time.monotonic(); interfaces = frozen_interfaces(config['interfaces'], device); cells = []; audits = []
    frozen_before = interfaces_state_hashes(interfaces)
    for distractors in config['eval_distractors']:
        data, public_inputs, labels = get_data(config['data']['validation'], distractors)
        rows = data['public']; predicted = propose(interfaces['lowerer'], rows, device)
        actual = build_returns(rows, predicted['primitive'], predicted['canonical_pointers'])
        oracle_lowering = build_returns(rows, labels['primitive'], labels['targets'])
        # Explicit privileged oracle-return control; never replaces actual errors.
        oracle_return = dict(indices=torch.arange(len(rows)), public=data['reference']['public'], reasons=['']*len(rows), total=len(rows))
        full_proposal = (predicted['primitive'] == labels['primitive']) & (predicted['canonical_pointers'] == labels['targets']).all(-1)
        audits.append(dict(distractors=distractors, public_sha256=tensor_digest(public_inputs), full_proposal=int(full_proposal.sum()), total=len(rows), refusals=len(rows)-len(actual['indices'])))
        for arm, bundle in (('hybrid', actual), ('oracle_lowering', oracle_lowering), ('oracle_return', oracle_return)):
            result = consume(interfaces, bundle, config['delays'], device, config['capture_batch_size'])
            for delay, outcome in result['cells'].items():
                cell = dict(arm=arm, distractors=distractors, delay=delay, **paired_metrics(outcome['predictions'], labels['task'], result['supplied']))
                if arm == 'hybrid':
                    cell['lowering_and_answer'] = dict(correct=int((full_proposal & (outcome['predictions'] == labels['task'])).sum()), total=len(rows))
                    cell['groups'] = stratified(outcome['predictions'], labels, public_inputs['query'])
                cells.append(cell)
            torch.save(dict(result=result, original=labels['task'], proposal=predicted if arm == 'hybrid' else None, full_proposal=full_proposal, reasons=bundle['reasons'], supplied_public=bundle['public']), output/f'{arm}-{distractors}.pt')
        copy_pred = exact_copy(interfaces, actual, device)
        cells.append(dict(arm='exact_copy', distractors=distractors, delay=None, **paired_metrics(copy_pred, labels['task'], consume_labels(actual))))
        torch.save(dict(predictions=copy_pred, original=labels['task'], supplied=consume_labels(actual)), output/f'exact-copy-{distractors}.pt')
        if distractors != 8: continue
        n = min(config['causal_size'], len(rows)); causal_rows = rows[:n]
        causal = build_returns(causal_rows, predicted['primitive'][:n], predicted['canonical_pointers'][:n])
        swap = make_lowering_batch(config['swap_seed'], n, distractors)['reference']['public']
        with torch.no_grad():
            query = torch.stack([row['query'] for row in causal_rows]).to(device)
            query_input = torch.cat((torch.zeros(n, 33, device=device), query/query.new_tensor([8., 1.])), -1)
            query_pred = interfaces['query_only'](query_input).argmax(-1).cpu()
        torch.save(dict(predictions=query_pred, original=labels['task'][:n]), output/'query-only.pt')
        for kind in ('correct', 'drop', 'wrong', 'swap'):
            bundle = manipulate(causal, kind, swap)
            result = consume(interfaces, bundle, config['intervention_delays'], device, config['capture_batch_size'])
            copied = exact_copy(interfaces, bundle, device)
            for delay, outcome in result['cells'].items():
                cells.append(dict(arm=kind, distractors=8, delay=delay, **paired_metrics(outcome['predictions'], labels['task'][:n], result['supplied'])))
            torch.save(dict(result=result, original=labels['task'][:n], exact_copy=copied, query_only=query_pred, reasons=bundle['reasons'], supplied_public=None if kind == 'drop' else bundle['public']), output/f'intervention-{kind}.pt')
        for i, kind in enumerate(config['public_controls']):
            altered = controlled_rows(rows, kind, config['control_seed']+i)
            altered_pred = propose(interfaces['lowerer'], altered, device)
            bundle = build_returns(altered, altered_pred['primitive'], altered_pred['canonical_pointers'])
            result = consume(interfaces, bundle, config['intervention_delays'], device, config['capture_batch_size'])
            requested = labels_from_public(altered)
            requested_bundle = build_returns(altered, requested['primitive'], requested['targets'])
            if len(requested_bundle['indices']) != len(rows): raise RuntimeError('public control requested execution outside declared contract')
            requested_answer = consume_labels(requested_bundle)
            for delay, outcome in result['cells'].items():
                cells.append(dict(arm='public_'+kind, distractors=8, delay=delay, requested=counts(outcome['predictions'], requested_answer), requested_changed=counts(outcome['predictions'], requested_answer, requested_answer != labels['task']), **paired_metrics(outcome['predictions'], labels['task'], result['supplied'])))
            torch.save(dict(result=result, original=labels['task'], requested=requested_answer, proposal=altered_pred, reasons=bundle['reasons'], supplied_public=bundle['public']), output/f'public-{kind}.pt')
    frozen_after = interfaces_state_hashes(interfaces)
    frozen = dict(before=frozen_before, after=frozen_after, unchanged=frozen_before == frozen_after)
    (output/'frozen-state.json').write_text(json.dumps(frozen,indent=2)+'\n')
    if not frozen['unchanged']: raise RuntimeError('hybrid frozen state changed')
    return dict(phase='hybrid', cells=cells, proposal_audits=audits, frozen_state=frozen, causal_checks=causal_checks(output, config['intervention_delays']), elapsed_seconds=time.monotonic()-start)


def causal_checks(output, delays):
    data = {kind: torch.load(output/f'intervention-{kind}.pt', map_location='cpu', weights_only=True) for kind in ('correct','drop','wrong','swap')}
    original = data['correct']['original']; n = len(original)
    query = (data['correct']['query_only'] == original).float().mean().item()
    oracle = (data['correct']['exact_copy'] == original).float().mean().item()
    checks = []
    for delay in delays:
        clean_bits = (data['correct']['result']['cells'][delay]['predictions'] == original).float()
        drop_bits = (data['drop']['result']['cells'][delay]['predictions'] == original).float()
        difference = clean_bits-drop_bits; effect = float(difference.mean())
        standard_error = float(difference.std(unbiased=True)/n**.5) if n > 1 else float('nan')
        gain = (float(clean_bits.mean())-query)/(oracle-query) if oracle > query else None
        interventions = {}
        for kind in ('wrong','swap'):
            supplied = data[kind]['result']['supplied']; mask = (supplied >= 0) & (supplied != original)
            interventions[kind] = counts(data[kind]['result']['cells'][delay]['predictions'], supplied, mask)
        passed = effect >= .15 and gain is not None and gain >= .8 and all(v['total'] >= 256 and v['correct']/v['total'] >= .9 for v in interventions.values())
        checks.append(dict(delay=delay, clean_accuracy=float(clean_bits.mean()), query_only_accuracy=query, exact_copy_accuracy=oracle,
                           clean_minus_drop=effect, paired_difference_normal95=[effect-1.96*standard_error,effect+1.96*standard_error],
                           normalized_gain=gain, changed_supplied=interventions, passed=passed))
    return checks


def consume_labels(bundle):
    labels = torch.full((bundle['total'],), -1, dtype=torch.long)
    if bundle['public'] is not None:
        public = bundle['public']; labels[bundle['indices']] = answer(public['event']['values'][:, 0], public['query'])
    return labels


def run(config, phase, output, device):
    if config['hidden'] != 1024 or config['key_dim'] != 32: raise ValueError('experiment hidden1024/key32 required')
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summary = hybrid(config, output, device) if phase == 'hybrid' else train_baseline(config, phase, output, device)
    if device.startswith('cuda'): torch.cuda.synchronize()
    files = ('campaign_composition.py','campaign_composition_acquire.py','campaign_composition_models.py','campaign_composition_runtime.py','campaign_composition_study.py','campaign_returns_use.py','retention_data.py','interface_proposals.py','thinking_runtime.py','return_memory.py','return_crossdelay.py','return_memory_study.py','thinking.py','retention.py','return_diagnostics_probe.py','campaign_returns.py','campaign_returns_balanced.py')
    summary.update(config=config, config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(), torch_version=torch.__version__, float32_matmul_precision=torch.get_float32_matmul_precision(), source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                   claim='one-development-population scheduled bounded composition; no learned planning or timing')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('hybrid','n1','n1_frozen','n2'), required=True); parser.add_argument('--device', required=True)
    args = parser.parse_args(); run(json.loads(Path(args.config).read_text()), args.phase, args.output, args.device)
