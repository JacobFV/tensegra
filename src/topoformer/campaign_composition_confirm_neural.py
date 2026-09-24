"""C04 fresh paired neural confirmation; fixed recipe, private overlap audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from .campaign_composition_study import get_data, baseline_evaluate, paired_metrics
from .campaign_composition_acquire import sliced, tensor_digest, controlled_rows, labels_from_public, score
from .campaign_composition_models import NeuralOperandBaseline, ContextualBaseline
from .campaign_composition_runtime import build_returns, load_checked
from .campaign_composition import model_inputs
from .campaign_composition_rekey import rekey
from .campaign_composition_confirm_data import numeric_overlap
from .campaign_composition_roles import reversed_supervision, choose_views
from .campaign_returns_use import answer
from .interface_proposals import proposal_loss


def evaluate(model, public, labels, device, chunk):
    metrics, logits = baseline_evaluate(model, public, labels, device, chunk)
    metrics['lowering'] = score(logits, public, labels)
    return metrics, logits


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, arm, output, device):
    if config['hidden'] != 1024 or config['key_dim'] != 32:
        raise ValueError('width1024/key32 required')
    torch.set_num_threads(4)
    tick = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    datasets = {}; manifests = {}; swapped = {}
    for split, spec in config['data'].items():
        data, public, labels = get_data(spec)
        datasets[split] = (public, labels, data['public'])
        manifests[split] = dict(**spec, public_sha256=tensor_digest(public), labels_sha256=tensor_digest(labels))
        swapped[split] = reversed_supervision(data['public'])
        manifests[split]['reversed_public_sha256'] = tensor_digest(swapped[split][0])
        manifests[split]['reversed_labels_sha256'] = tensor_digest(swapped[split][1])
        del data
    torch.manual_seed(config['seed'])
    model = (ContextualBaseline() if arm == 'n2_rekey' else NeuralOperandBaseline()).to(device)
    if arm != 'n2_rekey': model.lowerer.load_state_dict(load_checked(config['interfaces']['lowerer'], config['interfaces']['lowerer_sha256'], device))
    initial_hash = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
    sampler = torch.Generator().manual_seed(config['sample_seed'])
    view_rng = torch.Generator().manual_seed(config['role_seed'])
    code_rng = torch.Generator().manual_seed(config['rekey_seed'])
    public, labels, _ = datasets['train']
    visits = torch.zeros(len(labels['task']), dtype=torch.long)
    sample_hash = hashlib.sha256(); views_hash = hashlib.sha256(); code_hash = hashlib.sha256()
    swapped_visits = torch.zeros_like(visits)
    best = -1; best_step = None; curves = []; parity = []
    for step in range(config['updates']+1):
        if step in config['checkpoints']:
            metrics, logits = evaluate(model, *datasets['calibration'][:2], device, config['batch_size'])
            record = dict(step=step, calibration=metrics, optimizer_presentations=int(visits.sum()), unique_base_events=int((visits > 0).sum()))
            reverse_metrics, reverse_logits = evaluate(model, *swapped['calibration'], device, config['batch_size'])
            record['reversed_calibration_diagnostic'] = reverse_metrics
            torch.save(dict(logits=reverse_logits, labels=swapped['calibration'][1], public=swapped['calibration'][0]), output/f'calibration-reverse-{step}.pt')
            curves.append(record); print(json.dumps(record), flush=True)
            torch.save(dict(logits=logits, labels=datasets['calibration'][1]), output/f'calibration-{step}.pt')
            if metrics['answer']['correct'] > best:
                best = metrics['answer']['correct']; best_step = step
                torch.save(model.state_dict(), output/'selected.pt')
            (output/'curve.json').write_text(json.dumps(curves, indent=2)+'\n')
        if step == config['updates']: break
        model.train()
        idx = torch.randint(len(labels['task']), (config['batch_size'],), generator=sampler)
        visits += torch.bincount(idx, minlength=len(visits))
        sample_hash.update(idx.numpy().tobytes())
        targets = sliced(labels, idx, 'cpu')
        observed = sliced(public, idx, 'cpu')
        coin = torch.rand(config['batch_size'], generator=view_rng) < .5 if arm == 'n1_roles' else torch.zeros(config['batch_size'], dtype=torch.bool)
        if arm == 'n1_roles':
            observed = choose_views(observed, sliced(swapped['train'][0], idx, 'cpu'), coin)
            targets = choose_views(targets, sliced(swapped['train'][1], idx, 'cpu'), coin)
        if arm == 'n2_rekey': observed = rekey(observed, code_rng)
        swapped_visits += torch.bincount(idx[coin], minlength=len(visits))
        views_hash.update(coin.numpy().tobytes())
        code_hash.update(observed['keys'].numpy().tobytes())
        targets = {k:v.to(device) for k,v in targets.items()}
        pred = model({k:v.to(device) for k,v in observed.items()})
        loss = nn.functional.cross_entropy(pred['answer'], targets['task']) + nn.functional.cross_entropy(pred['value'], targets['value'])
        loss = loss + sum(proposal_loss(pred, targets).values())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), output/'endpoint.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.get_state(), role_rng=view_rng.get_state(), rekey_rng=code_rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [], visits=visits, swapped_visits=swapped_visits, step=config['updates']), output/'resume.pt')
    torch.save(dict(base=visits, swapped=swapped_visits), output/'visitation.pt')
    with gzip.open(output/'numeric-overlap.json.gz','wt') as stream:
        json.dump(numeric_overlap(datasets, swapped, visits, swapped_visits),stream,sort_keys=True)
    endpoints = {}
    for endpoint in ('endpoint', 'selected'):
        model.load_state_dict(torch.load(output/f'{endpoint}.pt', map_location=device, weights_only=True))
        report = {}
        for split in ('validation',):
            public, labels, rows = datasets[split]
            metrics, logits = evaluate(model, public, labels, device, config['batch_size'])
            report[split] = metrics
            torch.save(dict(logits=logits, labels=labels, public=public), output/f'{endpoint}-{split}.pt')
            changed, changed_labels = swapped[split]
            metrics, logits = evaluate(model, changed, changed_labels, device, config['batch_size'])
            report[split+'_reversed'] = dict(supplied=metrics, paired=paired_metrics(logits['answer'].argmax(-1), labels['task'], changed_labels['task']))
            torch.save(dict(logits=logits, labels=changed_labels, original_labels=labels, public=changed), output/f'{endpoint}-{split}-reversed.pt')
        # Same registered C01 public controls, on the fresh confirmation population.
        report['controls'] = {}
        for i, kind in enumerate(config['public_controls']):
            rows = controlled_rows(datasets['validation'][2], kind, config['control_seed']+i)
            oracle = labels_from_public(rows); bundle = build_returns(rows, oracle['primitive'], oracle['targets'])
            if len(bundle['indices']) != len(rows): raise RuntimeError('control outside supplied schema')
            event = bundle['public']['event']; query = bundle['public']['query']
            labels = dict(oracle, task=answer(event['values'][:,0], query), value=(event['values'][:,0]*2+16).long())
            observed = model_inputs(rows)
            metrics, logits = evaluate(model, observed, labels, device, config['batch_size'])
            report['controls'][kind] = dict(supplied=metrics, paired=paired_metrics(logits['answer'].argmax(-1), datasets['validation'][1]['task'], labels['task']))
            torch.save(dict(logits=logits, labels=labels, original_labels=datasets['validation'][1], public=observed), output/f'{endpoint}-control-{kind}.pt')
        endpoints[endpoint] = report
    if device.startswith('cuda'): torch.cuda.synchronize()
    sources = ('campaign_composition_confirm_neural.py','campaign_composition_confirm_data.py','campaign_composition_rekey.py','campaign_composition_roles.py','campaign_composition_models.py','campaign_composition_study.py','campaign_composition.py','campaign_composition_acquire.py','interface_proposals.py','retention_data.py','campaign_composition_runtime.py','campaign_returns_use.py','thinking_runtime.py')
    summary = dict(arm=arm, config=config, data=manifests, initial_state_sha256=initial_hash,
        parameter_count=sum(p.numel() for p in model.parameters()), selected_step=best_step,
        primary_endpoint_step=config['updates'], curves=curves, historical_calibration_parity=parity,
        rekeyed_presentations=int(visits.sum()) if arm == 'n2_rekey' else 0,
        optimizer_presentations=int(visits.sum()), unique_base_events=int((visits>0).sum()),
        swapped_presentations=int(swapped_visits.sum()), clean_presentations=int((visits-swapped_visits).sum()),
        visited_base_view_slots=int((swapped_visits>0).sum() + ((visits-swapped_visits)>0).sum()), refused_training_views=0,
        semantic_pool_size=config['data']['train']['count'], views_per_base_event_maximum=2 if arm == 'n1_roles' else 1,
        identity_code_views='fresh per presentation' if arm == 'n2_rekey' else 'fixed per base event',
        sample_index_stream_sha256=sample_hash.hexdigest(), presentation_swap_coin_stream_sha256=views_hash.hexdigest(), presented_key_stream_sha256=code_hash.hexdigest(),
        checkpoint_sha256={name:file_hash(output/f'{name}.pt') for name in ('endpoint','selected','resume')},
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in sources},
        torch_version=torch.__version__, float32_matmul_precision=torch.get_float32_matmul_precision(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        results=endpoints, elapsed_seconds=time.monotonic()-tick,
        claim='fresh paired finite-IID confirmation replicate; all outcomes retained, aggregate audit pending')
    summary['accuracy_gates'] = {ep: all((report['supplied'] if 'supplied' in report else report)['answer']['correct'] / (report['supplied'] if 'supplied' in report else report)['answer']['total'] >= .98 for key,report in results.items() if key in ('validation','validation_reversed')) for ep,results in endpoints.items()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=('n1_static','n1_roles','n2_rekey'), required=True); parser.add_argument('--device', required=True)
    args = parser.parse_args(); run(json.loads(Path(args.config).read_text()), args.arm, args.output, args.device)
