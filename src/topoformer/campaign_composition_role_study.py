"""C03 paired N1 replay and independently sampled public binary-role swaps."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from .campaign_composition_study import get_data, baseline_evaluate, paired_metrics
from .campaign_composition_acquire import sliced, tensor_digest, controlled_rows, labels_from_public, score
from .campaign_composition_models import NeuralOperandBaseline
from .campaign_composition_runtime import build_returns, load_checked
from .campaign_composition import model_inputs
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
    model = NeuralOperandBaseline().to(device)
    model.lowerer.load_state_dict(load_checked(config['interfaces']['lowerer'], config['interfaces']['lowerer_sha256'], device))
    initial_hash = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
    sampler = torch.Generator().manual_seed(config['sample_seed'])
    view_rng = torch.Generator().manual_seed(config['role_seed'])
    public, labels, _ = datasets['train']
    visits = torch.zeros(len(labels['task']), dtype=torch.long)
    sample_hash = hashlib.sha256(); views_hash = hashlib.sha256()
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
            if arm == 'static' and config.get('historical_directory'):
                old = Path(config['historical_directory'])/f"calibration-{config['learning_rate']}-{step}.pt"
                if old.exists():
                    prior = torch.load(old, map_location='cpu', weights_only=True)
                    parity.append(dict(step=step, historical_sha256=file_hash(old), labels_equal=all(torch.equal(v, prior['labels'][k]) for k,v in datasets['calibration'][1].items()),
                        logits_equal={k:torch.equal(v,prior['logits'][k]) for k,v in logits.items()},
                        max_abs_logit_difference={k:float((v-prior['logits'][k]).abs().max()) for k,v in logits.items()}))
                    if not parity[-1]['labels_equal']:
                        (output/'historical-parity-failure.json').write_text(json.dumps(parity[-1],indent=2)+'\n')
                        raise RuntimeError('historical calibration labels changed')
                else:
                    (output/'historical-parity-failure.json').write_text(json.dumps(dict(step=step,missing_historical_file=str(old)),indent=2)+'\n')
                    raise FileNotFoundError(f'required historical calibration file: {old}')
            (output/'curve.json').write_text(json.dumps(curves, indent=2)+'\n')
        if step == config['updates']: break
        model.train()
        idx = torch.randint(len(labels['task']), (config['batch_size'],), generator=sampler)
        visits += torch.bincount(idx, minlength=len(visits))
        sample_hash.update(idx.numpy().tobytes())
        targets = sliced(labels, idx, 'cpu')
        observed = sliced(public, idx, 'cpu')
        coin = torch.rand(config['batch_size'], generator=view_rng) < .5 if arm == 'roles' else torch.zeros(config['batch_size'], dtype=torch.bool)
        if arm == 'roles':
            observed = choose_views(observed, sliced(swapped['train'][0], idx, 'cpu'), coin)
            targets = choose_views(targets, sliced(swapped['train'][1], idx, 'cpu'), coin)
        swapped_visits += torch.bincount(idx[coin], minlength=len(visits))
        views_hash.update(coin.numpy().tobytes())
        targets = {k:v.to(device) for k,v in targets.items()}
        pred = model({k:v.to(device) for k,v in observed.items()})
        loss = nn.functional.cross_entropy(pred['answer'], targets['task']) + nn.functional.cross_entropy(pred['value'], targets['value'])
        loss = loss + sum(proposal_loss(pred, targets).values())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), output/'endpoint.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.get_state(), role_rng=view_rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [], visits=visits, swapped_visits=swapped_visits, step=config['updates']), output/'resume.pt')
    torch.save(dict(base=visits, swapped=swapped_visits), output/'visitation.pt')
    endpoints = {}
    for endpoint in ('endpoint', 'selected'):
        model.load_state_dict(torch.load(output/f'{endpoint}.pt', map_location=device, weights_only=True))
        report = {}
        for split in ('validation', 'fresh_validation'):
            public, labels, rows = datasets[split]
            metrics, logits = evaluate(model, public, labels, device, config['batch_size'])
            report[split] = metrics
            torch.save(dict(logits=logits, labels=labels, public=public), output/f'{endpoint}-{split}.pt')
            changed, changed_labels = swapped[split]
            metrics, logits = evaluate(model, changed, changed_labels, device, config['batch_size'])
            report[split+'_reversed'] = dict(supplied=metrics, paired=paired_metrics(logits['answer'].argmax(-1), labels['task'], changed_labels['task']))
            torch.save(dict(logits=logits, labels=changed_labels, original_labels=labels, public=changed), output/f'{endpoint}-{split}-reversed.pt')
        # Same registered C01 public controls, on the new development population.
        report['controls'] = {}
        for i, kind in enumerate(config['public_controls']):
            rows = controlled_rows(datasets['fresh_validation'][2], kind, config['control_seed']+i)
            oracle = labels_from_public(rows); bundle = build_returns(rows, oracle['primitive'], oracle['targets'])
            if len(bundle['indices']) != len(rows): raise RuntimeError('control outside supplied schema')
            event = bundle['public']['event']; query = bundle['public']['query']
            labels = dict(oracle, task=answer(event['values'][:,0], query), value=(event['values'][:,0]*2+16).long())
            observed = model_inputs(rows)
            metrics, logits = evaluate(model, observed, labels, device, config['batch_size'])
            report['controls'][kind] = dict(supplied=metrics, paired=paired_metrics(logits['answer'].argmax(-1), datasets['fresh_validation'][1]['task'], labels['task']))
            torch.save(dict(logits=logits, labels=labels, original_labels=datasets['fresh_validation'][1], public=observed), output/f'{endpoint}-control-{kind}.pt')
        endpoints[endpoint] = report
    if device.startswith('cuda'): torch.cuda.synchronize()
    sources = ('campaign_composition_role_study.py','campaign_composition_roles.py','campaign_composition_models.py','campaign_composition_study.py','campaign_composition.py','campaign_composition_acquire.py','interface_proposals.py','retention_data.py','campaign_composition_runtime.py','campaign_returns_use.py','thinking_runtime.py')
    summary = dict(arm=arm, config=config, data=manifests, initial_state_sha256=initial_hash,
        parameter_count=sum(p.numel() for p in model.parameters()), selected_step=best_step,
        primary_endpoint_step=config['updates'], curves=curves, historical_calibration_parity=parity,
        replay_numerical_deviation_requires_coordinator_review=any(not all(x['logits_equal'].values()) for x in parity),
        optimizer_presentations=int(visits.sum()), unique_base_events=int((visits>0).sum()),
        swapped_presentations=int(swapped_visits.sum()), clean_presentations=int((visits-swapped_visits).sum()),
        visited_base_view_slots=int((swapped_visits>0).sum() + ((visits-swapped_visits)>0).sum()), refused_training_views=0,
        semantic_pool_size=config['data']['train']['count'], views_per_base_event_maximum=2,
        sample_index_stream_sha256=sample_hash.hexdigest(), presentation_swap_coin_stream_sha256=views_hash.hexdigest(),
        checkpoint_sha256={name:file_hash(output/f'{name}.pt') for name in ('endpoint','selected','resume')},
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in sources},
        torch_version=torch.__version__, float32_matmul_precision=torch.get_float32_matmul_precision(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        results=endpoints, elapsed_seconds=time.monotonic()-tick,
        claim='single development paired binary-role augmentation; original failures retained; no confirmation or superiority claim')
    if arm == 'static' and config.get('historical_directory'):
        old = Path(config['historical_directory'])/'selected.pt'
        prior = torch.load(old, map_location='cpu', weights_only=True)
        selected = torch.load(output/'selected.pt', map_location='cpu', weights_only=True)
        summary['historical_selected_state_parity'] = dict(historical_sha256=file_hash(old), tensor_equal=all(torch.equal(v,prior[k]) for k,v in selected.items()))
        if not summary['historical_selected_state_parity']['tensor_equal']:
            summary['replay_numerical_deviation_requires_coordinator_review'] = True
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=('static','roles'), required=True); parser.add_argument('--device', required=True)
    args = parser.parse_args(); run(json.loads(Path(args.config).read_text()), args.arm, args.output, args.device)
