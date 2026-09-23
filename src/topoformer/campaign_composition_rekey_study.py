"""C02 paired static-code replay and presentation-level rekeying experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from .campaign_composition_study import get_data, baseline_evaluate, paired_metrics
from .campaign_composition_acquire import sliced, tensor_digest, controlled_rows, labels_from_public, score
from .campaign_composition_models import ContextualBaseline
from .campaign_composition_runtime import build_returns
from .campaign_composition import model_inputs
from .campaign_composition_rekey import rekey
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
    datasets = {}; manifests = {}
    for split, spec in config['data'].items():
        data, public, labels = get_data(spec)
        datasets[split] = (public, labels, data['public'])
        manifests[split] = dict(**spec, public_sha256=tensor_digest(public), labels_sha256=tensor_digest(labels))
        del data
    torch.manual_seed(config['seed'])
    model = ContextualBaseline().to(device)
    initial_hash = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=.01)
    sampler = torch.Generator().manual_seed(config['sample_seed'])
    code_rng = torch.Generator().manual_seed(config['rekey_seed'])
    public, labels, _ = datasets['train']
    visits = torch.zeros(len(labels['task']), dtype=torch.long)
    sample_hash = hashlib.sha256(); views_hash = hashlib.sha256()
    best = -1; best_step = None; curves = []; parity = []
    for step in range(config['updates']+1):
        if step in config['checkpoints']:
            metrics, logits = evaluate(model, *datasets['calibration'][:2], device, config['batch_size'])
            record = dict(step=step, calibration=metrics, optimizer_presentations=int(visits.sum()), unique_base_events=int((visits > 0).sum()))
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
                else:
                    parity.append(dict(step=step, missing_historical_file=str(old)))
            (output/'curve.json').write_text(json.dumps(curves, indent=2)+'\n')
        if step == config['updates']: break
        model.train()
        idx = torch.randint(len(labels['task']), (config['batch_size'],), generator=sampler)
        visits += torch.bincount(idx, minlength=len(visits))
        sample_hash.update(idx.numpy().tobytes())
        targets = sliced(labels, idx, device)
        observed = sliced(public, idx, 'cpu')
        if arm == 'rekey': observed = rekey(observed, code_rng)
        views_hash.update(observed['keys'].numpy().tobytes())
        pred = model({k:v.to(device) for k,v in observed.items()})
        loss = nn.functional.cross_entropy(pred['answer'], targets['task']) + nn.functional.cross_entropy(pred['value'], targets['value'])
        loss = loss + sum(proposal_loss(pred, targets).values())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), output/'endpoint.pt')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.get_state(), rekey_rng=code_rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [], visits=visits, step=config['updates']), output/'resume.pt')
    torch.save(visits, output/'visitation.pt')
    endpoints = {}
    for endpoint in ('endpoint', 'selected'):
        model.load_state_dict(torch.load(output/f'{endpoint}.pt', map_location=device, weights_only=True))
        report = {}
        for split in ('validation', 'fresh_validation'):
            public, labels, rows = datasets[split]
            metrics, logits = evaluate(model, public, labels, device, config['batch_size'])
            report[split] = metrics
            torch.save(dict(logits=logits, labels=labels, public=public), output/f'{endpoint}-{split}.pt')
            # Post-selection, fixed identical nuisance view for both arms/endpoints.
            changed = rekey(public, torch.Generator().manual_seed(config['eval_rekey_seed']))
            metrics, logits = evaluate(model, changed, labels, device, config['batch_size'])
            report[split+'_rekey'] = metrics
            torch.save(dict(logits=logits, labels=labels, public=changed), output/f'{endpoint}-{split}-rekey.pt')
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
    sources = ('campaign_composition_rekey_study.py','campaign_composition_rekey.py','campaign_composition_models.py','campaign_composition_study.py','campaign_composition.py','campaign_composition_acquire.py','interface_proposals.py','retention_data.py','campaign_composition_runtime.py','campaign_returns_use.py','thinking_runtime.py')
    summary = dict(arm=arm, config=config, data=manifests, initial_state_sha256=initial_hash,
        parameter_count=sum(p.numel() for p in model.parameters()), selected_step=best_step,
        primary_endpoint_step=config['updates'], curves=curves, historical_calibration_parity=parity,
        optimizer_presentations=int(visits.sum()), unique_base_events=int((visits>0).sum()),
        rekeyed_presentations=int(visits.sum()) if arm == 'rekey' else 0,
        semantic_pool_size=config['data']['train']['count'], unique_code_views_measured=False,
        sample_index_stream_sha256=sample_hash.hexdigest(), presented_key_stream_sha256=views_hash.hexdigest(),
        checkpoint_sha256={name:file_hash(output/f'{name}.pt') for name in ('endpoint','selected','resume')},
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in sources},
        torch_version=torch.__version__, float32_matmul_precision=torch.get_float32_matmul_precision(),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        results=endpoints, elapsed_seconds=time.monotonic()-tick,
        claim='single development paired nuisance-code intervention; no confirmation or superiority claim')
    if arm == 'static' and config.get('historical_directory'):
        old = Path(config['historical_directory'])/'selected.pt'
        prior = torch.load(old, map_location='cpu', weights_only=True)
        selected = torch.load(output/'selected.pt', map_location='cpu', weights_only=True)
        summary['historical_selected_state_parity'] = dict(historical_sha256=file_hash(old), tensor_equal=all(torch.equal(v,prior[k]) for k,v in selected.items()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=('static','rekey'), required=True); parser.add_argument('--device', required=True)
    args = parser.parse_args(); run(json.loads(Path(args.config).read_text()), args.arm, args.output, args.device)
