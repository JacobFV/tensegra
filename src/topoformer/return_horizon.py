"""Stage 11: existing return workspace acquisition under two timing schedules."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .return_memory import ReturnMemoryModel, FIELDS
from .return_memory_study import move
from .return_crossdelay import feature_batch, prediction_record, tensor_hash
from .retention_data import make_batch


def schedule_counts(schedule, updates):
    if not schedule or any(delay < 0 or delay > 16 for delay in schedule):
        raise ValueError('Training schedule must stay within zero through sixteen')
    return {str(delay): sum(schedule[step % len(schedule)] == delay for step in range(updates))
            for delay in sorted(set(schedule))}


def update(model, optimizer, batch, delay):
    model.train()
    output = model(batch['public'], delay, 'factorized', 'persistent')
    terms = {field: F.cross_entropy(output['logits'][field], batch['targets'][field]) for field in FIELDS}
    optimizer.zero_grad(set_to_none=True)
    sum(terms.values()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return {field: float(loss.detach()) for field, loss in terms.items()}


def evaluate(model, seed, size, delays, device, arm, split, distractors=(2,)):
    model.eval()
    rows = []
    for distractor in distractors:
        batch = feature_batch(model, seed, size, distractor, sorted(delays), 32, device)
        for delay in sorted(delays):
            row = prediction_record(0, arm, split, delay, distractor, batch,
                                    batch['original_predictions'][delay]['value'])
            row['event_row_hashes'] = batch['event_row_hashes']
            rows.append(row)
    return rows


def acquisition_gate(rows):
    return bool(rows) and all(row['counts']['joint']['correct'] == row['counts']['joint']['total'] for row in rows)


def run(config, output):
    torch.set_num_threads(2)
    device = config['device']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config['checkpoint']).expanduser()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    manifest = dict(config=config, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    source={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('return_horizon.py','return_memory.py','return_crossdelay.py',
                                         'return_diagnostics_probe.py','retention_data.py','thinking.py')},
                    config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                    environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     device=torch.cuda.get_device_name(),
                                     total_device_bytes=torch.cuda.get_device_properties(0).total_memory), runs=[])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if config['mode'] == 'profile':
        model = ReturnMemoryModel(width=1024, encoding='factorized').to(device)
        model.load_state_dict(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
        batch = move(make_batch(config['seed'], config['batch_size']), device)
        update(model, optimizer, batch, 0)
        torch.cuda.synchronize()
        for delay in (0,4,16):
            tick = time.monotonic()
            losses = update(model, optimizer, batch, delay)
            torch.cuda.synchronize()
            manifest['runs'].append(dict(delay=delay, kind='training', seconds=time.monotonic()-tick, losses=losses))
        tick = time.monotonic()
        with torch.no_grad(): model.eval()(batch['public'],32,'factorized','persistent')
        torch.cuda.synchronize()
        manifest['runs'].append(dict(delay=32,kind='inference',seconds=time.monotonic()-tick))
        manifest['parameters'] = sum(p.numel() for p in model.parameters())
    elif config['mode'] == 'acquisition':
        assert config['seed'] != config['fresh_seed']
        batch_cpu = make_batch(config['seed'], config['batch_size'])
        fresh = make_batch(config['fresh_seed'], config['fresh_size'])
        assert tensor_hash(batch_cpu['public']['event']) != tensor_hash(fresh['public']['event'])
        batch = move(batch_cpu, device)
        for arm, schedule in config['schedules'].items():
            model = ReturnMemoryModel(width=1024, encoding='factorized').to(device)
            model.load_state_dict(state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
            rows, curve = [], []
            tick = time.monotonic()
            for step in range(config['updates']+1):
                if step % config['check_every'] == 0 or step == config['updates']:
                    measured = evaluate(model, config['seed'], config['batch_size'], set(schedule),device,arm,'fixed')
                    for row in measured: row['update'] = step
                    rows.extend(measured)
                    print(arm, step, [(r['target_delay'],r['counts']['joint']['correct']) for r in measured], flush=True)
                if step == config['updates']: break
                terms = update(model, optimizer, batch, schedule[step % len(schedule)])
                curve.append(dict(update=step+1, delay=schedule[step % len(schedule)], loss=terms))
            final = [row for row in rows if row['update']==config['updates']]
            rows += evaluate(model,config['fresh_seed'],config['fresh_size'],set(schedule),device,arm,'fresh')
            path = output/f'{arm}-predictions.json.gz'
            path.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
            modelpath=output/f'{arm}-checkpoint.pt'; torch.save(model.state_dict(),modelpath)
            manifest['runs'].append(dict(arm=arm, passed_fixed_gate=acquisition_gate(final),curve=curve,
                schedule_counts=schedule_counts(schedule,config['updates']),presentations=config['updates']*config['batch_size'],
                unique_training_events=config['batch_size'],event_sha256=tensor_hash(batch_cpu['public']['event']),
                checkpoint_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),seconds=time.monotonic()-tick))
            del optimizer, model
            torch.cuda.empty_cache()
    else:
        raise ValueError('Unregistered mode')
    manifest.update(seconds=time.monotonic()-started, width=1024,
                    peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                    process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({key:manifest[key] for key in ('seconds','peak_cuda_allocated','process_peak_rss')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.output)
