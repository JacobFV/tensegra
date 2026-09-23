"""R08: fixed-capacity readout learning-rate acquisition screen."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .campaign_returns import sha, calibration, group_counts
from .campaign_returns_exposure import load_verified
from .return_crossdelay import prediction_record, save_cache


from .campaign_returns_capacity import ResidualReadout


def calibration_grid(cache, predictor, device):
    cells = []
    with torch.no_grad():
        for key, batch in cache.items():
            if not key.startswith("calibration_grid/"): continue
            for delay, features in batch["features"].items():
                pred = predictor(features.to(device)).argmax(-1).cpu()
                for typ in batch["targets"]["type"].unique().tolist():
                    for label in batch["labels"][batch["targets"]["type"]==typ].unique().tolist():
                        mask = (batch["targets"]["type"]==typ) & (batch["labels"]==label)
                        cells.append(dict(delay=delay, type=typ, label=label, correct=int((pred[mask]==label).sum()), total=int(mask.sum())))
    return cells


def run(cfg, out):
    torch.set_num_threads(2)
    torch.manual_seed(cfg['initialization_seed'])
    device = cfg['device']
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    cache = load_verified(cfg['feature_path'], cfg['feature_sha256'])
    assert all(32 not in b['features'] for b in cache.values())
    assert sha(cfg['checkpoint']) == cfg['checkpoint_sha256']
    original = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    linear = nn.Linear(1024, 33).to(device)
    linear.load_state_dict({'weight': original['scalar_heads.0.weight'], 'bias': original['scalar_heads.0.bias']})
    del original
    train = cache['train_balanced/2']
    x = torch.cat([train['features'][d] for d in cfg['delays']]).to(device)
    y = train['labels'].repeat(len(cfg['delays'])).to(device)
    assert y.unique().numel() == 33
    mean = x.mean(0); scale = x.std(0).clamp_min(.01); z = (x-mean)/scale
    with torch.no_grad():
        weight = linear.weight.clone(); linear.weight.mul_(scale); linear.bias.add_(weight@mean)
    initial = ResidualReadout(linear)
    heads = {f'residual_lr_{lr:g}': (copy.deepcopy(initial), lr) for lr in cfg['learning_rates']}
    for arm, spec in cfg['frozen_references'].items():
        assert sha(spec['path']) == spec['sha256']
        saved = torch.load(spec['path'], map_location=device, weights_only=True)
        head = copy.deepcopy(linear if arm=='linear_reference' else initial)
        head.load_state_dict(saved['state'])
        torch.testing.assert_close(mean,saved['mean'],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(scale,saved['scale'],atol=1e-6,rtol=1e-6)
        heads[arm] = (head, None)
    fits = []; rows = []; logits = {}
    for arm, (head, lr) in heads.items():
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr or .003)
        generator = torch.Generator(device=device).manual_seed(cfg['head_seed'])
        visited = torch.zeros(len(x), dtype=torch.bool, device=device)
        curve = []; losses = []
        updates = cfg['updates'] if lr is not None else 0
        for step in range(updates+1):
            if step % cfg['check_every'] == 0 or step == updates:
                curve.append({'step': step, 'cells': calibration(cache, lambda v: head((v-mean)/scale), device), 'grid':calibration_grid(cache, lambda v: head((v-mean)/scale), device)})
            if step == updates: break
            index = torch.randint(len(x), (cfg['batch_size'],), generator=generator, device=device)
            visited[index] = True
            loss = F.cross_entropy(head(z[index]), y[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        packed = np.packbits(visited.cpu().numpy(), bitorder='little').tobytes().hex()
        path = out/f'{arm}.pt'
        torch.save({'state': {k:v.cpu() for k,v in head.state_dict().items()}, 'mean':mean.cpu(), 'scale':scale.cpu()}, path)
        fits.append({'arm':arm,'learning_rate':lr,'trained_here':lr is not None,'parameters':sum(p.numel() for p in head.parameters()),'checkpoint_sha256':sha(path),
                     'curve':curve,'losses':losses,'optimizer_presentations':updates*cfg['batch_size'],
                     'visited_row_bits_little_endian':packed,'unique_rows_sampled':int(visited.sum()),
                     'unique_events_sampled':int(visited.reshape(len(cfg['delays']),-1).any(0).sum())})
        with torch.no_grad():
            for key, batch in cache.items():
                for delay, features in batch['features'].items():
                    scores = head((features.to(device)-mean)/scale)
                    logits[f'{key}/{delay}/{arm}'] = scores.cpu()
                    row = prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    row['groups'] = group_counts(row)
                    rows.append(row)
    assert len({f['visited_row_bits_little_endian'] for f in fits if f['trained_here']}) == 1
    candidates = [f for f in fits if f['trained_here']]
    def selection(f):
        last=f['curve'][-1]
        return (min(c['correct']/c['total'] for c in last['grid']), min(c['correct']/c['total'] for c in last['cells']))
    selected = max(candidates, key=selection)['arm']
    predpath = out/'predictions.json.gz'
    predpath.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest = {'config':cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                'source':{n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_optimizer.py','campaign_returns_capacity.py','campaign_returns.py','campaign_returns_exposure.py','return_crossdelay.py')},
                'fits':fits,'selected':selected,'selection_rule':'maximum minimum calibration grid, then mixture, then declared LR order','width':1024,'fit_events':len(train['labels']),'fit_rows':len(x),
                'training_event_hashes':train['event_row_hashes'],'label_support':torch.bincount(train['labels'],minlength=33).tolist(),
                'predictions_sha256':sha(predpath),'logits':save_cache(out/'logits.pt.gz',logits),
                'seconds':time.monotonic()-started,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
                'process_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()}}
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0))
    print(json.dumps({'seconds':manifest['seconds'],'selected':selected,'peak_cuda_allocated':manifest['peak_cuda_allocated']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)
