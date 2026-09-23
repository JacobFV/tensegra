"""R09: public binary-phase versus shared linear return consumers."""
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


class PhaseReadout(nn.Module):
    """Public ingestion-versus-recurrent selection, never a per-delay oracle."""
    def __init__(self, linear):
        super().__init__()
        self.ingestion = copy.deepcopy(linear)
        self.recurrent = copy.deepcopy(linear)

    def forward(self, x, recurrent):
        mask = torch.as_tensor(recurrent, dtype=torch.bool, device=x.device)
        if mask.ndim: mask = mask[:, None]
        return torch.where(mask, self.recurrent(x), self.ingestion(x))


def predict(head, x, recurrent):
    return head(x, recurrent) if isinstance(head, PhaseReadout) else head(x)


def calibration_cells(cache, head, mean, scale, device):
    cells = []; grid = []
    with torch.no_grad():
        for key, batch in cache.items():
            if not key.startswith(('calibration/', 'calibration_grid/')): continue
            for delay, features in batch['features'].items():
                pred = predict(head, (features.to(device)-mean)/scale, delay>0).argmax(-1).cpu()
                labels = batch['labels']
                if key.startswith('calibration/'):
                    cells.append(dict(delay=delay,distractors=batch['distractors'],correct=int((pred==labels).sum()),total=len(labels)))
                else:
                    for typ in batch['targets']['type'].unique().tolist():
                        for label in labels[batch['targets']['type']==typ].unique().tolist():
                            mask=(batch['targets']['type']==typ)&(labels==label)
                            grid.append(dict(delay=delay,type=typ,label=label,correct=int((pred[mask]==label).sum()),total=int(mask.sum())))
    return dict(cells=cells,grid=grid)


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
    heads = {'linear': linear, 'phase_linear': PhaseReadout(linear)}
    with torch.no_grad():
        torch.testing.assert_close(heads['linear'](z[:128]), heads['phase_linear'](z[:128], False), rtol=0, atol=0)
    fits = []; rows = []; logits = {}; replay = None
    for arm, head in heads.items():
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'])
        generator = torch.Generator(device=device).manual_seed(cfg['head_seed'])
        visited = torch.zeros(len(x), dtype=torch.bool, device=device)
        phase_presentations = [0, 0]
        curve = []; losses = []
        for step in range(cfg['updates']+1):
            if step % cfg['check_every'] == 0 or step == cfg['updates']:
                curve.append(dict(step=step, **calibration_cells(cache,head,mean,scale,device)))
            if step == cfg['updates']: break
            index = torch.randint(len(x), (cfg['batch_size'],), generator=generator, device=device)
            visited[index] = True
            recurrent = index // len(train['labels']) > 0
            phase_presentations[0] += int((~recurrent).sum())
            phase_presentations[1] += int(recurrent.sum())
            loss = F.cross_entropy(predict(head,z[index],recurrent), y[index])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        packed = np.packbits(visited.cpu().numpy(), bitorder='little').tobytes().hex()
        if arm == 'linear' and cfg['require_replay']:
            assert sha(cfg['reference_path']) == cfg['reference_sha256']
            reference = torch.load(cfg['reference_path'], map_location=device, weights_only=True)
            for name, value in head.state_dict().items():
                torch.testing.assert_close(value, reference['state'][name], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(mean, reference['mean'], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(scale, reference['scale'], atol=1e-6, rtol=1e-6)
            assert sha(cfg['reference_manifest']) == cfg['reference_manifest_sha256']
            old = json.load(gzip.open(cfg['reference_manifest'], 'rt'))
            assert packed == next(f for f in old['fits'] if f['arm']=='balanced')['visited_row_bits_little_endian']
            replay = {'max_parameter_difference': max(float((v-reference['state'][k]).abs().max()) for k,v in head.state_dict().items()), 'visited_bitset_equal': True}
        path = out/f'{arm}.pt'
        torch.save({'state': {k:v.cpu() for k,v in head.state_dict().items()}, 'mean':mean.cpu(), 'scale':scale.cpu()}, path)
        fits.append({'arm':arm,'parameters':sum(p.numel() for p in head.parameters()),'checkpoint_sha256':sha(path),
                     'curve':curve,'losses':losses,'phase_presentations':phase_presentations,'optimizer_presentations':cfg['updates']*cfg['batch_size'],
                     'visited_row_bits_little_endian':packed,'unique_rows_sampled':int(visited.sum()),
                     'unique_events_sampled':int(visited.reshape(len(cfg['delays']),-1).any(0).sum())})
        with torch.no_grad():
            for key, batch in cache.items():
                for delay, features in batch['features'].items():
                    scores = predict(head,(features.to(device)-mean)/scale,delay>0)
                    logits[f'{key}/{delay}/{arm}'] = scores.cpu()
                    row = prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    row['groups'] = group_counts(row)
                    rows.append(row)
    assert fits[0]['visited_row_bits_little_endian'] == fits[1]['visited_row_bits_little_endian']
    predpath = out/'predictions.json.gz'
    predpath.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest = {'config':cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                'source':{n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_phase.py','campaign_returns.py','campaign_returns_exposure.py','return_crossdelay.py')},
                'fits':fits,'replay':replay,'width':1024,'fit_events':len(train['labels']),'fit_rows':len(x),
                'training_event_hashes':train['event_row_hashes'],'label_support':torch.bincount(train['labels'],minlength=33).tolist(),
                'predictions_sha256':sha(predpath),'logits':save_cache(out/'logits.pt.gz',logits),
                'seconds':time.monotonic()-started,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
                'process_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()}}
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0))
    print(json.dumps({'seconds':manifest['seconds'],'replay':replay,'peak_cuda_allocated':manifest['peak_cuda_allocated']}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)
