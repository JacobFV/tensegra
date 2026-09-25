"""R03: exact replay then longer CE exposure on immutable R02 features."""
import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .campaign_returns import sha, calibration, selection_score, group_counts
from .return_crossdelay import prediction_record, save_cache


def load_verified(path, expected):
    assert sha(path)==expected
    return torch.load(io.BytesIO(gzip.decompress(Path(path).read_bytes())),map_location='cpu',weights_only=True)


def run(cfg,out):
    torch.set_num_threads(2);device=cfg['device'];torch.manual_seed(cfg['head_seed'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    assert cfg['endpoints'][0]==900 and 32 not in cfg['delays']
    cache=load_verified(cfg['feature_path'],cfg['feature_sha256'])
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    original=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    head=nn.Linear(1024,33).to(device)
    head.load_state_dict(dict(weight=original['scalar_heads.0.weight'],bias=original['scalar_heads.0.bias']));del original
    x=torch.cat([cache['train/2']['features'][d] for d in cfg['delays']]).to(device)
    y=cache['train/2']['labels'].repeat(len(cfg['delays'])).to(device)
    mean=x.mean(0);scale=x.std(0).clamp_min(.01);z=(x-mean)/scale
    with torch.no_grad():
        weight=head.weight.clone();head.weight.mul_(scale);head.bias.add_(weight@mean)
    assert sha(cfg['reference_path'])==cfg['reference_sha256']
    reference=torch.load(cfg['reference_path'],map_location=device,weights_only=True)
    assert sha(cfg['reference_manifest'])==cfg['reference_manifest_sha256']
    reference_manifest=json.load(gzip.open(cfg['reference_manifest'],'rt'))
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr']);generator=torch.Generator(device=device).manual_seed(cfg['head_seed'])
    visited=torch.zeros(len(x),dtype=torch.bool,device=device);records=[];snapshots={};losses=[];replay={}
    for step in range(1,max(cfg['endpoints'])+1):
        idx=torch.randint(len(x),(cfg['batch_size'],),generator=generator,device=device);visited[idx]=True
        loss=F.cross_entropy(head(z[idx]),y[idx]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if step not in cfg['endpoints']:continue
        packed=np.packbits(visited.cpu().numpy(),bitorder='little').tobytes().hex()
        if step==900:
            torch.testing.assert_close(mean,reference['mean'],atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(scale,reference['scale'],atol=1e-6,rtol=1e-6)
            for key,value in head.state_dict().items():torch.testing.assert_close(value,reference['state'][key],atol=1e-6,rtol=1e-6)
            assert packed==reference_manifest['fits'][-1]['visited_row_bits_little_endian']
            refhead=nn.Linear(1024,33).to(device);refhead.load_state_dict(reference['state']);differences=0
            with torch.no_grad():
                for batch in cache.values():
                    for features in batch['features'].values():
                        features=features.to(device);zs=(features-mean)/scale
                        differences+=int((head(zs).argmax(-1)!=refhead(zs).argmax(-1)).sum())
            assert differences==0
            replay=dict(parameter_tolerance=1e-6,max_parameter_absolute_difference=max(float((v-reference['state'][k]).abs().max()) for k,v in head.state_dict().items()),argmax_differences=differences,visited_bitset_equal=True)
        snap=copy.deepcopy(head).eval();snapshots[step]=snap
        cells=calibration(cache,lambda features:snap((features-mean)/scale),device)
        path=out/f'ce-{step}.pt';torch.save(dict(state={k:v.cpu() for k,v in snap.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),path)
        records.append(dict(step=step,cells=cells,selection_score=selection_score(cells),checkpoint_sha256=sha(path),visited_row_bits_little_endian=packed,
            unique_rows_sampled=int(visited.sum()),unique_events_sampled=int(visited.reshape(len(cfg['delays']),-1).any(0).sum()),optimizer_presentations=step*cfg['batch_size']))
    selected=max(records,key=lambda r:r['selection_score'])['step'];rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            for delay,features in batch['features'].items():
                zs=(features.to(device)-mean)/scale
                for step,snap in snapshots.items():
                    scores=snap(zs);arm=f'ce_{step}';logits[f'{key}/{delay}/{arm}']=scores.cpu()
                    row=prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    row['groups']=group_counts(row);top=scores.topk(2,dim=-1).values
                    row['scalar_margin']=(top[:,0]-top[:,1]).cpu().tolist();rows.append(row)
    path=out/'predictions.json.gz';path.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_exposure.py','campaign_returns.py','return_crossdelay.py')},
        replay900=replay,endpoints=records,selected_step=selected,losses=losses,fit_unique_events=len(cache['train/2']['labels']),fit_rows=len(x),width=1024,head_parameters=sum(p.numel() for p in head.parameters()),
        predictions_sha256=sha(path),logits=save_cache(out/'logits.pt.gz',logits),seconds=time.monotonic()-started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),total_device_bytes=torch.cuda.get_device_properties(0).total_memory))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(seconds=manifest['seconds'],selected_step=selected,replay=replay)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
