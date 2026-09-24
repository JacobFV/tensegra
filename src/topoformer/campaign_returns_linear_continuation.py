"""R11: exact replay, then paired optimizer-only forks on frozen R10 features."""
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

from .campaign_returns import sha, group_counts
from .campaign_returns_exposure import load_verified
from .campaign_returns_phase import calibration_cells
from .return_crossdelay import prediction_record, save_cache


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_tree(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_tree(v) for v in x)
    return x


def update(head, optimizer, generator, z, y, steps, batch_size, callback=None):
    visited=torch.zeros(len(z),dtype=torch.bool,device=z.device);losses=[];sampled=[];order=hashlib.sha256()
    for step in range(steps):
        index=torch.randint(len(z),(batch_size,),generator=generator,device=z.device)
        visited[index]=True;sampled.append(index.detach())
        loss=F.cross_entropy(head(z[index]),y[index])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
        if len(sampled)==128:
            order.update(torch.cat(sampled).cpu().numpy().astype("<i8",copy=False).tobytes());sampled.clear()
        if callback is not None:callback(step+1)
    if sampled:order.update(torch.cat(sampled).cpu().numpy().astype("<i8",copy=False).tobytes())
    return losses,visited,order.hexdigest()


def require_exact_replay(head, mean, scale, visited, reference, old_fit):
    for name,value in head.state_dict().items():
        if not torch.equal(value.cpu(),reference['state'][name].cpu()):raise ValueError('Replay parameter mismatch: '+name)
    if not torch.equal(mean.cpu(),reference['mean'].cpu()) or not torch.equal(scale.cpu(),reference['scale'].cpu()):
        raise ValueError('Replay normalization mismatch')
    bits=np.packbits(visited.cpu().numpy(),bitorder='little').tobytes().hex()
    if bits!=old_fit['visited_row_bits_little_endian']:raise ValueError('Replay sampled-row mismatch')
    return bits


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['initialization_seed'])
    device=cfg['device'];out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();timings={};torch.cuda.reset_peak_memory_stats()
    for field in ('checkpoint','reference_path','reference_manifest','reference_predictions'):
        if sha(cfg[field])!=cfg[field+'_sha256']:raise ValueError('Hash mismatch: '+field)
    cache=load_verified(cfg['feature_path'],cfg['feature_sha256'])
    assert all(set(b['features']).issubset({0,1,2,4,8,16}) for b in cache.values())
    train=cache['train/2'];assert len(train['labels'])==cfg['fit_events']
    initial=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    head=nn.Linear(1024,33).to(device)
    head.load_state_dict({'weight':initial['scalar_heads.0.weight'],'bias':initial['scalar_heads.0.bias']});del initial
    x=torch.cat([train['features'][d] for d in cfg['delays']]).to(device)
    y=train['labels'].repeat(len(cfg['delays'])).to(device)
    mean=x.mean(0);scale=x.std(0).clamp_min(.01);z=(x-mean)/scale
    with torch.no_grad():
        weight=head.weight.clone();head.weight.mul_(scale);head.bias.add_(weight@mean)
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['initial_lr'])
    generator=torch.Generator(device=device).manual_seed(cfg['head_seed'])
    torch.cuda.synchronize();timings['load_and_prepare_seconds']=time.monotonic()-started;tick=time.monotonic()
    replay_losses,visited,replay_order=update(head,optimizer,generator,z,y,cfg['replay_steps'],cfg['batch_size'])
    reference=torch.load(cfg['reference_path'],map_location='cpu',weights_only=True)
    old=json.load(gzip.open(cfg['reference_manifest'],'rt'))
    old_fit=next(f for f in old['fits'] if f['arm']==cfg['reference_arm'])
    old_rows=json.load(gzip.open(cfg['reference_predictions'],'rt'))
    old_reference={(r['split'],r['target_delay'],r['distractors']):r for r in old_rows if r['head']==cfg['reference_arm']}
    bits=require_exact_replay(head,mean,scale,visited,reference,old_fit)
    if replay_losses!=old_fit['losses']:raise ValueError('Replay loss sequence mismatch')
    torch.cuda.synchronize();timings['exact_replay_seconds']=time.monotonic()-tick;tick=time.monotonic()
    endpoint={'state':cpu_tree(head.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'rng_state':generator.get_state().cpu(),
              'mean':mean.cpu(),'scale':scale.cpu(),'step':cfg['replay_steps'],'sampled_index_sha256':replay_order}
    torch.save(endpoint,out/'replayed_reference.pt')
    heads={'reference':copy.deepcopy(head).eval()};records=[];curves={}
    for arm,lr in cfg['forks'].items():
        h=copy.deepcopy(head);opt=torch.optim.AdamW(h.parameters(),lr=cfg['initial_lr']);opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for group in opt.param_groups:group['lr']=lr
        rng=torch.Generator(device=device);rng.set_state(generator.get_state())
        curve=[]
        def observe(step):
            if step%cfg['check_every']==0 or step==cfg['extra_steps']:
                curve.append(dict(step=cfg['replay_steps']+step,**calibration_cells(cache,h,mean,scale,device)))
        losses,seen,sampled_order=update(h,opt,rng,z,y,cfg['extra_steps'],cfg['batch_size'],observe)
        checkpoint=out/f'{arm}.pt'
        torch.save(dict(state=cpu_tree(h.state_dict()),optimizer=cpu_tree(opt.state_dict()),rng_state=rng.get_state().cpu(),
                        mean=mean.cpu(),scale=scale.cpu(),step=cfg['replay_steps']+cfg['extra_steps'],sampled_index_sha256=sampled_order),checkpoint)
        packed=np.packbits(seen.cpu().numpy(),bitorder='little').tobytes().hex()
        records.append(dict(arm=arm,lr=lr,losses=losses,curve=curve,checkpoint_sha256=sha(checkpoint),
                            optimizer_presentations=cfg['extra_steps']*cfg['batch_size'],visited_row_bits_little_endian=packed,
                            sampled_index_sha256=sampled_order,final_rng_sha256=hashlib.sha256(rng.get_state().cpu().numpy().tobytes()).hexdigest(),
                            unique_rows_sampled=int(seen.sum()),unique_events_sampled=int(seen.reshape(len(cfg['delays']),-1).any(0).sum())))
        heads[arm]=h.eval()
    assert len({r['visited_row_bits_little_endian'] for r in records})==1
    assert len({r['sampled_index_sha256'] for r in records})==1
    assert len({r['final_rng_sha256'] for r in records})==1
    torch.cuda.synchronize();timings['fork_fit_seconds']=time.monotonic()-tick;tick=time.monotonic()
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            for delay,features in batch['features'].items():
                normalized=(features.to(device)-mean)/scale
                for arm,h in heads.items():
                    scores=h(normalized);logits[f'{key}/{delay}/{arm}']=scores.cpu()
                    row=prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    if arm=='reference':
                        historical=old_reference[(row['split'],delay,row['distractors'])]
                        for field in ('targets','predictions','event_sha256'):
                            if row[field]!=historical[field]:raise ValueError('Historical prediction replay mismatch: '+field)
                    row['groups']=group_counts(row);rows.append(row)
    torch.cuda.synchronize();timings['endpoint_evaluation_seconds']=time.monotonic()-tick
    path=out/'predictions.json.gz';path.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={n:sha(Path(__file__).with_name(n))for n in ('campaign_returns_linear_continuation.py','campaign_returns.py','campaign_returns_exposure.py','campaign_returns_phase.py','return_crossdelay.py')},
        replay=dict(exact_reference_predictions=True,exact_parameters=True,exact_normalization=True,exact_loss_sequence=True,visited_row_bits_little_endian=bits,
                    sampled_index_sha256=replay_order,historical_index_order_hash_available=False,
                    final_rng_sha256=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
                    checkpoint_sha256=sha(out/'replayed_reference.pt')),fits=records,phase_seconds=timings,
        width=1024,parameters=sum(p.numel()for p in head.parameters()),fit_events=len(train['labels']),fit_rows=len(x),
        predictions_sha256=sha(path),logits=save_cache(out/'logits.pt.gz',logits),seconds=time.monotonic()-started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0))
    print(json.dumps(dict(seconds=manifest['seconds'],phase_seconds=timings,replay='exact')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output)
