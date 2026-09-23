"""R02: nested fresh frozen-feature pools with fixed CE exposure."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .return_memory import ReturnMemoryModel
from .return_crossdelay import feature_batch, prediction_record, save_cache, tensor_hash
from .campaign_returns import sha, calibration, group_counts


def fit_consumer(model, cache, count, cfg, device):
    train=cache['train/2'];delays=cfg['delays']
    x=torch.cat([train['features'][d][:count] for d in delays]).to(device)
    y=train['labels'][:count].repeat(len(delays)).to(device)
    assert y.unique().numel()==33
    mean=x.mean(0);scale=x.std(0).clamp_min(.01);z=(x-mean)/scale
    head=copy.deepcopy(model.scalar_heads[0]);head.requires_grad_(True)
    with torch.no_grad():
        weight=head.weight.clone();head.weight.mul_(scale);head.bias.add_(weight@mean)
        torch.testing.assert_close(head(z[:128]),model.scalar_heads[0](x[:128]),atol=1e-4,rtol=1e-4)
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['ce_lr'])
    generator=torch.Generator(device=device).manual_seed(cfg['head_seed'])
    visited=torch.zeros(len(x),dtype=torch.bool,device=device);curve=[];losses=[]
    for step in range(cfg['ce_updates']+1):
        if step%cfg['ce_check_every']==0 or step==cfg['ce_updates']:
            curve.append(dict(step=step,cells=calibration(cache,lambda v:head((v-mean)/scale),device)))
        if step==cfg['ce_updates']:break
        index=torch.randint(len(x),(cfg['ce_batch_size'],),generator=generator,device=device)
        visited[index]=True
        loss=F.cross_entropy(head(z[index]),y[index]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    return head.eval(),mean,scale,dict(unique_rows_sampled=int(visited.sum()),unique_events_sampled=int(visited.reshape(len(delays),count).any(0).sum()),
        training_event_hashes=train['event_row_hashes'][:count],label_counts=torch.bincount(train['labels'][:count],minlength=33).tolist(),
        rows=len(x),events=count,optimizer_presentations=cfg['ce_updates']*cfg['ce_batch_size'],curve=curve,losses=losses)


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['head_seed']);device=cfg['device']
    assert 32 not in cfg['delays']
    assert cfg['pool_sizes']==sorted(cfg['pool_sizes']) and max(cfg['pool_sizes'])==cfg['data']['train']['size']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    checkpoint=Path(cfg['checkpoint']);assert sha(checkpoint)==cfg['checkpoint_sha256']
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval()
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.requires_grad_(False)
    initial=tensor_hash(model.state_dict());cache={}
    for split,spec in cfg['data'].items():
        for d in ([2] if split=='train' else cfg['eval_distractors']):
            cache[f'{split}/{d}']=feature_batch(model,spec['seed'],spec['size'],d,cfg['delays'],cfg['batch_size'],device)
    sets=[set(cache[f'{split}/2']['event_row_hashes']) for split in cfg['data']]
    assert all(not(a&b) for i,a in enumerate(sets) for b in sets[i+1:])
    for split in cfg['data']:
        for key,batch in cache.items():
            if key.startswith(split+'/'):assert batch['event_row_hashes']==cache[f'{split}/2']['event_row_hashes']
    fits={};records=[]
    for count in cfg['pool_sizes']:
        tick=time.monotonic();head,mean,scale,record=fit_consumer(model,cache,count,cfg,device)
        path=out/f'ce-{count}.pt';torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),path)
        record.update(seconds=time.monotonic()-tick,checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in head.parameters()))
        records.append(record);fits[count]=(head,mean,scale)
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            split=key.split('/')[0]
            for delay,features in batch['features'].items():
                features=features.to(device)
                outputs={'unchanged':model.scalar_heads[0](features)}
                for count,(head,mean,scale) in fits.items():outputs[f'ce_{count}']=head((features-mean)/scale)
                for arm,scores in outputs.items():
                    logits[f'{key}/{delay}/{arm}']=scores.cpu()
                    row=prediction_record(cfg['backbone_seed'],arm,split,delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    top=scores.topk(2,dim=-1).values
                    target=batch['labels'].to(device)
                    nontruth=scores.clone();nontruth.scatter_(1,target[:,None],float('-inf'))
                    row['scalar_margin']=(top[:,0]-top[:,1]).cpu().tolist()
                    row['scalar_true_margin']=(scores.gather(1,target[:,None])[:,0]-nontruth.max(-1).values).cpu().tolist()
                    row['groups']=group_counts(row)
                    # Also preserve genuinely in-pool training fit, not only the maximal population.
                    if split=='train' and arm!='unchanged':
                        count=int(arm.split('_')[1]);pred=scores.argmax(-1).cpu()
                        row['in_pool_value']=dict(correct=int((pred[:count]==batch['labels'][:count]).sum()),total=count)
                        if count<len(pred):row['heldout_prefix_tail_value']=dict(correct=int((pred[count:]==batch['labels'][count:]).sum()),total=len(pred)-count)
                    rows.append(row)
    assert tensor_hash(model.state_dict())==initial
    predictions=out/'predictions.json.gz';predictions.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_diversity.py','campaign_returns.py','return_crossdelay.py','return_memory.py','retention_data.py','thinking.py')},
        checkpoint_sha256=sha(checkpoint),backbone_state_sha256=initial,width=1024,backbone_parameters=sum(p.numel() for p in model.parameters()),
        fits=records,predictions_sha256=sha(predictions),features=save_cache(out/'features.pt.gz',cache),logits=save_cache(out/'logits.pt.gz',logits),
        seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),total_device_bytes=torch.cuda.get_device_properties(0).total_memory))
    raw=json.dumps(manifest,indent=2).encode();(out/'manifest.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    print(json.dumps(dict(seconds=manifest['seconds'],peak_cuda_allocated=manifest['peak_cuda_allocated'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
