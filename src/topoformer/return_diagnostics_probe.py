"""Frozen Stage8 boundary probes. No gradients or gold inputs enter the backbone."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .return_memory import ReturnMemoryModel
from .retention_data import make_batch
from .return_memory_study import move


def capture(model, public, encoding, availability, delays=(0,1,4,16,32)):
    event=public['event']; batch=event['values'].shape[0]
    raw=model.encoders[0]((event['values']/model.value_limit).unsqueeze(-1))[:,0]
    raw=F.pad(raw,(0,model.width-raw.shape[-1]))
    memory=model.encode(event,encoding)
    # Correct facet selection is privileged only as a probe boundary.
    result={'scalar_premix':raw,'memory':memory[:,0]}
    state=model.initial[None].expand(batch,-1,-1)
    state=state+model.initial_read(state,memory,memory,need_weights=False)[0]
    if 0 in delays: result['workspace_0']=model.norm(state)[:,0]
    for step in range(max(delays)):
        mem=model.features(public['distractors'][:,step])
        if availability=='persistent': mem=torch.cat((mem,memory),1)
        mask=torch.ones(mem.shape[:2],dtype=torch.bool,device=state.device)
        bias=state.new_zeros(batch,model.head_count,6,6)
        for block in model.blocks: state,_=block(state,mem,mask,bias)
        if step+1 in delays: result[f'workspace_{step+1}']=model.norm(state)[:,0]
    return result,state


def event_hash(event):
    h=hashlib.sha256()
    for name in sorted(event):
        h.update(name.encode()); h.update(event[name].contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def fit_ridge(x,y,regularization):
    mean=x.mean(0); scale=x.std(0).clamp_min(.01)
    z=(x-mean)/scale; z=torch.cat((z,torch.ones(len(z),1,device=z.device)),1)
    penalty=torch.eye(z.shape[1],device=z.device)*regularization; penalty[-1,-1]=0
    coef=torch.linalg.solve(z.T@z+penalty,z.T@y)
    return mean,scale,coef


def predict_ridge(x,fit):
    mean,scale,coef=fit
    return torch.cat(((x-mean)/scale,torch.ones(len(x),1,device=x.device)),1)@coef


def metrics(pred,target):
    err=(pred-target).float()/2
    return dict(correct=int((pred==target).sum()),total=len(target),mae=float(err.abs().mean()),signed_error=float(err.mean()),within_half=int((err.abs()<=.5).sum()))


def run(config, output):
    torch.set_num_threads(2); output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=config.get('device','cuda'); started=time.monotonic(); records=[]
    if len({spec[0] for spec in config['splits'].values()}) != len(config['splits']): raise ValueError('split seeds overlap')
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('return_diagnostics_probe.py','return_memory.py','retention_data.py','thinking.py')}
    manifest=dict(config=config,source=source,runs=[])
    for seed in config['seeds']:
      for encoding in config['encodings']:
       for availability in config['availability']:
        name=f'{seed}-{encoding}-{availability}'; path=Path(config['checkpoint_dir'])/(name+'.pt')
        model=ReturnMemoryModel(width=1024,encoding=encoding).to(device)
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
        checkpoint_sha=hashlib.sha256(path.read_bytes()).hexdigest(); splits={}; split_hashes={}
        historical=json.loads((Path(config['checkpoint_dir'])/'manifest.json').read_text())
        expected=next(r for r in historical['runs'] if r['seed']==seed and r['encoding']==encoding and r['availability']==availability)
        if checkpoint_sha != expected['checkpoint_sha256']: raise ValueError('Historical checkpoint hash mismatch')
        torch.cuda.reset_peak_memory_stats(); run_start=time.monotonic()
        for split, (data_seed,size) in config['splits'].items():
            batch=make_batch(data_seed,size,distractors=config.get('distractors',8))
            split_hashes[split]=event_hash(batch['public']['event'])
            chunks={}; baseline={}
            with torch.no_grad():
             for start in range(0,size,config.get('batch_size',64)):
                def sl(tree):return {k:sl(v) if isinstance(v,dict) else v[start:start+config.get('batch_size',64)] for k,v in tree.items()}
                public=move(sl(batch['public']),device)
                captured,state=capture(model,public,encoding,availability,tuple(config['delays']))
                for key,value in captured.items(): chunks.setdefault(key,[]).append(value.cpu())
                for key,value in captured.items():
                    if key.startswith('workspace_'): baseline.setdefault(key,[]).append(model.scalar_heads[0](value).argmax(-1).cpu())
            splits[split]=({k:torch.cat(v).to(device) for k,v in chunks.items()},batch['targets']['value'].to(device))
            for key,values in baseline.items():
                records.append(dict(run=name,split=split,boundary=key,family='historical_decoder',metrics=metrics(torch.cat(values),batch['targets']['value']),predictions=torch.cat(values).tolist(),targets=batch['targets']['value'].tolist()))
        if len(set(split_hashes.values())) != len(split_hashes): raise ValueError('event split collision')
        if not config.get('profile_only',False):
         train,y=splits['train']; val,vy=splits['validation']; test,ty=splits['test']
         for boundary,x in train.items():
          for family in ('categorical_ridge','numerical_ridge'):
            target=F.one_hot(y,33).float() if family=='categorical_ridge' else y.float()[:,None]
            fits=[]
            for alpha in config['ridge_grid']:
                fit=fit_ridge(x,target,alpha); raw=predict_ridge(val[boundary],fit)
                pred=raw.argmax(-1) if family=='categorical_ridge' else raw[:,0].round().long().clamp(0,32)
                fits.append((int((pred==vy).sum()),alpha,fit))
            _,alpha,fit=max(fits,key=lambda z:z[0])
            coefficients=output/f'{name}-{boundary}-{family}.pt'; torch.save(tuple(v.cpu() for v in fit),coefficients)
            selection=dict(validation_grid=[dict(alpha=a,correct=c,total=len(vy)) for c,a,_ in fits],parameters=int(fit[2].numel()),coefficient_sha256=hashlib.sha256(coefficients.read_bytes()).hexdigest(),train_label_support=sorted(y.unique().tolist()))
            for split,(features,targets) in splits.items():
                raw=predict_ridge(features[boundary],fit)
                pred=raw.argmax(-1) if family=='categorical_ridge' else raw[:,0].round().long().clamp(0,32)
                records.append(dict(run=name,split=split,boundary=boundary,family=family,alpha=alpha,selection=selection,metrics=metrics(pred,targets),predictions=pred.tolist(),targets=targets.tolist()))
        torch.cuda.synchronize()
        manifest['runs'].append(dict(run=name,checkpoint_sha256=checkpoint_sha,event_hashes=split_hashes,seconds=time.monotonic()-run_start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),width=1024))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'predictions.json').write_text(json.dumps(records,separators=(',',':')))
        print(name,manifest['runs'][-1]['seconds'],flush=True)
    manifest['seconds']=time.monotonic()-started;(output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),args.output)
