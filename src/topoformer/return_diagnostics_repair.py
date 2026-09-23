"""Stage9 minimal frozen-workspace scalar readout repair, with matched CE refit."""
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
from .return_memory import ReturnMemoryModel,FIELDS
from .retention_data import make_batch
from .return_memory_study import move,counts
from .return_diagnostics_probe import fit_ridge,predict_ridge,event_hash


def run(cfg,out):
    torch.set_num_threads(2); device=cfg['device']
    ds=[cfg[k] for k in ('head_train_seed','validation_seed','test_seed')]
    if len(set(ds))!=3: raise ValueError('head split seeds overlap')
    for seed in cfg['seeds']:
        if any(9600000+(seed-10)*10000 <= s < 9600000+(seed-10)*10000+cfg['updates'] for s in ds): raise ValueError('backbone/head data overlap')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),source={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('return_diagnostics_repair.py','return_diagnostics_probe.py','return_memory.py','retention_data.py','thinking.py')},runs=[])
    for seed in cfg['seeds']:
        started=time.monotonic();torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats()
        model=ReturnMemoryModel(width=1024,encoding='factorized').to(device)
        initial_hash=hashlib.sha256()
        for k,v in model.state_dict().items():initial_hash.update(k.encode());initial_hash.update(v.detach().cpu().numpy().tobytes())
        opt=torch.optim.AdamW(model.parameters(),lr=.0003);curve=[]
        def data(s,n,d=2):return move(make_batch(s,n,distractors=d),device)
        for step in range(cfg['updates']):
            batch=data(9600000+(seed-10)*10000+step,cfg['batch_size'])
            length=(0,1,2,4)[step%4]
            pred=model(batch['public'],length,'factorized','persistent')
            terms={k:F.cross_entropy(pred['logits'][k],batch['targets'][k]) for k in FIELDS};loss=sum(terms.values())
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%100==0:curve.append(dict(step=step,loss={k:float(v.detach()) for k,v in terms.items()}))
        model.eval();checkpoint=out/f'{seed}-backbone.pt';torch.save(model.state_dict(),checkpoint)
        for p in model.parameters():p.requires_grad_(False)
        @torch.no_grad()
        def collect(ds,n,delays,distractors,intervention='none'):
            results=[]
            for d in distractors:
                batch=data(ds,n,d)
                for delay in delays:
                    parts=[];xs=[];supplied=[]
                    replacement=data(ds+1000,n,d)
                    for start in range(0,n,cfg['eval_batch_size']):
                        def sl(tree):return {k:sl(v) if isinstance(v,dict) else v[start:start+cfg['eval_batch_size']] for k,v in tree.items()}
                        public=sl(batch['public']);re=sl(replacement['public']['event'])
                        if intervention=='overwrite':
                            public=dict(public,argument_keys=sl(replacement['public'])['argument_keys'],provenance_keys=sl(replacement['public'])['provenance_keys'])
                        p=model(public,delay,'factorized','persistent',intervention,re)
                        parts.append(p['logits']);xs.append(model.norm(p['state'])[:,0])
                    target=replacement['targets'] if intervention=='overwrite' else batch['targets']
                    results.append(dict(distractors=d,steps=delay,intervention=intervention,x=torch.cat(xs),logits={k:torch.cat([p[k] for p in parts]) for k in FIELDS},targets=target,event_sha256=event_hash(batch['public']['event'])))
            return results
        headtrain=collect(cfg['head_train_seed'],cfg['head_train_size'],cfg['head_delays'],[2])
        headval=collect(cfg['validation_seed'],cfg['eval_size'],cfg['head_delays'],cfg['eval_distractors'])
        x=torch.cat([r['x'] for r in headtrain]); y=torch.cat([r['targets']['value'] for r in headtrain])
        vx=torch.cat([r['x'] for r in headval]);vy=torch.cat([r['targets']['value'] for r in headval])
        fits=[]
        for alpha in cfg['ridge_grid']:
            fit=fit_ridge(x,F.one_hot(y,33).float(),alpha)
            fits.append((int((predict_ridge(vx,fit).argmax(-1)==vy).sum()),alpha,fit))
        _,alpha,ridge=max(fits,key=lambda z:z[0]);ridgepath=out/f'{seed}-ridge.pt';torch.save(tuple(v.cpu() for v in ridge),ridgepath)
        head=copy.deepcopy(model.scalar_heads[0]);head.requires_grad_(True)
        ceopt=torch.optim.AdamW(head.parameters(),lr=cfg['head_lr']);best=-1;cecurve=[];beststep=0;beststate=None
        generator=torch.Generator(device=device).manual_seed(seed+9000000)
        for step in range(cfg['head_updates']+1):
            if step%cfg['head_validate_every']==0 or step==cfg['head_updates']:
                with torch.no_grad():correct=int((head(vx).argmax(-1)==vy).sum())
                cecurve.append(dict(step=step,correct=correct,total=len(vy)))
                if correct>best:best=correct;beststep=step;beststate=copy.deepcopy(head.state_dict())
            if step==cfg['head_updates']:break
            idx=torch.randint(len(x),(cfg['head_batch_size'],),generator=generator,device=device)
            loss=F.cross_entropy(head(x[idx]),y[idx]);ceopt.zero_grad();loss.backward();ceopt.step()
        head.load_state_dict(beststate);head.eval();cepath=out/f'{seed}-ce.pt';torch.save(head.state_dict(),cepath)
        rows=[]
        for split,ds in [('validation',cfg['validation_seed']),('test',cfg['test_seed'])]:
            evaluated=collect(ds,cfg['eval_size'],cfg['eval_delays'],cfg['eval_distractors'])
            for intervention in cfg.get('interventions',[]): evaluated+=collect(ds,cfg['eval_size'],[16],cfg['eval_distractors'],intervention)
            for r in evaluated:
                predictions={'unchanged':r['logits']['value'],'ridge':predict_ridge(r['x'],ridge),'ce_refit':head(r['x'])}
                for arm,value in predictions.items():
                    logits=dict(r['logits'],value=value);target=r['targets']
                    row=dict(seed=seed,split=split,arm=arm,distractors=r['distractors'],steps=r['steps'],intervention=r['intervention'],event_sha256=r['event_sha256'],counts=counts(dict(logits=logits),target),predictions={k:v.argmax(-1).tolist() for k,v in logits.items()},targets={k:target[k].tolist() for k in FIELDS})
                    supplied=dict(target)
                    if r['intervention']=='wrong_value': supplied['value']=torch.where(target['value']!=16,32-target['value'],torch.full_like(target['value'],18))
                    if r['intervention']=='wrong_type': supplied['type']=(target['type']+1)%3
                    if r['intervention']=='wrong_provenance': supplied['provenance']=(target['provenance']+1)%4
                    if r['intervention']!='event_drop':
                        row['supplied_fact_counts']=counts(dict(logits=logits),supplied)
                        row['supplied_targets']={k:supplied[k].tolist() for k in FIELDS}
                    rows.append(row)
        (out/f'{seed}-predictions.json.gz').write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
        meta=dict(seed=seed,initial_state_sha256=initial_hash.hexdigest(),head_train_event_hashes=sorted({r['event_sha256'] for r in headtrain}),head_train_label_support=sorted(y.unique().tolist()),environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),device_capacity=torch.cuda.get_device_properties(0).total_memory),width=1024,parameters=sum(p.numel() for p in model.parameters()),head_parameters=sum(p.numel() for p in head.parameters()),backbone_presentations=cfg['updates']*cfg['batch_size'],head_unique_events=cfg['head_train_size'],head_feature_rows=len(x),head_ce_presentations=cfg['head_updates']*cfg['head_batch_size'],ridge_selection=[dict(correct=c,total=len(vy),alpha=a) for c,a,_ in fits],chosen_alpha=alpha,ce_curve=cecurve,chosen_ce_step=beststep,backbone_curve=curve,checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint,ridgepath,cepath)},seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        manifest['runs'].append(meta);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(seed,meta['seconds'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
