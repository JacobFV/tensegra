"""Reproducible isolated belief acquisition; no readiness/runtime composition."""
import argparse
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import time
import torch
from .belief_state import BeliefModel,make_episodes,collate,loss,oracle


def evaluate(model, count, seed, candidates, condition, raw_path=None):
    batch=collate(make_episodes(count,seed,candidates,condition=condition))
    if model is not None:
        device=next(model.parameters()).device
        batch={group:{k:v.to(device) for k,v in fields.items()} for group,fields in batch.items()}
    with torch.no_grad():
        p=oracle(batch['public']) if model is None else model(batch['public'])['logits'].softmax(-1)
    q=batch['targets']['posterior']; predicted=p.argmax(-1)
    if raw_path is not None:
        with gzip.open(raw_path,'wt') as handle:
            json.dump({'posterior':p.tolist(),'target':q.tolist(),'seed':seed,'condition':condition,'candidates':candidates,'public_examples':{k:v[:8].tolist() for k,v in batch['public'].items()}},handle)
    expected_correct=q.gather(-1,predicted.unsqueeze(-1)).squeeze(-1)
    correct=expected_correct>0
    impossible=(p*(q==0)).sum(-1); entropy=-(p*p.clamp_min(1e-12).log()).sum(-1)
    frames=[]
    for t in range(p.shape[1]):
        conf=p[:,t].max(-1).values
        bins=[]
        for lo in range(10):
            mask=(conf>=lo/10)&(conf<(lo+1)/10 if lo<9 else conf<=1)
            bins.append({'lower':lo/10,'count':int(mask.sum()),'confidence':float(conf[mask].mean()) if mask.any() else None,'accuracy':float(expected_correct[mask,t].mean()) if mask.any() else None})
        frames.append({'frame':t,'mean_support_size':float((q[:,t]>0).sum(-1).float().mean()),'ambiguous_count':int(((q[:,t]>0).sum(-1)>1).sum()),'support_accuracy':float(correct[:,t].float().mean()),'impossible_mass':float(impossible[:,t].mean()),'entropy':float(entropy[:,t].mean()),'posterior_l1':float((p[:,t]-q[:,t]).abs().sum(-1).mean()),'calibration':bins})
    return {'count':count,'condition':condition,'candidates':candidates,'frames':frames,
            'examples':[{'posterior':p[i].tolist(),'target':q[i].tolist(),'predicted':predicted[i].tolist()} for i in range(min(count,8))]}


def gate(rows, expected):
    if not expected or not rows: return False
    indexed={(r['seed'],r['condition'],r['candidates']):r for r in rows if r.get('split')=='validation'}
    for key in expected:
        if key not in indexed: return False
        row=indexed[key]; frames=row.get('frames',[])
        if row.get('count',0)<512 or not frames: return False
        regime=row.get('regime')
        if regime not in ('iid','moderate_ood'): return False
        accuracy=.98 if regime=='iid' else .95
        if frames[-1]['support_accuracy']<=accuracy: return False
        if sum(f['posterior_l1'] for f in frames)/len(frames)>=.05: return False
        if sum(f['impossible_mass'] for f in frames)/len(frames)>=.01: return False
    return True


def run(config,out):
    torch.set_num_threads(config.get('threads',2)); torch.manual_seed(config['seed'])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(config.get('device','cpu'))
    model=BeliefModel(config['mode'],width=config.get('width',1024),inner=config.get('inner',2048),observation_id_features=config.get('observation_id_features',False)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.0003))
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('belief_state.py'))}
    manifest={'config':config,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'source_sha256':source,'git':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'parameters':sum(p.numel() for p in model.parameters()),'architecture_supplied':['typed roles','candidate records','observation identity','protected ledger for protected arm'],'composition_allowed':False,'device':str(device),'cuda_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    curve=[]; start=time.perf_counter()
    for step in range(config['steps']+1):
        if step in (0,config['steps']) or step%config.get('log_every',100)==0:
            row=evaluate(model,config.get('probe_count',64),100000+config['seed'],config['candidates'],'clean'); row.update(step=step,seconds=time.perf_counter()-start,cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None); curve.append(row)
            (out/'curve.json').write_text(json.dumps(curve)); print(json.dumps({'step':step,'seconds':row['seconds'],'final':row['frames'][-1]}),flush=True)
        if step==config['steps']: break
        # Fresh episodes each update; no memorized finite trajectory bank.
        condition=('clean','partial','contradiction','retract','duplicate','reorder')[step%6]
        b=collate(make_episodes(config['batch'],config['seed']*10000000+step,config['candidates'],condition=condition))
        b={group:{k:v.to(device) for k,v in fields.items()} for group,fields in b.items()}
        opt.zero_grad(); losses=loss(model(b['public']),b); total=losses['posterior']+config.get('compatibility_weight',1.)*losses['compatibility']; total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if step%config.get('log_every',100)==0:
            with (out/'loss.jsonl').open('a') as f: f.write(json.dumps({'step':step,**{k:float(v.detach()) for k,v in losses.items()}})+'\n')
    rows=[]
    for n in config.get('eval_candidates',[config['candidates'],config['candidates']*2]):
        for condition in ('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty'):
            for split in config.get('eval_splits',['validation','test']):
                if split not in ('validation','test'): raise ValueError('unknown evaluation split')
                offset={'validation':200000,'test':300000}[split]
                row=evaluate(model,config.get('eval_count',128),offset+config['seed'],n,condition,raw_path=out/f'raw-{split}-{n}-{condition}.json.gz'); row.update(seed=config['seed'],split=split,regime='iid' if n==config['candidates'] else 'moderate_ood'); rows.append(row)
    (out/'metrics.json').write_text(json.dumps(rows))
    oracle_rows=[evaluate(None,config.get('eval_count',128),200000+config['seed'],n,c) for n in config.get('eval_candidates',[config['candidates']]) for c in ('clean','reorder','duplicate','long_duplicate','contradiction','retract','partial','empty')]
    (out/'oracle.json').write_text(json.dumps(oracle_rows))
    torch.save(model.state_dict(),out/'model.pt')
    manifest['checkpoint_sha256']=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--out',required=True); a=parser.parse_args()
    run(json.loads(Path(a.config).read_text()),a.out)
