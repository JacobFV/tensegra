"""Isolated Stage8 return acquisition; no downstream use or composition."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .return_memory import ReturnMemoryModel, FIELDS
from .retention_data import make_batch


def move(value, device):
    if torch.is_tensor(value): return value.to(device)
    return {k:move(v,device) for k,v in value.items()}


def counts(output, targets):
    matches = {k:output['logits'][k].argmax(-1)==targets[k] for k in FIELDS}
    matches['identity_joint'] = torch.stack([matches[k] for k in FIELDS[3:]]).all(0)
    matches['scalar_joint'] = torch.stack([matches[k] for k in FIELDS[:3]]).all(0)
    matches['joint'] = torch.stack([matches[k] for k in FIELDS]).all(0)
    return {k:{'correct':int(v.sum()),'total':v.numel()} for k,v in matches.items()}


def gate(rows, seeds, distractors):
    if not seeds or not distractors: return False
    for seed in seeds:
        for d in distractors:
            selected = [r for r in rows if r['seed']==seed and r['distractors']==d and r['steps']==16 and r['intervention']=='none']
            if len(selected)!=1: return False
            for field in FIELDS:
                c=selected[0]['counts'][field]
                if c['total'] < 512 or c['correct']/c['total'] <= (.99 if field in ('type','operation') else .98): return False
    return True


def run(config, output):
    torch.set_num_threads(min(2,config.get('threads',2)))
    device=torch.device(config.get('device','cpu'))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    source={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('return_memory.py','return_memory_study.py','retention_data.py','retention.py','thinking.py')}
    manifest=dict(source=source,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),runs=[])
    if set(config.get('validation_seeds',[])) & set(config.get('test_seeds',[])): raise ValueError('split seed collision')
    for seed in config.get('seeds',[0]):
        for encoding in config.get('encodings',['mixed','compressed','factorized']):
            for availability in config.get('availability',['once','persistent']):
                torch.manual_seed(seed)
                model=ReturnMemoryModel(width=config.get('width',1024),feature_dim=config.get('feature_dim',32),value_limit=config.get('value_limit',8),heads=config.get('heads',8),encoding=encoding).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.0003))
                name=f'{seed}-{encoding}-{availability}'
                started=time.monotonic()
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                def data(s,n,d=2): return move(make_batch(s,n,config.get('feature_dim',32),d,config.get('value_limit',8)),device)
                with (output/(name+'.jsonl')).open('w') as log:
                    model.eval()
                    with torch.no_grad():
                        for es in config.get('validation_seeds',[]):
                            batch=data(es,config.get('eval_size',64))
                            for length in (0,1):
                                pred=model(batch['public'],length,encoding,availability)
                                log.write(json.dumps(dict(phase='initialized_validation',step=0,data_seed=es,steps=length,counts=counts(pred,batch['targets'])))+'\n')
                    model.train()
                    for step in range(config.get('updates',0)+1):
                        batch=data(10000+seed*100000+(0 if config.get('fixed_set',False) else step),config.get('batch_size',8))
                        length=config.get('train_steps',[0,1])[step%len(config.get('train_steps',[0,1]))]
                        pred=model(batch['public'],length,encoding,availability)
                        terms={k:F.cross_entropy(pred['logits'][k],batch['targets'][k]) for k in FIELDS}
                        loss=sum(terms.values())
                        if step%config.get('log_every',10)==0 or step==config.get('updates',0):
                            log.write(json.dumps(dict(phase='train',step=step,steps=length,loss={k:float(v.detach()) for k,v in terms.items()},counts=counts(pred,batch['targets']),elapsed=time.monotonic()-started))+'\n'); log.flush()
                        if step==config.get('updates',0): break
                        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                    model.eval(); rows=[]
                    with torch.no_grad():
                        for split in ('validation','test'):
                            for es in config.get(split+'_seeds',[]):
                                for d in config.get('eval_distractors',[2,8]):
                                    for length in config.get('eval_steps',[0,1,2,4,8,16,32]):
                                        for intervention in config.get('interventions',['none']):
                                            batch=data(es,config.get('eval_size',64),d)
                                            replacement=data(es+1000000,config.get('eval_size',64),d)
                                            target=batch['targets']; public=batch['public']
                                            if intervention=='overwrite':
                                                public=dict(public,argument_keys=replacement['public']['argument_keys'],provenance_keys=replacement['public']['provenance_keys'])
                                                target=replacement['targets']
                                            # No lifecycle command is issued when zero recurrent steps execute.
                                            if length==0 and intervention in ('release','overwrite'): continue
                                            pred=model(public,length,encoding,availability,intervention,replacement['public']['event'])
                                            row=dict(phase='eval',split=split,seed=es,distractors=d,steps=length,intervention=intervention,counts=counts(pred,target),predictions={k:v.argmax(-1).tolist() for k,v in pred['logits'].items()},targets={k:target[k].tolist() for k in FIELDS},register_active=pred['register'] is not None)
                                            if pred['register'] is not None:
                                                supplied=dict(target)
                                                event=pred['register']
                                                supplied['value']=(2*event['values'][:,0]+2*config.get('value_limit',8)).long()
                                                supplied['type']=event['types'][:,0]
                                                supplied['operation']=event['operations'][:,0]
                                                for j,field in enumerate(('argument0','argument1')):
                                                    supplied[field]=(public['argument_keys']-event['arguments'][:,:,j]).square().sum(-1).argmin(-1)
                                                supplied['provenance']=(public['provenance_keys']-event['provenance']).square().sum(-1).argmin(-1)
                                                row['supplied_fact_counts']=counts(pred,supplied)
                                            rows.append(row); log.write(json.dumps(row)+'\n'); log.flush()
                    checkpoint=output/(name+'.pt'); torch.save(model.state_dict(),checkpoint)
                    manifest['runs'].append(dict(seed=seed,encoding=encoding,availability=availability,width=model.width,parameters=sum(p.numel() for p in model.parameters()),facet_coordinates=sum(model.sizes),memory_tokens=6 if encoding=='factorized' else 1,allocated_memory_coordinates=model.width*(6 if encoding=='factorized' else 1),elapsed=time.monotonic()-started,updates=config.get('updates',0),examples_seen=config.get('updates',0)*config.get('batch_size',8),peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),retention_gate=gate([r for r in rows if r['split']=='validation'],config.get('validation_seeds',[]),config.get('eval_distractors',[2,8])),composition_allowed=False))
                    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)
