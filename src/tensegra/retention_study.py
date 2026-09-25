"""Remote bounded Track C acquisition and matched intervention runner."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .retention import ReturnRetentionModel, MODES, INTERVENTIONS
from .retention_data import make_batch

FIELDS=('value','type','operation','argument0','argument1','provenance')


def digest(value):
    stream=io.BytesIO(); torch.save(value,stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def loss(output,targets):
    terms={key:F.cross_entropy(output['before'][key],targets[key]) for key in FIELDS}
    return sum(terms.values()),terms


def measure(output,targets,flip_query=False):
    matches={key:output['before'][key].argmax(-1)==targets[key] for key in FIELDS}
    matches['joint']=torch.stack(list(matches.values())).all(0)
    matches['task']=output['after']['task'].argmax(-1)==(1-targets['task'] if flip_query else targets['task'])
    result={key:dict(correct=int(value.sum()),total=value.numel()) for key,value in matches.items()}
    result['validity']=dict(correct=int((output['after']['validity'].argmax(-1)==int(output['register_active'])).sum()),total=targets['task'].numel())
    result['task_given_retained']=dict(correct=int((matches['task']&matches['joint']).sum()),total=int(matches['joint'].sum()))
    return result


def competence_gate(rows,seeds,distractors):
    if not seeds or not distractors: return False
    lookup={(r['seed'],r['distractors']):r for r in rows if r['delay']==16 and r.get('intervention','none')=='none'}
    for seed in seeds:
        for count in distractors:
            row=lookup.get((seed,count))
            if row is None: return False
            for key in FIELDS:
                c=row['counts'][key]
                if c['total']<1 or c['correct']/c['total'] <= (.99 if key in ('type','operation') else .98): return False
    return True


def usage_gate(rows,seeds,distractors):
    if not seeds or not distractors: return False
    for seed in seeds:
        for count in distractors:
            selected={r['intervention']:r['counts']['task'] for r in rows if r['seed']==seed and r['distractors']==count and r['delay']==16}
            if not all(k in selected for k in ('none','event_drop','wrong_value')): return False
            acc={k:v['correct']/v['total'] for k,v in selected.items()}
            if acc['none']<=.95 or min(acc['none']-acc[k] for k in ('event_drop','wrong_value'))<.20: return False
    return True


def evaluation_interventions(config,delay):
    selected=config.get('interventions',INTERVENTIONS)
    allowed=config.get('intervention_delays')
    return [name for name in selected if name=='none' or allowed is None or delay in allowed]


def run(config,output):
    if config.get('phase','C1')!='C1': raise ValueError('C2 requires root gate audit and is not enabled in this runner')
    if set(config['validation_seeds']) & set(config.get('test_seeds',[])): raise ValueError('validation/test seeds must be disjoint')
    torch.set_num_threads(min(2,config.get('threads',2)))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('retention.py','retention_data.py','retention_study.py','thinking.py')}
    manifest=dict(source=source,config=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),runs=[])
    f=config.get('feature_dim',32); width=config.get('width',32); limit=config.get('value_limit',8)
    for seed in config['seeds']:
        for mode in config.get('modes',('once','protected','gated')):
            if mode=='persistent': raise ValueError('persistent is a frozen gated-checkpoint lifecycle evaluation, not a distinct training arm')
            torch.manual_seed(seed)
            model=ReturnRetentionModel(f,width,limit)
            initial=digest(model.state_dict())
            optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('lr',.001))
            prefix=f'seed{seed}-{mode}'
            log=open(output/(prefix+'.jsonl'),'w')
            started=time.monotonic()
            fixed=make_batch(10000+seed,config['batch_size'],f,value_limit=limit)
            for step in range(config['steps']+1):
                data=fixed if config.get('fixed_set',False) else make_batch(10000+seed*100000+step,config['batch_size'],f,value_limit=limit)
                delay=config.get('train_delays',[1,2,4])[step%len(config.get('train_delays',[1,2,4]))]
                prediction=model(data['public'],delay,mode)
                total,terms=loss(prediction,data['targets'])
                if step%config.get('log_every',10)==0 or step==config['steps']:
                    log.write(json.dumps(dict(step=step,delay=delay,loss={k:float(v.detach()) for k,v in terms.items()},counts=measure(prediction,data['targets']),data_hash=digest(data),elapsed=time.monotonic()-started))+'\n'); log.flush()
                if step==config['steps']: break
                optimizer.zero_grad(); total.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                acquisition=[measure(model(fixed['public'],d,mode),fixed['targets']) for d in config.get('train_delays',[1,2,4])]
            acquisition_pass=all(row[k]['correct']/row[k]['total']>=.99 for row in acquisition for k in ('joint',))
            rows=[]
            with torch.no_grad():
                for eval_seed in config['validation_seeds']+config.get('test_seeds',[]):
                    for distractors in config.get('eval_distractors',[2,8]):
                        data=make_batch(eval_seed,config['eval_size'],f,distractors,limit)
                        for delay in config.get('eval_delays',[1,2,4,8,16,32]):
                            for intervention in evaluation_interventions(config,delay):
                                replacement=make_batch(eval_seed+1000000,config['eval_size'],f,distractors,limit)
                                public=data['public']
                                # Overwrite uses its own complete event/identity universe;
                                # only event is replaced, keys must contain replacement identities.
                                target=data['targets']
                                if intervention=='overwrite':
                                    public=dict(public,argument_keys=replacement['public']['argument_keys'],provenance_keys=replacement['public']['provenance_keys'])
                                    target=dict(replacement['targets'])
                                    val=replacement['public']['event']['values'][:,0]
                                    target['task']=((val>public['query'][:,0])^public['query'][:,1].bool()).long()
                                pred=model(public,delay,mode,intervention,replacement['public']['event'])
                                row=dict(split='validation' if eval_seed in config['validation_seeds'] else 'test',targets={k:v.tolist() for k,v in target.items()},seed=eval_seed,distractors=distractors,delay=delay,intervention=intervention,counts=measure(pred,target,intervention=='query_counterfactual'),register_active=pred['register_active'],drift=float(pred['drift']),data_hash=digest(data),predictions={k:v.argmax(-1).tolist() for k,v in pred['before'].items()},task_predictions=pred['after']['task'].argmax(-1).tolist())
                                rows.append(row); log.write(json.dumps(row)+'\n')
                                if mode=='gated' and intervention=='delay_memory_drop':
                                    durable_pred=model(public,delay,'persistent',intervention)
                                    log.write(json.dumps(dict(row,mode='persistent',counts=measure(durable_pred,target),predictions={k:v.argmax(-1).tolist() for k,v in durable_pred['before'].items()},task_predictions=durable_pred['after']['task'].argmax(-1).tolist(),drift=float(durable_pred['drift']),register_active=durable_pred['register_active']))+'\n')
            checkpoint=output/(prefix+'.pt'); torch.save(model.state_dict(),checkpoint)
            retention=competence_gate(rows,config['validation_seeds'],config.get('eval_distractors',[2,8]))
            use=usage_gate(rows,config['validation_seeds'],config.get('eval_distractors',[2,8]))
            record=dict(seed=seed,mode=mode,initial_hash=initial,checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),retention_gate=retention,return_use_gate=use,late_use_status='untrained_diagnostic_C1',acquisition_counts=acquisition,acquisition_gate=acquisition_pass,dependent_experiments='blocked_acquisition_only' if config.get('fixed_set',False) else ('not_implemented' if retention and use else 'blocked'),elapsed=time.monotonic()-started,optimizer_steps=config['steps'],examples_seen=config['steps']*config['batch_size'])
            manifest['runs'].append(record); log.close()
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)
