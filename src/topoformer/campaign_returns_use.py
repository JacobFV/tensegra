"""R05: frozen learned return access followed by a small learned decision."""
import argparse,copy,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .retention_data import make_batch
from .return_memory import ReturnMemoryModel
from .return_memory_study import move
from .return_crossdelay import save_cache,tensor_hash
from .campaign_returns import sha,selection_score
from .campaign_returns_balanced import balanced_indices


def answer(value,query):
    return ((value>query[:,0]) ^ query[:,1].bool()).long()


def alter_public(public,kind,swap=None):
    public=copy.deepcopy(public)
    if kind=='swap':
        for key in ('event','argument_keys','provenance_keys'):public[key]=copy.deepcopy(swap[key])
    elif kind=='wrong':
        event=public['event'];v=event['values'][:,0];typ=event['types'][:,0];op=event['operations'][:,0]
        v=torch.where(typ==2,1-v,torch.where(v!=0,-v,torch.full_like(v,8)))
        x=event['operand_values'][:,0,0].clone();y=v-x
        y[op==1]=x[op==1]-v[op==1];x[op==2]=v[op==2];y[op==2]=1
        x[op==3]=-v[op==3];y[op==3]=0;x[op==4]=1-v[op==4];y[op==4]=.5
        event['values']=v[:,None];event['operand_values']=torch.stack((x,y),1)[:,None]
    elif kind not in ('correct','drop'):raise ValueError(kind)
    return public


def workspace(model,public,delays,drop=False):
    state=model.initial[None].expand(len(public['event']['values']),-1,-1)
    memory=None if drop else model.encode(public['event'],'factorized')
    if memory is not None:state=state+model.initial_read(state,memory,memory,need_weights=False)[0]
    result={}
    if 0 in delays:result[0]=model.norm(state)[:,0]
    for step in range(max(delays)):
        mem=model.features(public['distractors'][:,step])
        if memory is not None:mem=torch.cat((mem,memory),1)
        mask=torch.ones(mem.shape[:2],dtype=torch.bool,device=state.device)
        bias=state.new_zeros(len(state),model.head_count,6,6)
        for block in model.blocks:state,_=block(state,mem,mask,bias)
        if step+1 in delays:result[step+1]=model.norm(state)[:,0]
    return result


def capture_batch(model,accessor,batch,delays,device,chunk,kind='correct',swap=None):
    original=batch['public'];public=alter_public(original,kind,swap)
    values=public['event']['values'][:,0];query=original['query'];parts={d:[] for d in delays}
    with torch.no_grad():
        for start in range(0,len(values),chunk):
            def section(tree):return {k:section(v) if isinstance(v,dict) else v[start:start+chunk] for k,v in tree.items()}
            out=workspace(model,move(section(public),device),delays,kind=='drop')
            for d,x in out.items():parts[d].append(x.cpu())
    features={d:torch.cat(v) for d,v in parts.items()};scores={}
    with torch.no_grad():
        for d,x in features.items():scores[d]=accessor(x.to(device)).cpu()
    return dict(features=features,scores=scores,query=query,original_targets=answer(original['event']['values'][:,0],query),
                supplied_targets=None if kind=='drop' else answer(values,query),supplied_value_labels=None if kind=='drop' else (2*values+16).long(),
                original_value_labels=batch['targets']['value'],types=batch['targets']['type'],operations=batch['targets']['operation'],
                supplied_types=None if kind=='drop' else public['event']['types'][:,0],supplied_operations=None if kind=='drop' else public['event']['operations'][:,0],
                original_event_hashes=[tensor_hash({k:v[i] for k,v in original['event'].items()}) for i in range(len(values))],
                supplied_event_hashes=None if kind=='drop' else [tensor_hash({k:v[i] for k,v in public['event'].items()}) for i in range(len(values))],
                public_sha256=tensor_hash(public),kind=kind)


def consumer_input(batch,delay,arm,device):
    scores=batch['scores'][delay].to(device)
    if arm=='learned':value=scores.softmax(-1)
    elif arm=='query_only' or batch['supplied_value_labels'] is None:value=torch.zeros_like(scores)
    elif arm=='oracle':value=F.one_hot(batch['supplied_value_labels'].to(device),33).float()
    else:raise ValueError(arm)
    query=batch['query'].to(device).clone();query[:,0]/=8
    return torch.cat((value,query),1)


def evaluate(head,arm,cache,device,prefix='calibration/'):
    cells=[]
    with torch.no_grad():
        for key,batch in cache.items():
            if not key.startswith(prefix):continue
            for delay in batch['scores']:
                logits=head(consumer_input(batch,delay,arm,device));pred=logits.argmax(-1).cpu();target=batch['original_targets']
                cells.append(dict(key=key,delay=delay,correct=int((pred==target).sum()),total=len(target)))
    return cells


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['consumer_seed']);device=cfg['device'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();assert 32 not in cfg['delays']
    checkpoint=Path(cfg['checkpoint']);readout=Path(cfg['readout']);assert sha(checkpoint)==cfg['checkpoint_sha256'];assert sha(readout)==cfg['readout_sha256']
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval();model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.requires_grad_(False)
    state_hash=tensor_hash(model.state_dict());saved=torch.load(readout,map_location=device,weights_only=True)
    head=nn.Linear(1024,33).to(device);head.load_state_dict(saved['state']);head.requires_grad_(False)
    def accessor(x):return head((x-saved['mean'])/saved['scale'])
    cache={};batches={}
    for split,spec in cfg['data'].items():
        if split=='test':continue
        for distractors in ([2] if split=='train' else cfg['eval_distractors']):
            batch=make_batch(spec['seed'],spec['size'],distractors=distractors);batches[f'{split}/{distractors}']=batch
            cache[f'{split}/{distractors}']=capture_batch(model,accessor,batch,cfg['delays'],device,cfg['batch_size'])
    populations=[set(cache[f'{split}/2']['original_event_hashes']) for split in cfg['data'] if split!='test']
    assert all(not(a&b) for i,a in enumerate(populations) for b in populations[i+1:])
    initial=nn.Sequential(nn.Linear(35,1024),nn.GELU(),nn.Linear(1024,2)).to(device)
    fitted={};fits=[];train=cache['train/2'];labels=train['original_targets'].repeat(len(cfg['delays'])).to(device)
    for arm in ('learned','query_only','oracle'):
        consumer=copy.deepcopy(initial);optimizer=torch.optim.AdamW(consumer.parameters(),lr=cfg['lr']);generator=torch.Generator(device=device).manual_seed(cfg['consumer_seed'])
        x=torch.cat([consumer_input(train,d,arm,device) for d in cfg['delays']]);best=None;curve=[];losses=[];visited=torch.zeros(len(x),dtype=torch.bool,device=device)
        arm_updates=cfg.get('fixed_steps',{}).get(arm,cfg['updates'])
        for step in range(arm_updates+1):
            if step in cfg['checkpoints'] or step==arm_updates:
                cells=evaluate(consumer,arm,cache,device);score=selection_score(cells);curve.append(dict(step=step,cells=cells,selection_score=score))
                if ('fixed_steps' in cfg and step==arm_updates) or ('fixed_steps' not in cfg and (best is None or score>best[0])):best=(score,step,copy.deepcopy(consumer.state_dict()))
            if step==arm_updates:break
            ids=torch.randint(len(x),(cfg['consumer_batch_size'],),device=device,generator=generator)
            visited[ids]=True
            loss=F.cross_entropy(consumer(x[ids]),labels[ids]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        consumer.load_state_dict(best[2]);consumer.eval();fitted[arm]=consumer
        path=out/f'{arm}.pt';torch.save({k:v.cpu() for k,v in best[2].items()},path)
        fits.append(dict(arm=arm,selected_step=best[1],curve=curve,losses=losses,checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in consumer.parameters()),optimizer_presentations=arm_updates*cfg['consumer_batch_size'],fit_rows=len(x),fit_events=len(train['original_targets']),visited_row_bits_little_endian=np.packbits(visited.cpu().numpy(),bitorder='little').tobytes().hex(),unique_rows_sampled=int(visited.sum()),unique_events_sampled=int(visited.reshape(len(cfg['delays']),-1).any(0).sum())))
    # All further captures are post-selection and cannot affect any fit.
    if 'test' in cfg['data']:
        spec=cfg['data']['test']
        for distractors in cfg['eval_distractors']:
            batch=make_batch(spec['seed'],spec['size'],distractors=distractors)
            cache[f'test/{distractors}']=capture_batch(model,accessor,batch,cfg['test_delays'],device,cfg['batch_size'])
        test_ids=set(cache['test/2']['original_event_hashes']);assert all(not(test_ids & ids) for ids in populations);populations.append(test_ids)
    if not cfg.get('profile_only',False):
        original=batches['validation/8']
        if 'causal_size' in cfg:
            def prefix(tree):return {k:prefix(v) if isinstance(v,dict) else v[:cfg['causal_size']] for k,v in tree.items()}
            original=prefix(original)
        swap=make_batch(cfg['swap_seed'],len(original['targets']['value']),distractors=8)['public']
        for kind in ('wrong','drop','swap'):
            cache[f'intervention_{kind}/8']=capture_batch(model,accessor,original,cfg['intervention_delays'],device,cfg['batch_size'],kind,swap)
        spec=cfg['balanced'];pool=make_batch(spec['seed'],spec['pool_size'],distractors=8);ids=balanced_indices(pool,spec['per_cell'])
        def subset(tree):return {k:subset(v) if isinstance(v,dict) else v[ids] for k,v in tree.items()}
        cache['balanced/8']=capture_batch(model,accessor,subset(pool),spec.get('delays',cfg['intervention_delays']),device,cfg['batch_size'])
        cache['balanced/8']['pool_indices']=ids
        fresh_sets=[set(cache['intervention_swap/8']['supplied_event_hashes']),set(cache['balanced/8']['original_event_hashes'])]
        assert not(fresh_sets[0]&fresh_sets[1])
        assert all(not(a&b) for a in fresh_sets for b in populations)
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            for d in batch['scores']:
                for arm,consumer in fitted.items():
                    scores=consumer(consumer_input(batch,d,arm,device));pred=scores.argmax(-1).cpu();target=batch['original_targets'];supplied=batch['supplied_targets']
                    row=dict(arm=arm,key=key,delay=d,predictions=pred.tolist(),original_targets=target.tolist(),supplied_targets=None if supplied is None else supplied.tolist(),
                             original_correct=int((pred==target).sum()),total=len(pred),positive_labels=int(target.sum()),types=batch['types'].tolist(),values=batch['original_value_labels'].tolist(),operations=batch['operations'].tolist(),supplied_types=None if batch['supplied_types'] is None else batch['supplied_types'].tolist(),supplied_operations=None if batch['supplied_operations'] is None else batch['supplied_operations'].tolist())
                    if supplied is not None:
                        changed=supplied!=target;row.update(supplied_correct=int((pred==supplied).sum()),changed_total=int(changed.sum()),changed_supplied_correct=int(((pred==supplied)&changed).sum()),changed_original_correct=int(((pred==target)&changed).sum()))
                    rows.append(row);logits[f'{key}/{d}/{arm}']=scores.cpu()
    assert tensor_hash(model.state_dict())==state_hash
    raw=out/'predictions.json.gz';raw.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_use.py','retention_data.py','return_memory.py','campaign_returns_balanced.py','campaign_returns.py','return_crossdelay.py','return_memory_study.py','thinking.py')},
                  environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()),width=1024,backbone_parameters=sum(p.numel() for p in model.parameters()),memory_tokens=6,backbone_state_sha256=state_hash,consumer_initial_sha256=tensor_hash(initial.state_dict()),fits=fits,cache=save_cache(out/'cache.pt.gz',cache),logits=save_cache(out/'logits.pt.gz',logits),predictions_sha256=sha(raw),
                  seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0));print(json.dumps(dict(seconds=manifest['seconds'],steps={r['arm']:r['selected_step'] for r in fits})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if 'runs' in cfg:
        for entry in cfg['runs']:
            merged={k:v for k,v in cfg.items() if k!='runs'};merged.update(entry);run(merged,Path(a.output)/str(entry['backbone_seed']))
    else:run(cfg,a.output)
