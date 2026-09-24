"""S15 v2 paired shape exposure; public English inputs, fixed S11 warmstart."""
from __future__ import annotations
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from torch import nn
from .campaign_semantics import Dataset,digest,write_gzip,calibrated_evaluation
from .campaign_semantics_data import load_cache
from .campaign_semantics_continue import next_indices
from .campaign_semantics_lr import override_learning_rate
from .semantic_curriculum import SemanticCurriculumActor,sampled_pairs,curriculum_weights
from .semantic_text_acquisition import corrected_losses
from .thinking_language import state_hash
PARENT='3799ade595500a083b9a558b6a5bc97c5cbc1b8bdce6e7e4b86d97b937ea373e'

def presentation_seed(number):
    return 150150000+number

def validate_config(c):
    if c.get('budget_status')!='frozen':raise ValueError('separate source/budget freeze required')
    if c['arm'] not in ('control','mixed') or (c['learning_rate'],c['batch_size'],c['negative_pairs'],c['schedule_seed'])!=(1e-5,8,128,15115):raise ValueError('registered recipe changed')
    schedule={'profile':(20,[0,20]),'main':(4096,[0,1024,2048,4096])}
    if c['job'] not in schedule or (c['added_updates'],c['checkpoints'])!=schedule[c['job']]:raise ValueError('unregistered schedule; extension requires separate version')
    if c['parent_checkpoint_sha256']!=PARENT:raise ValueError('fixed independently selected parent changed')

def run(c):
    validate_config(c);start=time.monotonic();torch.set_num_threads(2)
    for name,sha in c['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('source changed: '+name)
    data=Path(c['data_dir']);audit=json.loads((data/'audit.json').read_text())
    if digest(data/'audit.json')!=c['audit_sha256']:raise ValueError('data audit changed')
    for split in ('train_control','train_mixed','development','confirmation'):
        if digest(data/(split+'.jsonl.gz'))!=c['cache_sha256'][split]:raise ValueError('cache changed')
    if digest(c['calibration_cache'])!=c['calibration_cache_sha256'] or digest(c['data_audit'])!=c['data_audit_sha256']:raise ValueError('historical calibration/vocabulary changed')
    vocab=json.loads(Path(c['data_audit']).read_text())['value_vocabulary'];rows=load_cache(data/f"train_{c['arm']}.jsonl.gz");dev_rows=load_cache(data/'development.jsonl.gz')
    if len(rows)!=4096 or len(dev_rows)!=2048:raise ValueError('fixed support changed')
    if c['job']=='profile':
        dev_rows=[r for a in (3,4) for f in (3,4) for r in [x for x in dev_rows if (x['arity'],x['facts'])==(a,f)][:16]]
    train=Dataset(rows,vocab);dev=Dataset(dev_rows,vocab);cal=Dataset(load_cache(c['calibration_cache'])[:128],vocab)
    if digest(c['parent_checkpoint'])!=PARENT or digest(c['parent_evaluation'])!=c['parent_evaluation_sha256']:raise ValueError('parent bytes changed')
    parent=torch.load(c['parent_checkpoint'],map_location='cpu',weights_only=True)
    if parent['update']!=24576 or sum(parent['visits'])!=196608:raise ValueError('parent exposure changed')
    torch.manual_seed(201);model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(c['device']);model.load_state_dict(parent['model'])
    if any(isinstance(m,nn.Dropout) and m.p for m in model.modules()):raise ValueError('untracked stochastic actor')
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate']);optimizer.load_state_dict(parent['optimizer']);lr=override_learning_rate(optimizer,c['learning_rate'])
    steps=sorted({float(s['step']) for s in optimizer.state.values()})
    if steps!=[24576.]:raise ValueError('parent AdamW step changed')
    # New shared corpus: reset only its sampling schedule, preserve model and AdamW.
    schedule=torch.Generator().manual_seed(c['schedule_seed']);order=torch.randperm(4096,generator=schedule).tolist();position=0
    visits=[0]*4096;sequence=hashlib.sha256();pair_sequence=hashlib.sha256();common_sequence=hashlib.sha256();common_count=0
    other=load_cache(data/f"train_{'mixed' if c['arm']=='control' else 'control'}.jsonl.gz");common=[r['alpha_sha256']==s['alpha_sha256'] for r,s in zip(rows,other)]
    if sum(common)!=2048:raise ValueError('common index alignment changed')
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=False);curves=[];losses=[];train_seconds=0.;tokens=nodes=edges=0
    manifest=dict(config=c,initial_state_sha256=state_hash(model),inherited_optimizer_steps=steps,inherited_presentations=196608,learning_rate_override=lr,curves=curves,losses=losses,parameters=sum(p.numel() for p in model.parameters()))
    del parent;torch.cuda.reset_peak_memory_stats()
    for step in range(c['added_updates']+1):
        update=24576+step
        if step in c['checkpoints']:
            result=calibrated_evaluation(model,cal,dev,out,update,128,len(dev));saved=json.load(gzip.open(out/result['artifact'],'rt'));cells={}
            if step==0:
                original=json.load(gzip.open(c['parent_evaluation'],'rt'))
                if saved['thresholds']!=original['thresholds'] or saved['train_rows']!=original['train_rows']:raise ValueError('initial historical TRAIN128 replay changed')
                manifest['initial_calibration_replay_exact']=True
            for a in (3,4):
                for f in (3,4):
                    metrics=[r for r,d in zip(saved['rows'],dev_rows) if (d['arity'],d['facts'])==(a,f)]
                    cells[f'{a}x{f}']=dict(examples=len(metrics),raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in metrics),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in metrics))
            curve=dict(added_update=step,update=update,evaluation=result,cells=cells,added_presentations=sum(visits),tokens=tokens,nodes=nodes,edges=edges);curves.append(curve)
            path=out/f'model-u{update}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=update,added_update=step,presentation_number=step*8),path);curve['checkpoint_sha256']=digest(path)
            print(json.dumps(dict(event='evaluation',**curve)),flush=True)
        if step==c['added_updates']:break
        indices,order,position=next_indices(order,position,schedule,8);batch=[train[i] for i in indices];queries=[]
        for offset,(index,(_,gold,row)) in enumerate(zip(indices,batch)):
            number=step*8+offset;query=sampled_pairs(gold,torch.Generator().manual_seed(presentation_seed(number)),128);queries.append(query);pair_sequence.update(query.numpy().tobytes())
            if common[index]:common_sequence.update(number.to_bytes(8,'little'));common_sequence.update(query.numpy().tobytes());common_count+=1
            visits[index]+=1;tokens+=row['tokens'];nodes+=len(row['nodes']);edges+=len(row['edges'])
        sequence.update(json.dumps(indices).encode());torch.cuda.synchronize();tick=time.monotonic();optimizer.zero_grad()
        outputs=model.forward_batch([p for p,_,_ in batch],pairs=queries);weights=curriculum_weights(update*8,1000,2000);parts=[corrected_losses(o,g,q) for o,(_,g,_),q in zip(outputs,batch,queries)];loss=torch.stack([sum(weights[k]*v for k,v in item.items()) for item in parts]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();train_seconds+=time.monotonic()-tick
        if (step+1)%128==0:
            values=torch.stack([torch.stack([r[k].detach() for k in parts[0]]) for r in parts]).mean(0).cpu().tolist()
            point=dict(added_update=step+1,training_seconds=train_seconds,parts=dict(zip(parts[0],values)));losses.append(point);print(json.dumps(dict(event='progress',**point)),flush=True)
    if c['job']=='main' and any(v!=8 for v in visits):raise ValueError('matched eight-epoch exposure incomplete')
    manifest.update(visits=visits,added_presentations=sum(visits),optimizer_tokens=tokens,optimizer_nodes=nodes,optimizer_edges=edges,construction_sequence_sha256=sequence.hexdigest(),pair_sequence_sha256=pair_sequence.hexdigest(),common_pair_sequence_sha256=common_sequence.hexdigest(),common_presentations=common_count,training_seconds=train_seconds,process_seconds=time.monotonic()-start,final_state_sha256=state_hash(model),peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_gzip(out/'manifest.json.gz',manifest);return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
