"""S11: exact optimizer/sampling continuation with one declared LR change."""
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from torch import nn
from .campaign_semantics import Dataset,digest,write_gzip,calibrated_evaluation
from .campaign_semantics_data import load_cache
from .campaign_semantics_continue import assert_replay,next_indices,first_batch_probe
from .campaign_semantics_lr import override_learning_rate,calibration_losses
from .semantic_curriculum import SemanticCurriculumActor,sampled_pairs,curriculum_weights
from .semantic_text_acquisition import corrected_losses
from .thinking_language import state_hash

def run(config):
    if config.get('budget_status')!='frozen':raise ValueError('LR-only budget not frozen')
    start=time.monotonic();torch.set_num_threads(2);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    data=Path(config['data_dir']);audit=json.loads((data/'audit.json').read_text());vocab=audit['value_vocabulary']
    assert digest(data/'train.jsonl.gz')==audit['cache_sha256']['train'];assert digest(data/'development.jsonl.gz')==audit['cache_sha256']['development']
    train=Dataset(load_cache(data/'train.jsonl.gz'),vocab);dev=Dataset(load_cache(data/'development.jsonl.gz'),vocab)
    if config.get('train_limit'):train=Dataset(train.rows[:config['train_limit']],vocab)
    if digest(config['parent_checkpoint'])!=config['parent_checkpoint_sha256']:raise ValueError('parent checkpoint SHA mismatch')
    if digest(config['parent_evaluation'])!=config['parent_evaluation_sha256']:raise ValueError('parent evaluation SHA mismatch')
    checkpoint=torch.load(config['parent_checkpoint'],map_location='cpu',weights_only=True)
    torch.manual_seed(config['seed'])
    model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
    model.load_state_dict(checkpoint['model'])
    if any(isinstance(m,nn.Dropout) and m.p for m in model.modules()):raise ValueError('historical checkpoint lacks global dropout RNG state')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']);optimizer.load_state_dict(checkpoint['optimizer'])
    lr_override=override_learning_rate(optimizer,config['learning_rate'])
    generator=torch.Generator();generator.set_state(checkpoint['generator']);schedule=torch.Generator();schedule.set_state(checkpoint['schedule'])
    order=checkpoint['order'];position=checkpoint['position'];visits=checkpoint['visits'];start_update=checkpoint['update']
    if len(visits)!=len(train) or sum(visits)!=start_update*config['batch_size']:raise ValueError('inconsistent inherited exposure')
    tokens=sum(v*r['tokens'] for v,r in zip(visits,train.rows));losses=[];curves=[];train_seconds=0.;initial=state_hash(model)
    inherited_steps=sorted({float(state['step']) for state in optimizer.state.values()})
    expected_first_batch=first_batch_probe(checkpoint,train,config)
    comparator=json.load(gzip.open(config['comparator_manifest'],'rt'))
    if digest(config['comparator_manifest'])!=config['comparator_manifest_sha256']:raise ValueError('comparator manifest changed')
    if expected_first_batch!=comparator['first_batch_replay']:raise ValueError('continuation does not match S04 sampler')
    del checkpoint
    torch.cuda.reset_peak_memory_stats()
    source_names=('campaign_semantics_continue.py','campaign_semantics_lr.py','campaign_semantics_lr_train.py','campaign_semantics.py','campaign_semantics_data.py','semantic_curriculum.py','semantic_text_acquisition.py','semantic_scaling.py','semantic_contracts.py','thinking.py','thinking_language.py','semantic_graph.py','tcn_data.py')
    manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256={n:digest(Path(__file__).with_name(n)) for n in source_names},data_audit_sha256=digest(data/'audit.json'),parameters=sum(p.numel() for p in model.parameters()),width=1024,workspace_rows=8,initial_state_sha256=initial,parent_checkpoint_sha256=config['parent_checkpoint_sha256'],inherited_optimizer_steps=inherited_steps,start_update=start_update,learning_rate_override=lr_override,comparator_first_batch_matches=True,curves=curves,losses=losses)
    for update in range(start_update,config['updates']+1):
        if update in config['checkpoints']:
            record=calibrated_evaluation(model,train,dev,out,update,config['calibration_count'],config.get('evaluation_count'))
            if update==start_update:manifest['initial_prediction_replay_exact']=assert_replay(config['parent_evaluation'],out/record['artifact'])
            record.update(presentations=sum(visits),distinct_visited=sum(v>0 for v in visits),tokens=tokens,visit_min=min(visits),visit_max=max(visits))
            record['calibration_losses']=calibration_losses(out/f'calibration-u{update}.npz')
            curves.append(record);print(json.dumps(dict(event='evaluation',**record)),flush=True)
            checkpoint=out/f'model-u{update}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=update),checkpoint);record['checkpoint_sha256']=digest(checkpoint)
        if update==config['updates']:break
        indices,order,position=next_indices(order,position,schedule,config['batch_size'])
        for index in indices:visits[index]+=1;tokens+=train.rows[index]['tokens']
        batch=[train[i] for i in indices];queries=[sampled_pairs(gold,generator,config['negative_pairs']) for _,gold,_ in batch]
        if update==start_update:
            actual=dict(indices=indices,seeds=[train.rows[i]['seed'] for i in indices],pair_sha256=[hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in queries])
            if actual!=expected_first_batch:raise ValueError('first continuation batch or sampled-pair RNG mismatch')
            manifest['first_batch_replay']=actual
        torch.cuda.synchronize();tick=time.monotonic();optimizer.zero_grad()
        outputs=model.forward_batch([p for p,_,_ in batch],pairs=queries)
        weights=curriculum_weights(update*config['batch_size'],1000,2000)
        parts=[corrected_losses(o,gold,pairs) for o,(_,gold,_),pairs in zip(outputs,batch,queries)]
        loss=torch.stack([sum(weights[k]*v for k,v in row.items()) for row in parts]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();train_seconds+=time.monotonic()-tick
        if (update+1)%128==0 or update==0:
            values=torch.stack([torch.stack([r[k].detach() for k in parts[0]]) for r in parts]).mean(0).cpu().tolist()
            point=dict(update=update+1,presentations=sum(visits),parts=dict(zip(parts[0],values)),train_seconds=train_seconds,process_seconds=time.monotonic()-start);losses.append(point);print(json.dumps(dict(event='progress',**point)),flush=True)
    manifest.update(optimizer_presentations=sum(visits),actual_unique_visited=sum(v>0 for v in visits),optimizer_tokens=tokens,presentation_counts=visits,training_seconds=train_seconds,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,final_state_sha256=state_hash(model))
    write_gzip(out/'manifest.json.gz',manifest)
    (out/'completion.json').write_text(json.dumps(dict(process_seconds=time.monotonic()-start,training_seconds=train_seconds,success=True),indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
