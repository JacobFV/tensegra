"""S01 current-actor procedural acquisition with train-only calibration."""
from __future__ import annotations
import argparse,gzip,hashlib,json,resource,time
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn
from . import semantic_scaling as base
from .campaign_semantics_data import load_cache,target
from .semantic_curriculum import SemanticCurriculumActor,sampled_pairs,curriculum_weights,pack_graph
from .semantic_text_acquisition import corrected_losses
from .semantic_contracts import train_relation_thresholds
from .thinking_language import ActorInput,ROLES,state_hash


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gzip(path,value):
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:f.write(json.dumps(value,separators=(',',':')).encode())


class Dataset:
    def __init__(self,rows,vocab):self.rows,self.vocab=rows,vocab
    def __len__(self):return len(self.rows)
    @lru_cache(maxsize=512)
    def __getitem__(self,index):
        row=self.rows[index];return ActorInput(row['text'],()),target(row,self.vocab),row


def decode(output,public,thresholds):
    cpu={k:v.cpu() for k,v in output.items()};raw=base.decode(cpu,public)
    return raw,{**raw,'edges':cpu['edges'].gt(thresholds.cpu())}


def calibrated_evaluation(model,train,dev,out,update,calibration_count=128,eval_count=None):
    model.eval();scores=[];truth=[];calibration_rows=[];train_cache=[]
    tick=time.monotonic()
    with torch.no_grad():
        for index in range(min(len(train),calibration_count)):
            public,gold,row=train[index];output=model(public);cpu={k:v.cpu() for k,v in output.items()};pred=base.decode(cpu,public)
            active=pred['presence'][:,None]&pred['presence'][None,:]
            scores.append(cpu['edges'][active]);truth.append(gold['edges'][active]);train_cache.append((pred,cpu['edges'],gold,row))
            calibration_rows.append(dict(seed=row['seed'],pairs=active.nonzero().tolist(),scores=scores[-1].tolist(),targets=truth[-1].tolist()))
        s=torch.cat(scores);y=torch.cat(truth)
        if len(s):thresholds,calibration=train_relation_thresholds(s,y)
        else:thresholds=torch.zeros(len(ROLES));calibration=[dict(threshold=0.,empty_support=True,positive=0,negative=0,train_errors=0) for _ in ROLES]
        train_metrics=[dict(raw=base.metrics(pred,gold),calibrated=base.metrics({**pred,'edges':edge.gt(thresholds)},gold)) for pred,edge,gold,row in train_cache]
        rows=[]
        for index in range(min(len(dev),eval_count or len(dev))):
            public,gold,row=dev[index];output=model(public);raw,cal=decode(output,public,thresholds)
            rows.append(dict(seed=row['seed'],semantic_sha256=row['semantic_sha256'],graph_sha256=row['graph_sha256'],raw=pack_graph(raw),calibrated_edges=pack_graph(cal)['edges'],target=pack_graph(gold),raw_metrics=base.metrics(raw,gold),calibrated_metrics=base.metrics(cal,gold)))
    result=dict(update=update,thresholds=thresholds.tolist(),calibration=calibration,calibration_data=calibration_rows,train_metrics=train_metrics,rows=rows,evaluation_seconds=time.monotonic()-tick)
    path=out/f'evaluation-u{update}.json.gz';write_gzip(path,result)
    def summary(rows,key):
        def micro(kind):
            counts={k:sum(r[key][kind][k] for r in rows) for k in ('true_positive','predicted_count','gold_count')}
            return 2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])
        return dict(exact=sum(r[key]['semantic_equivalence'] for r in rows),examples=len(rows),copy=sum(r[key]['identity_copy_accuracy'] for r in rows)/len(rows),typed_edge_f1=micro('typed_edge'),ordered_edge_f1=micro('ordered_edge'))
    metadata=dict(update=update,train_raw=summary(train_metrics,'raw'),train_calibrated=summary(train_metrics,'calibrated'),dev_raw=summary(rows,'raw_metrics'),dev_calibrated=summary(rows,'calibrated_metrics'),seconds=result['evaluation_seconds'],artifact=path.name,sha256=digest(path))
    model.train();return metadata


def run(config):
    start=time.monotonic();torch.set_num_threads(2);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    data=Path(config['data_dir']);audit=json.loads((data/'audit.json').read_text());vocab=audit['value_vocabulary']
    assert digest(data/'train.jsonl.gz')==audit['cache_sha256']['train'];assert digest(data/'development.jsonl.gz')==audit['cache_sha256']['development']
    train=Dataset(load_cache(data/'train.jsonl.gz'),vocab);dev=Dataset(load_cache(data/'development.jsonl.gz'),vocab)
    if config.get('train_limit'):train=Dataset(train.rows[:config['train_limit']],vocab)
    torch.manual_seed(config['seed'])
    model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']);generator=torch.Generator().manual_seed(config['seed']+1729)
    schedule=torch.Generator().manual_seed(config['seed']+123);order=torch.randperm(len(train),generator=schedule).tolist();position=0
    visits=[0]*len(train);tokens=0;losses=[];curves=[];train_seconds=0.;initial=state_hash(model)
    torch.cuda.reset_peak_memory_stats()
    source_names=('campaign_semantics.py','campaign_semantics_data.py','semantic_curriculum.py','semantic_text_acquisition.py','semantic_scaling.py','semantic_contracts.py','thinking.py','thinking_language.py','semantic_graph.py','tcn_data.py')
    manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256={n:digest(Path(__file__).with_name(n)) for n in source_names},data_audit_sha256=digest(data/'audit.json'),parameters=sum(p.numel() for p in model.parameters()),width=1024,workspace_rows=8,initial_state_sha256=initial,curves=curves,losses=losses)
    for update in range(config['updates']+1):
        if update in config['checkpoints']:
            record=calibrated_evaluation(model,train,dev,out,update,config['calibration_count'],config.get('evaluation_count'))
            record.update(presentations=sum(visits),distinct_visited=sum(v>0 for v in visits),tokens=tokens,visit_min=min(visits),visit_max=max(visits))
            curves.append(record);print(json.dumps(dict(event='evaluation',**record)),flush=True)
            checkpoint=out/f'model-u{update}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),schedule=schedule.get_state(),order=order,position=position,visits=visits,update=update),checkpoint);record['checkpoint_sha256']=digest(checkpoint)
        if update==config['updates']:break
        indices=[]
        for _ in range(config['batch_size']):
            if position==len(order):order=torch.randperm(len(train),generator=schedule).tolist();position=0
            index=order[position];position+=1;indices.append(index);visits[index]+=1;tokens+=train.rows[index]['tokens']
        batch=[train[i] for i in indices];queries=[sampled_pairs(gold,generator,config['negative_pairs']) for _,gold,_ in batch]
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
