"""S02 warm existing edge head on frozen public-text-derived node features."""
from __future__ import annotations
import argparse,hashlib,json,time,gzip,resource
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .campaign_semantics import Dataset,digest,write_gzip
from .campaign_semantics_data import load_cache
from .semantic_curriculum import SemanticCurriculumActor,sampled_pairs,pack_graph,unpack_graph
from .semantic_scaling import decode,metrics
from .semantic_contracts import train_relation_thresholds
from .thinking_language import ROLES


def edge_scores(model,nodes,pairs=None):
    with torch.autocast(nodes.device.type,dtype=torch.bfloat16):
        source=model.edge_source(nodes).reshape(len(nodes),128,len(ROLES),model.edge_width);target=model.edge_target(nodes);result=[]
        for b in range(len(nodes)):
            if pairs is None:out=torch.einsum('nrd,md->nmr',source[b],target[b])/(model.edge_width**.5)
            else:
                i,j=pairs[b].to(nodes.device).unbind(-1);out=(source[b,i]*target[b,j,None,:]).sum(-1)/(model.edge_width**.5)
            result.append(out.float())
    return result


def frozen_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        if not name.startswith(('edge_source.','edge_target.')):h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture(model,dataset,count):
    model.eval();nodes=[];fixed=[];max_error=0.;captured=[]
    hook=model.decode_nodes.register_forward_hook(lambda module,args,out:captured.append((out[0]+model.queries).detach()))
    with torch.no_grad():
        for index in range(min(count,len(dataset))):
            public,gold,row=dataset[index];original=model(public);state=captured.pop();state=state.cpu().to(next(model.parameters()).device);scored=edge_scores(model,state)[0]
            error=float((scored-original['edges']).abs().max());max_error=max(max_error,error)
            if not torch.equal(scored,original['edges']):raise ValueError(f'cached-node edge reconstruction differs: {error}')
            pred=decode({k:v.cpu() for k,v in original.items()},public)
            nodes.append(state[0].cpu());fixed.append({k:v for k,v in pred.items() if k!='edges'})
    hook.remove();return dict(nodes=torch.stack(nodes),fixed=fixed,max_edge_reproduction_error=max_error)


def evaluate(model,cache,sets,out,update):
    scores=[];labels=[];pairs=[];offsets=[0];saved={}
    with torch.no_grad():
        for group in ('train','development'):
            rows=[]
            for index in range(len(cache[group]['nodes'])):
                logits=edge_scores(model,cache[group]['nodes'][index:index+1].to('cuda'))[0].cpu();gold=sets[group][index][1];fixed=cache[group]['fixed'][index];active=fixed['presence'][:,None]&fixed['presence'][None,:]
                if group=='train':scores.append(logits[active]);labels.append(gold['edges'][active]);pairs.append(active.nonzero().short());offsets.append(offsets[-1]+int(active.sum()))
                rows.append((logits,gold,fixed,sets[group].rows[index]))
            saved[group]=rows
        joined=torch.cat(scores);truth=torch.cat(labels);thresholds,calibration=train_relation_thresholds(joined,truth)
        records={};summary={}
        for group,rows in saved.items():
            records[group]=[]
            for logits,gold,fixed,row in rows:
                raw={**fixed,'edges':logits.gt(0)};cal={**fixed,'edges':logits.gt(thresholds)};active=fixed['presence'][:,None,None]&fixed['presence'][None,:,None]
                records[group].append(dict(seed=row['seed'],raw=pack_graph(raw),calibrated_edges=pack_graph(cal)['edges'],target=pack_graph(gold),raw_metrics=metrics(raw,gold),calibrated_metrics=metrics(cal,gold),raw_exact_edges=bool((raw['edges']&active).eq(gold['edges']).all()),calibrated_exact_edges=bool((cal['edges']&active).eq(gold['edges']).all())))
            summary[group]={d:dict(exact_edges=sum(r[d+'_exact_edges'] for r in records[group]),exact_graphs=sum(r[d+'_metrics']['semantic_equivalence'] for r in records[group]),examples=len(records[group])) for d in ('raw','calibrated')}
    cp=out/f'calibration-u{update}.npz';np.savez_compressed(cp,scores=joined.numpy(),targets=truth.numpy(),pairs=torch.cat(pairs).numpy(),offsets=np.asarray(offsets))
    path=out/f'evaluation-u{update}.json.gz';write_gzip(path,dict(update=update,thresholds=thresholds.tolist(),calibration=calibration,calibration_sha256=digest(cp),rows=records,summary=summary))
    print(json.dumps(dict(event='evaluation',update=update,**summary)),flush=True)
    return dict(update=update,summary=summary,artifact=path.name,sha256=digest(path))


def run(config):
    torch.set_num_threads(2);start=time.monotonic();out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False);data=Path(config['data_dir']);audit=json.loads((data/'audit.json').read_text());vocab=audit['value_vocabulary']
    sets={g:Dataset(load_cache(data/('train.jsonl.gz' if g=='train' else 'development.jsonl.gz'))[:config['train_count'] if g=='train' else config['evaluation_count']],vocab) for g in ('train','development')}
    checkpoint=torch.load(config['parent_checkpoint'],map_location='cuda',weights_only=True)
    model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').cuda();model.load_state_dict(checkpoint['model']);model.eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.load_state_dict(checkpoint['optimizer'])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('edge_source.','edge_target.')))
    optimizer_steps={name:float(optimizer.state[p]['step']) for name,p in model.named_parameters() if p.requires_grad};before=frozen_hash(model);tick=time.monotonic();cache={g:capture(model,sets[g],len(sets[g])) for g in sets};capture_seconds=time.monotonic()-tick
    cache_path=out/'nodes.pt';torch.save(cache,cache_path)
    generator=torch.Generator();generator.set_state(checkpoint['generator'].cpu());schedule=torch.Generator();schedule.set_state(checkpoint['schedule'].cpu());order=checkpoint['order'];position=checkpoint['position']
    if len(order)!=len(sets['train']):
        if not config.get('mechanical_profile'):raise ValueError('parent schedule does not match frozen TRAIN set')
        order=list(range(len(sets['train'])));position=0
    curves=[];losses=[];train_seconds=0.;torch.cuda.reset_peak_memory_stats()
    for update in range(config['updates']+1):
        if update in config['checkpoints']:
            curves.append(evaluate(model,cache,sets,out,update));path=out/f'edge-u{update}.pt';torch.save({k:v for k,v in model.state_dict().items() if k.startswith(('edge_source.','edge_target.'))},path);curves[-1]['checkpoint_sha256']=digest(path)
        if update==config['updates']:break
        indices=[]
        for _ in range(8):
            if position==len(order):order=torch.randperm(len(order),generator=schedule).tolist();position=0
            indices.append(order[position]);position+=1
        golds=[sets['train'][i][1] for i in indices];queries=[sampled_pairs(g,generator,128) for g in golds]
        nodes=cache['train']['nodes'][indices].cuda();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic();outputs=edge_scores(model,nodes,queries);batch=[]
        for logits,gold,q in zip(outputs,golds,queries):
            i,j=q.unbind(-1);truth=gold['edges'][i,j].cuda();raw=F.binary_cross_entropy_with_logits(logits,truth.float(),reduction='none');batch.append(torch.stack([raw[m].mean() for m in (truth,~truth) if m.any()]).mean())
        loss=torch.stack(batch).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();train_seconds+=time.monotonic()-tick
        if (update+1)%128==0 or update==0:losses.append(dict(update=update+1,loss=float(loss),training_seconds=train_seconds));print(json.dumps(dict(event='progress',**losses[-1])),flush=True)
    after=frozen_hash(model)
    if before!=after:raise ValueError('frozen non-edge parameters changed')
    source_names=['campaign_semantics_edge_probe.py','campaign_semantics.py','campaign_semantics_data.py','semantic_curriculum.py','semantic_scaling.py','semantic_contracts.py','thinking_language.py'];result=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),dependencies_sha256={n:digest(Path(__file__).with_name(n)) for n in source_names},data_sha256={n:digest(data/n) for n in ('audit.json','train.jsonl.gz','development.jsonl.gz')},inherited_edge_optimizer_steps=optimizer_steps,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),cache_parity_max_error={g:c['max_edge_reproduction_error'] for g,c in cache.items()},parent_checkpoint_sha256=digest(config['parent_checkpoint']),node_cache_sha256=digest(cache_path),node_width=1024,train_nodes=len(cache['train']['nodes']),development_nodes=len(cache['development']['nodes']),frozen_before_sha256=before,frozen_after_sha256=after,capture_seconds=capture_seconds,training_seconds=train_seconds,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,curves=curves,losses=losses,source_sha256=digest(__file__))
    write_gzip(out/'manifest.json.gz',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
