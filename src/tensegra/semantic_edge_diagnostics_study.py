"""Stage 10 exposure-only fixed-node fitting, one representative slot arm."""
from __future__ import annotations
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .semantic_contracts_study import fixtures,FixedNodeDecoder,evaluate
from .semantic_contracts import slot_objective,train_relation_thresholds
from .thinking_language import state_hash


def tensor_record(tensor):
    return tensor.detach().cpu().tolist()


def checkpoint(model,examples,update,out,seed):
    with torch.no_grad():
        logits=[model(n,p)[0] for n,p,_,_ in examples]
        truths=[g['edges'].flatten(0,1).to(logits[0].device) for _,_,g,_ in examples]
        thresholds,records=train_relation_thresholds(torch.cat(logits),torch.cat(truths))
        raw=evaluate(model,examples);calibrated=evaluate(model,examples,thresholds)
        # Exact FP32 score arrays and labels preserve every per-error trajectory.
        data=dict(seed=seed,update=update,relations=records,graphs=[dict(graph_seed=a['seed'],nodes=len(n),scores=tensor_record(logit),truth=tensor_record(truth)) for (n,_,_,a),logit,truth in zip(examples,logits,truths)])
    scores_path=out/f'scores-seed{seed}-u{update}.json.gz'
    with gzip.GzipFile(filename=str(scores_path),mode='wb',mtime=0) as stream:stream.write(json.dumps(data,separators=(',',':')).encode())
    path=out/f'model-seed{seed}-u{update}.pt';torch.save(model.state_dict(),path)
    return dict(update=update,raw=raw,calibrated=calibrated,thresholds=records,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),score_file=scores_path.name,score_sha256=hashlib.sha256(scores_path.read_bytes()).hexdigest(),state_sha256=state_hash(model))


def run(config):
    torch.set_num_threads(2);device=config['device'];out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    examples=fixtures(8,config['width'],config['data_seed'],device);runs=[]
    manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('semantic_edge_diagnostics_study.py','semantic_contracts_study.py','semantic_contracts.py')},graph_audits=[a for _,_,_,a in examples],runs=runs,
      environment=dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name() if device=='cuda' else 'cpu'),
      interpretation='Privileged fixed-node acquisition only; paired checkpoints of same trajectory, not independent arms. No text or generalization. Gold TRAIN edges fit thresholds only.')
    for seed in config['seeds']:
        torch.manual_seed(seed);model=FixedNodeDecoder(config['width'],rank=config['rank'],edge_affine=True).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);start=time.monotonic();curves=[];losses=[];training_seconds=0.
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        initial_hash=state_hash(model)
        for update in range(config['updates']+1):
            if update in config['checkpoints']:curves.append(checkpoint(model,examples,update,out,seed))
            if update==config['updates']:break
            if device=='cuda':torch.cuda.synchronize()
            stepstart=time.monotonic();optimizer.zero_grad();parts=[]
            for nodes,pairs,gold,_ in examples:
                edge,slots=model(nodes,pairs);truth=gold['edges'].to(device).flatten(0,1);labels=gold['slots'].to(device).flatten()+1
                raw=F.binary_cross_entropy_with_logits(edge,truth.float(),reduction='none')
                parts.append(.5*(raw[truth].mean()+raw[~truth].mean())+slot_objective(slots,labels,truth.any(-1),edge_conditional=True))
            loss=torch.stack(parts).mean();loss.backward();optimizer.step()
            if device=='cuda':torch.cuda.synchronize()
            training_seconds+=time.monotonic()-stepstart;losses.append(float(loss.detach()))
        runs.append(dict(seed=seed,initial_state_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),width=config['width'],head_rank=config['rank'],actual_unique_graphs=8,optimizer_presentations=8*config['updates'],optimizer_updates=config['updates'],training_seconds=training_seconds,wall_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None,process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,curves=curves,losses=losses))
        with gzip.GzipFile(filename=str(out/'results.json.gz'),mode='wb',mtime=0) as stream:stream.write(json.dumps(manifest,separators=(',',':')).encode())
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
