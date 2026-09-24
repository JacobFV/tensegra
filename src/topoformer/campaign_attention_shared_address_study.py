"""A13 inference-only shared-address localization; no parameter updates or selection."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from .campaign_attention_records import RecordAttention,tokenize
from .campaign_attention_shared_address import forward,POLICIES
from .campaign_attention_selector import generate,targets,oracle_successors,metrics
from .campaign_attention_study import write,sync


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    device=cfg.get('device','cuda');torch.set_num_threads(2)
    checkpoint=Path(cfg['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:raise ValueError('Checkpoint SHA mismatch')
    model=RecordAttention(width=1024).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
    digest=lambda:hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
    initial=digest()
    if initial!=cfg['tensor_sha256']:raise ValueError('Tensor SHA mismatch')
    files=['campaign_attention_records.py','campaign_attention_shared_address.py','campaign_attention_shared_address_study.py','campaign_attention_selector.py','campaign_attention.py']
    write(out/'source.json',{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files})
    start=time.monotonic();public={};raw={p:{} for p in POLICIES};rows={p:[] for p in POLICIES}
    for ci,c in enumerate(cfg['conditions']):
        collected={p:{} for p in POLICIES};shared={};elapsed={p:0. for p in POLICIES}
        for offset in range(0,cfg['examples'],cfg['batch']):
            count=min(cfg['batch'],cfg['examples']-offset)
            batch=generate(count,c['nodes'],c['depth'],groups=c['groups'],seed=cfg['data_seed']+ci*100000+offset,device=device,balanced=True)
            g=torch.Generator(device=device).manual_seed(cfg['order_seed']+ci*100000+offset)
            order=torch.rand(count,3*c['nodes']*c['groups'],generator=g,device=device).argsort(-1)
            records=tokenize(batch,order)
            indices=batch.adjacency.bool().nonzero().reshape(count,-1,4)[:,:,1:]
            indices=indices.gather(1,order[...,None].expand(-1,-1,3))
            gold=targets(batch);successor=oracle_successors(batch)
            for name,value in dict(gold=gold,successor=successor,start=batch.starts,relation=batch.relations,values=batch.values).items():
                shared.setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
            for policy in POLICIES:
                sync(device);begin=time.monotonic()
                result=forward(model,batch,records,indices,successor,policy)
                sync(device);elapsed[policy]+=time.monotonic()-begin
                scores=metrics(result,gold,batch)
                fields=dict(pred=result['logits'].argmax(-1),route=result['routes'])
                for name,value in fields.items():collected[policy].setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
                for name,value in scores.items():collected[policy].setdefault(name,[]).append(value.cpu().numpy())
                for name,value in result['diagnostics'].items():collected[policy].setdefault('diagnostic_'+name,[]).append(value.cpu().numpy())
        for name,pieces in shared.items():public[f'c{ci}_{name}']=np.concatenate(pieces)
        for policy in POLICIES:
            joined={name:np.concatenate(pieces) for name,pieces in collected[policy].items()}
            raw[policy].update({f'c{ci}_{name}':value for name,value in joined.items()})
            rows[policy].append(dict(condition=c,examples=cfg['examples'],instrumented_forward_seconds=elapsed[policy],
                                     **{name:float(joined[name].mean()) for name in scores},
                                     diagnostics={name:float(value.mean()) for name,value in joined.items() if name.startswith('diagnostic_')}))
    np.savez_compressed(out/'public.npz',**public)
    for policy in POLICIES:
        np.savez_compressed(out/f'{policy}.npz',**raw[policy]);write(out/f'{policy}.json',dict(rows=rows[policy]))
    final=digest()
    if initial!=final:raise RuntimeError('Frozen model mutated')
    write(out/'manifest.json',dict(initial_tensor_sha256=initial,final_tensor_sha256=final,checkpoint_sha256=cfg['checkpoint_sha256'],optimizer_updates=0,policies=list(POLICIES),wall_seconds=time.monotonic()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(completed=str(out),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
