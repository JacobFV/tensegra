"""Frozen A08 inference; no model selection or optimizer."""
import argparse,hashlib,json,time,resource
from pathlib import Path
import numpy as np
import torch
from .campaign_attention_selector import SelectorModel,generate,targets,metrics,oracle_successors
from .campaign_attention_corruption import replace_edges
from .campaign_attention_study import write,sync


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    device=cfg.get('device','cuda');torch.set_num_threads(2);start=time.monotonic()
    write(out/'source.json',{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['campaign_attention_corruption.py','campaign_attention_corruption_study.py','campaign_attention_selector.py','campaign_attention.py']})
    receipts=[];public={}
    for checkpoint in cfg['checkpoints']:
        path=Path(checkpoint['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=checkpoint['sha256']:raise ValueError('Checkpoint SHA mismatch')
        model=SelectorModel(width=1024).to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval()
        tensor_hash=lambda:hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
        initial=tensor_hash();raw={};rows=[]
        for ci,condition in enumerate(cfg['conditions']):
            pieces={};seconds=0.
            for offset in range(0,cfg['examples'],cfg['batch']):
                count=min(cfg['batch'],cfg['examples']-offset)
                base=generate(count,condition['nodes'],condition['depth'],groups=condition['groups'],seed=cfg['data_seed']+condition['data_group']*100000+offset,device=device,balanced=True)
                given=replace_edges(base,condition['fraction'],cfg['replacement_seed']+condition['data_group']*100000+offset)
                sync(device);t=time.monotonic()
                with torch.no_grad():result=model(given,checkpoint['mode'],selector_scale_override=16.,context_scale_override=16. if checkpoint['mode']=='context' else None)
                sync(device);seconds+=time.monotonic()-t
                gold=targets(base);supplied=targets(given);score=metrics(result,gold,base)
                pred=result['logits'].argmax(-1);bi=torch.arange(count,device=device)
                score.update(agreement_supplied_task=pred[:,-1][bi,base.starts]==supplied[:,-1][bi,base.starts],corrupt_pointer_task=supplied[:,-1][bi,base.starts]==gold[:,-1][bi,base.starts],removed_fraction=(base.adjacency*(1-given.adjacency)).sum((1,2,3))/base.adjacency.sum((1,2,3)),added_fraction=(given.adjacency*(1-base.adjacency)).sum((1,2,3))/base.adjacency.sum((1,2,3)))
                fields=dict(pred=pred,gold=gold,supplied_gold=supplied,route=result['routes'],start=base.starts,relation=base.relations,successor=oracle_successors(base),supplied_successor=oracle_successors(given),adjacency=base.adjacency,supplied_adjacency=given.adjacency)
                for name,x in {**score,**fields}.items():pieces.setdefault(name,[]).append(x.detach().cpu().numpy().astype('uint8') if name in fields else x.detach().cpu().numpy())
            joined={name:np.concatenate(x) for name,x in pieces.items()}
            for name,x in joined.items():
                if name in fields and name not in {'pred','route'}:
                    if not receipts:public[f'c{ci}_{name}']=x
                else:raw[f'c{ci}_{name}']=x
            rows.append(dict(condition=condition,examples=cfg['examples'],forward_seconds=seconds,**{name:float(joined[name].mean()) for name in score}))
        tag=f"{checkpoint['seed']}-{checkpoint['mode']}"
        np.savez_compressed(out/f'{tag}.npz',**raw);write(out/f'{tag}.json',dict(rows=rows))
        receipts.append(dict(**checkpoint,initial_tensor_sha256=initial,final_tensor_sha256=tensor_hash(),parameters=sum(p.numel() for p in model.parameters())))
        del model
    np.savez_compressed(out/'public.npz',**public)
    write(out/'manifest.json',dict(checkpoints=receipts,optimizer_updates=0,examples_per_cell=cfg['examples'],wall_seconds=time.monotonic()-start,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(completed=str(out),seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
