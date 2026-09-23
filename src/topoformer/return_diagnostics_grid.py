"""Bounded scalar-grid diagnostic through frozen Stage8 encoders, no recurrence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .return_memory import ReturnMemoryModel
from .retention_data import make_batch
from .return_memory_study import move
from .return_diagnostics_probe import fit_ridge, predict_ridge, metrics, event_hash


def batch(seed, contexts, device):
    data=move(make_batch(seed,33*contexts,distractors=0),device)
    y=torch.arange(33,device=device).repeat(contexts)
    event=data['public']['event']; event['values']=(y.float()/2-8)[:,None]
    # Float/add produces an auditable full bounded grid. Operand witnesses follow.
    event['types'].fill_(1);event['operations'].fill_(0)
    event['operand_values'][:,:,0]=0;event['operand_values'][:,:,1]=event['values']
    event['argument_mask'].fill_(True)
    # make_batch unary absence is repaired by selecting a public nonzero identity.
    missing=event['arguments'][:,:,1].square().sum(-1)==0
    event['arguments'][:,:,1]=torch.where(missing[:,:,None],data['public']['argument_keys'][:,1:2],event['arguments'][:,:,1])
    return data['public'],y


def run(config,out):
    torch.set_num_threads(2);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic()
    for seed in config['seeds']:
      for encoding in config['encodings']:
       for access in config['availability']:
        name=f'{seed}-{encoding}-{access}'; path=Path(config['checkpoint_dir'])/(name+'.pt')
        model=ReturnMemoryModel(width=1024,encoding=encoding).to(config['device']).eval()
        model.load_state_dict(torch.load(path,map_location=config['device'],weights_only=True)); parts={}
        with torch.no_grad():
         for split,ds in config['data_seeds'].items():
            public,y=batch(ds,config['contexts'],config['device']);e=public['event']
            raw=model.encoders[0]((e['values']/8).unsqueeze(-1))[:,0]
            padded=F.pad(raw,(0,1024-raw.shape[-1]));memory=model.encode(e,encoding)[:,0]
            parts[split]=(dict(raw=padded,memory=memory),y,event_hash(e))
         for boundary in ('raw','memory'):
            x,y,_=parts['train']; fit=fit_ridge(x[boundary],y.float()[:,None],config['ridge'])
            for split,(features,target,event_id) in parts.items():
                pred=predict_ridge(features[boundary],fit)[:,0].round().long().clamp(0,32)
                rows.append(dict(run=name,boundary=boundary,split=split,metrics=metrics(pred,target),predictions=pred.tolist(),targets=target.tolist(),event_sha256=event_id))
         # Identical nuisance context, varying only scalar, tests actual geometry.
         public,y=batch(9500101,1,config['device']);e=public['event']
         for key in e:
            if key!='values': e[key]=e[key][:1].expand_as(e[key]).clone()
         e['operand_values'][:,:,1]=e['values']
         raw=model.encoders[0]((e['values']/8).unsqueeze(-1))[:,0]; mem=model.encode(e,encoding)[:,0]
         for boundary,vectors in [('raw',raw),('memory',mem)]:
            distance=torch.cdist(vectors,vectors);distance.fill_diagonal_(float('inf'))
            rows.append(dict(run=name,boundary=boundary,kind='fixed_nuisance_geometry',minimum_pair_distance=float(distance.min()),adjacent_distances=(vectors[1:]-vectors[:-1]).norm(dim=-1).tolist(),norms=vectors.norm(dim=-1).tolist()))
    (out/'results.json').write_text(json.dumps(rows,separators=(',',':')))
    (out/'manifest.json').write_text(json.dumps(dict(config=config,seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpretation='privileged frozen encoder numerical readout; full bounded grid repeated in fresh nuisance contexts; not unseen numerical labels or complete workspace interface'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
