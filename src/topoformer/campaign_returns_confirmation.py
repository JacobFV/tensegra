"""R04: prospective paired diversity confirmation, with test captured after fitting."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from .return_memory import ReturnMemoryModel
from .return_crossdelay import feature_batch, prediction_record, save_cache, tensor_hash
from .campaign_returns import sha, calibration, group_counts


from .campaign_returns_diversity import fit_consumer
from .campaign_returns_balanced import capture_balanced


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['head_seed']);device=cfg['device']
    assert 32 not in cfg['delays']
    assert cfg['pool_sizes']==sorted(cfg['pool_sizes']) and max(cfg['pool_sizes'])==cfg['data']['train']['size']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    checkpoint=Path(cfg['checkpoint']);assert sha(checkpoint)==cfg['checkpoint_sha256']
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval()
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.requires_grad_(False)
    initial=tensor_hash(model.state_dict());cache={}
    for split,spec in cfg['data'].items():
        if split == 'test': continue
        for d in ([2] if split=='train' else cfg['eval_distractors']):
            cache[f'{split}/{d}']=feature_batch(model,spec['seed'],spec['size'],d,cfg['delays'],cfg['batch_size'],device)
    sets=[set(cache[f'{split}/2']['event_row_hashes']) for split in cfg['data'] if split != 'test']
    assert all(not(a&b) for i,a in enumerate(sets) for b in sets[i+1:])
    for split in cfg['data']:
        for key,batch in cache.items():
            if key.startswith(split+'/'):assert batch['event_row_hashes']==cache[f'{split}/2']['event_row_hashes']
    fits={};records=[]
    for count in cfg['pool_sizes']:
        tick=time.monotonic();head,mean,scale,record=fit_consumer(model,cache,count,cfg,device)
        path=out/f'ce-{count}.pt';torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),path)
        record.update(seconds=time.monotonic()-tick,checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in head.parameters()))
        records.append(record);fits[count]=(head,mean,scale)
    # Fixed consumers exist before any test feature capture; no test-driven selection.
    spec=cfg['data']['test']
    for d in cfg['eval_distractors']:
        cache[f'test/{d}']=feature_batch(model,spec['seed'],spec['size'],d,cfg['test_delays'],cfg['batch_size'],device)
        assert cache[f'test/{d}']['event_row_hashes']==cache['test/2']['event_row_hashes']
    test_ids=set(cache['test/2']['event_row_hashes'])
    assert all(not(test_ids & ids) for ids in sets)
    grid=cfg['balanced_grid']
    cache['balanced/8']=capture_balanced(model,grid['seed'],grid['pool_size'],grid['per_cell'],8,grid['delays'],cfg['batch_size'],device)
    grid_ids=set(cache['balanced/8']['event_row_hashes'])
    assert all(not(grid_ids & ids) for ids in sets+[test_ids])
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            split=key.split('/')[0]
            for delay,features in batch['features'].items():
                features=features.to(device)
                outputs={'unchanged':model.scalar_heads[0](features)}
                for count,(head,mean,scale) in fits.items():outputs[f'ce_{count}']=head((features-mean)/scale)
                for arm,scores in outputs.items():
                    logits[f'{key}/{delay}/{arm}']=scores.cpu()
                    row=prediction_record(cfg['backbone_seed'],arm,split,delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    top=scores.topk(2,dim=-1).values
                    target=batch['labels'].to(device)
                    nontruth=scores.clone();nontruth.scatter_(1,target[:,None],float('-inf'))
                    row['scalar_margin']=(top[:,0]-top[:,1]).cpu().tolist()
                    row['scalar_true_margin']=(scores.gather(1,target[:,None])[:,0]-nontruth.max(-1).values).cpu().tolist()
                    row['groups']=group_counts(row)
                    # Also preserve genuinely in-pool training fit, not only the maximal population.
                    if split=='train' and arm!='unchanged':
                        count=int(arm.split('_')[1]);pred=scores.argmax(-1).cpu()
                        row['in_pool_value']=dict(correct=int((pred[:count]==batch['labels'][:count]).sum()),total=count)
                        if count<len(pred):row['heldout_prefix_tail_value']=dict(correct=int((pred[count:]==batch['labels'][count:]).sum()),total=len(pred)-count)
                    rows.append(row)
    assert tensor_hash(model.state_dict())==initial
    predictions=out/'predictions.json.gz';predictions.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_balanced.py','campaign_returns_confirmation.py','campaign_returns_diversity.py','campaign_returns.py','return_crossdelay.py','return_memory.py','retention_data.py','thinking.py')},
        checkpoint_sha256=sha(checkpoint),backbone_state_sha256=initial,width=1024,backbone_parameters=sum(p.numel() for p in model.parameters()),
        fits=records,predictions_sha256=sha(predictions),features=save_cache(out/'features.pt.gz',cache),logits=save_cache(out/'logits.pt.gz',logits),
        seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),total_device_bytes=torch.cuda.get_device_properties(0).total_memory))
    raw=json.dumps(manifest,indent=2).encode();(out/'manifest.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    print(json.dumps(dict(seconds=manifest['seconds'],peak_cuda_allocated=manifest['peak_cuda_allocated'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    for entry in cfg['runs']:
        merged={k:v for k,v in cfg.items() if k!='runs'};merged.update(entry)
        run(merged,Path(a.output)/str(entry['backbone_seed']))
