"""R06: change fitting-population strata, keeping return backbone/readout recipe fixed."""
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from .retention_data import make_batch
from .return_memory import ReturnMemoryModel
from .return_crossdelay import feature_batch,prediction_record,save_cache,tensor_hash
from .campaign_returns import sha,group_counts
from .campaign_returns_balanced import capture_selected,capture_balanced
from .campaign_returns_diversity import fit_consumer


def fitting_indices(batch,count,seed):
    strata=[(typ,label) for typ,labels in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))) for label in labels]
    base,remainder=divmod(count,len(strata));extra=set(torch.randperm(len(strata),generator=torch.Generator().manual_seed(seed))[:remainder].tolist())
    selected=[]
    for i,(typ,label) in enumerate(strata):
        available=((batch['targets']['type']==typ)&(batch['targets']['value']==label)).nonzero().flatten();needed=base+int(i in extra)
        if len(available)<needed:raise ValueError(f'insufficient fixed-pool support: {typ}/{label}')
        selected.append(available[:needed])
    indices=torch.cat(selected)
    assert len(indices)==count and len(indices.unique())==count
    return indices


def subset(batch,indices):
    return {k:subset(v,indices) if isinstance(v,dict) else v[indices] for k,v in batch.items()}


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['head_seed']);device=cfg['device'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();assert 32 not in cfg['delays'] and 32 not in cfg['grid_delays']
    path=Path(cfg['checkpoint']);assert sha(path)==cfg['checkpoint_sha256']
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval();model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.requires_grad_(False)
    initial=tensor_hash(model.state_dict());pool=make_batch(cfg['pool_seed'],cfg['pool_size'],distractors=2)
    original_ids=torch.arange(cfg['fit_events']);balanced_ids=fitting_indices(pool,cfg['fit_events'],cfg['strata_seed'])
    indices={'original':original_ids,'balanced':balanced_ids};cache={}
    for arm,ids in indices.items():
        batch=subset(pool,ids)
        assert torch.equal(batch['targets']['value'],(2*batch['public']['event']['values'][:,0]+16).long())
        cache[f'train_{arm}/2']=capture_selected(model,batch,cfg['pool_seed'],ids,2,cfg['delays'],cfg['batch_size'],device)
    del pool
    for split,spec in cfg['data'].items():
        for distractors in cfg['eval_distractors']:
            cache[f'{split}/{distractors}']=feature_batch(model,spec['seed'],spec['size'],distractors,cfg['delays'],cfg['batch_size'],device)
    for split,spec in cfg['grids'].items():
        cache[f'{split}/8']=capture_balanced(model,spec['seed'],spec['pool_size'],spec['per_cell'],8,cfg['grid_delays'],cfg['batch_size'],device)
    train_sets=[set(cache[f'train_{arm}/2']['event_row_hashes']) for arm in indices]
    heldout=[set(cache[f'{split}/2']['event_row_hashes']) for split in cfg['data']]+[set(cache[f'{split}/8']['event_row_hashes']) for split in cfg['grids']]
    assert all(not(a&b) for a in train_sets for b in heldout)
    assert all(not(a&b) for i,a in enumerate(heldout) for b in heldout[i+1:])
    fits={};records=[]
    for arm in indices:
        fitcache={k:v for k,v in cache.items() if k.startswith('calibration/')};fitcache['train/2']=cache[f'train_{arm}/2']
        head,mean,scale,record=fit_consumer(model,fitcache,cfg['fit_events'],cfg,device)
        path=out/f'{arm}.pt';torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),path)
        record.update(arm=arm,checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in head.parameters()));records.append(record);fits[arm]=(head,mean,scale)
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            for delay,x in batch['features'].items():
                x=x.to(device);outputs={'unchanged':model.scalar_heads[0](x)}
                for arm,(head,mean,scale) in fits.items():outputs[arm]=head((x-mean)/scale)
                for arm,scores in outputs.items():
                    row=prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu());row['groups']=group_counts(row);rows.append(row);logits[f'{key}/{delay}/{arm}']=scores.cpu()
    assert tensor_hash(model.state_dict())==initial
    raw=out/'predictions.json.gz';raw.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_balanced_fit.py','campaign_returns_balanced.py','campaign_returns_diversity.py','campaign_returns.py','return_crossdelay.py','return_memory.py','return_memory_study.py','return_diagnostics_probe.py','retention_data.py','thinking.py')},
                  backbone_state_sha256=initial,backbone_parameters=sum(p.numel() for p in model.parameters()),width=1024,memory_tokens=6,fit_population_overlap=len(train_sets[0]&train_sets[1]),fits=records,
                  cache=save_cache(out/'cache.pt.gz',cache),logits=save_cache(out/'logits.pt.gz',logits),predictions_sha256=sha(raw),seconds=time.monotonic()-started,
                  peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0));print(json.dumps(dict(seconds=manifest['seconds'],peak_cuda_allocated=manifest['peak_cuda_allocated'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
