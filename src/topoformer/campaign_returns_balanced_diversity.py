"""R10: nested balanced-context diversity at matched scalar-consumer exposure."""
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import torch
from .retention_data import make_batch
from .return_memory import ReturnMemoryModel
from .return_crossdelay import feature_batch,prediction_record,save_cache,tensor_hash
from .campaign_returns import sha,group_counts
from .campaign_returns_balanced import capture_selected,capture_balanced
from .campaign_returns_diversity import fit_consumer


def nested_indices(batch,count,seed):
    """Round-robin legal strata: every prefix has balanced counts and exact nesting."""
    strata=[(typ,label) for typ,labels in ((0,range(0,33,2)),(1,range(33)),(2,(16,18))) for label in labels]
    order=torch.randperm(len(strata),generator=torch.Generator().manual_seed(seed)).tolist()
    strata=[strata[i] for i in order]
    available=[((batch['targets']['type']==typ)&(batch['targets']['value']==label)).nonzero().flatten() for typ,label in strata]
    for i,ids in enumerate(available):
        needed=count//52+int(i<count%52)
        if len(ids)<needed:raise ValueError(f'insufficient fixed-pool support: {strata[i]}')
    indices=torch.tensor([int(available[i%52][i//52]) for i in range(count)])
    assert len(indices.unique())==count
    return indices


def prefix_cache(batch,count):
    """View the captured maximum pool without duplicating it on disk."""
    return dict(batch,features={d:x[:count] for d,x in batch['features'].items()},
                labels=batch['labels'][:count],targets={k:v[:count] for k,v in batch['targets'].items()},
                event_row_hashes=batch['event_row_hashes'][:count])


def subset(batch,indices):
    return {k:subset(v,indices) if isinstance(v,dict) else v[indices] for k,v in batch.items()}


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['head_seed']);device=cfg['device'];out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();timings={};torch.cuda.reset_peak_memory_stats();assert 32 not in cfg['delays'] and 32 not in cfg['grid_delays']
    path=Path(cfg['checkpoint']);assert sha(path)==cfg['checkpoint_sha256']
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval();model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.requires_grad_(False)
    initial=tensor_hash(model.state_dict());pool=make_batch(cfg['pool_seed'],cfg['pool_size'],distractors=2)
    assert cfg['fit_sizes']==sorted(cfg['fit_sizes'])
    ids=nested_indices(pool,max(cfg['fit_sizes']),cfg['strata_seed'])
    batch=subset(pool,ids)
    cache={'train/2':capture_selected(model,batch,cfg['pool_seed'],ids,2,cfg['delays'],cfg['batch_size'],device)}
    del pool
    torch.cuda.synchronize();timings['training_capture_seconds']=time.monotonic()-started;tick=time.monotonic()
    for split,spec in cfg['data'].items():
        for distractors in cfg['eval_distractors']:
            cache[f'{split}/{distractors}']=feature_batch(model,spec['seed'],spec['size'],distractors,cfg['delays'],cfg['batch_size'],device)
    for split,spec in cfg['grids'].items():
        cache[f'{split}/8']=capture_balanced(model,spec['seed'],spec['pool_size'],spec['per_cell'],8,cfg['grid_delays'],cfg['batch_size'],device)
    torch.cuda.synchronize();timings['evaluation_capture_seconds']=time.monotonic()-tick;tick=time.monotonic()
    for split in cfg['data']:
        reference=cache[f'{split}/2']['event_row_hashes']
        for distractors in cfg['eval_distractors']:
            assert cache[f'{split}/{distractors}']['event_row_hashes']==reference
    train_sets=[set(cache['train/2']['event_row_hashes'][:n]) for n in cfg['fit_sizes']]
    heldout=[set(cache[f'{split}/2']['event_row_hashes']) for split in cfg['data']]+[set(cache[f'{split}/8']['event_row_hashes']) for split in cfg['grids']]
    assert all(not(a&b) for a in train_sets for b in heldout)
    assert all(not(a&b) for i,a in enumerate(heldout) for b in heldout[i+1:])
    fits={};records=[]
    for count in cfg['fit_sizes']:
        arm=f'balanced_{count}'
        fitcache={k:v for k,v in cache.items() if k.startswith('calibration/')};fitcache['train/2']=prefix_cache(cache['train/2'],count)
        head,mean,scale,record=fit_consumer(model,fitcache,count,cfg,device)
        path=out/f'{arm}.pt';torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),path)
        record.update(arm=arm,checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in head.parameters()));records.append(record);fits[arm]=(head,mean,scale)
    torch.cuda.synchronize();timings['fit_seconds']=time.monotonic()-tick;tick=time.monotonic()
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            for delay,x in batch['features'].items():
                x=x.to(device);outputs={'unchanged':model.scalar_heads[0](x)}
                for arm,(head,mean,scale) in fits.items():outputs[arm]=head((x-mean)/scale)
                for arm,scores in outputs.items():
                    row=prediction_record(cfg['backbone_seed'],arm,key.split('/')[0],delay,batch['distractors'],batch,scores.argmax(-1).cpu());row['groups']=group_counts(row)
                    if key=='train/2' and arm!='unchanged':
                        n=int(arm.split('_')[1]);pred=scores.argmax(-1).cpu()
                        row['in_pool_value']=dict(correct=int((pred[:n]==batch['labels'][:n]).sum()),total=n)
                        if n<len(pred):row['heldout_prefix_tail_value']=dict(correct=int((pred[n:]==batch['labels'][n:]).sum()),total=len(pred)-n)
                    rows.append(row);logits[f'{key}/{delay}/{arm}']=scores.cpu()
    torch.cuda.synchronize();timings['readout_evaluation_seconds']=time.monotonic()-tick
    assert tensor_hash(model.state_dict())==initial
    raw=out/'predictions.json.gz';raw.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),source={n:sha(Path(__file__).with_name(n)) for n in ('campaign_returns_balanced_diversity.py','campaign_returns_balanced.py','campaign_returns_diversity.py','campaign_returns.py','return_crossdelay.py','return_memory.py','return_memory_study.py','return_diagnostics_probe.py','retention_data.py','thinking.py','campaign_returns_balanced_fit.py')},
                  phase_seconds=timings,backbone_state_sha256=initial,backbone_parameters=sum(p.numel() for p in model.parameters()),width=1024,memory_tokens=6,fit_population_overlap=len(train_sets[0]&train_sets[1]),fits=records,
                  cache=save_cache(out/'cache.pt.gz',cache),logits=save_cache(out/'logits.pt.gz',logits),predictions_sha256=sha(raw),seconds=time.monotonic()-started,
                  peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    (out/'manifest.json.gz').write_bytes(gzip.compress(json.dumps(manifest,indent=2).encode(),mtime=0));print(json.dumps(dict(seconds=manifest['seconds'],peak_cuda_allocated=manifest['peak_cuda_allocated'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
