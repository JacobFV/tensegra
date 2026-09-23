"""R01 frozen wide-horizon scalar consumers, with adaptive campaign provenance."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .return_memory import ReturnMemoryModel
from .return_crossdelay import feature_batch, prediction_record, save_cache, tensor_hash
from .return_diagnostics_probe import fit_ridge, predict_ridge


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_score(cells):
    """All cells have the same support; minimum first, then total exact count."""
    if not cells or len({cell['total'] for cell in cells}) != 1:
        raise ValueError('Selection requires equal nonempty cell support')
    return min(cell['correct'] for cell in cells), sum(cell['correct'] for cell in cells)


def calibration(cache, predictor, device):
    cells=[]
    with torch.no_grad():
        for key,batch in cache.items():
            if not key.startswith('calibration/'):continue
            for delay,x in batch['features'].items():
                p=predictor(x.to(device)).argmax(-1).cpu()
                cells.append(dict(distractors=batch['distractors'],delay=delay,
                                  correct=int((p==batch['labels']).sum()),total=len(p)))
    return cells


def group_counts(row):
    targets=row['targets']; predictions=row['predictions']['value']; y=targets['value']
    result={}
    def add(group,key,index):
        cell=result.setdefault(group,{}).setdefault(str(key),dict(correct=0,total=0,signed_error_sum=0.,absolute_error_sum=0.))
        error=(predictions[index]-y[index])/2
        cell['correct']+=int(predictions[index]==y[index]);cell['total']+=1
        cell['signed_error_sum']+=error;cell['absolute_error_sum']+=abs(error)
    for i,label in enumerate(y):
        value=(label-16)/2
        for field in targets:add(field,targets[field][i],i)
        add('value_type_operation',f'{label}/{targets["type"][i]}/{targets["operation"][i]}',i)
        add('sign',-1 if value<0 else int(value>0),i);add('magnitude',abs(value),i)
    return result


def run(cfg,out):
    torch.set_num_threads(2);torch.manual_seed(cfg['head_seed']);device=cfg['device']
    if 32 in cfg['delays']:raise ValueError('R01 development forbids delay32 exposure')
    if len({s['seed'] for s in cfg['data'].values()})!=len(cfg['data']):raise ValueError('Overlapping split seeds')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(cfg['checkpoint']);assert sha(checkpoint)==cfg['checkpoint_sha256']
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval()
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.requires_grad_(False)
    cache={};capture_times={}
    for split,spec in cfg['data'].items():
        for distractor in ([2] if split=='train' else cfg['eval_distractors']):
            tick=time.monotonic()
            cache[f'{split}/{distractor}']=feature_batch(model,spec['seed'],spec['size'],distractor,cfg['delays'],cfg['batch_size'],device)
            torch.cuda.synchronize();capture_times[f'{split}/{distractor}']=time.monotonic()-tick
    sets=[set(cache[f'{split}/2']['event_row_hashes']) for split in cfg['data']]
    assert all(not(left&right) for i,left in enumerate(sets) for right in sets[i+1:])
    for split in cfg['data']:
        reference=cache[f'{split}/2']
        for key,batch in cache.items():
            if key.startswith(split+'/'):assert batch['event_row_hashes']==reference['event_row_hashes']
    train=cache['train/2'];x=torch.cat([train['features'][d] for d in cfg['delays']]).to(device)
    y=train['labels'].repeat(len(cfg['delays'])).to(device)
    if cfg['mode']!='profile':assert y.unique().numel()==33
    candidates=[];best_score=None;ridge=None;selected_alpha=None;fit_times=[]
    for alpha in cfg['ridge_grid']:
        tick=time.monotonic();fit=fit_ridge(x,F.one_hot(y,33).float(),alpha)
        cells=calibration(cache,lambda z:predict_ridge(z,fit),device);score=selection_score(cells)
        fit_times.append(time.monotonic()-tick);candidates.append(dict(alpha=alpha,cells=cells,score=score))
        if best_score is None or score>best_score:best_score=score;ridge=fit;selected_alpha=alpha
    ridgepath=out/'ridge.pt';torch.save(tuple(v.cpu() for v in ridge),ridgepath)
    mean,scale,_=ridge
    head=copy.deepcopy(model.scalar_heads[0]);head.requires_grad_(True)
    with torch.no_grad():
        weight=head.weight.clone();head.weight.mul_(scale);head.bias.add_(weight@mean)
        torch.testing.assert_close(head((x[:128]-mean)/scale),model.scalar_heads[0](x[:128]),atol=1e-4,rtol=1e-4)
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['ce_lr']);z=(x-mean)/scale
    generator=torch.Generator(device=device).manual_seed(cfg['head_seed'])
    ce_curve=[];ce_losses=[];best=None;best_state=None;best_step=None;tick=time.monotonic()
    for step in range(cfg['ce_updates']+1):
        if step%cfg['ce_check_every']==0 or step==cfg['ce_updates']:
            cells=calibration(cache,lambda features:head((features-mean)/scale),device)
            score=selection_score(cells);ce_curve.append(dict(step=step,cells=cells,score=score))
            if best is None or score>best:best=score;best_state=copy.deepcopy(head.state_dict());best_step=step
        if step==cfg['ce_updates']:break
        index=torch.randint(len(x),(cfg['ce_batch_size'],),device=device,generator=generator)
        loss=F.cross_entropy(head(z[index]),y[index]);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        ce_losses.append(float(loss.detach()))
    ce_seconds=time.monotonic()-tick;head.load_state_dict(best_state);head.eval()
    cepath=out/'ce.pt';torch.save(dict(state={k:v.cpu() for k,v in head.state_dict().items()},mean=mean.cpu(),scale=scale.cpu()),cepath)
    rows=[];logits={}
    with torch.no_grad():
        for key,batch in cache.items():
            split=key.split('/')[0]
            for delay,features in batch['features'].items():
                features=features.to(device)
                outputs=dict(unchanged=model.scalar_heads[0](features),ridge=predict_ridge(features,ridge),ce=head((features-mean)/scale))
                for arm,scores in outputs.items():
                    logits[f'{key}/{delay}/{arm}']=scores.cpu()
                    row=prediction_record(cfg['backbone_seed'],arm,split,delay,batch['distractors'],batch,scores.argmax(-1).cpu())
                    row['groups']=group_counts(row);rows.append(row)
    predictions=out/'predictions.json.gz';predictions.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
    feature_record=save_cache(out/'features.pt.gz',cache);logit_record=save_cache(out/'logits.pt.gz',logits)
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={name:sha(Path(__file__).with_name(name)) for name in ('campaign_returns.py','return_crossdelay.py','return_diagnostics_probe.py','return_memory.py','retention_data.py','thinking.py')},
        checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),head_parameters=sum(p.numel() for p in head.parameters()),
        width=1024,memory_tokens=6,memory_coordinates=6144,fit_rows=len(y),fit_unique_events=len(set(train['event_row_hashes'])),
        label_counts=torch.bincount(train['labels'],minlength=33).tolist(),feature_cache=feature_record,logit_cache=logit_record,
        initial_backbone_state_sha256=tensor_hash(model.state_dict()),fit_candidates=candidates,selected_alpha=selected_alpha,
        ce_curve=ce_curve,ce_losses=ce_losses,selected_ce_step=best_step,ce_optimizer_presentations=cfg['ce_updates']*cfg['ce_batch_size'],
        readout_hashes={p.name:sha(p) for p in (ridgepath,cepath)},predictions_sha256=sha(predictions),
        capture_seconds=capture_times,ridge_seconds=fit_times,ce_seconds=ce_seconds,seconds=time.monotonic()-started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),total_device_bytes=torch.cuda.get_device_properties(0).total_memory))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(seconds=manifest['seconds'],selected_alpha=selected_alpha,selected_ce_step=best_step,capture_seconds=capture_times)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
