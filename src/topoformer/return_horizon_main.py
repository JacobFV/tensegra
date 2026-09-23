"""Frozen paired continuation protocol; no new architecture or readout fitting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from .return_horizon import update, schedule_counts
from .return_memory import ReturnMemoryModel
from .return_memory_study import move
from .return_crossdelay import feature_batch, prediction_record, tensor_hash
from .return_diagnostics_probe import predict_ridge
from .retention_data import make_batch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def run(cfg, output):
    torch.set_num_threads(2)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    device=cfg['device']
    assert cfg['validation_seed'] != cfg['test_seed']
    assert 32 not in set(sum(cfg['schedules'].values(),[]))
    old=json.loads((Path(cfg['checkpoint_dir'])/'manifest.json').read_text())
    ridge_manifest=json.loads((Path(cfg['ridge_dir'])/'manifest.json').read_text())
    manifest=dict(config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        source={name:sha(Path(__file__).with_name(name)) for name in ('return_horizon_main.py','return_horizon.py',
           'return_memory.py','return_crossdelay.py','return_diagnostics_probe.py','retention_data.py','thinking.py')},
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
            total_device_bytes=torch.cuda.get_device_properties(0).total_memory),runs=[])
    for pair,seed in enumerate(cfg['backbone_seeds']):
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        path=Path(cfg['checkpoint_dir'])/f'{seed}-backbone.pt'
        expected=next(r for r in old['runs'] if r['seed']==seed)['checkpoint_hashes'][path.name]
        assert sha(path)==expected
        initial=torch.load(path,map_location='cpu',weights_only=True)
        initial_hash=tensor_hash(initial)
        ridge_path=Path(cfg['ridge_dir'])/f'{seed}-shared_pool-fit.pt'
        ridge_expected=next(f for r in ridge_manifest['runs'] if r['seed']==seed
                            for f in r['ridge_fit_records'] if f['head']=='shared_pool')['fit_sha256']
        assert sha(ridge_path)==ridge_expected
        ridge=tuple(v.to(device) for v in torch.load(ridge_path,map_location='cpu',weights_only=True))
        rows=[]; curves=[]; arm_meta=[]; event_reference={}
        def evaluate(model,arm,split,delays):
            data_seed=cfg['validation_seed'] if split=='validation' else cfg['test_seed']
            for distractor in cfg['eval_distractors']:
                batch=feature_batch(model,data_seed,cfg['eval_size'],distractor,delays,cfg['eval_batch_size'],device)
                if split in event_reference:
                    assert event_reference[split]==batch['event_row_hashes']
                else:
                    event_reference[split]=batch['event_row_hashes']
                for delay in delays:
                    row=prediction_record(seed,arm,split,delay,distractor,batch,batch['original_predictions'][delay]['value'])
                    row['run_id']=30+pair;rows.append(row)
                    if arm=='frozen_original':
                        with torch.no_grad(): scalar=predict_ridge(batch['features'][delay].to(device),ridge).argmax(-1).cpu()
                        record=prediction_record(seed,'frozen_shared_readout',split,delay,distractor,batch,scalar)
                        record['run_id']=30+pair;rows.append(record)
        # Reference has no new optimizer exposure and is explicitly contextual.
        torch.manual_seed(seed)
        model=ReturnMemoryModel(width=1024,encoding='factorized').to(device).eval();model.load_state_dict(initial)
        for split in ('validation','test'): evaluate(model,'frozen_original',split,cfg['eval_delays'])
        del model
        paired_batches=None
        for arm,schedule in cfg['schedules'].items():
            tick=time.monotonic()
            torch.manual_seed(seed)
            model=ReturnMemoryModel(width=1024,encoding='factorized').to(device);model.load_state_dict(initial)
            assert tensor_hash(model.state_dict())==initial_hash
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
            batches=[]; losses=[]; training_hashes=[]
            for step in range(cfg['updates']+1):
                if step%cfg['curve_every']==0 or step==cfg['updates']:
                    model.eval()
                    batch=feature_batch(model,cfg['validation_seed'],cfg['eval_size'],2,cfg['curve_delays'],cfg['eval_batch_size'],device)
                    for delay in cfg['curve_delays']:
                        record=prediction_record(seed,arm,'validation_curve',delay,2,batch,batch['original_predictions'][delay]['value'])
                        record['update']=step;curves.append(record)
                if step==cfg['updates']:break
                ds=cfg['training_seed_base']+pair*10000+step
                assert ds not in (cfg['validation_seed'],cfg['test_seed'])
                cpu=make_batch(ds,cfg['batch_size'])
                row_hashes=[tensor_hash({key:value[i] for key,value in cpu['public']['event'].items()}) for i in range(cfg['batch_size'])]
                training_hashes.extend(row_hashes)
                batches.append(dict(seed=ds,event_sha256=tensor_hash(cpu['public']['event']),event_row_hashes=row_hashes))
                terms=update(model,optimizer,move(cpu,device),schedule[step%len(schedule)])
                losses.append(dict(update=step+1,delay=schedule[step%len(schedule)],loss=terms))
                if (step+1)%250==0: print(seed,arm,step+1,terms['value'],flush=True)
            if paired_batches is None: paired_batches=batches
            else: assert paired_batches==batches
            checkpoint=out/f'{seed}-{arm}.pt';torch.save(model.state_dict(),checkpoint)
            model.eval()
            for split in ('validation','test'):evaluate(model,arm,split,cfg['eval_delays'])
            assert not(set(training_hashes) & set(event_reference['validation']))
            assert not(set(training_hashes) & set(event_reference['test']))
            counts=schedule_counts(schedule,cfg['updates'])
            arm_meta.append(dict(arm=arm,initial_state_sha256=initial_hash,checkpoint_sha256=sha(checkpoint),
                optimizer_reset=True,torch_seed=seed,parameters=sum(p.numel() for p in model.parameters()),width=1024,memory_tokens=6,
                allocated_memory_coordinates=6144,training_batches=batches,losses=losses,delay_updates=counts,
                presentations=cfg['updates']*cfg['batch_size'],unique_training_events=len(set(training_hashes)),
                recurrent_example_microsteps=sum(int(d)*n*cfg['batch_size'] for d,n in counts.items()),
                seconds=time.monotonic()-tick))
            del optimizer,model;torch.cuda.empty_cache()
        assert not(set(event_reference['validation']) & set(event_reference['test']))
        predictions=out/f'{seed}-predictions.json.gz'
        predictions.write_bytes(gzip.compress(json.dumps(rows,separators=(',',':')).encode(),mtime=0))
        curvepath=out/f'{seed}-curves.json.gz'
        curvepath.write_bytes(gzip.compress(json.dumps(curves,separators=(',',':')).encode(),mtime=0))
        manifest['runs'].append(dict(backbone_seed=seed,run_id=30+pair,initial_checkpoint_sha256=expected,
             initial_state_sha256=initial_hash,frozen_ridge_sha256=ridge_expected,event_row_hashes=event_reference,
             arms=arm_meta,predictions_sha256=sha(predictions),curves_sha256=sha(curvepath),
             seconds=time.monotonic()-started,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
             process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        write_json(out/'manifest.json',manifest)
        print(seed,'complete',manifest['runs'][-1]['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
