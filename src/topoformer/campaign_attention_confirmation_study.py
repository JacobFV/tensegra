"""A14 fixed seeds, budgets and policies; no outcome-driven stopping."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from .campaign_attention_records import RecordAttention,tokenize
from .campaign_attention_selector import SelectorModel,generate,targets,oracle_successors,metrics
from .campaign_attention_selector_study import evaluate as development_evaluate
from .campaign_attention_confirmation_read import forward,POLICIES
from .campaign_attention_study import write,sync


def tensor_hash(model):
    return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.state_dict().values())).hexdigest()


def evaluate(model,cfg,out,device,reference_mode=None):
    out.mkdir(parents=True,exist_ok=False);model.eval();initial=tensor_hash(model)
    policies=POLICIES if reference_mode is None else ('a06_successful',)
    public={};input_hashes={};order_hashes={};raw={p:{} for p in policies};rows={p:[] for p in policies}
    for ci,c in enumerate(cfg['conditions']):
        inputs={name:hashlib.sha256() for name in ['keys','attributes','instructions','adjacency']};order_digest=hashlib.sha256()
        collected={p:{} for p in policies};shared={};elapsed={p:0. for p in policies}
        for offset in range(0,cfg['examples'],cfg['batch']):
            count=min(cfg['batch'],cfg['examples']-offset)
            batch=generate(count,c['nodes'],c['depth'],groups=c['groups'],seed=cfg['data_seed']+ci*100000+offset,device=device,balanced=True)
            for name,digest in inputs.items():digest.update(getattr(batch,name).cpu().numpy().tobytes())
            gold=targets(batch);successor=oracle_successors(batch)
            if reference_mode is None:
                g=torch.Generator(device=device).manual_seed(cfg['order_seed']+ci*100000+offset)
                order=torch.rand(count,3*c['nodes']*c['groups'],generator=g,device=device).argsort(-1)
                order_digest.update(order.cpu().numpy().tobytes())
                records=tokenize(batch,order)
                indices=batch.adjacency.bool().nonzero().reshape(count,-1,4)[:,:,1:].gather(1,order[...,None].expand(-1,-1,3))
            for name,value in dict(gold=gold,successor=successor,start=batch.starts,relation=batch.relations,values=batch.values).items():
                shared.setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
            for policy in policies:
                sync(device);begin=time.monotonic()
                with torch.no_grad():
                    if reference_mode is None:result=forward(model,batch,records,indices,successor,policy)
                    else:result=model(batch,reference_mode,selector_scale_override=16.,context_scale_override=16. if reference_mode=='context' else None)
                sync(device);elapsed[policy]+=time.monotonic()-begin
                scores=metrics(result,gold,batch)
                for name,value in dict(pred=result['logits'].argmax(-1),route=result['routes']).items():collected[policy].setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
                for name,value in scores.items():collected[policy].setdefault(name,[]).append(value.cpu().numpy())
                for name,value in result.get('diagnostics',{}).items():collected[policy].setdefault('diagnostic_'+name,[]).append(value.cpu().numpy())
        for name,pieces in shared.items():public[f'c{ci}_{name}']=np.concatenate(pieces)
        input_hashes.update({f'c{ci}_input_{name}':digest.hexdigest() for name,digest in inputs.items()})
        if reference_mode is None:order_hashes[f'c{ci}']=order_digest.hexdigest()
        for policy in policies:
            joined={name:np.concatenate(pieces) for name,pieces in collected[policy].items()}
            raw[policy].update({f'c{ci}_{name}':value for name,value in joined.items()})
            rows[policy].append(dict(condition=c,examples=cfg['examples'],forward_seconds=elapsed[policy],timing_scope='instrumented' if reference_mode is None else 'model_forward',
                **{name:float(joined[name].mean()) for name in scores},diagnostics={name:float(value.mean()) for name,value in joined.items() if name.startswith('diagnostic_')}))
    # All seeds/references must have identical decompressed public array hashes.
    public_hash={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in public.items()}
    public_hash.update(input_hashes)
    if cfg.get('export_public',True):np.savez_compressed(out/'public.npz',**public)
    write(out/'public-sha256.json',public_hash)
    for policy in policies:
        np.savez_compressed(out/f'{policy}.npz',**raw[policy]);write(out/f'{policy}.json',dict(rows=rows[policy]))
    final=tensor_hash(model)
    if final!=initial:raise RuntimeError('Confirmation inference mutated model')
    return dict(initial_tensor_sha256=initial,final_tensor_sha256=final,policies=list(policies),public_sha256=public_hash,record_order_sha256=order_hashes,reference_mode=reference_mode)


def benchmark(model,cfg,device):
    """Separate uninstrumented profile workload; reports warmup cost explicitly."""
    c=cfg['conditions'][-1];count=cfg['batch']
    begin=time.monotonic()
    batch=generate(count,c['nodes'],c['depth'],groups=c['groups'],seed=cfg['benchmark_seed'],device=device,balanced=True)
    records=tokenize(batch);indices=batch.adjacency.bool().nonzero().reshape(count,-1,4)[:,:,1:];successor=oracle_successors(batch);sync(device)
    preparation=time.monotonic()-begin;result={}
    for policy in POLICIES:
        sync(device);begin=time.monotonic();forward(model,batch,records,indices,successor,policy,instrument=False);sync(device)
        warmup=time.monotonic()-begin;times=[]
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            sync(device);begin=time.monotonic();forward(model,batch,records,indices,successor,policy,instrument=False);sync(device);times.append(time.monotonic()-begin)
        result[policy]=dict(warmup_seconds=warmup,repeated_forward_seconds=times,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
    return dict(condition=c,batch=count,preparation_seconds=preparation,policies=result,
        scope='Uninstrumented model forward with pretokenized input; setup, warmup and repeats charged separately; no sparse-kernel or cross-architecture efficiency claim')


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    device=cfg.get('device','cuda');torch.set_num_threads(2);start=time.monotonic()
    files=['campaign_attention_confirmation_read.py','campaign_attention_confirmation_study.py','campaign_attention_records.py','campaign_attention_selector.py','campaign_attention_selector_study.py','campaign_attention.py','campaign_attention_study.py']
    write(out/'source.json',{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files})
    if cfg['kind']=='references':
        manifests=[]
        for i,ref in enumerate(cfg['checkpoints']):
            path=Path(ref['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:raise ValueError('Reference checkpoint SHA mismatch')
            model=SelectorModel(width=1024).to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
            ecfg=dict(cfg,export_public=i==0)
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            manifests.append(dict(**ref,**evaluate(model,ecfg,out/f"{ref['seed']}-{ref['mode']}",device,ref['mode']),
                parameters_allocated=sum(p.numel() for p in model.parameters()),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0))
            del model
        write(out/'manifest.json',dict(references=manifests,optimizer_updates=0,wall_seconds=time.monotonic()-start,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    else:
        torch.manual_seed(cfg['seed']);model=RecordAttention(width=1024).to(device);initial=tensor_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4)
        losses=[];curves=[];monitor_seconds=0.;begin=time.monotonic()
        monitor=out/'development';monitor.mkdir()
        for step in range(cfg['steps']):
            depth=1+step%4
            batch=generate(16,32,depth,groups=4,seed=171000000+step,device=device,train=True,balanced=True)
            logits=model(batch)['logits'];gold=targets(batch)
            loss=F.cross_entropy(logits.flatten(0,2),gold.flatten())
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(loss.detach())
            if step+1 in cfg['monitor_steps']:
                sync(device);ms=time.monotonic()
                # Development observations cannot alter training's RNG stream.
                devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(cfg['monitor_order_seed']+step+1)
                    ecfg=dict(mode='records',generator='block_permutation_v2',eval_seed=173000000,eval_examples=cfg['monitor_examples'],eval_batch=32,
                        conditions=[dict(nodes=32,depth=4,groups=4,data_group=0),dict(nodes=64,depth=8,groups=4,data_group=1)])
                    curves.append(development_evaluate(model,ecfg,monitor,step+1,device))
                sync(device);monitor_seconds+=time.monotonic()-ms
        sync(device);training_and_monitor_seconds=time.monotonic()-begin
        torch.save(model.state_dict(),out/'checkpoint.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=cfg['steps'],cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],config=cfg),out/'training-state.pt')
        train_peak=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0
        gradient_count=sum(p.numel() for p in model.parameters() if p.grad is not None)
        nonzero_gradient_count=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().any().item())
        loss_values=torch.stack(losses).cpu().tolist()
        del optimizer,losses,logits,gold,batch,loss
        model.zero_grad(set_to_none=True);model.eval()
        if cfg.get('benchmark'):write(out/'benchmark.json',benchmark(model,cfg,device))
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        ev=evaluate(model,cfg,out/'confirmation',device)
        write(out/'manifest.json',dict(seed=cfg['seed'],steps=cfg['steps'],presentations=cfg['steps']*16,
            initial_tensor_sha256=initial,final_tensor_sha256=tensor_hash(model),checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
            training_state_sha256=hashlib.sha256((out/'training-state.pt').read_bytes()).hexdigest(),
            parameters_allocated=sum(p.numel() for p in model.parameters()),parameters_with_gradient=gradient_count,
            parameters_nonzero_gradient_last_update=nonzero_gradient_count,
            training_and_monitor_seconds=training_and_monitor_seconds,monitor_seconds=monitor_seconds,training_excluding_monitor_seconds=training_and_monitor_seconds-monitor_seconds,
            curves=curves,losses=loss_values,confirmation=ev,training_cuda_peak_allocated_bytes=train_peak,
            inference_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.monotonic()-start))
    print(json.dumps(dict(completed=str(out),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
