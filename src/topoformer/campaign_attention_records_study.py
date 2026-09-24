"""A07 versioned runner reusing audited evaluator, with explicit model factory."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .campaign_attention_records import RecordAttention
from .campaign_attention_selector import generate,targets
from .campaign_attention_selector_study import evaluate
from .campaign_attention_study import write,sync


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    files=['campaign_attention_records.py','campaign_attention_records_study.py',
           'campaign_attention_selector.py','campaign_attention_selector_study.py',
           'campaign_attention.py','campaign_attention_study.py']
    write(out/'source.json',{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files})
    device=cfg.get('device','cuda');torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    model=RecordAttention(width=cfg['width']).to(device)
    tensor_hash=lambda:hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    initial=tensor_hash();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-4)
    start_step=0
    if 'resume_state' in cfg:
        state_path=Path(cfg['resume_state'])
        if hashlib.sha256(state_path.read_bytes()).hexdigest()!=cfg['resume_state_sha256']:
            raise ValueError('Resume state SHA mismatch')
        state=torch.load(state_path,map_location=device,weights_only=False)
        for key in ['seed','width','mode','batch','nodes','groups','lr','train_seed','generator']:
            if state['config'][key]!=cfg[key]:raise ValueError(f'Resume contract mismatch: {key}')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        if tensor_hash()!=cfg['resume_tensor_sha256']:raise ValueError('Resume tensor SHA mismatch')
        start_step=state['step']
        if start_step!=cfg['resume_step']:raise ValueError('Resume step mismatch')
        torch.set_rng_state(state['cpu_rng'].cpu())
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        initial=tensor_hash()
    start=time.monotonic();curves=[];losses=[];node_steps=0;replay=None
    for step in range(start_step,cfg['steps']+1):
        if step==cfg.get('prefix_steps',-1):
            reference=Path(cfg['prefix_checkpoint'])
            if hashlib.sha256(reference.read_bytes()).hexdigest()!=cfg['prefix_checkpoint_sha256']:
                raise ValueError('Prefix checkpoint SHA mismatch')
            actual=tensor_hash()
            if actual!=cfg['prefix_tensor_sha256']:
                raise RuntimeError('Exact prefix replay failed before extension')
            devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
            with torch.random.fork_rng(devices=devices),torch.no_grad():
                reference_model=RecordAttention(width=cfg['width']).to(device)
                reference_model.load_state_dict(torch.load(reference,map_location=device,weights_only=True))
                fixture=generate(4,32,4,groups=4,seed=177000000,device=device,balanced=True)
                torch.manual_seed(901);left=model(fixture)['logits']
                torch.manual_seed(901);right=reference_model(fixture)['logits']
                if not torch.equal(left,right):raise RuntimeError('Prefix checkpoint logit replay failed')
                replay=dict(step=step,tensor_sha256=actual,logits_exact=True,max_logit_error=float((left-right).abs().max()))
                del reference_model
            write(out/'prefix-replay.json',replay)
        if step in cfg['checkpoints'] and not ('resume_state' in cfg and step==start_step):
            ecfg=dict(cfg)
            if step!=cfg['steps'] and 'curve_conditions' in cfg:
                ecfg.update(conditions=cfg['curve_conditions'],eval_seed=cfg['curve_eval_seed'],eval_examples=cfg['curve_examples'])
            curves.append(evaluate(model,ecfg,out,step,device))
            if cfg.get('save_training_state') and step in cfg.get('state_checkpoints',[cfg['steps']]):
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                                config=cfg),out/f'training-state-{step:05d}.pt')
        if step==cfg['steps']:break
        depth=1+step%4
        batch=generate(cfg['batch'],cfg['nodes'],depth,groups=cfg['groups'],seed=cfg['train_seed']+step,device=device,train=True,balanced=True)
        result=model(batch);gold=targets(batch)
        loss=F.cross_entropy(result['logits'].flatten(0,2),gold.flatten())
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(loss.detach());node_steps+=cfg['batch']*cfg['nodes']*depth
    torch.save(model.state_dict(),out/'checkpoint.pt');sync(device)
    write(out/'manifest.json',dict(seed=cfg['seed'],mode=cfg['mode'],width=cfg['width'],parameters_allocated=sum(p.numel() for p in model.parameters()),
         initial_tensor_sha256=initial,final_tensor_sha256=tensor_hash(),checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
         config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
         start_step=start_step,end_step=cfg['steps'],updates_this_run=cfg['steps']-start_step,
         presentations=(cfg['steps']-start_step)*cfg['batch'],generated_graph_draws=(cfg['steps']-start_step)*cfg['batch'],unique_canonical_graphs=None,
         node_microsteps=node_steps,edge_tokens_per_training_graph=3*cfg['nodes']*cfg['groups'],
         prefix_replay=replay,curves=curves,losses=torch.stack(losses).cpu().tolist() if losses else [],
         wall_seconds_including_eval_export=time.monotonic()-start,
         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
         process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         record_score_multiplier=float(model.record_log_scale.exp().detach()),address_scale=float(model.context_log_scale.exp().detach()),
         parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),
         parameters_nonzero_gradient_last_update=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().any().item()),
         parameter_note='Inherited unused comparator tensors remain allocated; gradient counts are participation, not effective capacity.'))
    print(json.dumps(dict(completed=str(out),seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
