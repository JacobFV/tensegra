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
    start=time.monotonic();curves=[];losses=[];node_steps=0
    for step in range(cfg['steps']+1):
        if step in cfg['checkpoints']:
            ecfg=dict(cfg)
            if step!=cfg['steps'] and 'curve_conditions' in cfg:
                ecfg.update(conditions=cfg['curve_conditions'],eval_seed=cfg['curve_eval_seed'],eval_examples=cfg['curve_examples'])
            curves.append(evaluate(model,ecfg,out,step,device))
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
         presentations=cfg['steps']*cfg['batch'],generated_graph_draws=cfg['steps']*cfg['batch'],unique_canonical_graphs=None,
         node_microsteps=node_steps,edge_tokens_per_training_graph=3*cfg['nodes']*cfg['groups'],
         curves=curves,losses=torch.stack(losses).cpu().tolist() if losses else [],
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
