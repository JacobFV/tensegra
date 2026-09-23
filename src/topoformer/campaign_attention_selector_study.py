"""Versioned A04 development runner; no route labels in training."""
import argparse
import hashlib
import json
import resource
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .campaign_attention_selector import SelectorModel,generate,targets,oracle_successors,metrics,swap_instruction,permute_nodes,corrupt
from .campaign_attention import restore_node_order
from .campaign_attention_study import write,sync

@torch.no_grad()
def evaluate(model,cfg,out,step,device):
    model.eval();rows=[];raw={}
    for ci,c in enumerate(cfg['conditions']):
        collected={};elapsed=0.;timers=[]
        for offset in range(0,cfg['eval_examples'],cfg['eval_batch']):
            count=min(cfg['eval_batch'],cfg['eval_examples']-offset)
            batch=generate(count,c['nodes'],c['depth'],groups=c.get('groups',4),seed=cfg['eval_seed']+c.get('data_group',ci)*100000+offset,device=device,heldout_composition=c.get('composition',False),balanced=cfg.get('generator','independent_v1')=='block_permutation_v2')
            original=batch
            if c.get('instruction_swap'):batch=swap_instruction(batch)
            gold=targets(batch);given=corrupt(batch,c.get('corruption','clean'),cfg['eval_seed']+ci*100000+offset+50000);order=None
            if c.get('node_permutation'):
                g=torch.Generator(device=device).manual_seed(cfg['eval_seed']+offset+777)
                order=torch.rand(count,c['nodes'],generator=g,device=device).argsort(-1);given=permute_nodes(given,order)
            if c.get('wrong_instruction'):given=swap_instruction(given)
            if str(device).startswith('cuda'):
                t0=torch.cuda.Event(enable_timing=True);t1=torch.cuda.Event(enable_timing=True);t0.record()
            else:t0=time.monotonic()
            result=model(given,cfg['mode'],zero_strength=c.get('zero_strength',False),zero_content=c.get('zero_content',False),selector_scale_override=c.get('selector_scale_override'))
            if str(device).startswith('cuda'):t1.record();timers.append((t0,t1))
            else:elapsed+=time.monotonic()-t0
            if order is not None:result=restore_node_order(result,order)
            scores=metrics(result,gold,batch)
            supplied_gold=targets(given)
            if order is not None:
                supplied_gold=supplied_gold.gather(2,order.argsort(-1)[:,None,:].expand(-1,c['depth'],-1))
            scores['agreement_supplied_task']=(result['logits'].argmax(-1)[:,-1].gather(1,batch.starts[:,None]).squeeze(1)==supplied_gold[:,-1].gather(1,batch.starts[:,None]).squeeze(1))
            successor=oracle_successors(batch)
            def terminal(source,successors):
                pointer=source.starts.clone();bi=torch.arange(len(pointer),device=pointer.device)
                for t in range(successors.shape[1]):pointer=successors[bi,t,pointer]
                return pointer
            current_terminal=terminal(batch,successor)
            original_terminal=terminal(original,oracle_successors(original)) if c.get('instruction_swap') else current_terminal
            bi=torch.arange(count,device=device)
            scores['changed_terminal']=current_terminal!=original_terminal
            scores['changed_answer']=batch.values[bi,current_terminal]!=original.values[bi,original_terminal]
            metric_names=set(scores)
            for name,value in scores.items():collected.setdefault(name,[]).append(value.cpu().numpy())
            for name,value in dict(supplied_final_target=supplied_gold[:,-1].gather(1,batch.starts[:,None]).squeeze(1),oracle_terminal=current_terminal,original_terminal=original_terminal).items():
                collected.setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
            for name,value in dict(pred=result['logits'].argmax(-1),gold=gold,route=result['routes'],start=batch.starts,relation=batch.relations,successor=successor).items():
                collected.setdefault(name,[]).append(value.cpu().numpy().astype('uint8'))
        sync(device)
        if timers:elapsed=sum(t0.elapsed_time(t1)/1000 for t0,t1 in timers)
        collected={k:np.concatenate(v) for k,v in collected.items()}
        row=dict(condition=c,examples=cfg['eval_examples'],forward_seconds=elapsed)
        for k,v in collected.items():
            raw[f'c{ci}_{k}']=v
            if k in metric_names:row[k]=float(v.mean())
        for kind in ['terminal','answer']:
            mask=collected['changed_'+kind].astype(bool)
            row['changed_'+kind+'_count']=int(mask.sum())
            row['task_given_changed_'+kind]=float(collected['task'][mask].mean()) if mask.any() else None
        row['forward_seconds_per_correct']=elapsed/int(collected['task'].sum()) if collected['task'].sum() else None
        rows.append(row)
    np.savez_compressed(out/f'eval-{step:05d}.npz',**raw);write(out/f'eval-{step:05d}.json',dict(rows=rows,eval_seed=cfg['eval_seed']))
    model.train();return dict(step=step,rows=rows)


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg)
    sources=[Path(__file__),Path(__file__).with_name('campaign_attention_selector.py'),Path(__file__).with_name('campaign_attention.py'),Path(__file__).with_name('campaign_attention_study.py')]
    write(out/'source.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    device=cfg.get('device','cuda');torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    model=SelectorModel(width=cfg.get('width',1024),strength=cfg.get('strength',8)).to(device)
    initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',.0003),weight_decay=1e-4)
    start=time.monotonic();curves=[];losses=[];draws=0;node_steps=0
    for step in range(cfg['steps']+1):
        if step in cfg['checkpoints']:
            evaluation=dict(cfg)
            if step!=cfg['steps'] and 'curve_conditions' in cfg:
                evaluation.update(conditions=cfg['curve_conditions'],eval_seed=cfg['curve_eval_seed'],eval_examples=cfg.get('curve_examples',256))
            curves.append(evaluate(model,evaluation,out,step,device))
        if step==cfg['steps']:break
        depth=1+step%4
        batch=generate(cfg['batch'],cfg['nodes'],depth,groups=cfg.get('groups',4),seed=cfg['train_seed']+step,device=device,train=True,balanced=cfg.get('generator','independent_v1')=='block_permutation_v2')
        result=model(batch,cfg['mode']);gold=targets(batch)
        loss=F.cross_entropy(result['logits'].flatten(0,2),gold.flatten())
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        losses.append(loss.detach());draws+=cfg['batch'];node_steps+=cfg['batch']*cfg['nodes']*depth
    torch.save(model.state_dict(),out/'checkpoint.pt');sync(device)
    manifest=dict(seed=cfg['seed'],mode=cfg['mode'],width=cfg.get('width',1024),parameters_allocated=sum(p.numel() for p in model.parameters()),
        initial_tensor_sha256=initial,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),presentations=draws,generated_graph_draws=draws,unique_canonical_graphs=None,node_microsteps=node_steps,
        curves=curves,losses=torch.stack(losses).cpu().tolist(),wall_seconds_including_eval_export=time.monotonic()-start,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        learned_strength=model.strength.detach().cpu().tolist(),selector_scale=float(model.selector_log_scale.exp().detach()),
        parameters_with_gradient=sum(p.numel() for p in model.parameters() if p.grad is not None),
        parameters_nonzero_gradient_last_update=sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().any().item()),
        gradient_count_note='Autograd participation and last-update nonzero tensors, not functional capacity or acquired learning')
    write(out/'manifest.json',manifest);print(json.dumps(dict(completed=str(out),seconds=manifest['wall_seconds_including_eval_export'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
