"""C04 separately scheduled end-to-end inference timing on fixed public rows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from .campaign_composition import model_inputs
from .campaign_composition_study import get_data
from .campaign_composition_models import NeuralOperandBaseline,ContextualBaseline
from .campaign_composition_runtime import frozen_interfaces,interfaces_state_hashes,load_checked,propose,build_returns,consume,exact_copy


def infer(path,model,interfaces,rows,device,batch,delay):
    predictions=[]
    for start in range(0,len(rows),batch):
        chunk=rows[start:start+batch]
        if path in ('workspace','supplied_copy'):
            proposed=propose(interfaces['lowerer'],chunk,device,chunk=batch)
            returned=build_returns(chunk,proposed['primitive'],proposed['canonical_pointers'])
            pred=exact_copy(interfaces,returned,device) if path=='supplied_copy' else consume(interfaces,returned,[delay],device,chunk=batch)['cells'][delay]['predictions']
        else:
            public={k:v.to(device) for k,v in model_inputs(chunk).items()}
            with torch.no_grad(): pred=model(public)['answer'].argmax(-1).cpu()
        predictions.append(pred)
    return torch.cat(predictions)


def run(config,output,device):
    if config['hidden']!=1024 or config['key_dim']!=32: raise ValueError('width1024/key32 required')
    torch.set_num_threads(4);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    setup=time.monotonic();interfaces=frozen_interfaces(config['interfaces'],device)
    data,_,labels=get_data(config['data']['validation'],8)
    # Labels stay outside infer; original full generation then fixed-index prefix.
    rows=data['public'][:config['timing_examples']];truth=labels['task'][:len(rows)]
    inherited_before=interfaces_state_hashes(interfaces);setup_seconds=time.monotonic()-setup
    cells=[];loaded={};models={}
    for path in ('n1_static','n1_roles','n2_rekey'):
        tick=time.monotonic();model=(ContextualBaseline() if path=='n2_rekey' else NeuralOperandBaseline()).to(device)
        spec=config['neural_checkpoints'][path]
        model.load_state_dict(load_checked(spec['path'],spec['sha256'],device));model.eval().requires_grad_(False)
        loaded[path]=dict(load_seconds=time.monotonic()-tick,parameters=sum(p.numel() for p in model.parameters()));models[path]=model
    component_counts={k:sum(p.numel() for p in v.parameters()) for k,v in interfaces.items() if isinstance(v,torch.nn.Module)}
    for path in ('workspace','supplied_copy','n1_static','n1_roles','n2_rekey'):
        model=models.get(path)
        for batch in config['timing_batch_sizes']:
            for delay in (config['timing_delays'] if path=='workspace' else [None]):
                infer(path,model,interfaces,rows[:config['warmup_examples']],device,batch,delay)
                passes=[]
                for repeat in range(config['timing_repeats']):
                    if device.startswith('cuda'):
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    tick=time.monotonic();prediction=infer(path,model,interfaces,rows,device,batch,delay)
                    if device.startswith('cuda'):torch.cuda.synchronize()
                    elapsed=time.monotonic()-tick;correct=int((prediction==truth).sum())
                    record=dict(repeat=repeat,seconds=elapsed,attempted=len(rows),correct=correct,refused=int((prediction<0).sum()),seconds_per_attempt=elapsed/len(rows),seconds_per_correct=elapsed/correct if correct else None,examples_per_second=len(rows)/elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0)
                    passes.append(record)
                    torch.save(dict(predictions=prediction,targets=truth),output/f'{path}-b{batch}-d{delay}-r{repeat}.pt')
                params=(component_counts['lowerer']+component_counts['backbone']+component_counts['accessor']+component_counts['consumer']) if path=='workspace' else (component_counts['lowerer']+component_counts['oracle']) if path=='supplied_copy' else loaded[path]['parameters']
                cells.append(dict(path=path,batch=batch,delay=delay,executed_parameters=params,passes=passes,median_seconds=statistics.median(x['seconds'] for x in passes),min_seconds=min(x['seconds'] for x in passes),max_seconds=max(x['seconds'] for x in passes)))
    after=interfaces_state_hashes(interfaces)
    if after!=inherited_before:raise RuntimeError('frozen timing interfaces changed')
    result=dict(config=config,setup_seconds=setup_seconds,neural_loads=loaded,components=component_counts,resident_parameters=sum(component_counts.values())+sum(x['parameters'] for x in loaded.values()),cells=cells,frozen_interfaces_unchanged=True,
                scope='clean first fixed test-prefix; full independent paths include input preparation/transfers/CPU runtime; disk loading excluded; role accuracy reported separately',
                source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('campaign_composition_confirm_timing.py','campaign_composition.py','campaign_composition_runtime.py','campaign_composition_models.py','campaign_returns_use.py','return_memory.py','return_crossdelay.py','return_memory_study.py','thinking.py','thinking_runtime.py')})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.output,args.device)
