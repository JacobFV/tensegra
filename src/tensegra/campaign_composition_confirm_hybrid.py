"""C04 frozen actual-lowering workspace/copy paths and causal controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .campaign_composition import make_lowering_batch, model_inputs
from .campaign_composition_acquire import controlled_rows, tensor_digest
from .campaign_composition_study import get_data, counts, paired_metrics, stratified, consume_labels
from .campaign_composition_runtime import frozen_interfaces, interfaces_state_hashes, propose, build_returns, consume, exact_copy, manipulate
from .campaign_composition_confirm_data import requested_labels, full_proposal, causal_gate


def evaluate(interfaces, rows, labels, delays, config, output, name, device, reference=None):
    predicted=propose(interfaces['lowerer'],rows,device)
    actual=build_returns(rows,predicted['primitive'],predicted['canonical_pointers'])
    full=full_proposal(predicted,labels)
    result=consume(interfaces,actual,delays,device,config['capture_batch_size'])
    copied=exact_copy(interfaces,actual,device)
    cells=[]
    for delay,cell in result['cells'].items():
        cells.append(dict(path='workspace',delay=delay,**paired_metrics(cell['predictions'],labels['task'],result['supplied']),
                          joint=counts(full & (cell['predictions']==labels['task']),torch.ones_like(full)),
                          groups=stratified(cell['predictions'],labels,model_inputs(rows)['query'])))
    cells.append(dict(path='supplied_copy',delay=None,**paired_metrics(copied,labels['task'],result['supplied']),
                      joint=counts(full & (copied==labels['task']),torch.ones_like(full)),
                      groups=stratified(copied,labels,model_inputs(rows)['query'])))
    torch.save(dict(result=result,original=labels['task'],labels=labels,proposal=predicted,full_proposal=full,reasons=actual['reasons'],supplied_public=actual['public'],exact_copy=copied),output/f'{name}.pt')
    if reference is not None:
        _, oracle=requested_labels(rows)
        for kind,bundle in (('oracle_lowering',oracle),('oracle_return',reference)):
            outcome=consume(interfaces,bundle,delays,device,config['capture_batch_size'])
            torch.save(dict(result=outcome,original=labels['task'],reasons=bundle['reasons'],supplied_public=bundle['public']),output/f'{name}-{kind}.pt')
            for delay,cell in outcome['cells'].items():
                cells.append(dict(path=kind,delay=delay,**paired_metrics(cell['predictions'],labels['task'],outcome['supplied'])))
    return cells,predicted


def causal(interfaces,rows,labels,predicted,config,output,name,device,swap_seed):
    n=min(config['causal_size'],len(rows)); rows=rows[:n]; original=labels['task'][:n]
    actual=build_returns(rows,predicted['primitive'][:n],predicted['canonical_pointers'][:n])
    swap=make_lowering_batch(swap_seed,n,8)['reference']['public']
    query=torch.stack([row['query'] for row in rows]).to(device)
    with torch.no_grad():
        query_input=torch.cat((torch.zeros(n,33,device=device),query/query.new_tensor([8.,1.])), -1)
        query_pred=interfaces['query_only'](query_input).argmax(-1).cpu()
    outcomes={}; copies={}; cells=[]
    for kind in ('correct','drop','wrong','swap'):
        changed=manipulate(actual,kind,swap)
        result=consume(interfaces,changed,config['intervention_delays'],device,config['capture_batch_size'])
        copied=exact_copy(interfaces,changed,device); outcomes[kind]=result; copies[kind]=copied
        torch.save(dict(result=result,original=original,exact_copy=copied,query_only=query_pred,reasons=changed['reasons'],supplied_public=None if kind=='drop' else changed['public']),output/f'{name}-{kind}.pt')
        for delay,cell in result['cells'].items():
            cells.append(dict(path='workspace',kind=kind,delay=delay,**paired_metrics(cell['predictions'],original,result['supplied'])))
        cells.append(dict(path='supplied_copy',kind=kind,delay=None,**paired_metrics(copied,original,result['supplied'])))
    checks=[]
    for delay in config['intervention_delays']:
        controls={k:(outcomes[k]['cells'][delay]['predictions'],outcomes[k]['supplied']) for k in ('wrong','swap')}
        checks.append(dict(path='workspace',delay=delay,**causal_gate(outcomes['correct']['cells'][delay]['predictions'],outcomes['drop']['cells'][delay]['predictions'],query_pred,copies['correct'],original,controls)))
    checks.append(dict(path='supplied_copy',delay=None,**causal_gate(copies['correct'],copies['drop'],query_pred,copies['correct'],original,{k:(copies[k],outcomes[k]['supplied']) for k in ('wrong','swap')})))
    return dict(cells=cells,checks=checks)


def run(config,output,device):
    if config['hidden']!=1024 or config['key_dim']!=32: raise ValueError('width1024/key32 required')
    torch.set_num_threads(4); output=Path(output);output.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    interfaces=frozen_interfaces(config['interfaces'],device);before=interfaces_state_hashes(interfaces)
    cells=[];controls=[];causal_results=[];manifests=[]
    for distractors in config['eval_distractors']:
        data,public,clean_labels=get_data(config['data']['validation'],distractors)
        for view in ('clean','reversed'):
            rows=data['public'] if view=='clean' else controlled_rows(data['public'],'reverse_roles',0)
            labels,requested=requested_labels(rows)
            if view=='clean':
                if any(not torch.equal(labels[k],clean_labels[k]) for k in labels): raise RuntimeError('clean actual requested labels differ from generator')
                reference=dict(total=len(rows),indices=torch.arange(len(rows)),reasons=['']*len(rows),public=data['reference']['public'])
            else: reference=requested
            name=f'{view}-{distractors}'
            found,predicted=evaluate(interfaces,rows,labels,config['delays'],config,output,name,device,reference)
            cells.extend(dict(view=view,distractors=distractors,**x) for x in found)
            manifests.append(dict(view=view,distractors=distractors,public_sha256=tensor_digest(model_inputs(rows)),labels_sha256=tensor_digest(labels)))
            if distractors!=8: continue
            causal_results.append(dict(view=view,**causal(interfaces,rows,labels,predicted,config,output,f'{view}-causal',device,config['swap_seed']+(view=='reversed'))))
            for i,kind in enumerate(config['public_controls']):
                altered=controlled_rows(rows,kind,config['control_seed']+i)
                wanted,_=requested_labels(altered)
                found,_=evaluate(interfaces,altered,wanted,config['intervention_delays'],config,output,f'{view}-public-{kind}',device)
                controls.extend(dict(view=view,control=kind,**x) for x in found)
    after=interfaces_state_hashes(interfaces)
    if before!=after: raise RuntimeError('frozen interfaces changed')
    if device.startswith('cuda'): torch.cuda.synchronize()
    gates={path:all(x['joint' if path=='workspace' else 'original']['correct']/x['joint' if path=='workspace' else 'original']['total']>=.98 for x in cells if x['path']==path) for path in ('workspace','supplied_copy')}
    source_files=('campaign_composition_confirm_hybrid.py','campaign_composition_confirm_data.py','campaign_composition.py','campaign_composition_acquire.py','campaign_composition_runtime.py','campaign_composition_study.py','campaign_returns_use.py','campaign_returns.py','campaign_returns_balanced.py','interface_proposals.py','thinking_runtime.py','retention_data.py','return_memory.py','return_crossdelay.py','return_memory_study.py','thinking.py','retention.py','return_diagnostics_probe.py')
    summary=dict(phase='hybrid',config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),data=manifests,cells=cells,controls=controls,causal=causal_results,accuracy_gates=gates,
                 frozen_state=dict(before=before,after=after,unchanged=True),parameters={k:sum(p.numel() for p in v.parameters()) for k,v in interfaces.items() if isinstance(v,torch.nn.Module)},
                 source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in source_files},elapsed_seconds=time.monotonic()-tick,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,claim='one of three fresh finite-domain replicates; no pooled pass or efficiency claim')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',required=True)
    args=parser.parse_args();run(json.loads(Path(args.config).read_text()),args.output,args.device)
