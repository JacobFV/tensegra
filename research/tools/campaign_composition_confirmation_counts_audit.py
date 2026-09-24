"""Independent CPU C04 raw primary counts; no neural inference."""
import json,hashlib,time,argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--prefix',default='c04-profile');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();torch.set_num_threads(2);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();summaries={};files=0;cells=0
inventory=json.loads(a.manifest.read_text());inventory=inventory.get('phase_files',inventory)
for phase,entries in inventory.items():
 r=a.root/f'{a.prefix}-{phase}';summaries[phase]=json.loads((r/'summary.json').read_text())
 for item in entries:
  f=r/item['name'];assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];files+=1

def count(p,y,mask=None):
 mask=torch.ones_like(y,dtype=torch.bool) if mask is None else mask
 return dict(correct=int(((p==y)&mask).sum()),total=int(mask.sum()))
def paired(p,y,s):
 valid=s>=0;changed=valid&(s!=y)
 return dict(original=count(p,y),supplied=count(p,s,valid),changed_original=count(p,y,changed),changed_supplied=count(p,s,changed),refused=int((p<0).sum()),supplied_absent=int((~valid).sum()))
r=a.root/(a.prefix+'-hybrid');m=summaries['hybrid'];assert m['frozen_state']['before']==m['frozen_state']['after'];load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
def check(c,name):
 global cells
 raw=load(r/(name+'.pt'));p=raw['exact_copy'] if c['path']=='supplied_copy' else raw['result']['cells'][c['delay']]['predictions'];y=raw['original'];s=raw['result']['supplied']
 for k,v in paired(p,y,s).items():assert c[k]==v,(name,k)
 if 'joint'in c:assert c['joint']==count(raw['full_proposal']&(p==y),torch.ones_like(y,dtype=torch.bool))
 cells+=1
for c in m['cells']:check(c,f"{c['view']}-{c['distractors']}"+('-'+c['path'] if c['path'].startswith('oracle_') else ''))
for c in m['controls']:check(c,f"{c['view']}-public-{c['control']}")
for group in m['causal']:
 for c in group['cells']:check(c,f"{group['view']}-causal-{c['kind']}")
for arm in ('n1_static','n1_roles','n2_rekey'):
 r=a.root/f'{a.prefix}-{arm}';m=summaries[arm];assert m['primary_endpoint_step']==m['config']['updates']
 for policy,results in m['results'].items():
  for key in ('validation','validation_reversed'):
   x=load(r/f"{policy}-{key.replace('_','-')}.pt");rec=results[key];rec=rec.get('supplied',rec)
   for head,label in [('answer','task'),('value','value')]:assert count(x['logits'][head].argmax(-1),x['labels'][label])==rec[head]
   cells+=1
  for name,rec in results['controls'].items():
   x=load(r/f'{policy}-control-{name}.pt');pred=x['logits']['answer'].argmax(-1);assert paired(pred,x['original_labels']['task'],x['labels']['task'])==rec['paired'];cells+=1
 for c in m['curves']:
  x=load(r/f"calibration-{c['step']}.pt");assert count(x['logits']['answer'].argmax(-1),x['labels']['task'])==c['calibration']['answer']
 assert m['selected_step']==max(m['curves'],key=lambda c:(c['calibration']['answer']['correct'],-c['step']))['step']
for c in summaries['timing']['cells']:
 for rec in c['passes']:
  x=load(a.root/(a.prefix+'-timing')/f"{c['path']}-b{c['batch']}-d{c['delay']}-r{rec['repeat']}.pt");assert count(x['predictions'],x['targets'])==dict(correct=rec['correct'],total=rec['attempted']);assert int((x['predictions']<0).sum())==rec['refused'];cells+=1
assert summaries['timing']['frozen_interfaces_unchanged'];assert len({summaries[a]['sample_index_stream_sha256'] for a in ('n1_static','n1_roles','n2_rekey')})==1;assert summaries['n1_static']['initial_state_sha256']==summaries['n1_roles']['initial_state_sha256']
out=dict(raw_files_verified=files,primary_cells_reconstructed=cells,paired_sample_stream=True,n1_initial_states_equal=True,cpu_audit_wall_seconds=time.monotonic()-t,scope='Independent raw primary counts, calibration selection, all artifact hashes and timing outcome counts. Arithmetic/full causal/overlap reconstruction in separate receipt.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
