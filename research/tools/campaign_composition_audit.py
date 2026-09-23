"""Independent C01 archived counts and neural selection; no producer metrics."""
import argparse,json,time,hashlib,torch
from pathlib import Path

def count(p,y,mask=None):
 mask=torch.ones_like(y,dtype=torch.bool)if mask is None else mask
 return dict(correct=int(((p==y)&mask).sum()),total=int(mask.sum()))
def paired(p,y,s):
 valid=s>=0;c=valid&(s!=y)
 return dict(original=count(p,y),supplied=count(p,s,valid),changed_original=count(p,y,c),changed_supplied=count(p,s,c),refused=int((p<0).sum()),supplied_absent=int((~valid).sum()))
def run(path):
 start=time.monotonic();torch.set_num_threads(2);root=Path(path);m=json.loads((root/'summary.json').read_text());load=lambda n:torch.load(root/n,map_location='cpu',weights_only=False);checked=0
 if m['phase']=='hybrid':
  assert m['frozen_state']['before']==m['frozen_state']['after'] and m['frozen_state']['unchanged']
  for cell in m['cells']:
   arm=cell['arm'];d=cell['delay'];dist=cell['distractors']
   if arm=='exact_copy':
    raw=load(f'exact-copy-{dist}.pt');p,y,s=raw['predictions'],raw['original'],raw['supplied']
   else:
    name=f'{arm}-{dist}.pt'if arm in ('hybrid','oracle_lowering','oracle_return')else f'public-{arm[7:]}.pt'if arm.startswith('public_')else f'intervention-{arm}.pt'
    raw=load(name);p=raw['result']['cells'][d]['predictions'];y=raw['original'];s=raw['result']['supplied']
   for k,v in paired(p,y,s).items():assert cell[k]==v,(arm,d,k)
   if arm=='hybrid':assert cell['lowering_and_answer']==dict(correct=int((raw['full_proposal']&(p==y)).sum()),total=len(y))
   if arm.startswith('public_'):
    assert cell['requested']==count(p,raw['requested']);assert cell['requested_changed']==count(p,raw['requested'],raw['requested']!=y)
   checked+=1
  causal=[]
  for rec in m['causal_checks']:
   d=rec['delay'];clean=load('intervention-correct.pt');y=clean['original'];cleanacc=float((clean['result']['cells'][d]['predictions']==y).float().mean());drop=load('intervention-drop.pt');dropacc=float((drop['result']['cells'][d]['predictions']==y).float().mean());q=float((clean['query_only']==y).float().mean());oracle=float((clean['exact_copy']==y).float().mean());gain=(cleanacc-q)/(oracle-q)if oracle>q else None
   assert abs(rec['clean_minus_drop']-(cleanacc-dropacc))<1e-6;assert rec['normalized_gain']==gain
   valid=[]
   for kind in ('wrong','swap'):
    raw=load(f'intervention-{kind}.pt');s=raw['result']['supplied'];c=count(raw['result']['cells'][d]['predictions'],s,(s>=0)&(s!=y));assert c==rec['changed_supplied'][kind];valid.append(c['total']>=256 and c['correct']/max(1,c['total'])>=.9)
   passed=cleanacc-dropacc>=.15 and gain is not None and gain>=.8 and all(valid);assert passed==rec['passed'];causal.append(rec)
  return dict(phase='hybrid',raw_cells=checked,causal=causal,clean_joint=[c['lowering_and_answer']for c in m['cells']if c['arm']=='hybrid'],frozen_state_verified=True,cpu_audit_wall_seconds=time.monotonic()-start)
 # Baseline LR/step selection evaluated solely against calibration labels.
 best=None
 for rec in m['curves']:
  raw=load(f"calibration-{rec['lr']}-{rec['step']}.pt");v=count(raw['logits']['answer'].argmax(-1),raw['labels']['task']);assert v==rec['calibration']['answer'];candidate=(v['correct'],-rec['step'],-m['config']['learning_rates'].index(rec['lr']))
  if best is None or candidate>best[0]:best=(candidate,rec['lr'],rec['step'])
 assert (best[1],best[2])==(m['selected_lr'],m['selected_step']);assert hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest()==m['selected_checkpoint_sha256']
 raw=load('validation.pt');assert count(raw['logits']['answer'].argmax(-1),raw['labels']['task'])==m['validation']['answer'];assert count(raw['logits']['value'].argmax(-1),raw['labels']['value'])==m['validation']['value'];checked=1
 for name,rec in m['controls'].items():
  raw=load(f'control-{name}.pt');p=raw['logits']['answer'].argmax(-1);assert count(p,raw['labels']['task'])==rec['supplied']['answer'];assert paired(p,raw['original_labels']['task'],raw['labels']['task'])==rec['paired'];checked+=1
 return dict(phase=m['phase'],raw_cells=checked,selected_lr=best[1],selected_step=best[2],validation=m['validation'],cpu_audit_wall_seconds=time.monotonic()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();r=run(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items()if k not in ('causal','validation')})
