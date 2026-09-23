"""Independent P01 typed proposal, calibration and control count audit."""
import torch,json,hashlib,time,argparse
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(blob):
 out=blob['logits'];gold=blob['labels'];public=blob['public'];op=out['primitive'].argmax(-1);raw=out['pointers'].argmax(-1);ptr=raw.clone();ptr[op==3,2]=public['keys'].abs().sum(-1).argmin(-1)[op==3]
 y=gold['primitive'];target=gold['targets'];b=y!=3;non=(y==1)|(y==4);eq=raw==target;canon=ptr==target
 def count(mask,choose=None):
  choose=torch.ones_like(mask) if choose is None else choose;return {'correct':int((mask&choose).sum()),'total':int(choose.sum())}
 idx=torch.arange(len(op));types=public['operand_types'];valid=(types[idx,ptr[:,0]]==-1)&(public['keys'][idx,ptr[:,0]].abs().sum(-1)!=0)&(types[idx,ptr[:,1]]>=0)&((op==3)|(types[idx,ptr[:,2]]>=0))
 return dict(full_semantic=count((op==y)&canon.all(-1)),predicted_public_schema_valid=count(valid),primitive=count(op==y),destination=count(eq[:,0]),operand0=count(eq[:,1]),required_operand1=count(eq[:,2],b),raw_null_pointer_diagnostic=count(eq[:,2],~b),canonical_null_schema=count(canon[:,2],~b),noncommutative_order=count(eq[:,1:].all(-1),non))
def run(path):
 start=time.monotonic();torch.set_num_threads(2);p=Path(path);m=json.loads((p/'summary.json').read_text());load=lambda f:torch.load(p/f,map_location='cpu',weights_only=False)
 assert sha(p/'selected.pt')==m['selected_checkpoint_sha256'];assert sha(p/'training-visitation.pt')==m['training_visitation_sha256'];visits=load('training-visitation.pt');assert int(visits.sum())==m['optimizer_presentations'];assert int((visits>0).sum())==m['unique_training_events_visited']
 validation=load('validation.pt');assert score(validation)==m['validation'];count=1
 # Calibration has no public tensor, but correct unary null is the target public null.
 # Reconstruct canonical semantic counts only: this is a supplied arity convention.
 best=None
 for c in m['curve']:
  b=load(f"calibration-{c['step']}.pt");o=b['logits'];g=b['labels'];op=o['primitive'].argmax(-1);ptr=o['pointers'].argmax(-1).clone();ptr[op==3,2]=g['targets'][op==3,2];n=int(((op==g['primitive'])&(ptr==g['targets']).all(-1)).sum());assert n==c['calibration']['full_semantic']['correct']
  if best is None or n>best[0]:best=(n,c['step'])
 assert best[1]==m['selected_step']
 for name,summary in m['controls'].items():
  b=load('control-'+name+'.pt');assert score(b)==summary['supplied'];count+=1
  if b['original_labels_comparable']:assert score(dict(b,labels=b['original_labels']))==summary['original']
  if name=='fresh_names':
   assert all(torch.equal(b['public'][k],validation['public'][k]) for k in validation['public']);assert all(torch.equal(b['logits'][k],validation['logits'][k]) for k in validation['logits'])
  if name=='inventory_order':
   # Remap pointers to actual public keys to verify unchanged semantic identity.
   for i in range(len(b['labels']['primitive'])):
    assert torch.equal(b['public']['keys'][i,b['labels']['targets'][i]],validation['public']['keys'][i,validation['labels']['targets'][i]])
 return dict(selected_step=best[1],validation=m['validation'],controls=m['controls'],raw_cells=count,checkpoint_bytes=True,visitation_checked=True,cpu_audit_wall_seconds=time.monotonic()-start,scope='Raw learned required-role and supplied-arity canonical scores; fresh names is actor-input no-op. Calibration null target used only in independent score reconstruction, not actor.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();r=run(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='controls'}))
