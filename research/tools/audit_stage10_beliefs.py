"""Independent posterior audit of Stage10 belief role separation."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
def load(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def audit(root,require_complete=True):
 runs=[];frames=0;cells=0;pairing={}
 for path in sorted(root.glob('*/manifest.json')):
  manifest=load(path);cfg=manifest['config'];seed=cfg['seed'];arm=cfg['arm']
  if seed not in (20,21,22):continue
  assert manifest['workspace_width']==1024
  pairing.setdefault(seed,[]).append((manifest['initial_tensor_sha256'],manifest['semantic_training_stream_sha256']))
  rows=load(path.parent/'metrics.json');expected={(s,n,c) for s in cfg['split_seeds'] for n in cfg['eval_candidates'] for c in cfg['conditions']};seen=set();gate=[];raws={}
  for row in rows:
   key=(row['split'],row['candidates'],row['condition']);assert key not in seen;seen.add(key)
   file=path.parent/row['raw'];assert hashlib.sha256(file.read_bytes()).hexdigest()==row['raw_sha256'];raw=load(file);raws[key]=raw
   assert len(raw['posterior'])==512
   l1=[];imp=[];final=0
   for ep,tar,empty in zip(raw['posterior'],raw['target'],raw['empty_ledger']):
    for p,q,e in zip(ep,tar,empty):
     frames+=1;assert all(math.isfinite(v) and v>=0 for v in p);assert abs(sum(p)-1)<1e-5
     if e:assert max(abs(v-1/row['candidates']) for v in p[:-1])<1e-7 and p[-1]==0
     l1.append(sum(abs(a-b) for a,b in zip(p,q)));imp.append(sum(a for a,b in zip(p,q) if b==0))
    final+=tar[-1][max(range(len(ep[-1])),key=ep[-1].__getitem__)]>0
   a=sum(l1)/len(l1);b=sum(imp)/len(imp);acc=final/512
   assert abs(a-row['mean_l1'])<1e-6 and abs(b-row['mean_impossible'])<1e-6 and acc==row['final_support_accuracy']
   passed=a<.05 and b<.01 and acc>(.98 if row['candidates']==8 else .95)
   assert passed==row['passed'];cells+=1
   gate.append(dict(split=key[0],candidates=key[1],condition=key[2],passed=passed,mean_l1=a,mean_impossible=b,final_correct=final))
  assert seen==expected
  deltas=[]
  for (split,n,condition),raw in raws.items():
   if condition!='clean':continue
   renamed=raws[(split,n,'id_rename')];assert raw['target']==renamed['target']
   delta=max(abs(a-b) for ep,eq in zip(raw['posterior'],renamed['posterior']) for p,q in zip(ep,eq) for a,b in zip(p,q));deltas.append(delta)
  if arm=='ledger_only':assert max(deltas)==0
  runs.append(dict(seed=seed,arm=arm,cells=len(rows),max_renaming_delta=max(deltas),cells_detail=gate))
 if require_complete:assert len(runs)==9 and cells==810
 assert all((len(v)==3 or not require_complete) and len(set(v))==1 for v in pairing.values())
 return dict(runs=runs,cells_verified=cells,frames_verified=frames,paired_initial_tensors_and_semantic_streams=True,complete_matrix=len(runs)==9 and cells==810)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();out=audit(a.root,not a.partial);a.output.write_text(json.dumps(out,indent=2)+'\n');print('Verified',out['cells_verified'],'cells')
