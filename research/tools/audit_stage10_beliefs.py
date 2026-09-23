"""Independent posterior audit of Stage10 belief role separation."""
import argparse,gzip,hashlib,json,math,struct
from pathlib import Path
def load(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def audit(root,require_complete=True):
 runs=[];frames=0;cells=0;pairing={};calibration_checked=0
 for path in sorted(root.glob('*/manifest.json')):
  manifest=load(path);cfg=manifest['config'];seed=cfg['seed'];arm=cfg['arm']
  if seed not in (20,21,22):continue
  assert manifest['workspace_width']==1024
  assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==manifest['config_sha256']
  for name,h in manifest['source_sha256'].items():assert hashlib.sha256((root.parent/'frozen-source'/name).read_bytes()).hexdigest()==h
  inventory_path=path.parent/'training-id-inventory.json';inventory=load(inventory_path)
  assert hashlib.sha256(inventory_path.read_bytes()).hexdigest()==manifest['training_id_inventory_sha256']
  coverage=manifest['training_id_coverage'];assert len(inventory)==coverage['unique'] and len(set(inventory))==len(inventory)
  for handle,seen_handle in coverage['historical_handles_seen'].items():assert (int(handle) in inventory)==seen_handle
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
   for frame in row['frames']:
    t=frame['frame'];ps=[ep[t] for ep in raw['posterior']];qs=[ep[t] for ep in raw['target']]
    predictions=[max(range(len(p)),key=p.__getitem__) for p in ps]
    truth=[q[-1] for q in qs];null=[p[-1] for p in ps]
    support=sum(q[j]>0 for q,j in zip(qs,predictions))
    assert support==frame['support_correct']
    assert sum(sum(v>0 for v in q)>1 for q in qs)==frame['ambiguous']
    assert sum(truth)==frame['null_targets']
    assert abs(sum((a-b)**2 for a,b in zip(null,truth))/512-frame['null_brier'])<1e-6
    assert abs(sum(a-b for a,b in zip(null,truth))/512-frame['null_signed_error'])<1e-6
    assert sum(j==len(p)-1 and q[-1]==1 for p,q,j in zip(ps,qs,predictions))==frame['no_match_true_positive']
    assert sum(j==len(p)-1 and q[-1]==0 for p,q,j in zip(ps,qs,predictions))==frame['no_match_false_positive']
    for binrow in frame['calibration']:
     k=round(binrow['lower']*10)
     lo=struct.unpack('f',struct.pack('f',k/10))[0];hi=struct.unpack('f',struct.pack('f',(k+1)/10))[0]
     chosen=[(p[j],q[j]) for p,q,j in zip(ps,qs,predictions) if p[j]>=lo and (p[j]<hi if k<9 else p[j]<=1)]
     calibration_checked+=1
     assert len(chosen)==binrow['count']
     if chosen:
      assert abs(sum(a for a,b in chosen)/len(chosen)-binrow['confidence'])<1e-6
      assert abs(sum(b for a,b in chosen)/len(chosen)-binrow['expected_correctness'])<1e-6
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
 return dict(runs=runs,cells_verified=cells,frames_verified=frames,calibration_bins_verified=calibration_checked,paired_initial_tensors_and_semantic_streams=True,complete_matrix=len(runs)==9 and cells==810)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();out=audit(a.root,not a.partial);a.output.write_text(json.dumps(out,indent=2)+'\n');print('Verified',out['cells_verified'],'cells')
