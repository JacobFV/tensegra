"""Independent compact R05 outcome and validation selection reconstruction."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np

def run(root):
 start=time.monotonic();root=Path(root)
 m=json.loads(gzip.decompress((root/'manifest.json.gz').read_bytes()))
 raw=(root/'predictions.json.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['predictions_sha256']
 rows=json.loads(gzip.decompress(raw));lookup={};out=[]
 for r in rows:
  k=(r['arm'],r['key'],r['delay']);assert k not in lookup;lookup[k]=r
  p=np.array(r['predictions']);y=np.array(r['original_targets']);n=len(y)
  assert n==r['total'] and int(y.sum())==r['positive_labels']
  assert int((p==y).sum())==r['original_correct']
  for field in ('types','values','operations'):assert len(r[field])==n
  if r['supplied_targets'] is not None:
   s=np.array(r['supplied_targets']);c=s!=y
   for f,v in dict(supplied_correct=(p==s).sum(),changed_total=c.sum(),changed_supplied_correct=((p==s)&c).sum(),changed_original_correct=((p==y)&c).sum()).items():assert r[f]==int(v),(k,f)
  out.append({f:r[f] for f in ('arm','key','delay','total','original_correct','supplied_correct','changed_total','changed_supplied_correct') if f in r})
 selected={}
 for f in m['fits']:
  scored=[]
  for c in f['curve']:
   vals=[x['correct'] for x in c['cells']];score=[min(vals),sum(vals)]
   assert score==c['selection_score'];scored.append((score,-c['step'],c))
  best=max(scored,key=lambda x:(x[0],x[1]))[2]
  step=m['config'].get('fixed_steps',{}).get(f['arm'],best['step']);assert step==f['selected_step']
  selected[f['arm']]=step
  chosen=next(c for c in f['curve'] if c['step']==step)
  for c in chosen['cells']:assert lookup[f['arm'],c['key'],c['delay']]['original_correct']==c['correct']
  bits=np.unpackbits(np.frombuffer(bytes.fromhex(f['visited_row_bits_little_endian']),dtype=np.uint8),bitorder='little')[:f['fit_rows']]
  assert int(bits.sum())==f['unique_rows_sampled']
  assert int(bits.reshape(-1,f['fit_events']).any(0).sum())==f['unique_events_sampled']
 # All predictions share targets across comparator arms. Query-only ignores return and delay.
 for (arm,key,d),r in lookup.items():
  q=lookup['query_only',key,d];assert r['original_targets']==q['original_targets']
  if key.startswith(('validation/','intervention_')):
   base=lookup['query_only','validation/8',0]['predictions'][:r['total']]
   assert q['predictions']==base
 clean=[];causal=[]
 for d in m['config']['delays']:
  for distractors in m['config']['eval_distractors']:
   key=f'validation/{distractors}'; rr=[lookup[a,key,d] for a in ('learned','query_only','oracle')];acc=[r['original_correct']/r['total'] for r in rr]
   clean.append(dict(delay=d,distractors=distractors,learned=acc[0],query_only=acc[1],oracle=acc[2],relative_gain=(acc[0]-acc[1])/(acc[2]-acc[1]) if acc[2]>acc[1] else None))
 for d in m['config'].get('intervention_delays',[]):
  drop=lookup['learned','intervention_drop/8',d];base=lookup['learned','validation/8',d]
  bp=np.array(base['predictions'][:drop['total']]);by=np.array(base['original_targets'][:drop['total']]);dropdiff=float((bp==by).mean()-drop['original_correct']/drop['total'])
  r=dict(delay=d,drop_difference=dropdiff)
  for kind in ('wrong','swap'):
   x=lookup['learned',f'intervention_{kind}/8',d];r[kind]=dict(correct=x['changed_supplied_correct'],total=x['changed_total'])
  causal.append(r)
 return dict(raw_cells=len(rows),selected=selected,clean=clean,causal=causal,rows=out,cpu_audit_wall_seconds=time.monotonic()-start,scope='Independent compact predictions, endpoint selection, exact counts and paired controls. Actor feature/cache replay is a separate audit.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();r=run(a.root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
