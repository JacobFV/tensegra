"""Separate balanced-grid type/value counts; never used for primary selection."""
import argparse,gzip,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();out=[]
for seed in (10,11,12):
 for row in json.load(gzip.open(a.root/str(seed)/'predictions.json.gz','rt')):
  if row['split']!='balanced':continue
  counts={}
  for pred,y,typ in zip(row['predictions']['value'],row['targets']['value'],row['targets']['type']):
   key=typ,(y-16)/2;counts.setdefault(key,[0,0]);counts[key][0]+=pred==y;counts[key][1]+=1
  assert len(counts)==52 and all(v[1]==64 for v in counts.values())
  out.append(dict(seed=seed,head=row['head'],delay=row['target_delay'],value=dict(correct=sum(v[0]for v in counts.values()),total=sum(v[1]for v in counts.values())),strata=[dict(type=k[0],value=k[1],correct=v[0],total=v[1])for k,v in counts.items()]))
a.output.write_text(json.dumps(dict(cells=out,cpu_audit_wall_seconds=time.monotonic()-start,scope='Postfreeze diagnostic52 legal type/value strata,64 contexts each, paired across delays and consumers. Not a new promotion gate and insufficient support for precise per-value population guarantees.'),indent=2)+'\n')
