"""Posthoc error stratification by public primitive value; no model inference."""
import argparse,gzip,json
from pathlib import Path
from topoformer.belief_contracts import contract_episodes
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=[]
for folder in sorted(a.root.glob('*-[012]')):
 for path in sorted(folder.glob('supplied_empty_prior-*.json.gz')):
  with gzip.open(path,'rt') as f:r=json.load(f)
  es=contract_episodes(len(r['posterior']),r['event_seed'],r['candidates'],r['condition'])
  for value in (0,1):
   inds=[i for i,e in enumerate(es) if e['events'][1][3]==value]
   final=[];errors=[]
   for i in inds:
    ps,qs=r['posterior'][i],r['target'][i]
    final.append(qs[-1][max(range(len(ps[-1])),key=ps[-1].__getitem__)]>0)
    errors.extend(sum(abs(x-y) for x,y in zip(p,q)) for p,q in zip(ps,qs))
   rows.append(dict(model=folder.name,condition=r['condition'],primitive_value=value,count=len(inds),final_correct=sum(final),mean_l1=sum(errors)/len(errors)))
(a.root/'primitive-strata.json').write_text(json.dumps(rows,indent=2))
