"""Reconstruct R08 final-only selection and retain every fitting/grid outcome."""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
rows=json.load(gzip.open(a.results/'predictions.json.gz','rt'));manifest=json.load(gzip.open(a.results/'manifest.json.gz','rt'))
out={'arms':{},'grid_cells':[],'selected':manifest['selected'],'scope':'reused development; fixed900, calibration-only selection'}
for arm in dict.fromkeys(r['head'] for r in rows):
 info={}
 for split in ('train_balanced','train_original','calibration','validation'):
  cells=[r for r in rows if r['head']==arm and r['split']==split];worst=min(cells,key=lambda r:r['counts']['value']['correct']/r['counts']['value']['total'])
  info[split]={'minimum':worst['counts']['value'],'delay':worst['target_delay'],'distractors':worst['distractors']}
 for split in ('calibration_grid','validation_grid'):
  cells=[]
  for r in rows:
   if r['head']!=arm or r['split']!=split:continue
   counts=defaultdict(lambda:[0,0])
   for pred,target,typ in zip(r['predictions']['value'],r['targets']['value'],r['targets']['type']):
    counts[(typ,target)][0]+=int(pred==target);counts[(typ,target)][1]+=1
   for (typ,label),(correct,total) in counts.items():cells.append(dict(arm=arm,split=split,delay=r['target_delay'],type=typ,value=(label-16)/2,correct=correct,total=total))
  out['grid_cells'].extend(cells);info[split]=min(cells,key=lambda x:x['correct']/x['total'])
 out['arms'][arm]=info
names=[f['arm'] for f in manifest['fits'] if f['trained_here']]
def score(name):
 x=out['arms'][name];g=x['calibration_grid'];m=x['calibration']['minimum'];return(g['correct']/g['total'],m['correct']/m['total'])
assert max(names,key=score)==out['selected']
g,m=score(out['selected']);reference=score('linear_reference')[0]
out['advancement']={'strict_grid_improvement':g>reference,'mixture_contract':m>=.98};out['advances']=all(out['advancement'].values())
(a.results/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
