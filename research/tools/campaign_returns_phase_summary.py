"""Reconstruct the prespecified R09 calibration advancement criterion."""
import argparse,gzip,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();rows=json.load(gzip.open(a.results/'predictions.json.gz','rt'))
out={'arms':{},'grid_cells':[],'scope':'development only; fixed900 endpoint; calibration-only advancement'}
for arm in ('linear','phase_linear'):
 info={}
 for split in ('calibration','validation'):
  cells=[r for r in rows if r['head']==arm and r['split']==split];worst=min(cells,key=lambda r:r['counts']['value']['correct']/r['counts']['value']['total'])
  info[split]={'minimum':worst['counts']['value'],'delay':worst['target_delay'],'distractors':worst['distractors']}
 for split in ('calibration_grid','validation_grid'):
  cells=[]
  for r in rows:
   if r['head']!=arm or r['split']!=split:continue
   counts=defaultdict(lambda:[0,0])
   for pred,target,typ in zip(r['predictions']['value'],r['targets']['value'],r['targets']['type']):counts[(typ,target)][0]+=int(pred==target);counts[(typ,target)][1]+=1
   for (typ,label),(correct,total) in counts.items():cells.append(dict(arm=arm,split=split,delay=r['target_delay'],type=typ,value=(label-16)/2,correct=correct,total=total))
  out['grid_cells'].extend(cells);info[split]=min(cells,key=lambda x:x['correct']/x['total'])
 out['arms'][arm]=info
orig=out['arms']['linear'];bal=out['arms']['phase_linear'];c=bal['calibration']['minimum'];og=orig['calibration_grid'];bg=bal['calibration_grid']
out['calibration_advancement']=dict(strict_grid_improvement=bg['correct']/bg['total']>og['correct']/og['total'],original_mixture_contract=c['correct']/c['total']>=.98)
out['advances']=all(out['calibration_advancement'].values());(a.results/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
