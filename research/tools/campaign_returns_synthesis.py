"""Reproduce compact return-campaign comparisons from committed raw outcomes."""
import argparse,gzip,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('research/results/campaign-01/returns'));a=p.parse_args()
specs=[('R06 original','r06-development','original'),('R06 balanced','r06-development','balanced'),('R07 residual .003','r07-development','residual_mlp'),('R08 residual .0003','r08-development','residual_lr_0.0003'),('R08 residual .0001','r08-development','residual_lr_0.0001'),('R08 residual .00003','r08-development','residual_lr_3e-05'),('R09 binary phase','r09-development','phase_linear')]
out={'scope':'same R06 development populations; no fresh confirmation or independent replication','rows':[]}
for label,stage,arm in specs:
 rows=json.load(gzip.open(a.root/stage/'predictions.json.gz','rt'))
 def cell(split,d):return next(r for r in rows if r['head']==arm and r['split']==split and r['target_delay']==d and r['distractors']==(2 if split.startswith('train') else 8))
 row={'label':label,'stage':stage,'arm':arm,'counts':{},'scalar_error_distances':{}}
 for split in ['train_balanced','train_original','validation','validation_grid']:
  row['counts'][split]={str(d):cell(split,d)['counts'] for d in [0,1,16]}
 for split in ['calibration_grid','validation_grid']:
  cells=[]
  for r in rows:
   if r['head']!=arm or r['split']!=split:continue
   for typ,val in sorted(set(zip(r['targets']['type'],r['targets']['value']))):
    indices=[i for i,(t,v) in enumerate(zip(r['targets']['type'],r['targets']['value'])) if (t,v)==(typ,val)]
    cells.append({'type':typ,'value':(val-16)/2,'delay':r['target_delay'],'correct':sum(r['predictions']['value'][i]==val for i in indices),'total':len(indices)})
  row[split+'_minimum']=min(cells,key=lambda c:c['correct']/c['total'])
 r=cell('validation_grid',16)
 row['scalar_error_distances']=dict(Counter(str(abs(pred-target)/2) for pred,target in zip(r['predictions']['value'],r['targets']['value']) if pred!=target))
 out['rows'].append(row)
path=a.root/'synthesis';path.mkdir(exist_ok=True);(path/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
