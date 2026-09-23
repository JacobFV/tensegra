"""Reconstruct R04 contract and paired readout outcomes from compact predictions."""
import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
summary={'runs':[],'historical_registry_changed':False}
for seed in (10,11,12):
 rows=json.load(gzip.open(args.results/str(seed)/'predictions.json.gz','rt'))
 run={'seed':seed,'cells':[],'contracts':{},'paired':[]}
 for row in rows:
  if row['split']=='train':continue
  run['cells'].append({k:row[k] for k in ('head','split','target_delay','distractors','counts')})
 for head in ('unchanged','ce_4096','ce_16384'):
  cells=[r for r in rows if r['head']==head and r['split']=='validation']
  failed=[{'delay':r['target_delay'],'distractors':r['distractors'],'count':r['counts']['value']} for r in cells if r['counts']['value']['correct']/r['counts']['value']['total']<.98]
  run['contracts'][head]={'passed':not failed,'failed_cells':failed,'worst_correct':min(r['counts']['value']['correct'] for r in cells),'total_per_cell':4096}
 for a in rows:
  if a['head']!='ce_4096' or a['split']=='train':continue
  b=next(r for r in rows if r['head']=='ce_16384' and (r['split'],r['target_delay'],r['distractors'])==(a['split'],a['target_delay'],a['distractors']))
  assert a['targets']==b['targets'] and a['event_sha256']==b['event_sha256']
  aa=[p==t for p,t in zip(a['predictions']['value'],a['targets']['value'])];bb=[p==t for p,t in zip(b['predictions']['value'],b['targets']['value'])]
  run['paired'].append({'split':a['split'],'delay':a['target_delay'],'distractors':a['distractors'],'cc':sum(x and y for x,y in zip(aa,bb)),'cw':sum(x and not y for x,y in zip(aa,bb)),'wc':sum(not x and y for x,y in zip(aa,bb)),'ww':sum(not x and not y for x,y in zip(aa,bb))})
 summary['runs'].append(run)
(args.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
