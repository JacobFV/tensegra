"""Aggregate archived public-text graph scores without inference or calibration."""
import json,gzip,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('localization');p.add_argument('output');a=p.parse_args()
x=json.load(gzip.open(a.archive,'rt'));loc=json.load(gzip.open(a.localization,'rt'));summary=[]
for run in x['runs']:
 for decoder in ('raw','calibrated'):
  rows=[r[decoder+'_metrics'] for r in run['rows']];records=[r for r in loc['rows'] if r['seed']==run['seed'] and r['decoder']==decoder]
  z={k:sum(r[k] for r in rows)/len(rows) for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence','semantic_equivalence')}
  for k in ('node','typed_edge','ordered_edge'):
   q={v:sum(r[k][v] for r in rows) for v in ('true_positive','predicted_count','gold_count')};q.update(precision=q['true_positive']/max(1,q['predicted_count']),recall=q['true_positive']/max(1,q['gold_count']),f1=2*q['true_positive']/max(1,q['predicted_count']+q['gold_count']));z[k]=q
  z['oracle_component_exact']={k:sum(r['oracle_component_exact'][k] for r in records) for k in records[0]['oracle_component_exact']}
  z['slot_accuracy_given_correct_edge']={k:sum(r['slot_accuracy_given_correct_edge'][k] for r in records) for k in ('correct','denominator')}
  z['by_graph_size']={}
  for n in sorted({sum(r['target']['presence']) for r in run['rows']}):
   group=[r[decoder+'_metrics'] for r in run['rows'] if sum(r['target']['presence'])==n]
   z['by_graph_size'][str(n)]=dict(examples=len(group),exact=sum(r['semantic_equivalence'] for r in group),mean_type_accuracy=sum(r['node_type_accuracy'] for r in group)/len(group),mean_copy_accuracy=sum(r['identity_copy_accuracy'] for r in group)/len(group))
  summary.append(dict(seed=run['seed'],decoder=decoder,examples=len(rows),**z))
Path(a.output).write_text(json.dumps(summary,indent=2)+'\n')
