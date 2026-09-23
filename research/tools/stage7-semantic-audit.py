"""Independent corrected semantic exposure/provenance and compact raw audit."""
import gzip,hashlib,json
from pathlib import Path
root=Path('research/results/stage7-semantic');manifests={};checked=0
for n in (1000,10000):
 p=root/f'scaling-corrected-{n}';m=json.loads((p/'manifest.json').read_text());manifests[n]=m
 assert hashlib.sha256((p/'semantic_scaling-source.py').read_bytes()).hexdigest()==m['source_sha256']['semantic_scaling.py']
 assert hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()==m['config_sha256']
 assert not m['gate_f']['passed'];assert m['corpus']['actual_unique_graphs']==n+64
 assert m['corpus']['attempts']==m['corpus']['actual_unique_graphs']+m['corpus']['duplicates']
 rows=[json.loads(l) for l in (p/'curves.jsonl').read_text().splitlines()]
 assert len(rows)==25
 for r in rows:
  if r['arm']!='frequency':
   assert r['renderer_exposure']=={'english':((r['step']+1)//2)*2,'spanish':(r['step']//2)*2}
   assert r['optimizer_examples']==2*r['step'];assert r['actual_unique_graphs_seen']==min(n,2*r['step'])
  for result in r['evaluation'].values():
   assert result['examples']==64;assert result['semantic_equivalence']==0
   for kind in ('node','typed_edge','ordered_edge'):
    c=result[kind];assert abs(c['f1']-2*c['true_positive']/max(1,c['predicted_count']+c['gold_count']))<1e-10
 for seed in (0,1,2):
  runs=[r for r in m['runs'] if r['seed']==seed];assert len(runs)==2;assert len({r['initial_sha256'] for r in runs})==1;assert len({r['parameters'] for r in runs})==1
 with gzip.open(p/'final-failures.jsonl.gz','rt') as f:raws=[json.loads(l) for l in f]
 assert len(raws)==96
 for r in raws:
  pred,target=r['prediction'],r['target'];pg={i for i,v in enumerate(pred['presence']) if v};tg={i for i,v in enumerate(target['presence']) if v};pe={tuple(e) for e in pred['edges'] if e[0] in pg and e[1] in pg};te={tuple(e) for e in target['edges']};po={e for e in pe if pred['slots'][e[0]][e[1]]>=0};to={e for e in te if target['slots'][e[0]][e[1]]>=0}
  for key,a,b,tp in [('node',pg,tg,len(pg&tg)),('typed_edge',pe,te,len(pe&te)),('ordered_edge',po,to,sum(pred['slots'][e[0]][e[1]]==target['slots'][e[0]][e[1]] for e in po&to))]:
   calc=dict(true_positive=tp,predicted_count=len(a),gold_count=len(b),precision=tp/len(a) if a else 0.,recall=tp/len(b) if b else 0.,f1=2*tp/(len(a)+len(b)) if a or b else 1.)
   for k,v in calc.items():assert abs(v-r['metrics'][key][k])<1e-7
  copied=[i for i,v in enumerate(target['copy']) if v>=0];kind=sum(pred['kind'][i]==target['kind'][i] for i in tg)/len(tg);copy=sum(pred['copy'][i]==target['copy'][i] for i in copied)/len(copied) if copied else 1.;eq=sum((pred['copy'][i]==pred['copy'][j])==(target['copy'][i]==target['copy'][j]) for i in copied for j in copied)/len(copied)**2 if copied else 1.
  semantic=pg==tg and pe==te and all(pred['kind'][i]==target['kind'][i] for i in tg) and all(v!=0 and pred['value'][i]==v for i,v in enumerate(target['value']) if v>=0) and all(pred['copy'][i]==target['copy'][i] for i in copied) and all(pred['slots'][e[0]][e[1]]==target['slots'][e[0]][e[1]] for e in te)
  for k,v in [('node_type_accuracy',kind),('identity_copy_accuracy',copy),('entity_equivalence',eq),('semantic_equivalence',semantic)]:assert abs(v-r['metrics'][k])<1e-7
  checked+=1
assert manifests[1000]['split_semantic_sha256']['eval']==manifests[10000]['split_semantic_sha256']['eval']
assert manifests[1000]['source_sha256']==manifests[10000]['source_sha256']
for seed in (0,1,2):
 for arm in ('semantic','no_input'):
  a=next(r for r in manifests[1000]['runs'] if r['seed']==seed and r['arm']==arm);b=next(r for r in manifests[10000]['runs'] if r['seed']==seed and r['arm']==arm);assert a['initial_sha256']==b['initial_sha256'];assert a['parameters']==b['parameters']
print('PASS two budgets, 12 arms, matched source/initializations/eval/renderer exposures; 50 curve rows; compact raw rows',checked)
