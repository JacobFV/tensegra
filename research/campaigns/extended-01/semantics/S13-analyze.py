"""Fixed paired S13 development summary, CPU archived predictions only."""
import collections,gzip,hashlib,json
from pathlib import Path
base=Path('research/results/campaign-01/semantics');load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results={};manifests={};endpoints={};digests={}
for arm in ('english','mixed'):
 root=base/f's13-{arm}-main';m=load(root/'manifest.json.gz');manifests[arm]=m;digests[arm]=sha(root/'manifest.json.gz');curves=[]
 assert m['added_presentations']==32768 and set(m['added_visits'])=={4}
 assert all(v==([4,0] if arm=='english' else [2,2]) for v in m['renderer_visits'])
 for curve in m['curves']:
  records={};surface={}
  for language in ('english','spanish'):
   meta=curve[language];p=root/meta['artifact'];assert sha(p)==meta['sha256'];d=load(p);records[language]=d;assert len(d['rows'])==512
   if language=='spanish':assert d['thresholds']==records['english']['thresholds']
   summary={}
   for policy in ('raw','calibrated'):
    metrics=[r[policy+'_metrics'] for r in d['rows']];edges={}
    for key in ('typed_edge','ordered_edge'):
     tp=sum(x[key]['true_positive'] for x in metrics);pred=sum(x[key]['predicted_count'] for x in metrics);gold=sum(x[key]['gold_count'] for x in metrics)
     edges[key]=dict(true_positive=tp,predicted=pred,gold=gold,precision=tp/pred if pred else 0.,recall=tp/gold if gold else 0.,f1=2*tp/(pred+gold) if pred+gold else 1.)
    sizes=collections.defaultdict(lambda:[0,0])
    for r in d['rows']:
     n=sum(r['target']['presence']);sizes[n][0]+=1;sizes[n][1]+=r[policy+'_metrics']['semantic_equivalence']
    summary[policy]=dict(exact=sum(x['semantic_equivalence'] for x in metrics),exact_copy=sum(x['identity_copy_accuracy']==1 for x in metrics),mean_copy=sum(x['identity_copy_accuracy'] for x in metrics)/512,edges=edges,by_node_count={str(n):dict(examples=v[0],exact=v[1]) for n,v in sizes.items()})
   surface[language]=summary
  transitions={p:collections.Counter() for p in ('raw','calibrated')}
  for en,es in zip(records['english']['rows'],records['spanish']['rows']):
   assert en['semantic_sha256']==es['semantic_sha256']
   for p in transitions:transitions[p][f"{int(en[p+'_metrics']['semantic_equivalence'])}->{int(es[p+'_metrics']['semantic_equivalence'])}"]+=1
  curves.append(dict(added_update=curve['added_update'],surfaces=surface,english_to_spanish={p:dict(v) for p,v in transitions.items()}))
  if curve['added_update']==4096:endpoints[arm]=records
 results[arm]=dict(curves=curves,optimizer_tokens=m['optimizer_tokens'],training_seconds=m['training_seconds'])
for field in ('initial_state_sha256','inherited_optimizer_steps','construction_sequence_sha256','pair_sequence_sha256','first_batch_replay','renderer_phases'):
 assert manifests['english'][field]==manifests['mixed'][field],field
paired={}
for language in ('english','spanish'):
 paired[language]={}
 for p in ('raw','calibrated'):
  transitions=collections.Counter()
  for a,b in zip(endpoints['english'][language]['rows'],endpoints['mixed'][language]['rows']):
   assert a['semantic_sha256']==b['semantic_sha256'];transitions[f"{int(a[p+'_metrics']['semantic_equivalence'])}->{int(b[p+'_metrics']['semantic_equivalence'])}"]+=1
  paired[language][p]=dict(transitions=dict(transitions),mixed_minus_english=transitions['0->1']-transitions['1->0'])
final=results['mixed']['curves'][-1]['surfaces'];earlier=next(c for c in results['mixed']['curves'] if c['added_update']==2048)['surfaces']
criteria=dict(spanish_at_least_52=final['spanish']['calibrated']['exact']>=52,spanish_gain_at_least_26=paired['spanish']['calibrated']['mixed_minus_english']>=26,english_loss_at_most_26=paired['english']['calibrated']['mixed_minus_english']>=-26)
result=dict(source_sha256=sha(Path(__file__)),manifest_sha256=digests,results=results,paired_endpoint=paired,promotion_criteria=criteria,promotion_pass=all(criteria.values()),spanish_2048_to_4096_gain=final['spanish']['calibrated']['exact']-earlier['spanish']['calibrated']['exact'],scope='Single fixed development pair; not confirmation or new-motif transfer; extension requires finite-loss audit and separate authorization')
(base/'s13-paired-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('results',)},indent=2))
