"""Fixed lexical diagnostic: CPU-only paired outcomes, no fitting."""
import gzip,hashlib,json,collections
from pathlib import Path
root=Path('research/results/campaign-01/semantics');folder=root/'s12-renamed-inference'
load=lambda p:json.load(gzip.open(p,'rt'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(folder/'manifest.json.gz');results=[]
for entry in manifest['artifacts']:
 p=folder/entry['artifact'];assert sha(p)==entry['sha256'];data=load(p)
 primary_path=root/f"s12-{entry['arm']}-{entry['seed']}"/'evaluation-u24576.json.gz';assert sha(primary_path)==data['primary_sha256'];primary=load(primary_path)
 assert primary['thresholds']==data['thresholds'] and len(data['rows'])==1024
 summary=dict(seed=entry['seed'],arm=entry['arm'],artifact_sha256=sha(p),policies={})
 for policy in ('raw','calibrated'):
  transitions=collections.Counter();copy_exact=0;copy_sum=0.;counts={k:[0,0,0] for k in ('typed_edge','ordered_edge')}
  for old,new in zip(primary['rows'],data['rows']):
   assert old['semantic_sha256']==new['semantic_sha256']
   a=bool(old[policy+'_metrics']['semantic_equivalence']);b=bool(new[policy+'_metrics']['semantic_equivalence']);transitions[f'{int(a)}->{int(b)}']+=1
   m=new[policy+'_metrics'];copy_sum+=m['identity_copy_accuracy'];copy_exact+=m['identity_copy_accuracy']==1.
   for k in counts:
    for i,n in enumerate(('true_positive','predicted_count','gold_count')):counts[k][i]+=m[k][n]
  summary['policies'][policy]=dict(transitions=dict(transitions),original_exact=transitions['1->0']+transitions['1->1'],renamed_exact=transitions['0->1']+transitions['1->1'],paired_delta=transitions['0->1']-transitions['1->0'],exact_copy=copy_exact,mean_copy_accuracy=copy_sum/1024,edges={k:dict(true_positive=v[0],predicted=v[1],gold=v[2],f1=2*v[0]/(v[1]+v[2])) for k,v in counts.items()})
 results.append(summary);print(entry['seed'],entry['arm'],summary['policies'])
(folder/'analysis.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),manifest_sha256=sha(folder/'manifest.json.gz'),results=results),indent=2)+'\n')
