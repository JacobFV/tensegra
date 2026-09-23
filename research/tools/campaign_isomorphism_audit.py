"""Necessary attributed-graph invariants, independently from graph-isomorphism code."""
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import edge_set
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();x=json.load(gzip.open(a.root/'s08-isomorphism/predictions.json.gz'));indexed={(r['arm'],r['split'],r['policy'],r['seed']):r for r in x['rows']};counts=collections.Counter()
def invariants(g):
 present={i for i,v in enumerate(g['presence'])if v};nodes=collections.Counter((g['kind'][i],g['copy'][i]if g['kind'][i]in (1,5)else None,g['value'][i]if g['kind'][i]not in(1,5)else None)for i in present)
 pairs=collections.defaultdict(list)
 for i,j,r in sorted(edge_set(g)):
  if i in present and j in present:pairs[i,j].append((r,g['slots'][i][j]))
 return nodes,collections.Counter(tuple(v)for v in pairs.values())
for arm in x['config']['arms']:
 path=a.root/Path(arm['archive']).relative_to('results');assert hashlib.sha256(path.read_bytes()).hexdigest()==x['archive_sha256'][arm['name']];d=json.load(gzip.open(path))
 for split,key in [('train','train_rows'),('development','rows')]:
  for row in d[key]:
   gn,ge=invariants(row['target'])
   for policy in ('raw','calibrated'):
    pred=row['raw']if policy=='raw'else dict(row['raw'],edges=row['calibrated_edges']);pn,pe=invariants(pred);reason='node_attribute_multiset'if pn!=gn else 'edge_label_multiset'if pe!=ge else None;assert reason is not None,'Requires full isomorphism rather than invariant rejection';r=indexed[arm['name'],split,policy,row['seed']];assert not r['isomorphic']and r['reason']==reason;counts[f"{arm['name']}/{split}/{policy}/{reason}"]+=1
out=dict(records_verified=sum(counts.values()),counts=dict(counts),cpu_audit_wall_seconds=time.monotonic()-t,scope='All saved negative decisions follow necessary attributed-node or relation/slot multiset mismatch; no approximate matching or relaxed semantics. No full search is needed for this population.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['records_verified'])
