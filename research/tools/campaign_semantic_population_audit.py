"""Independent compact semantic population collision/capacity audit, no model."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();public={};alpha={};splits={};slot_conflicts=0
for split in ('train','development','reserved_confirmation'):
 rows=0;keys=set();path=a.root/f'{split}.jsonl.gz'
 with gzip.open(path,'rt')as f:
  for line in f:
   r=json.loads(line);rows+=1;assert len(r['nodes'])<=128;ids={};nodes=[];slots={}
   for kind,value in r['nodes']:
    if kind in ('ident','entity'):
     ids.setdefault(value,len(ids));value=['identity',ids[value]]
    nodes.append([kind,value])
   for i,j,relation,slot in r['edges']:
    assert 0<=i<len(nodes)and 0<=j<len(nodes)
    if slot is not None:
     assert slot>=0
     if (i,j)in slots and slots[i,j]!=slot:slot_conflicts+=1
     slots[i,j]=slot
   signature=json.dumps([r['nodes'],sorted(r['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3]is None else e[3]))],sort_keys=True)
   assert r['text']not in public or public[r['text']]==signature;public[r['text']]=signature
   # Dropping roots makes this stricter for overlap detection, not an inference of graph isomorphism.
   key=json.dumps([nodes,sorted(r['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3]is None else e[3]))],sort_keys=True)
   assert key not in alpha;alpha[key]=split;keys.add(key)
 splits[split]=dict(examples=rows,distinct_alpha_node_edge_keys=len(keys),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
assert slot_conflicts==0
out=dict(splits=splits,distinct_public_texts=len(public),distinct_alpha_node_edge_keys=len(alpha),conflicting_slot_labels=slot_conflicts,incompatible_identical_public_targets=0,cpu_audit_wall_seconds=time.monotonic()-start,scope='Compact node/edge labels independently checked; canonical node order retained, identity spellings alpha-normalized, roots omitted conservatively for overlap. Not a proof that every target distinction is linguistically identifiable.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
