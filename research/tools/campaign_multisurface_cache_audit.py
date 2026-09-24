"""Independent S13 compact surface/feature/target audit, using NumPy only."""
import argparse,gzip,hashlib,json,math,re,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--old-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();meta=json.loads((a.root/'audit.json').read_text());public={};features={};keys={};seen=set();counts={};parity=0
sha=lambda b:hashlib.sha256(b).hexdigest()
def rows(path):return [json.loads(s) for s in gzip.open(path,'rt')]
def alpha(r):
 ids={};ns=[]
 for k,v in r['nodes']:
  if k in ('ident','entity'):ids.setdefault(v,len(ids));v=['identity',ids[v]]
  ns.append([k,v])
 return json.dumps([ns,sorted(r['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3] is None else e[3]))],sort_keys=True)
for split,expected in meta['config']['exclude_cache_sha256'].items():assert sha((a.old_data/(split+'.jsonl.gz')).read_bytes())==expected
old={s:rows(a.old_data/(s+'.jsonl.gz')) for s in ('train','development','reserved_confirmation')};excluded={alpha(r) for rs in old.values() for r in rs}
for split in ('train','development'):
 path=a.root/(split+'.jsonl.gz');assert sha(path.read_bytes())==meta['cache_sha256'][split];rs=rows(path);counts[split]=len(rs)
 for j,r in enumerate(rs):
  key=alpha(r);assert key not in keys;keys[key]=split
  assert set(r['surfaces'])=={'english','spanish'}
  assert r['surfaces']['english']['text']==r['text']
  if split=='train':
   assert all(r[k]==old['train'][j][k] for k in old['train'][j]);parity+=1
  else:assert key not in excluded
  assert len(r['nodes'])<=128;slots={}
  for i,k,relation,slot in r['edges']:
   assert 0<=i<len(r['nodes']) and 0<=k<len(r['nodes']) and (slot is None or 0<=slot<32)
   value=-1 if slot is None else slot
   assert (i,k) not in slots or slots[i,k]==value;slots[i,k]=value
  for language,s in r['surfaces'].items():
   assert language in ('english','spanish');text=s['text'];tok=re.findall(r'\w+|[^\w\s]',text,re.UNICODE);assert len(tok)==s['tokens'];assert sha(text.encode())==s['public_sha256']
   identities={v for k,v in r['nodes'] if k in ('ident','entity')};positions={v:tok.index(v) for v in identities};assert len(set(positions.values()))==len(positions)
   # The pinned unification family uses these unchanged proper-name/variable forms in both renderers.
   assert identities<=set(['alice','bob','carol','dave','erin','frank','A','B','C','D','E'])
   nodes=[(k,None,positions[v]) if k in ('ident','entity') else (k,v,None) for k,v in r['nodes']]
   assert all(json.dumps(v,sort_keys=True) in meta['value_vocabulary'][1:] for k,v in r['nodes'] if k not in ('ident','entity'))
   edges=sorted((i,k,rel,-1 if slot is None else slot) for i,k,rel,slot in r['edges']);target=sha(json.dumps(dict(nodes=nodes,edges=edges),sort_keys=True,separators=(',',':')).encode());assert target==s['target_sha256']
   assert text not in public or public[text]==target;public[text]=target
   vectors=[]
   for i,t in enumerate(tok):
    bits=hashlib.sha256(json.dumps(t,ensure_ascii=False,sort_keys=True).encode()).digest()[:8]
    vectors.append([float((byte>>b)&1) for byte in bits for b in range(8)]+[i/len(tok),math.sin(i),math.cos(i),0.])
   f=np.asarray(vectors,dtype=np.float32)[None];u=f.view(np.uint32);bf=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
   for precision,array in [('float32',f),('bfloat16',bf)]:
    h=sha(array.tobytes());assert h==s['actor_feature_sha256'][precision];fk=precision+':'+h;assert fk not in features or features[fk]==target;features[fk]=target
assert counts=={'train':8192,'development':512};assert len(public)==17408
out=dict(cache_sha256=meta['cache_sha256'],historical_cache_sha256=meta['config']['exclude_cache_sha256'],counts=counts,historical_train_exact=parity,unique_alpha_node_edge_keys=len(keys),distinct_public_texts=len(public),distinct_features={p:sum(k.startswith(p+':') for k in features) for p in ('float32','bfloat16')},all_target_and_feature_hashes_reconstructed=True,development_old_overlap=0,conflicting_slots=0,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Producer-independent NumPy reconstruction of all compact labels, both precision feature hashes, surface-local targets, historical English parity and semantic disjointness. Rendered prose semantics rely on pinned generator; not a language-model result.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
