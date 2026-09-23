"""Independent lexical-bijection and indexed-target equivalence, no model call."""
import argparse,gzip,hashlib,json,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();data=a.root/'s01-data';rename=a.root/'s12-rename-case-audit';m=json.loads((rename/'audit.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(data/'reserved_confirmation.jsonl.gz')==m['original_cache_sha256'];assert sha(rename/'renamed_confirmation.jsonl.gz')==m['renamed_cache_sha256'];read=lambda p:[json.loads(v)for v in gzip.open(p,'rt')];old=read(data/'reserved_confirmation.jsonl.gz');new=read(rename/'renamed_confirmation.jsonl.gz');train=read(data/'train.jsonl.gz');mapping=m['mapping'];assert len(mapping)==len(set(mapping.values()))==11;assert all(x.isupper()==y.isupper()for x,y in mapping.items());tokens=lambda s:re.findall(r'\w+|[^\w\s]',s);vocab={v for r in train for v in tokens(r['text'])};assert not set(mapping.values())&vocab;hashes=[hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).digest()[:8]for v in vocab|set(mapping.values())];assert len(hashes)==len(set(hashes));copies=0
for x,y in zip(old,new):
 assert x['seed']==y['seed'];assert x['edges']==y['edges'];assert len(x['nodes'])==len(y['nodes']);ot,nt=tokens(x['text']),tokens(y['text']);assert len(ot)==len(nt)==x['tokens'];assert nt==[mapping.get(v,v)for v in ot];assert y['public_sha256']==hashlib.sha256(y['text'].encode()).hexdigest()
 for (k,v),(kk,vv) in zip(x['nodes'],y['nodes']):
  assert k==kk
  if k in('ident','entity'):
   assert vv==mapping[v];assert [i for i,z in enumerate(ot)if z==v]==[i for i,z in enumerate(nt)if z==vv];assert v in ot and vv in nt;copies+=1
  else:assert v==vv
assert len(old)==len(new)==1024
out=dict(examples=1024,identity_nodes_verified=copies,novel_tokens=11,case_preserved=True,all_indexed_copy_targets_and_graph_edges_unchanged=True,visible_bijection_verified=True,original_sha256=m['original_cache_sha256'],renamed_sha256=m['renamed_cache_sha256'],cpu_audit_wall_seconds=time.monotonic()-t,scope='Same graph equality patterns and ordered indices under visible token bijection; no actor forward. Novel strings become novel lexical hashes, without explicit case features.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
