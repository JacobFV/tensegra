"""Independent compact confirmation target binding, without actor inference."""
import argparse,ast,base64,gzip,hashlib,json,re,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('cache',type=Path);p.add_argument('train',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic()
module=ast.parse((Path(__file__).resolve().parents[2]/'src/topoformer/thinking_language.py').read_text());constants={n.targets[0].id:ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES')}
train=[json.loads(s) for s in gzip.open(a.train,'rt')];vocab=['<unknown>']+sorted({json.dumps(v,sort_keys=True) for row in train for k,v in row['nodes'] if k not in ('ident','entity')})
cache=[json.loads(s) for s in gzip.open(a.cache,'rt')];rows=json.load(gzip.open(a.artifact,'rt'))['rows'];assert len(rows)==len(cache)==1024
for row,c in zip(rows,cache):
 assert all(row[k]==c[k] for k in ('seed','graph_sha256','semantic_sha256'));g=row['target'];n=len(c['nodes']);assert g['presence']==[i<n for i in range(128)]
 tok=re.findall(r'\w+|[^\w\s]',c['text'],re.UNICODE);kinds=[0]*128;values=[-1]*128;copy=[-1]*128
 for i,(kind,value) in enumerate(c['nodes']):
  kinds[i]=constants['KINDS'].index(kind)
  if kind in ('ident','entity'):copy[i]=tok.index(str(value))
  else:values[i]=vocab.index(json.dumps(value,sort_keys=True))
 assert g['kind']==kinds and g['value']==values and g['copy']==copy
 edges=np.zeros((128,128,len(constants['ROLES'])),bool);slots=np.full((128,128),-1)
 for i,j,r,s in c['edges']:
  edges[i,j,constants['ROLES'].index(r)]=True
  if s is not None:slots[i,j]=s
 packed=g['edges'];actual=np.unpackbits(np.frombuffer(base64.b64decode(packed['packed_b64']),dtype=np.uint8),bitorder='little')[:edges.size].reshape(edges.shape)
 assert np.array_equal(actual,edges) and np.array_equal(g['slots'],slots)
assert not ({x['semantic_sha256'] for x in cache}&{x['semantic_sha256'] for x in train})
result=dict(examples=len(rows),all_targets_reconstructed=True,train_overlap=0,cache_sha256=hashlib.sha256(a.cache.read_bytes()).hexdigest(),cpu_audit_wall_seconds=time.monotonic()-tick,scope='English surface-local copy and every graph target reconstructed independently from frozen compact cache; no neural inference.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
