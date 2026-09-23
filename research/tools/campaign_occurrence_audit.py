"""Independent visible-position alignment audit for the narrow S07 corpus."""
import argparse,collections,gzip,hashlib,json,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('receipt');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();old=json.loads(Path(a.receipt).read_text());out={}
for split in ('train','development'):
 path=Path(a.data)/(split+'.jsonl.gz');h=hashlib.sha256();graphs=occs=changed=0
 assert hashlib.sha256(path.read_bytes()).hexdigest()==old['splits'][split]['data_sha256']
 for line in gzip.open(path,'rt'):
  row=json.loads(line);tok=re.findall(r'\w+|[^\w\s]',row['text'],re.UNICODE);positions=collections.defaultdict(collections.deque)
  for i,t in enumerate(tok):positions[t].append(i)
  mapping=[]
  for i,(kind,value) in enumerate(row['nodes']):
   if kind=='ident':
    assert positions[value];j=positions[value].popleft();mapping.append((i,j));changed+=j!=tok.index(value)
  assert [j for _,j in mapping]==sorted(j for _,j in mapping)
  names={v for k,v in row['nodes'] if k=='ident'};assert all(not positions[n] for n in names)
  h.update(json.dumps([row['semantic_sha256'],mapping],separators=(',',':')).encode());graphs+=1;occs+=len(mapping)
 expected=old['splits'][split];assert h.hexdigest()==expected['alignment_sha256'];assert changed==expected['counts']['occurrence_target_differs_from_identity_first']
 out[split]=dict(graphs=graphs,occurrences=occs,changed_from_first_identity=changed,alignment_sha256=h.hexdigest())
r=dict(splits=out,cpu_audit_wall_seconds=time.monotonic()-start,scope='TRAIN/DEV ordered English unification occurrences only. Programmed privileged span targets, no claim this mapping survives arbitrary renderer reorderings.')
Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(r)
