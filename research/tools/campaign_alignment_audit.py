"""Independent S08 frozen copy logits, occurrence semantics and edge association."""
import argparse,gzip,json,hashlib,time,re,collections
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import edge_set
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('semantic_root');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();root=Path(a.root);base=Path(a.semantic_root);x=json.load(gzip.open(root/'predictions.json.gz','rt'));assert hashlib.sha256((root/'copy-logits.npz').read_bytes()).hexdigest()==x['logits_sha256'];z=np.load(root/'copy-logits.npz');logits=z['logits'];offsets=z['offsets'];data={}
for split in ('train','development'):
 for line in gzip.open(base/'s01-data'/f'{split}.jsonl.gz','rt'):
  r=json.loads(line);data[split,r['seed']]=r
archives={};summary={};nodes=0
for m in x['models']:
 # Archive path determined by public arm label via frozen manifest receipt.
 name=m['arm'];folder='s01-current-n8192-dev201'if 's01'in name.lower() else 's07-occurrence-n8192-dev201';raw=(base/folder/'evaluation-u8192.json.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==m['archive_sha256'];arc=json.loads(gzip.decompress(raw));archives[name]={r['seed']:r for key in ('train_rows','rows')for r in arc[key]}
for index,r in enumerate(x['records']):
 row=data[r['split'],r['seed']];tok=re.findall(r'\w+|[^\w\s]',row['text'],re.UNICODE);ids=np.array([tok.index(t)for t in tok]);assert ids.tolist()==r['token_identity'];assert [r['logit_start'],r['logit_stop']]==offsets[index:index+2].tolist();arr=logits[r['logit_start']:r['logit_stop']].reshape(r['copy_shape']);old=archives[r['arm']][r['seed']];gold=old['target'];pred=old['raw'];edges=edge_set(dict(pred,edges=old['calibrated_edges']));edges={e for e in edges if pred['presence'][e[0]]and pred['presence'][e[1]]};ge=edge_set(gold)
 queues=collections.defaultdict(collections.deque)
 for j,t in enumerate(tok):queues[t].append(j)
 occ={}
 for i,(kind,value)in enumerate(row['nodes']):
  if kind=='ident':occ[i]=queues[value].popleft()
 for n in r['nodes']:
  i=n['node'];first=gold['copy'][i];occurrence=occ.get(i,first);assert n['first_target']==first and n['occurrence_target']==occurrence
  raw=arr[i];prob=np.exp(raw.astype(np.float64)-float(raw.max()));prob/=prob.sum();arg=int(raw.argmax());assert n['raw_argmax']==arg;assert n['first_correct']==(arg==first);assert n['occurrence_correct']==(arg==occurrence);assert n['canonical_correct']==(ids[arg]==ids[first]);assert pred['copy'][i]==int(ids[arg])
  vals={'first_mass':prob[first],'occurrence_mass':prob[occurrence],'identity_mass':prob[ids==ids[first]].sum(),'entropy':-(prob*np.log(np.maximum(prob,1e-30))).sum()}
  for k,v in vals.items():assert abs(n[k]-v)<2e-6,(k,n[k],v)
  required={(s,t,rr)for s,t,rr in ge if t==i and gold['slots'][s][t]>=0};correct=sum(e in edges and pred['slots'][e[0]][e[1]]==gold['slots'][e[0]][e[1]]for e in required)
  assert n['required_incoming_ordered_edges']==len(required)and n['correct_incoming_ordered_edges']==correct
  key=f"{r['arm']}/{r['split']}/{n['kind']}";v=summary.setdefault(key,{'nodes':0,'occurrence_correct':0,'canonical_correct':0,'aligned_required':0,'aligned_correct':0});v['nodes']+=1;v['occurrence_correct']+=n['occurrence_correct'];v['canonical_correct']+=n['canonical_correct']
  if n['occurrence_correct']:v['aligned_required']+=len(required);v['aligned_correct']+=correct
  nodes+=1
out=dict(records=len(x['records']),nodes=nodes,summary=summary,cpu_audit_wall_seconds=time.monotonic()-start,scope='Frozen archived copy logits independently scored against public occurrence alignment and gold edge targets; association is not causal proof.')
Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(out)
