"""Independent S05 programmed-reference reconstruction from packed predictions."""
import argparse,base64,gzip,hashlib,json,time
from pathlib import Path
from audit_stage11_semantic_text import components,same,edge_set
p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic();x=json.load(gzip.open(a.result,'rt'));raw=Path(a.archive).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['archive_sha256'];archive=json.loads(gzip.decompress(raw));records=iter(x['rows']);out={};checked=0
for split,key in [('train','train_rows'),('development','rows')]:
 for row in archive[key]:
  gold=row['target'];ge=edge_set(gold)
  for mode in ('raw','calibrated'):
   r=next(records);assert (r['split'],r['mode'],r['seed'])==(split,mode,row['seed'])
   pred=dict(row['raw']);
   if mode=='calibrated':pred['edges']=row['calibrated_edges']
   present=pred['presence'];ident=[i for i,v in enumerate(present)if v and pred['kind'][i]==5];entities=[i for i,v in enumerate(present)if v and pred['kind'][i]==1 and pred['copy'][i]>=0]
   relation={(i,j)for i in ident for j in entities if pred['copy'][i]>=0 and pred['copy'][i]==pred['copy'][j]}
   assert sorted(map(list,relation))==r['programmed_ref_indices']
   for arm in ('learned','programmed','gold_reference'):
    edges=edge_set(pred);refs={(i,j)for i,j,k in edges if k==2} if arm=='learned' else relation if arm=='programmed' else {(i,j)for i,j,k in ge if k==2}
    edges={e for e in edges if e[2]!=2}|{(i,j,2)for i,j in refs};shape=pred['edges']['shape'];n,b,rels=shape;packed=bytearray((n*b*rels+7)//8)
    for i,j,k in edges:
     z=(i*b+j)*rels+k;packed[z//8]|=1<<(z%8)
    g=dict(pred,edges=dict(shape=shape,bitorder='little',packed_b64=base64.b64encode(packed).decode()));metrics=components(g,gold);same(metrics,r['arms'][arm]['metrics'])
    active={(i,j)for i,j in refs if present[i] and present[j]};gr={(i,j)for i,j,k in ge if k==2};count=dict(true_positive=len(active&gr),predicted=len(active),gold=len(gr),exact=active==gr);assert count==r['arms'][arm]['reference']
    k=f'{split}/{mode}/{arm}';v=out.setdefault(k,dict(examples=0,complete=0,reference_exact=0));v['examples']+=1;v['complete']+=int(metrics['semantic_equivalence']);v['reference_exact']+=int(count['exact']);checked+=1
assert next(records,None) is None
result=dict(graph_decisions=checked,results=out,cpu_audit_wall_seconds=time.monotonic()-start,scope='Posthoc programmed equality on predicted identity fields; not learned reference acquisition.');Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
