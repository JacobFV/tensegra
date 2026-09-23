"""Independent set-based public-schema decoding and error reconstruction."""
import argparse,base64,collections,gzip,hashlib,json,time,subprocess
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import edge_set,components,same
K=('scope','entity','token','str','num','ident','nil','pred','rel','node','tuple','list','record','app')
R=('contains','declares','refers_to','argument','item','binds','binding_scope','field:query','field:substitution','field:pattern','field:fact','field:facts','field:scene')
def allowed(i,j,r,k):
 a,b=k[i],k[j];term=lambda v:v not in(0,1)
 return ((a==0 and term(b))if r==0 else (a==0 and b==1)if r==1 else(a==5 and b==1)if r==2 else(a in(7,8,9)and term(b))if r==3 else(a in(10,11)and term(b))if r==4 else(term(a)and term(b))if r==5 else(a==7 and b==0)if r==6 else(a in(12,13)and term(b)))
def packed(edges,shape):
 n,m,r=shape;bits=bytearray((n*m*r+7)//8)
 for i,j,k in edges:
  t=(i*m+j)*r+k;bits[t//8]|=1<<(t%8)
 return dict(shape=shape,bitorder='little',packed_b64=base64.b64encode(bits).decode())
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--result-name',default='s10-schema-decode-compact');a=p.parse_args();t=time.monotonic();x=json.load(gzip.open(a.root/a.result_name/'predictions.json.gz'));assert hashlib.sha256(subprocess.check_output(['git','show','2085053:src/topoformer/campaign_semantics_contract_decode.py'])).hexdigest()==x['source_sha256'];indexed={(r['checkpoint'],r['split'],r['policy'],r['variant'],r['seed']):r for r in x['rows']};sums=collections.defaultdict(collections.Counter);verified=0
for arm in x['config']['arms']:
 path=a.root/Path(arm['archive']).parent.name/Path(arm['archive']).name;assert hashlib.sha256(path.read_bytes()).hexdigest()==x['archive_sha256'][arm['name']];archive=json.load(gzip.open(path))
 for split,key in [('train','train_rows'),('development','rows')]:
  for row in archive[key]:
   gold=row['target'];ge=edge_set(gold)
   for policy in('raw','calibrated'):
    pred=row['raw']if policy=='raw'else dict(row['raw'],edges=row['calibrated_edges']);pe=edge_set(pred);active={i for i,v in enumerate(pred['presence'])if v};base={e for e in pe if e[0]in active and e[1]in active};old=base^ge;k=pred['kind'];scopes={i for i in active if k[i]==0}
    for variant in('baseline','schema','bookkeeping','combined'):
     mask=variant in('schema','combined');book=variant in('bookkeeping','combined');es=set(pe);slots=[v[:]for v in pred['slots']]
     if mask:es={e for e in es if e[0]in active and e[1]in active and allowed(*e,k)}
     if book:
      if len(scopes)==1:
       s=next(iter(scopes));es={e for e in es if e[2]not in(0,1)};es|={(s,j,1 if k[j]==1 else 0)for j in active if k[j]!=0}
      es={e for e in es if e[2]!=2};es|={(i,j,2)for i in active for j in active if k[i]==5 and k[j]==1 and pred['copy'][i]>=0 and pred['copy'][i]==pred['copy'][j]}
     structural={(i,j)for i,j,r in es if r>=3};bp={(i,j)for i,j,r in es if r<3}
     if mask or book:
      for i,j in bp-structural:slots[i][j]=-1
     record=indexed[arm['name'],split,policy,variant,row['seed']];assert record['edge_flip_count']==len(pe^es);assert record['slot_change_count']==int((np.asarray(slots)!=np.asarray(pred['slots'])).sum());dense=np.zeros(pred['edges']['shape'],dtype=np.bool_)
     if es:dense[tuple(np.asarray(sorted(es)).T)]=True
     assert hashlib.sha256(dense.tobytes()).hexdigest()==record['edge_sha256'];assert hashlib.sha256(np.asarray(slots,dtype=np.int64).tobytes()).hexdigest()==record['slot_sha256']
     out=dict(pred,edges=packed(es,pred['edges']['shape']),slots=slots);got=components(out,gold);same(got,record['metrics']);new={e for e in es if e[0]in active and e[1]in active}^ge;assert record['removed_errors']==len(old-new)and record['introduced_errors']==len(new-old);baseexact=components(pred,gold)['semantic_equivalence'];assert record['repair']==bool(got['semantic_equivalence']and not baseexact);assert record['regression']==bool(baseexact and not got['semantic_equivalence']);key2='/'.join((arm['name'],split,policy,variant));sums[key2].update(exact=int(got['semantic_equivalence']),removed=record['removed_errors'],introduced=record['introduced_errors'],examples=1);verified+=1
out=dict(source_ref='2085053',source_sha256=x['source_sha256'],records_verified=verified,summary={k:dict(v)for k,v in sums.items()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Set-based predicted-only schema and bookkeeping reconstruction. Programmed compiler guarantee, not learned semantics; all archived raw/calibrated variants retained.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(verified)
