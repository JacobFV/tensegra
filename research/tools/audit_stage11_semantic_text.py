"""Independent public-text graph count and calibration reconstruction (stdlib)."""
import argparse,base64,gzip,hashlib,itertools,json,re,subprocess
from pathlib import Path
from collections import Counter

def edge_set(graph):
 code=graph['edges'];a,b,r=code['shape'];payload=base64.b64decode(code['packed_b64']);assert code['bitorder']=='little'
 result=set()
 for index,byte in enumerate(payload):
  while byte:
   bit=byte&-byte;k=index*8+bit.bit_length()-1
   if k<a*b*r:result.add((k//(b*r),(k//r)%b,k%r))
   byte-=bit
 return result
def components(pred,gold):
 p=pred['presence'];q=gold['presence'];ge=edge_set(gold);pe={e for e in edge_set(pred) if p[e[0]] and p[e[1]]}
 def counts(a,b):
  tp=len(a&b);np=len(a);ng=len(b);return dict(true_positive=tp,predicted_count=np,gold_count=ng,precision=tp/np if np else 0.,recall=tp/ng if ng else 0.,f1=2*tp/(np+ng) if np+ng else 1.)
 present={i for i,v in enumerate(q)if v};copied={i for i,v in enumerate(gold['copy'])if v>=0}
 ordered_pred={(i,j,r,pred['slots'][i][j])for i,j,r in pe if pred['slots'][i][j]>=0};ordered_gold={(i,j,r,gold['slots'][i][j])for i,j,r in ge if gold['slots'][i][j]>=0}
 correct=lambda key,mask:all(pred[key][i]==gold[key][i] for i in mask)
 exact=0 not in gold['value'] and p==q and correct('kind',present) and pe==ge and correct('value',{i for i,v in enumerate(gold['value'])if v>=0}) and correct('copy',copied) and all(pred['slots'][i][j]==gold['slots'][i][j]for i,j,r in ge)
 return dict(node=counts({i for i,v in enumerate(p)if v},present),typed_edge=counts(pe,ge),ordered_edge=counts(ordered_pred,ordered_gold),node_type_accuracy=sum(pred['kind'][i]==gold['kind'][i]for i in present)/len(present),identity_copy_accuracy=sum(pred['copy'][i]==gold['copy'][i]for i in copied)/len(copied)if copied else 1.,entity_equivalence=sum((pred['copy'][i]==pred['copy'][j])==(gold['copy'][i]==gold['copy'][j])for i in copied for j in copied)/len(copied)**2 if copied else 1.,semantic_equivalence=float(exact))
def same(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[same(a[k],b[k])for k in a]
 else:assert abs(a-b)<1e-6,(a,b)
def threshold(pairs):
 pairs=sorted(pairs);err=sum(not y for _,y in pairs);best=(err,pairs[0][0]-1)
 for score,g in itertools.groupby(pairs,key=lambda x:x[0]):
  ys=[y for _,y in g];err+=sum(ys)-(len(ys)-sum(ys))
  if err<best[0]:best=(err,score)
 return best

def audit(path,source_ref=None):
 with gzip.open(path,'rt')as f:x=json.load(f)
 assert hashlib.sha256(json.dumps(x['config'],sort_keys=True).encode()).hexdigest()==x['config_sha256']
 if source_ref:
  for name,h in x['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',source_ref+':src/topoformer/'+name],cwd=Path(__file__).resolve().parents[2])).hexdigest()==h
 total=thresholds=0;final=[]
 target_graphs=[row['target'] for row in x['runs'][0]['curves'][0]['rows']]
 def mode(values):
  c=Counter(values);return min(c,key=lambda v:(-c[v],v))
 freq={key:[mode(v)for v in zip(*(g[key]for g in target_graphs))]for key in ('presence','kind','value','copy')}
 size=len(freq['presence']);freq['slots']=[[mode([g['slots'][i][j]for g in target_graphs])for j in range(size)]for i in range(size)]
 shape=target_graphs[0]['edges']['shape'];counts=Counter(e for g in target_graphs for e in edge_set(g));payload=bytearray((shape[0]*shape[1]*shape[2]+7)//8)
 for (i,j,r),count in counts.items():
  if count>len(target_graphs)/2:
   k=(i*shape[1]+j)*shape[2]+r;payload[k//8]|=1<<(k%8)
 freq['edges']=dict(shape=shape,bitorder='little',packed_b64=base64.b64encode(payload).decode())
 for g,record in zip(target_graphs,x['frequency']):same(components(freq,g),record['metrics'])
 for run in x['runs']:
  assert run['width']==1024
  for curve in run['curves']:
   assert len(curve['rows'])==x['config']['graphs'];raw=cal=0
   for row,data in zip(curve['rows'],curve['calibration_data']):
    assert row['graph_seed']==data['graph_seed'];gold=row['target'];p=row['raw']['presence'];n=len(p);ge=edge_set(gold)
    tokens=re.findall(r'\w+|[^\w\s]',row['public_text'],re.UNICODE)or['']
    assert all(0<=i<len(tokens)and tokens.index(tokens[i])==i for i in row['raw']['copy'])
    assert all(i<0 or (i<len(tokens)and tokens.index(tokens[i])==i)for i in gold['copy'])
    assert data['pairs']==[[i,j]for i in range(n)for j in range(n)if p[i]and p[j]]
    for (i,j),ys in zip(data['pairs'],data['targets']):assert ys==[(i,j,r)in ge for r in range(len(ys))]
    for policy in ('raw','calibrated'):
     got=components(row[policy],gold);same(got,row[policy+'_metrics']);total+=1
     if policy=='raw':raw+=got['semantic_equivalence']
     else:cal+=got['semantic_equivalence']
     pred={(i,j,r)for (i,j),scores in zip(data['pairs'],data['scores'])for r,v in enumerate(scores)if v>(0 if policy=='raw'else curve['calibration'][r]['threshold'])}
     assert pred=={e for e in edge_set(row[policy])if p[e[0]]and p[e[1]]}
   assert raw==curve['raw_exact']and cal==curve['calibrated_exact']
   for r,record in enumerate(curve['calibration']):
    pairs=[(a[r],bool(b[r]))for d in curve['calibration_data']for a,b in zip(d['scores'],d['targets'])]
    if not pairs:assert record['threshold']==0 and record['positive']==record['negative']==0
    else:
     errors,cut=threshold(pairs);assert errors==record['train_errors']and abs(cut-record['threshold'])<1e-5
     assert sum(y for _,y in pairs)==record['positive']and sum(not y for _,y in pairs)==record['negative']
    thresholds+=1
   if curve['update']==x['config']['updates']:final.append(dict(seed=run['seed'],raw=int(raw),calibrated=int(cal),graphs=len(curve['rows'])))
 return dict(source_ref_verified=source_ref,frequency_graphs_verified=len(target_graphs),graph_decisions_verified=total,threshold_records_verified=thresholds,final=final,all_prescribed_seeds_present=sorted(r['seed']for r in x['runs'])==sorted(x['config']['seeds']),restricted_final_fit=all(r['calibrated']==r['graphs']for r in final)and len(final)==len(x['config']['seeds']),scope='Fixed public-text acquisition only; independent archived prediction/score reconstruction, not inference.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('path',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--source-ref');a=p.parse_args();out=audit(a.path,a.source_ref);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
