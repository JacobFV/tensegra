"""Independent compact S01 graph metrics and TRAIN threshold reconstruction."""
import argparse,ast,gzip,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from audit_stage11_semantic_text import components,same,edge_set

def load(p):
 with (gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cutoff(scores,truth):
 if not len(scores):return 0.,0
 order=np.argsort(scores,kind='stable');v=scores[order];y=truth[order].astype(np.int64);ends=np.flatnonzero(np.r_[v[1:]!=v[:-1],True]);errors=np.r_[int((~truth).sum()),int((~truth).sum())+np.cumsum(2*y-1)[ends]];thresholds=np.r_[v[:1]-1,v[ends]];j=int(errors.argmin());return float(thresholds[j]),int(errors[j])
def audit(root,ref,data):
 started=time.monotonic();m=load(root/'manifest.json.gz');cfg=m['config'];assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['config_sha256']
 for name,want in m['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{ref}:src/topoformer/{name}'])).hexdigest()==want
 roles=next(ast.literal_eval(n.value)for n in ast.parse((Path(__file__).resolve().parents[2]/'src/topoformer/thinking_language.py').read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ROLES'for t in n.targets))
 train={r['seed']:r for r in (json.loads(line)for line in gzip.open(data/'train.jsonl.gz','rt'))};cells=[];nthresholds=0
 for point in m['curves']:
  p=root/point['artifact'];assert sha(p)==point['sha256'];x=load(p);npz=root/x['calibration_data_artifact'];assert sha(npz)==x['calibration_data_sha256'];cal=np.load(npz)
  for r,record in enumerate(x['calibration']):
   truth=cal['targets'][:,r].astype(bool);cut,error=cutoff(cal['scores'][:,r],truth);assert abs(cut-record['threshold'])<1e-6 and error==record['train_errors'];assert int(truth.sum())==record['positive']and int((~truth).sum())==record['negative'];nthresholds+=1
  for j,record in enumerate(x['calibration_records']):
   a,b=record['start'],record['stop'];assert [a,b]==cal['offsets'][j:j+2].tolist();g=train[record['seed']];edges={(i,k,roles.index(role))for i,k,role,slot in g['edges']}
   labels=np.array([[(int(i),int(k),r)in edges for r in range(len(roles))]for i,k in cal['pairs'][a:b]],dtype=bool).reshape(b-a,len(roles));assert np.array_equal(labels,cal['targets'][a:b])
  for row in x['rows']:
   gold=row['target'];calgraph=dict(row['raw'],edges=row['calibrated_edges'])
   for policy,pred in [('raw',row['raw']),('calibrated',calgraph)]:same(components(pred,gold),row[policy+'_metrics'])
  for policy in ('raw','calibrated'):
   rows=x['rows'];key=policy+'_metrics';summary=point['dev_'+policy]
   assert sum(r[key]['semantic_equivalence']for r in rows)==summary['exact']and len(rows)==summary['examples']
   assert abs(sum(r[key]['identity_copy_accuracy']for r in rows)/len(rows)-summary['copy'])<1e-6
   for kind in ('typed_edge','ordered_edge'):
    tp=sum(r[key][kind]['true_positive']for r in rows);den=sum(r[key][kind]['predicted_count']+r[key][kind]['gold_count']for r in rows);assert abs(2*tp/max(1,den)-summary[kind+'_f1'])<1e-6
  cells.append(dict(update=x['update'],graphs=len(x['rows']),raw_exact=point['dev_raw']['exact'],calibrated_exact=point['dev_calibrated']['exact']))
 return dict(cpu_audit_wall_seconds=time.monotonic()-started,source_ref=ref,curves=cells,threshold_records_verified=nthresholds,scope='Independent archived dev graph metrics, compact TRAIN labels/thresholds and micro aggregates. No neural inference.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=audit(a.root,a.source_ref,a.data);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
