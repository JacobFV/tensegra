"""Independent frozen learned-node edge-refit metrics and calibration audit."""
import argparse,hashlib,json,subprocess,time,base64
from pathlib import Path
import numpy as np
from campaign_semantics_audit import load,sha,cutoff,components,same,edge_set
def active_graph(graph):
 code=graph['edges'];bits=np.unpackbits(np.frombuffer(base64.b64decode(code['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(code['shape'])].reshape(code['shape']);presence=np.asarray(graph['presence']);bits&=presence[:,None,None]&presence[None,:,None];packed=base64.b64encode(np.packbits(bits.reshape(-1),bitorder='little').tobytes()).decode();return dict(graph,edges=dict(code,packed_b64=packed))

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();m=load(a.root/'manifest.json.gz');c=m['config'];assert hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==m['config_sha256']
for name,want in m['dependencies_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',f'{a.source_ref}:src/topoformer/{name}'])).hexdigest()==want
assert m['parent_checkpoint_sha256']==c['parent_checkpoint_sha256'];assert m['frozen_before_sha256']==m['frozen_after_sha256'];assert all(v==0 for v in m['cache_parity_max_error'].values());baseline={};results=[];thresholds_checked=0;graphs=0
for point in m['curves']:
 x=load(a.root/point['artifact']);assert sha(a.root/point['artifact'])==point['sha256'];calfile=a.root/f"calibration-u{x['update']}.npz";assert sha(calfile)==x['calibration_sha256'];cal={key:value for key,value in np.load(calfile).items()}
 for r,record in enumerate(x['calibration']):
  truth=cal['targets'][:,r].astype(bool);cut,error=cutoff(cal['scores'][:,r],truth);assert abs(cut-record['threshold'])<1e-6 and error==record['train_errors'];assert int(truth.sum())==record['positive'];thresholds_checked+=1
 for group,rows in x['rows'].items():
  for j,row in enumerate(rows):
   key=group,row['seed'];fixed={k:v for k,v in row['raw'].items()if k!='edges'}
   if key not in baseline:baseline[key]=fixed
   else:assert fixed==baseline[key]
   if group=='train':
    lo,hi=cal['offsets'][j:j+2];presence=np.asarray(row['raw']['presence']);assert np.array_equal(cal['pairs'][lo:hi],np.argwhere(presence[:,None]&presence[None,:]));gold=edge_set(row['target']);want=np.asarray([[(int(i),int(k),r)in gold for r in range(cal['targets'].shape[1])]for i,k in cal['pairs'][lo:hi]]);assert np.array_equal(want,cal['targets'][lo:hi])
   for policy,pred in [('raw',row['raw']),('calibrated',dict(row['raw'],edges=row['calibrated_edges']))]:
    pred=active_graph(pred);measured=components(pred,row['target']);same(measured,row[policy+'_metrics']);assert ({e for e in edge_set(pred)if pred['presence'][e[0]]and pred['presence'][e[1]]}==edge_set(row['target']))==row[policy+'_exact_edges'];graphs+=1
  for policy in ('raw','calibrated'):
   summary=x['summary'][group][policy];assert summary==dict(exact_edges=sum(r[policy+'_exact_edges']for r in rows),exact_graphs=sum(r[policy+'_metrics']['semantic_equivalence']for r in rows),examples=len(rows));assert summary==point['summary'][group][policy]
 results.append(dict(update=x['update'],summary=x['summary']))
out=dict(source_ref=a.source_ref,graphs_verified=graphs,threshold_records_verified=thresholds_checked,curves=results,unchanged_nonedge_predictions=True,cpu_audit_wall_seconds=time.monotonic()-start,scope='Archived edge/wholegraph metrics and TRAIN-only threshold fitting. Cache parity/frozen-state hashes checked as producer records; independent checkpoint bytes/cache replay separate. Fixed-set diagnostic only, no peak substitution.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
