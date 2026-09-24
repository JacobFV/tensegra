"""Independent closed-main S20 workspace targets/calibration/metrics, CPU only."""
import argparse,ast,base64,collections,gzip,hashlib,json,re,time,sys
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('repo',type=Path);parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();root=args.repo;review=Path(__file__).resolve().parents[2];sys.path.insert(0,str(review/'research/tools'));from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same
start=time.monotonic();archive=args.archive;run=archive/'s20-main-v1';load=lambda p:json.load(gzip.open(p,'rt'));read=lambda p:list(map(json.loads,gzip.open(p,'rt')));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();base=root/'research/results/campaign-01/semantics';m=load(run/'manifest.json.gz');receipt=json.loads((archive/'s20-main-launch.occupancy.json').read_text());assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['gpu_processes']=='';assert receipt['config_sha256']=='4e648a9a274a7cabcbb84c0b2e6e6a2dcbf37925e2146b0758702086245f6c8b';assert [(r['seed'],r['arm']) for r in m['results']]==[(seed,arm) for seed in (701,702,703) for arm in ('original','context','record')];assert all(r['added_update']==4096 for r in m['results']);assert m['config']==json.loads((root/'configs/campaign-s20-main-frozen-v1.json').read_text())
# No confirmation file is opened until closed complete-nine manifest/receipt guards above pass.
vocab=json.loads((base/'s15-bindings/original-data-audit.json').read_text())['value_vocabulary'];const={}
for n in ast.parse((root/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):const[n.targets[0].id]=ast.literal_eval(n.value)
code=ast.parse((review/'research/tools/campaign_semantic_shape_main_audit.py').read_text());node=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='gold');exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-gold>','exec'))
rows=read(base/'s15-shape-cache-v2/confirmation.jsonl.gz');dev=[r for a in (3,4) for f in (3,4) for r in rows if (r['arity'],r['facts'])==(a,f)];assert len(dev)==2048 and len({r['semantic_sha256'] for r in dev})==2048;mixed=read(base/'s15-shape-cache-v2/train_mixed.jsonl.gz');sel=json.loads((base/'s17-calibration-selection/selection.json').read_text())['mixed'];train=[mixed[r['index']] for r in sel];graphs=cuts=0;flags={str(seed):{} for seed in (701,702,703)}
for result in m['results']:
 if result['arm']=='record':continue
 arm=result['arm'];seed=result['seed'];folder=run/f'seed-{seed}'/arm;flags[str(seed)][arm]={'raw':{},'matched':{}};assert result==load(folder/'manifest.json.gz');entry=result['matched'];path=folder/result['matched_directory']/entry['artifact'];assert sha(path)==entry['sha256'];d=load(path);npz=path.parent/d['calibration_data_artifact'];assert sha(npz)==d['calibration_data_sha256'];cal=np.load(npz);assert d['thresholds']==[x['threshold'] for x in d['calibration']]
 for j,r in enumerate(d['calibration']):
  cut,err=cutoff(cal['scores'][:,j],cal['targets'][:,j].astype(bool));assert abs(cut-r['threshold'])<1e-6 and err==r['train_errors'];assert r['positive']==int(cal['targets'][:,j].sum()) and r['negative']==len(cal['targets'])-r['positive'];cuts+=1
 for i,(row,source) in enumerate(zip(d['train_rows'],train,strict=True)):
  assert row['seed']==source['seed'] and row['target']==gold(source);lo,hi=cal['offsets'][i:i+2];pairs=cal['pairs'][lo:hi];presence=np.asarray(row['raw']['presence']);assert np.array_equal(pairs,np.argwhere(presence[:,None]&presence[None,:]));edge=row['target']['edges'];target=np.unpackbits(np.frombuffer(base64.b64decode(edge['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge['shape'])].reshape(edge['shape']);assert np.array_equal(cal['targets'][lo:hi],target[pairs[:,0],pairs[:,1]])
  for edge_data,threshold in [(row['raw']['edges'],0),(row['calibrated_edges'],np.asarray(d['thresholds']))]:
   observed=np.unpackbits(np.frombuffer(base64.b64decode(edge_data['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge_data['shape'])].reshape(edge_data['shape']);assert np.array_equal(observed[pairs[:,0],pairs[:,1]],cal['scores'][lo:hi]>threshold)
  for label,pred in [('raw',row['raw']),('calibrated',{**row['raw'],'edges':row['calibrated_edges']})]:same(components(pred,row['target']),d['train_metrics'][i][label])
  graphs+=1
 for row,source in zip(d['rows'],dev,strict=True):
  assert row['semantic_sha256']==source['semantic_sha256'] and row['target']==gold(source)
  for label,pred in [('raw',row['raw']),('calibrated',{**row['raw'],'edges':row['calibrated_edges']})]:
   got=components(pred,row['target']);same(got,row[label+'_metrics']);cell=f"{source['arity']}x{source['facts']}";policy='raw' if label=='raw' else 'matched';flags[str(seed)][arm][policy].setdefault(cell,{})[row['semantic_sha256']]=bool(got['semantic_equivalence'])
  graphs+=1
 for label,rr,key in [('dev_raw',d['rows'],'raw_metrics'),('dev_calibrated',d['rows'],'calibrated_metrics'),('train_raw',d['train_metrics'],'raw'),('train_calibrated',d['train_metrics'],'calibrated')]:
  stat=entry[label];assert stat['examples']==len(rr) and stat['exact']==sum(r[key]['semantic_equivalence'] for r in rr);assert abs(stat['copy']-sum(r[key]['identity_copy_accuracy'] for r in rr)/len(rr))<1e-12
  for kind in ('typed_edge','ordered_edge'):
   tp=sum(r[key][kind]['true_positive'] for r in rr);den=sum(r[key][kind]['predicted_count']+r[key][kind]['gold_count'] for r in rr);assert abs(stat[kind+'_f1']-2*tp/max(1,den))<1e-12
 assert abs(sum(entry['timing'].values())-entry['whole_evaluation_seconds'])<.01
for rel,rec in json.loads((archive/'archive-inventory.json').read_text()).items():assert sha(archive/rel)==rec['sha256'] and (archive/rel).stat().st_size==rec['bytes']
r=dict(status='pass',seconds=time.monotonic()-start,graphs=graphs,global_cutoffs=cuts,flags=flags,calibration_npz_labels_masks_verified=True,archive_hashes_verified=True,input_sha256={str(q)[str(q).index('research/'):]:sha(q) for q in archive.rglob('*') if q.is_file()},scope='Closed complete S20 main only; all6workspace arms raw/calibrated targets/metrics/cutoffs independently reconstructed; allflags retained for paired bootstrap. No actor/model calls.')
args.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ('input_sha256','flags')})
