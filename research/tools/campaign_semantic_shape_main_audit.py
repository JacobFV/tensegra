"""Independent S15 main fixed public targets, raw metrics and TRAIN calibration."""
import argparse,ast,base64,gzip,hashlib,json,math,re,time
from pathlib import Path
import numpy as np
from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same,edge_set
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--arm',choices=('control','mixed'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();base=a.repo/'research/results/campaign-01/semantics';root=base/f's15-{a.arm}-main-v3';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));read=lambda p:[json.loads(s) for s in gzip.open(p,'rt')];m=load(root/'manifest.json.gz');cfg=m['config'];dev=read(base/'s15-shape-cache-v2/development.jsonl.gz');train=read(base/'s01-data/train.jsonl.gz')[:128];vocab=json.loads((base/'s15-bindings/original-data-audit.json').read_text())['value_vocabulary'];const={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):const[n.targets[0].id]=ast.literal_eval(n.value)
assert cfg==json.loads((a.repo/f'configs/campaign-s15-{a.arm}-main-frozen-v3.json').read_text());assert cfg['job']=='main' and cfg['added_updates']==4096 and cfg['checkpoints']==[0,1024,2048,4096];assert m['added_presentations']==32768 and m['visits']==[8]*4096 and m['common_presentations']==16384 and m['inherited_optimizer_steps']==[24576.]
assert [r['added_update'] for r in m['losses']]==list(range(128,4097,128));assert all(math.isfinite(v) for r in m['losses'] for v in r['parts'].values());assert [p['added_update'] for p in m['curves']]==[0,1024,2048,4096]
for group in ('source_sha256','launch_source_sha256'):
 for name,h in cfg[group].items():assert sha(a.repo/'src/topoformer'/name)==h
for split,h in cfg['cache_sha256'].items():assert sha(base/'s15-shape-cache-v2'/(split+'.jsonl.gz'))==h
assert sha(base/'s01-data/train.jsonl.gz')==cfg['calibration_cache_sha256'];assert sha(base/'s15-bindings/original-data-audit.json')==cfg['data_audit_sha256']
occ=json.loads((root/f's15-{a.arm}-main-v3.occupancy.json').read_text());assert occ['exit_code']==0 and not occ['timed_out'] and occ['cap_seconds']==1100;assert occ['config_sha256']==sha(a.repo/f'configs/campaign-s15-{a.arm}-main-frozen-v3.json');assert occ['wrapper_sha256']==cfg['launch_source_sha256']['campaign_semantics_shape_launch.py'];assert occ['primary_matrix']['parent_checkpoint_sha256']==cfg['parent_checkpoint_sha256']
parent=load(base/'s11-lr-decay-n8192-196k-dev201/evaluation-u24576.json.gz');assert sha(base/'s11-lr-decay-n8192-196k-dev201/evaluation-u24576.json.gz')==cfg['parent_evaluation_sha256']
# Reconstruct dense packed targets directly from already independently audited cache.
def gold(row):
 n=128;tok=re.findall(r'\w+|[^\w\s]',row['text']);g=dict(presence=[i<len(row['nodes']) for i in range(n)],kind=[0]*n,value=[-1]*n,copy=[-1]*n,slots=[[-1]*n for _ in range(n)]);edges=np.zeros((n,n,len(const['ROLES'])),dtype=np.uint8)
 for i,(kind,value) in enumerate(row['nodes']):
  g['kind'][i]=const['KINDS'].index(kind)
  if kind in ('ident','entity'):g['copy'][i]=tok.index(value)
  else:g['value'][i]=vocab.index(json.dumps(value,sort_keys=True))
 for i,j,r,s in row['edges']:
  edges[i,j,const['ROLES'].index(r)]=1
  if s is not None:g['slots'][i][j]=s
 g['edges']=dict(shape=list(edges.shape),bitorder='little',packed_b64=base64.b64encode(np.packbits(edges.reshape(-1),bitorder='little').tobytes()).decode());return g
expected={r['seed']:hashlib.sha256(json.dumps(gold(r),sort_keys=True).encode()).hexdigest() for r in dev+train};graphs=thresholds=calrows=0;curves=[]
for point in m['curves']:
 ep=root/point['evaluation']['artifact'];assert sha(ep)==point['evaluation']['sha256'];e=load(ep);assert len(e['rows'])==2048 and len(e['train_rows'])==128;assert e['thresholds']==[r['threshold'] for r in e['calibration']]
 if point['added_update']==0:assert e['thresholds']==parent['thresholds'] and e['train_rows']==parent['train_rows']
 cp=root/e['calibration_data_artifact'];assert sha(cp)==e['calibration_data_sha256'];cal=np.load(cp)
 for j,r in enumerate(e['calibration']):
  cut,err=cutoff(cal['scores'][:,j],cal['targets'][:,j].astype(bool));assert abs(cut-r['threshold'])<1e-6 and err==r['train_errors'];thresholds+=1
 for i,(r,data) in enumerate(zip(e['train_rows'],train)):
  assert r['seed']==data['seed'] and hashlib.sha256(json.dumps(r['target'],sort_keys=True).encode()).hexdigest()==expected[data['seed']]
  lo,hi=cal['offsets'][i:i+2];pairs=cal['pairs'][lo:hi];presence=np.asarray(r['raw']['presence']);assert np.array_equal(pairs,np.argwhere(presence[:,None]&presence[None,:]));edge=r['target']['edges'];g=np.unpackbits(np.frombuffer(base64.b64decode(edge['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge['shape'])].reshape(edge['shape']);assert np.array_equal(cal['targets'][lo:hi],g[pairs[:,0],pairs[:,1]]);calrows+=1
 for r,data in zip(e['rows'],dev):
  assert r['seed']==data['seed'] and r['semantic_sha256']==data['semantic_sha256'];assert hashlib.sha256(json.dumps(r['target'],sort_keys=True).encode()).hexdigest()==expected[data['seed']]
  for mode,pred in [('raw',r['raw']),('calibrated',{**r['raw'],'edges':r['calibrated_edges']})]:same(components(pred,r['target']),r[mode+'_metrics'])
  graphs+=1
 for arity in (3,4):
  for facts in (3,4):
   chosen=[r for r,d in zip(e['rows'],dev) if (d['arity'],d['facts'])==(arity,facts)];assert point['cells'][f'{arity}x{facts}']==dict(examples=512,raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in chosen),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in chosen))
 curves.append(dict(added_update=point['added_update'],cells=point['cells']))
out=dict(arm=a.arm,graphs_reconstructed=graphs,unique_targets_reconstructed=len(expected),train_threshold_records=thresholds,train_calibration_target_masks=calrows,initial_parent_TRAIN128_predictions_thresholds_exact=True,curves=curves,full_occupancy_seconds=occ['process_occupancy_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in root.iterdir() if p.is_file()},cpu_audit_wall_seconds=time.monotonic()-t,scope='Fixed single development arm, all2048 public DEV examples/four curves, historicalTRAIN128 calibration. Paired decisions wait both independently audited endpoints; no confirmation inference or successor authorization.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='input_sha256'})
