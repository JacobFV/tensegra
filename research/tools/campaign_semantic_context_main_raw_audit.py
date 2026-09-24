"""Independent S18 full-arm raw/calibration metrics and event flags; CPU only."""
import argparse,ast,base64,collections,gzip,hashlib,json,re,time
from pathlib import Path
import numpy as np
from campaign_semantics_audit import cutoff
from audit_stage11_semantic_text import components,same
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--archive',type=Path,required=True);p.add_argument('--arm',choices=('context','workspace_control'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.load(gzip.open(p,'rt'));read=lambda p:[json.loads(s) for s in gzip.open(p,'rt')];base=a.repo/'research/results/campaign-01/semantics';run=a.archive/'s18-main-v1';m=load(run/'manifest.json.gz');c=m['config'];assert hashlib.sha256(json.dumps(c,indent=2).encode()+b'\n').hexdigest()=='cfe0359c80b3ed55a3a643b5dd7f823808eadbf256f50d18610501a5596462e0';occ=json.loads((a.archive/'s18-main-v2-launch.occupancy.json').read_text());clock=dict(s.split('=') for s in (a.archive/'s18-main-v2-launch.outer.txt').read_text().splitlines() if '=' in s);assert occ['exit_code']==int(clock['exit_code'])==0 and not occ['timed_out'];assert occ['config_sha256']=='cfe0359c80b3ed55a3a643b5dd7f823808eadbf256f50d18610501a5596462e0';assert occ['wrapper_sha256']==c['source_sha256']['campaign_semantics_s18_launch.py'] and occ['cap_seconds']==4000;assert [x['arm'] for x in m['results']]==['context','workspace_control'];result=next(x for x in m['results'] if x['arm']==a.arm);assert result==load(run/a.arm/'manifest.json.gz');assert [r['added_update'] for r in result['curves']]==[0,1024,2048,4096];vocab=json.loads((base/'s15-bindings/original-data-audit.json').read_text())['value_vocabulary'];const={}
for n in ast.parse((a.repo/'src/topoformer/thinking_language.py').read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('KINDS','ROLES'):const[n.targets[0].id]=ast.literal_eval(n.value)
code=ast.parse(Path(__file__).with_name('campaign_semantic_shape_main_audit.py').read_text());node=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='gold');exec(compile(ast.Module(body=[node],type_ignores=[]),'<independent-gold>','exec'))
rows=read(base/'s15-shape-cache-v2/development.jsonl.gz');dev={r['semantic_sha256']:r for r in rows};mixed=read(base/'s15-shape-cache-v2/train_mixed.jsonl.gz');sel=json.loads((base/'s17-calibration-selection/selection.json').read_text())['mixed'];train={'matched':[mixed[r['index']] for r in sel],'historical':read(base/'s01-data/train.jsonl.gz')[:128]};graphs=cuts=0;results={};raw={};paths=[run/'manifest.json.gz',run/a.arm/'manifest.json.gz',a.archive/'s18-main-v2-launch.occupancy.json',a.archive/'s18-main-v2-launch.outer.txt'];target_cache={k:gold(v) for k,v in dev.items()}
for curve in result['curves']:
 step=curve['added_update'];u=24576+step;results[str(step)]={}
 for policy in ('matched','historical'):
  rec=curve[policy];path=run/a.arm/f'{policy}-u{u}'/rec['artifact'];assert sha(path)==rec['sha256'];d=load(path);N={r['semantic_sha256']:r for r in d['rows']};assert len(N)==len(d['rows'])==2048 and N.keys()==dev.keys();assert len(d['train_rows'])==len(d['train_metrics'])==128;assert d['thresholds']==[x['threshold'] for x in d['calibration']];npz=path.parent/d['calibration_data_artifact'];assert sha(npz)==d['calibration_data_sha256'];cal=np.load(npz);paths.extend([path,npz])
  for j,r in enumerate(d['calibration']):
   cut,err=cutoff(cal['scores'][:,j],cal['targets'][:,j].astype(bool));assert abs(cut-r['threshold'])<1e-6 and err==r['train_errors'];assert r['positive']==int(cal['targets'][:,j].sum()) and r['negative']==len(cal['targets'])-r['positive'];cuts+=1
  for i,(row,source) in enumerate(zip(d['train_rows'],train[policy],strict=True)):
   assert row['seed']==source['seed'] and row['target']==gold(source);lo,hi=cal['offsets'][i:i+2];pairs=cal['pairs'][lo:hi];presence=np.asarray(row['raw']['presence']);assert np.array_equal(pairs,np.argwhere(presence[:,None]&presence[None,:]));edge=row['target']['edges'];target=np.unpackbits(np.frombuffer(base64.b64decode(edge['packed_b64']),dtype=np.uint8),bitorder='little')[:np.prod(edge['shape'])].reshape(edge['shape']);assert np.array_equal(cal['targets'][lo:hi],target[pairs[:,0],pairs[:,1]])
   for label,pred in [('raw',row['raw']),('calibrated',{**row['raw'],'edges':row['calibrated_edges']})]:same(components(pred,row['target']),d['train_metrics'][i][label])
   graphs+=1
  flags={cell:dict(event_ids=[],raw=[],calibrated=[]) for cell in ('3x3','3x4','4x3','4x4')}
  for event in sorted(N):
   row=N[event];source=dev[event];assert row['target']==target_cache[event] and row['seed']==source['seed'];key=(step,event);rh=hashlib.sha256(json.dumps(row['raw'],sort_keys=True).encode()).hexdigest();assert key not in raw or raw[key]==rh;raw[key]=rh;cell=f"{source['arity']}x{source['facts']}";flags[cell]['event_ids'].append(event)
   for label,pred in [('raw',row['raw']),('calibrated',{**row['raw'],'edges':row['calibrated_edges']})]:
    got=components(pred,row['target']);same(got,row[label+'_metrics']);flags[cell][label].append(int(got['semantic_equivalence']))
   graphs+=1
  for label,rr,key in [('dev_raw',d['rows'],'raw_metrics'),('dev_calibrated',d['rows'],'calibrated_metrics'),('train_raw',d['train_metrics'],'raw'),('train_calibrated',d['train_metrics'],'calibrated')]:
   stat=rec[label];assert stat['examples']==len(rr) and stat['exact']==sum(r[key]['semantic_equivalence'] for r in rr);assert abs(stat['copy']-sum(r[key]['identity_copy_accuracy'] for r in rr)/len(rr))<1e-12
   for kind in ('typed_edge','ordered_edge'):
    tp=sum(r[key][kind]['true_positive'] for r in rr);den=sum(r[key][kind]['predicted_count']+r[key][kind]['gold_count'] for r in rr);assert abs(stat[kind+'_f1']-2*tp/max(1,den))<1e-12
  results[str(step)][policy]=flags
  print(a.arm,step,policy,'verified',flush=True)
out=dict(arm=a.arm,graphs_reconstructed=graphs,relation_cutoffs_recomputed=cuts,both_policy_raw_targets_exact=True,results=results,outer_seconds=float(clock['elapsed_seconds']),conservative_charge_seconds=round(float(clock['elapsed_seconds'])+.01,2),input_sha256={str(p)[str(p).index('research/'):]:sha(p) for p in paths},cpu_audit_wall_seconds=time.monotonic()-tick,scope='All four fixed checkpoints, both calibration populations128 and2048DEV each. Independently reconstructed graph metrics, labels, TRAIN pair masks and global thresholds; event-aligned flags retained for paired gates/bootstrap. No actor/forward or outcome-based selection.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('input_sha256','results')})
