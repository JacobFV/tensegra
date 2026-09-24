"""Independent A14 task/path and archived diagnostic contract reconstruction."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();cfg=json.loads((a.root/'config.json').read_text());training=json.loads((a.root/'manifest.json').read_text());m=training['confirmation'];policies=['unchanged','both_hard','shared_soft','shared_hard','oracle_common'];assert m['policies']==policies;assert m['initial_tensor_sha256']==m['final_tensor_sha256']==training['final_tensor_sha256'];m['checkpoint_sha256']=training['checkpoint_sha256']
for n,h in json.loads((a.root/'source.json').read_text()).items():assert hashlib.sha256(subprocess.check_output(['git','show',a.source_ref+':src/topoformer/'+n])).hexdigest()==h
assert training['steps']==cfg['steps'] and training['presentations']==cfg['steps']*16
a.root=a.root/'confirmation'
pub=np.load(a.root/'public.npz');cells=[];diagnostics=0
for policy in policies:
 raw=np.load(a.root/f'{policy}.npz');report=json.loads((a.root/f'{policy}.json').read_text());assert [r['condition'] for r in report['rows']]==cfg['conditions']
 for ci,row in enumerate(report['rows']):
  get=lambda k:raw[f'c{ci}_{k}'];shared=lambda k:pub[f'c{ci}_{k}'];pred,route=get('pred'),get('route');gold,succ,start,values=[shared(k) for k in ('gold','successor','start','values')];b,d,n=gold.shape;assert b==cfg['examples'];bi=np.arange(b);state=values.copy()
  for t in reversed(range(d)):state=np.take_along_axis(state,succ[:,t],axis=1);assert np.array_equal(state,gold[:,d-1-t])
  ok=pred==gold;actual=start.copy();proposed=start.copy();path=np.ones(b,bool)
  for t in range(d):actual=succ[bi,t,actual];proposed=route[bi,d-1-t,proposed];path&=actual==proposed
  expected=dict(task=ok[:,-1][bi,start],all_node=ok.mean((1,2)),suffix_value_trajectory=ok.all((1,2)),exact_pointer_path=path)
  for k,v in expected.items():np.testing.assert_allclose(v,get(k),atol=1e-7,rtol=1e-6);assert abs(v.mean()-row[k])<1e-6
  for k in ('selected_mass','edge_mass','supplied_edge_mass'):
   v=get(k);assert np.all((v>=-1e-6)&(v<=1+1e-6));assert abs(v.mean()-row[k])<1e-6
  for k,mean in row['diagnostics'].items():
   v=get(k);assert v.shape==(b,d,8) and np.isfinite(v).all();assert abs(float(v.mean())-mean)<1e-6*max(1,abs(mean));assert (v>=-1e-7).all();diagnostics+=1
   if any(s in k for s in ('mass','argmax_correct','all_heads_same')):assert (v<=1+1e-6).all()
   if 'all_heads_same'in k:assert np.array_equal(v,np.repeat(v[:,:,:1],8,axis=2))
  dg=lambda name:get('diagnostic_'+name)
  support=dg('correct_destination_count');assert np.all((support>=0)&(support<=n));assert (support==support.astype(int)).all();assert np.all(dg('payload_mse_correct_destination_sum')[support==0]==0)
  if policy in ('shared_soft','shared_hard','oracle_common'):assert np.all(dg('used_head_address_discrepancy')==0)
  if policy in ('both_hard','shared_hard','oracle_common'):assert np.all(dg('payload_mse_to_destination_argmax')==0);assert np.all(dg('payload_mse_correct_destination_sum')==0)
  query=dg('query_mean_route_correct');assert np.array_equal(query,np.repeat(query[:,:,:1],8,axis=2));assert np.array_equal(query[:,:,0].all(1),path)
  if policy=='oracle_common':assert dg('query_used_destination_head_correct').all();assert dg('destination_argmax_correct').min()==1;assert np.all(dg('payload_mse_to_oracle')==0)
  if policy=='shared_hard':assert np.array_equal(route,np.load(a.root/'unchanged.npz')[f'c{ci}_route'])
  cells.append(dict(policy=policy,condition=row['condition'],examples=b,task_correct=int(expected['task'].sum()),path_correct=int(path.sum()),suffix_correct=int(expected['suffix_value_trajectory'].sum()),instrumented_forward_seconds=row['forward_seconds']))
out=dict(source_ref=a.source_ref,cells=cells,diagnostic_cells_checked=diagnostics,shared_targets_reconstructed_from_initial_values=True,shared_hard_own_argmax_zero_distortion=True,queried_mean_full_path_equals_existing_metric=True,frozen_checkpoint_sha256=m['checkpoint_sha256'],initial_final_tensor_sha256=m['initial_tensor_sha256'],inference_zero_updates=True,training_updates=training['steps'],instrumented_forward_seconds=sum(c['instrumented_forward_seconds'] for c in cells),cpu_audit_wall_seconds=time.monotonic()-tick,scope='Task/path/full suffix labels independently reconstructed; diagnostic means/bounds/shapes/hardening identities checked. Full weights/current hidden payloads absent, so diagnostic MSE/oracle accuracy derivations rely on reviewed source and mechanical tests. No policy selection or model inference.')
result_root=a.root.parent;run=result_root.parent
receipt=json.loads((run/'receipt.json').read_text());config_name='campaign-a14-profile.json' if cfg['seed']==1499 else f"campaign-a14-confirm-{cfg['seed']}.json"
assert hashlib.sha256((result_root/'config.json').read_bytes()).hexdigest()==receipt['config_sha256']==hashlib.sha256(subprocess.check_output(['git','show',a.source_ref+':configs/'+config_name])).hexdigest()
assert receipt['exit_code']==0 and receipt['source']==a.source_ref
for line in (run/'SHA256SUMS').read_text().splitlines():
 h,name=line.split(maxsplit=1);p=run/name
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
for key in pub.files:assert hashlib.sha256(pub[key].tobytes()).hexdigest()==m['public_sha256'][key]
paired=[]
for ci,c in enumerate(cfg['conditions']):
 base=np.load(a.root/'unchanged.npz')[f'c{ci}_task'].astype(bool)
 for policy in policies:
  raw=np.load(a.root/f'{policy}.npz');task=raw[f'c{ci}_task'].astype(bool);route=raw[f'c{ci}_diagnostic_query_mean_route_correct'][:,:,0].astype(bool).all(1);heads={}
  for kind in ('record','original_destination','used_destination'):
   correct=raw[f'c{ci}_diagnostic_query_{kind}_head_correct'].astype(bool).all(1);heads[kind]=dict(correct=int(correct.sum()),denominator=int(correct.size),events_all_heads=int(correct.all(1).sum()))
  paired.append(dict(condition=c,policy=policy,event_support=len(task),route_support=int(route.sum()),task_route_joint=[int((route&task).sum()),int((route&~task).sum()),int((~route&task).sum()),int((~route&~task).sum())],task_given_route=float((route&task).sum()/route.sum()) if route.any() else None,fixed=int((~base&task).sum()),broken=int((base&~task).sum()),head_paths=heads))
dev=0
for artifact in sorted((result_root/'development').glob('eval-*.npz')):
 raw=np.load(artifact);report=json.loads(artifact.with_suffix('.json').read_text())
 for ci,row in enumerate(report['rows']):
  g=raw[f'c{ci}_gold'];pred=raw[f'c{ci}_pred'];route=raw[f'c{ci}_route'];start=raw[f'c{ci}_start'];succ=raw[f'c{ci}_successor'];b,d,n=g.shape;ix=np.arange(b);ok=pred==g;actual=start.copy();proposed=start.copy();path=np.ones(b,bool)
  for t in range(d):actual=succ[ix,t,actual];proposed=route[ix,d-1-t,proposed];path&=actual==proposed
  for key,v in dict(task=ok[:,-1][ix,start],all_node=ok.mean((1,2)),suffix_value_trajectory=ok.all((1,2)),exact_pointer_path=path).items():np.testing.assert_allclose(v,raw[f'c{ci}_{key}']);assert abs(v.mean()-row[key])<1e-6
  dev+=1
repo=run
while repo.name!='research':repo=repo.parent
repo=repo.parent
out.update(status='raw_and_provenance_audited_state_receipt_separate',paired=paired,development_cells_verified=dev,full_process_occupancy_seconds=receipt['full_process_occupancy_seconds'],input_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()},all_seed_status='pending' if cfg['seed']!=1499 else 'mechanical_profile',cpu_audit_wall_seconds=time.monotonic()-tick)
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cells=len(cells),seconds=out['cpu_audit_wall_seconds']))
