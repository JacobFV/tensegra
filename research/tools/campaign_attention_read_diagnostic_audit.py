"""Independent A12 task/path and archived diagnostic contract reconstruction."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();cfg=json.loads((a.root/'config.json').read_text());m=json.loads((a.root/'manifest.json').read_text());policies=['unchanged','record_hard','destination_hard','both_hard'];assert m['policies']==policies and m['optimizer_updates']==0;assert m['initial_tensor_sha256']==m['final_tensor_sha256']==cfg['tensor_sha256'];assert m['checkpoint_sha256']==cfg['checkpoint_sha256']
for n,h in json.loads((a.root/'source.json').read_text()).items():assert hashlib.sha256(subprocess.check_output(['git','show',a.source_ref+':src/topoformer/'+n])).hexdigest()==h
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
  if policy in ('record_hard','both_hard'):assert np.all(dg('key_mse_to_record_argmax')==0);assert np.array_equal(dg('record_used_target_mass'),dg('record_argmax_correct'))
  if policy in ('destination_hard','both_hard'):assert np.all(dg('payload_mse_to_destination_argmax')==0);assert np.array_equal(dg('destination_used_target_mass'),dg('destination_argmax_correct'));assert np.all(dg('payload_mse_correct_destination_sum')==0)
  cells.append(dict(policy=policy,condition=row['condition'],examples=b,task_correct=int(expected['task'].sum()),path_correct=int(path.sum()),suffix_correct=int(expected['suffix_value_trajectory'].sum()),instrumented_forward_seconds=row['instrumented_forward_seconds']))
out=dict(source_ref=a.source_ref,cells=cells,diagnostic_cells_checked=diagnostics,shared_targets_reconstructed_from_initial_values=True,hardening_own_argmax_zero_distortion=True,frozen_checkpoint_sha256=m['checkpoint_sha256'],initial_final_tensor_sha256=m['initial_tensor_sha256'],zero_updates=True,instrumented_forward_seconds=sum(c['instrumented_forward_seconds'] for c in cells),cpu_audit_wall_seconds=time.monotonic()-tick,scope='Task/path/full suffix labels independently reconstructed; diagnostic means/bounds/shapes/hardening identities checked. Full weights/current hidden payloads absent, so diagnostic MSE/oracle accuracy derivations rely on reviewed source and mechanical tests. No policy selection or model inference.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cells=len(cells),seconds=out['cpu_audit_wall_seconds']))
