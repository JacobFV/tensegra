"""Independent A14 frozen references and shared-event paired bootstrap."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('repo',type=Path);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();t=time.monotonic();root=a.repo/'research/results/campaign-01/attention/a14';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result=json.loads((root/'confirmation-analysis.json').read_text());policies=['unchanged','both_hard','shared_soft','shared_hard','oracle_common'];seeds=[1401,1402,1403];arrays={};manifests={};gates={p:True for p in policies[:-1]}
for seed in seeds:
 audit=json.loads((a.review/f'A14-{seed}-audit.json').read_text())
 for name,h in audit['input_sha256'].items():assert sha(a.repo/name)==h
 run=root/str(seed)/'results';manifests[seed]=json.loads((run/'manifest.json').read_text());arrays[seed]={p:np.load(run/'confirmation'/f'{p}.npz') for p in policies}
 for name,h in result['input_sha256'][str(seed)].items():assert sha(run/name)==h
for ci,condition in enumerate(result['conditions']):
 assert condition['threshold']==[.98,.95,.98][ci]
 for seed in seeds:
  base=arrays[seed]['unchanged'][f'c{ci}_task'].astype(bool)
  for policy in policies:
   raw=arrays[seed][policy];task=raw[f'c{ci}_task'].astype(bool);route=raw[f'c{ci}_diagnostic_query_mean_route_correct'][:,:,0].all(1);r=condition['seed_rows'][str(seed)][policy];assert r['task_correct']==int(task.sum()) and r['events']==1024 and r['complete_all_node_suffix']==int(raw[f'c{ci}_suffix_value_trajectory'].sum());assert abs(r['all_node_value_accuracy']-raw[f'c{ci}_all_node'].mean())<1e-7;assert r['fixed']==int((task&~base).sum()) and r['broken']==int((~task&base).sum());assert r['route_correct']==int(route.sum());assert r['joint_counts']==dict(route_correct_task_correct=int((route&task).sum()),route_correct_task_wrong=int((route&~task).sum()),route_wrong_task_correct=int((~route&task).sum()),route_wrong_task_wrong=int((~route&~task).sum()));assert r['task_given_correct_queried_route']==float(task[route].mean());assert r['all_node_routes_equal_original']==np.array_equal(raw[f'c{ci}_route'],arrays[seed]['unchanged'][f'c{ci}_route'])
   for kind,reported in r['queried_head_fullpaths'].items():
    heads=raw[f'c{ci}_diagnostic_query_{kind}_head_correct'].all(1);assert reported==dict(correct=int(heads.sum()),head_event_denominator=8192,correct_per_head=heads.sum(0).tolist(),events_all_heads_correct=int(heads.all(1).sum()),event_denominator=1024)
   if policy!='oracle_common':gates[policy]&=float(task.mean())>=condition['threshold']
assert gates==result['competence_all_three_core_cells_all_seeds']
original=np.stack([arrays[s]['unchanged']['c2_task'] for s in seeds]).astype(int);shared=np.stack([arrays[s]['shared_hard']['c2_task'] for s in seeds]).astype(int);route=np.stack([arrays[s]['shared_hard']['c2_diagnostic_query_mean_route_correct'][:,:,0].all(1) for s in seeds]);gain=shared-original;support=route.sum(1);joint=(shared*route).sum(1);primary=result['joint_primary'];assert primary['per_seed_route_support']==support.tolist() and primary['per_seed_task_and_route_correct']==joint.tolist();np.testing.assert_allclose(primary['per_seed_gain'],gain.mean(1),rtol=0,atol=1e-15);assert primary['mean_gain']==gain.mean();assert primary['pooled_conditional_task']==joint.sum()/support.sum();assert primary['per_seed_conditional_task']==(joint/support).tolist()
rng=np.random.default_rng(291000000);draws=[];conditional=[]
for offset in range(0,10000,125):
 indices=rng.integers(0,1024,size=(min(125,10000-offset),1024));multiplicity=np.stack([np.bincount(row,minlength=1024) for row in indices]);draws.extend((multiplicity@gain.sum(0)/3072).tolist());conditional.extend(((multiplicity@(shared*route).sum(0))/(multiplicity@route.sum(0))).tolist())
ci=np.quantile(draws,[.025,.975]);cci=np.quantile(conditional,[.025,.975]);np.testing.assert_allclose(ci,primary['gain_percentile95'],rtol=0,atol=1e-14);np.testing.assert_allclose(cci,primary['conditional_percentile95'],rtol=0,atol=1e-14);assert primary['replicated_effect_pass']==bool((gain.mean(1)>0).all() and ci[0]>0);assert primary['conditional_mechanistic_pass']==bool((joint/support>=.98).all())
refs=root/'references/results';cfg=json.loads((refs/'config.json').read_text());assert sha(refs/'config.json')==hashlib.sha256(subprocess.check_output(['git','show','67dc519:configs/campaign-a14-engineering-references.json'])).hexdigest();rm=json.loads((refs/'manifest.json').read_text());assert rm['optimizer_updates']==0 and len(rm['references'])==9;historical=json.loads((a.review/'A06-confirmation-audit.json').read_text());old={(r['seed'],r['arm']):r for r in historical['checkpoints']};pubref=manifests[1401]['confirmation']['public_sha256'];refs_verified=[]
for n,h in json.loads((refs/'source.json').read_text()).items():assert hashlib.sha256(subprocess.check_output(['git','show','67dc519:src/topoformer/'+n])).hexdigest()==h
for rec in rm['references']:
 identity=rec['seed'],rec['mode'];assert rec['path']==old[identity]['path'] and rec['sha256']==old[identity]['sha256'];assert rec['initial_tensor_sha256']==rec['final_tensor_sha256'];assert rec['public_sha256']==pubref;tag=f'{identity[0]}-{identity[1]}';run=refs/tag;pub=np.load(refs/'601-soft/public.npz');assert json.loads((run/'public-sha256.json').read_text())==pubref;raw=np.load(run/'a06_successful.npz');report=json.loads((run/'a06_successful.json').read_text());assert result['engineering_references'][tag]==report
 for key in pub.files:assert hashlib.sha256(pub[key].tobytes()).hexdigest()==pubref[key]
 for ci,r in enumerate(report['rows']):
  get=lambda k:pub[f'c{ci}_{k}'];gold,succ,start,values=[get(k) for k in ('gold','successor','start','values')];state=values.copy();b,d,n=gold.shape;assert b==1024
  for step in reversed(range(d)):state=np.take_along_axis(state,succ[:,step],1);assert np.array_equal(state,gold[:,d-1-step])
  pred=raw[f'c{ci}_pred'];routes=raw[f'c{ci}_route'];ok=pred==gold;true=start.copy();proposed=start.copy();path=np.ones(b,bool);idx=np.arange(b)
  for step in range(d):true=succ[idx,step,true];proposed=routes[idx,d-1-step,proposed];path&=true==proposed
  expected=dict(task=ok[:,-1][idx,start],all_node=ok.mean((1,2)),suffix_value_trajectory=ok.all((1,2)),exact_pointer_path=path)
  for k,v in expected.items():np.testing.assert_allclose(v,raw[f'c{ci}_{k}']);assert abs(v.mean()-r[k])<1e-7;assert v.all()
  for k in ('selected_mass','edge_mass','supplied_edge_mass'):v=raw[f'c{ci}_{k}'];assert ((v>=0)&(v<=1+1e-6)).all() and abs(v.mean()-r[k])<1e-6
  refs_verified.append(dict(seed=identity[0],mode=identity[1],condition=r['condition'],task=1024,suffix=1024,path=1024))
assert len({(r['seed'],r['mode']) for r in refs_verified})==9
for name,h in result['engineering_reference_input_sha256'].items():assert sha(refs/name)==h
receipt=json.loads((root/'references/receipt.json').read_text());assert receipt['exit_code']==0 and receipt['source']=='67dc519';assert receipt['config_sha256']==sha(refs/'config.json')
out=dict(reference_cells_verified=27,reference_checkpoint_bindings_to_A06=True,all_three_previously_audited_inputs_rebound=True,primary_actual_joint_tables_verified=True,bootstrap_implementation='Independent event multiplicity matrices with shared indices across fixed seeds, 10000 replicates seed291000000; no seed resampling.',primary=primary,competence=gates,reference_full_occupancy_seconds=receipt['full_process_occupancy_seconds'],input_sha256={str(p.relative_to(a.repo)):sha(p) for p in [root/'confirmation-analysis.json']+list((root/'references').rglob('*.json'))+list((root/'references').rglob('*.npz'))},cpu_audit_wall_seconds=time.monotonic()-t,scope='Shared-hard supplied inference-policy effect and competence. Acquired original soft policy fails. A06 perfect references have different training and supplied-neighborhood histories; no architecture superiority or generalization beyond fixed populations follows.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(seconds=out['cpu_audit_wall_seconds'],primary=primary,competence=gates))
