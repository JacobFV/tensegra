"""Independent NumPy A08 graph replacement and original/supplied outcome audit."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--source-ref',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();tick=time.monotonic();cfg=json.loads((a.root/'config.json').read_text());m=json.loads((a.root/'manifest.json').read_text());assert m['optimizer_updates']==0
for n,h in json.loads((a.root/'source.json').read_text()).items():assert hashlib.sha256(subprocess.check_output(['git','show',a.source_ref+':src/topoformer/'+n])).hexdigest()==h
for c in m['checkpoints']:assert c['initial_tensor_sha256']==c['final_tensor_sha256']
public=np.load(a.root/'public.npz');raws={f.stem:np.load(f) for f in a.root.glob('*-*.npz')};reports={k:json.loads((a.root/(k+'.json')).read_text())['rows'] for k in raws};cells=[];invariants=[]
for ci,condition in enumerate(cfg['conditions']):
 get=lambda k:public[f'c{ci}_{k}'];gold,supplied,start,succ,ssucc,rel=[get(k) for k in ('gold','supplied_gold','start','successor','supplied_successor','relation')];adj,sadj=get('adjacency'),get('supplied_adjacency');b,d,n=gold.shape;bi=np.arange(b);assert b==cfg['examples']==1024;assert np.array_equal(adj.sum(-1),sadj.sum(-1)) and np.array_equal(adj.sum(-2),sadj.sum(-2));assert np.all(adj.sum(-1)==condition['groups']);assert np.all((adj==0)|(adj==1)) and np.all((sadj==0)|(sadj==1))
 removed=(adj*(1-sadj)).sum((1,2,3))/adj.sum((1,2,3));added=(sadj*(1-adj)).sum((1,2,3))/adj.sum((1,2,3));expected=0 if condition['fraction']==0 else max(2,round(n/condition['groups']*condition['fraction']))/(n/condition['groups']);assert np.all(removed==added) and np.all(removed==expected)
 clean=next(j for j,c in enumerate(cfg['conditions']) if c['data_group']==condition['data_group'] and c['fraction']==0)
 for field in ('gold','start','successor','relation','adjacency'):assert np.array_equal(get(field),public[f'c{clean}_{field}'])
 for graph,successors,targets in ((adj,succ,gold),(sadj,ssucc,supplied)):
  for t in range(d):assert np.all(graph[bi[:,None],rel[:,t,None],np.arange(n)[None],successors[:,t]]==1)
  for t in range(1,d):assert np.array_equal(targets[:,t],np.take_along_axis(targets[:,t-1],successors[:,d-1-t],axis=1))
 for tag,raw in raws.items():
  row=reports[tag][ci];assert row['condition']==condition;pred=raw[f'c{ci}_pred'];route=raw[f'c{ci}_route'];ok=pred==gold;actual=start.copy();proposed=start.copy();path=np.ones(b,dtype=bool)
  for t in range(d):actual=succ[bi,t,actual];proposed=route[bi,d-1-t,proposed];path&=actual==proposed
  values=dict(task=ok[:,-1][bi,start],all_node=ok.mean((1,2)),suffix_value_trajectory=ok.all((1,2)),exact_pointer_path=path,agreement_supplied_task=pred[:,-1][bi,start]==supplied[:,-1][bi,start],corrupt_pointer_task=supplied[:,-1][bi,start]==gold[:,-1][bi,start],removed_fraction=removed,added_fraction=added)
  for k,v in values.items():np.testing.assert_allclose(v,raw[f'c{ci}_{k}'],atol=1e-7,rtol=1e-6);assert abs(v.mean()-row[k])<1e-6
  for k in ('selected_mass','edge_mass','supplied_edge_mass'):
   v=raw[f'c{ci}_{k}'];assert np.all((v>=-1e-6)&(v<=1+1e-6));assert abs(v.mean()-row[k])<1e-6
  assert values['agreement_supplied_task'].all();cells.append(dict(model=tag,condition=condition,examples=b,original_correct=int(values['task'].sum()),supplied_correct=int(values['agreement_supplied_task'].sum()),original_path_correct=int(path.sum())))
 invariants.append(dict(condition=condition,removed_fraction=expected,added_fraction=expected,in_out_degree_and_size_preserved=True))
out=dict(source_ref=a.source_ref,cells=cells,invariants=invariants,frozen_tensor_receipts_equal=True,optimizer_updates=0,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Independent original/supplied metrics, recursive targets, successor-edge membership, matched base populations, degree/size and exact replacement counts. Shared public archive stores one copy across models; source establishes deterministic reuse. Attributes/initial labels absent: successor selection and first target step retain generator provenance. No full attention-weight replay. Joint edge replacement only; no pure-deletion or missing-address coverage.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(dict(cells=len(cells),seconds=out['cpu_audit_wall_seconds']))
