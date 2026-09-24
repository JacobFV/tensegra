import json,gzip,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--aggregate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2);tick=time.monotonic();root=args.root;agg=json.loads(args.aggregate.read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False);diffs={};paircount=0
for seed in range(3):
 base=root/f'c04-confirmation-{seed}';correct={}
 for view in ('clean','reversed'):
  for d in (2,8):
   x=load(Path(str(base)+'-hybrid')/f'{view}-{d}.pt');truth=x['original'];correct[f'supplied_copy/{view}/d{d}']=(x['exact_copy']==truth).numpy()
   for delay,cell in x['result']['cells'].items():correct[f'workspace/{view}/d{d}/t{delay}']=(cell['predictions']==truth).numpy()
  for arm in ('n1_static','n1_roles','n2_rekey'):
   for endpoint in ('endpoint','selected'):
    x=load(Path(str(base)+'-'+arm)/(endpoint+'-validation'+('-reversed' if view=='reversed' else '')+'.pt'));correct[f'{arm}/{endpoint}/{view}']=(x['logits']['answer'].argmax(-1)==x['labels']['task']).numpy()
 diffs[seed]={}
 for key,pair in agg['lineages'][str(seed)]['pairings'].items():
  left,right=map(correct.__getitem__,key.split(' minus '));expected=dict(left_only=int((left&~right).sum()),right_only=int((right&~left).sum()),both_correct=int((left&right).sum()),both_wrong=int((~left&~right).sum()),attempted=len(left),difference=float(left.mean()-right.mean()));assert pair==expected;diffs[seed][key]=left.astype(float)-right.astype(float);paircount+=1
keys=list(agg['comparisons']);reps=2000;draws=np.zeros((len(keys),reps))
for seed in range(3):
 matrix=np.stack([diffs[seed][k] for k in keys]);rng=np.random.default_rng(6042026+seed)
 for lo in range(0,reps,100):
  ix=rng.integers(0,4096,(100,4096));weights=np.stack([np.bincount(row,minlength=4096) for row in ix]).astype(float);draws[:,lo:lo+100]+=matrix@weights.T/(4096*3)
for j,k in enumerate(keys):
 c=agg['comparisons'][k];means=[float(diffs[s][k].mean()) for s in range(3)];assert c['seed_means']==means and c['seed_min']==min(means) and c['seed_max']==max(means);assert abs(c['equal_lineage_mean']-np.mean(means))<1e-12;assert abs(c['seed_sample_sd']-np.std(means,ddof=1))<1e-12;assert np.allclose(c['event_bootstrap_percentile95'],np.quantile(draws[j],[.025,.975]),rtol=0,atol=1e-12)
# Independent compact multiset overlap across distinct lineages.
audits={(s,a):json.load(gzip.open(Path(str(root/f'c04-confirmation-{s}')+'-'+a)/'numeric-overlap.json.gz','rt')) for s in range(3) for a in ('n1_static','n1_roles','n2_rekey')};cross=0
for kind in ('cross_lineage','cross_lineage_test_test'):
 for c in agg['semantic_overlap'][kind]:
  test=audits[c['test_lineage'],c['arm']]['audits'][c['signature']]['populations']['validation/'+c['view']]['signature_counts'];other=audits[c.get('train_lineage',c.get('other_test_lineage')),c['arm']]['audits'][c['signature']];keys2=other['actual_training_signature_counts'] if kind=='cross_lineage' else other['populations']['validation/'+c['view']]['signature_counts'];common=set(test)&set(keys2);assert c['overlap_events']==sum(test[k] for k in common) and c['shared_distinct']==len(common);cross+=1
assert agg['status']=='complete'
for seed,g in agg['aggregate_gates'].items():assert all(g[k] for k in ('workspace_joint','supplied_copy_answer','causal_workspace','causal_supplied_copy'));assert g['neural_fixed_endpoint']==dict(n1_static=False,n1_roles=True,n2_rekey=False)
out=dict(paired_tables_reconstructed=paircount,bootstrap_intervals_verified=len(keys),cross_lineage_overlap_cells_verified=cross,aggregate_gates_verified=True,cpu_audit_wall_seconds=time.monotonic()-tick,scope='Independent raw correctness vectors, all paired contingency tables, matrix-multiplicity implementation of prespecified event bootstrap, fixed-lineage dispersion and cross-lineage signature intersections. No model inference.')
args.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
