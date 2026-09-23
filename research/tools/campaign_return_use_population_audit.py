import torch,gzip,json,time,pathlib,itertools
start=time.monotonic();torch.set_num_threads(2);base=pathlib.Path.home()/'topoformer-campaign-01/returns';sets={}
for name in ['r05-development']+[f'r05-confirmation/{s}' for s in [10,11,12]]:
 with gzip.open(base/name/'cache.pt.gz','rb') as f:c=torch.load(f,map_location='cpu',weights_only=False)
 for split in ('train','calibration','validation','test'):
  if split+'/2' in c:
   ids=c[split+'/2']['original_event_hashes'];assert len(set(ids))==len(ids);sets[name+'/'+split]=set(ids)
   if split+'/8' in c:assert ids==c[split+'/8']['original_event_hashes']
 for key,field in [('balanced/8','original_event_hashes'),('intervention_swap/8','supplied_event_hashes')]:
  ids=c[key][field];assert len(set(ids))==len(ids);sets[name+'/'+key]=set(ids)
 for kind in ('wrong','drop','swap'):
  x=c['intervention_'+kind+'/8'];n=len(x['original_targets']);b=c['validation/8'];assert x['original_event_hashes']==b['original_event_hashes'][:n];assert torch.equal(x['query'],b['query'][:n])
 del c
for (a,x),(b,y) in itertools.combinations(sets.items(),2):assert not(x&y),(a,b)
out=dict(populations={k:len(v)for k,v in sets.items()},pairwise_disjoint=True,paired_distractor_event_identity=True,causal_query_original_event_prefix_equal=True,cpu_audit_wall_seconds=time.monotonic()-start)
pathlib.Path('/tmp/r05-population-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
