import ast,json,gzip,hashlib,tempfile,time,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.root;out=args.output;tick=time.monotonic()
paths=['src/topoformer/campaign_semantics_confirmation_analysis.py','research/campaigns/extended-01/semantics/S10-S12-analysis.py','research/campaigns/extended-01/semantics/S10-paired-report.py']
ns=[]
for path in paths:
 tree=ast.parse((root/path).read_text());tree.body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,))];env=dict(__file__=str(root/path),np=np,json=json,gzip=gzip,hashlib=hashlib,Path=Path,time=time,digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),POLICIES={'baseline':(False,False),'schema':(True,False),'bookkeeping':(False,True),'combined':(True,True)});exec(compile(tree,path,'exec'),env);ns.append(env)
v=np.tile([1.,-1.],512); delta=np.stack([v,-v,np.zeros(1024)])
a=ns[0]['paired_interval'](delta,257,12);b=ns[1]['paired_interval'](delta,257,12);assert a['shared_event_mean_interval']==b['shared_event_interval']==[0.,0.];assert a['per_seed_interval']==b['per_seed_interval']
with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp);pairs=[]
 for seed,num in [(701,66),(702,131),(703,150)]:
  pair=dict(seed=seed)
  for arm in ('constant','decay'):
   rows=[dict(semantic_sha256=str(i),raw_metrics=dict(semantic_equivalence=arm=='decay' and i<2),calibrated_metrics=dict(semantic_equivalence=arm=='decay' and i<num)) for i in range(1024)];p=tmp/f'{seed}-{arm}.gz'
   with gzip.open(p,'wt') as f:json.dump(dict(update=24576,rows=rows),f)
   pair[arm]=str(p)
  pairs.append(pair)
 cfg=dict(pairs=pairs,output=str(tmp/'result.json'));res=ns[0]['run'](cfg);assert not res['results']['calibrated_metrics']['all_decayed_competent'];assert res['results']['calibrated_metrics']['replicated_directional_advantage'];assert res['primary']=='calibrated_metrics' and 'supplied-schema' in res['secondary']
 try:ns[0]['run']({**cfg,'pairs':pairs[:2]});raise AssertionError('accepted incomplete')
 except ValueError:pass
 p=Path(pairs[2]['decay']);data=json.load(gzip.open(p,'rt'));data['rows'].reverse()
 with gzip.open(p,'wt') as f:json.dump(data,f)
 try:ns[0]['run'](cfg);raise AssertionError('accepted reordered')
 except ValueError:pass
out.write_text(json.dumps(dict(source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},fixture_checks=['shared-event cancellation across three fixed seeds, exact primary/secondary interval agreement','701 competence failure preserved despite positive direction and other competent seeds','incomplete matrix rejection','paired ordering rejection','calibrated primary/raw secondary/schema separate'],source_review=['S10 frequency output uses fixed TRAIN8192 prediction hash, never actor predictions, and exact confirmation identity/target pairing','All schema/bookkeeping variants remain secondary; no policy selection','Legacy S10 paired report provides descriptive counts only, not all-six promotion','Primary evaluator relies on prior immutable artifact audit for actual targets and producer lineage; semantic identity alone is not an independent target hash guard'],cpu_audit_wall_seconds=time.monotonic()-tick,scope='AST-extracted pure analysis functions and synthetic fixtures; no model imports, inference or future outcome access.'),indent=2)+'\n');print(out.read_text())
