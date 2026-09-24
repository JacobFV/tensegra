import ast,gzip,hashlib,json,subprocess,time
from pathlib import Path
t=time.monotonic();ref='d824318';read=lambda p:subprocess.check_output(['git','show',ref+':'+p]);source=read('src/topoformer/campaign_semantics_multisurface_train.py');tree=ast.parse(source);allowed=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in ('renderer_for','validate_config'):allowed.append(n)
 if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='S11_CHECKPOINT_SHA256' for a in n.targets):allowed.append(n)
ns={};exec(compile(ast.Module(body=allowed,type_ignores=[]),'<isolated guarded functions>','exec'),ns)
configs={};rejects=0
for arm in ('english','mixed'):
 for job in ('profile','main'):
  path=f'configs/campaign-s13-{arm}-{job}-prepared.json';c=json.loads(read(path));configs[arm,job]=c
  try:ns['validate_config'](c)
  except ValueError:rejects+=1
  else:raise AssertionError('prepared launch accepted')
  ns['validate_config'](dict(c,budget_status='frozen'))
for job in ('profile','main'):
 a,b=configs['english',job],configs['mixed',job];assert {k for k in a if a[k]!=b[k]}=={'arm','output_dir'}
for phase in (0,1):
 assert [ns['renderer_for']('english',0,[v],[phase]) for v in range(4)]==['english']*4
 assert sorted(ns['renderer_for']('mixed',0,[v],[phase]) for v in range(4))==['english']*2+['spanish']*2
m=json.load(gzip.open('../campaign/research/results/campaign-01/semantics/s11-lr-decay-n8192-196k-dev201/manifest.json.gz'));hashes={}
for name,want in m['source_sha256'].items():
 got=hashlib.sha256(read('src/topoformer/'+name)).hexdigest();assert got==want;hashes[name]=got
assert m['curves'][-1]['checkpoint_sha256']==ns['S11_CHECKPOINT_SHA256']
out=dict(source_ref=ref,trainer_sha256=hashlib.sha256(source).hexdigest(),inherited_s11_source_hashes_unchanged=hashes,parent_checkpoint_sha256=ns['S11_CHECKPOINT_SHA256'],prepared_configs_rejected=rejects,paired_configs_differ_only_arm_output=True,four_visit_renderer_balance=True,source_review=dict(target_free_hidden_state=True,gold_pairs_training_loss_indexing_only=True,restored_model_optimizer_sampler=True,renderer_rng_separate=True,english_train128_thresholds_shared_across_languages=True,raw_policy_retained=True,no_dev_checkpoint_selection=True),cpu_audit_wall_seconds=time.monotonic()-t,scope='Static source and isolated pure-function review, no torch/model/inference/GPU. Prospective bounded profile only after coordinator release and complete S12 matrix; main timing and numerical pairing remain untested. One inspected development parent; trained Spanish is not held-out language transfer.')
Path('research/campaigns/extended-01/review/S13-trainer-preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(out['cpu_audit_wall_seconds'])
