"""CPU selection/recipe/immutable failure checks; no model import or inference."""
import ast,collections,gzip,hashlib,importlib.util,json,os,subprocess,sys,tempfile
from pathlib import Path
source=Path('src/topoformer');namespace={};tree=ast.parse((source/'campaign_semantics_recalibrate.py').read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate'],type_ignores=[]),'<validate>','exec'),namespace)
c=json.loads(Path('configs/campaign-s17-calibration-profile-prepared-v1.json').read_text())
try:namespace['validate'](c)
except ValueError:pass
else:raise AssertionError('prepared allowed')
namespace['validate']({**c,'budget_status':'frozen'})
for key,value in [('dev_per_cell',512),('calibration_count',127),('runs',list(reversed(c['runs']))),('job','other')]:
 try:namespace['validate']({**c,'budget_status':'frozen',key:value})
 except ValueError:pass
 else:raise AssertionError('recipe mutation allowed')
base=Path('research/results/campaign-01/semantics');selected=json.loads((base/'s17-calibration-selection/selection.json').read_text())
for arm,quotas in [('control',{'4x3':64,'4x4':64}),('mixed',{'3x3':64,'4x3':32,'4x4':32})]:
 rows=[json.loads(line) for line in gzip.open(base/'s15-shape-cache-v2'/f'train_{arm}.jsonl.gz','rt')];remaining=quotas.copy();expected=[]
 for index,r in enumerate(rows):
  cell=f"{r['arity']}x{r['facts']}"
  if remaining.get(cell,0):remaining[cell]-=1;expected.append(index)
 assert [r['index'] for r in selected[arm]]==expected and len(expected)==128 and not any(remaining.values())
spec=importlib.util.spec_from_file_location('freeze',source/'campaign_semantics_recalibrate_freeze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);src=root/'source';src.mkdir();data=root/'data';data.mkdir()
 for n in ('campaign_semantics_recalibrate_freeze.py','campaign_semantics_recalibrate_launch.py'):(src/n).write_bytes((source/n).read_bytes())
 (src/'actor.py').write_text('fixture');(data/'train.jsonl.gz').write_text('fixture')
 fixture=dict(budget_status='prepared',job='profile',source_sha256={'actor.py':m.digest(src/'actor.py')},data_dir=str(data),cache_sha256={'train':m.digest(data/'train.jsonl.gz')},output_dir=str(root/'unused'),runs=[])
 for key in ('selection','selection_audit','data_audit'):
  p=root/key;p.write_text(key);fixture[key]=str(p);fixture[key+'_sha256']=m.digest(p)
 for arm in ('control','mixed'):
  r=dict(arm=arm)
  for key in ('checkpoint','evaluation','manifest'):
   p=root/(arm+key);p.write_text(arm+key);r[key]=str(p);r[key+'_sha256']=m.digest(p)
  fixture['runs'].append(r)
 prepared=root/'prepared';prepared.write_text(json.dumps(fixture));frozen=root/'frozen';f=m.freeze(prepared,frozen,src,90);m.verify(f,src,90)
 try:m.freeze(prepared,frozen,src,90)
 except FileExistsError:pass
 else:raise AssertionError('overwrite allowed')
 try:m.verify(f,src,91)
 except ValueError:pass
 else:raise AssertionError('cap mutation allowed')
 for target in (src/'actor.py',Path(f['selection']),Path(f['runs'][0]['checkpoint']),data/'train.jsonl.gz'):
  original=target.read_bytes();target.write_bytes(b'changed')
  try:m.verify(f,src,90)
  except ValueError:pass
  else:raise AssertionError('changed input accepted')
  target.write_bytes(original)
 binary=root/'bin';binary.mkdir();idle=binary/'nvidia-smi';idle.write_text('#!/bin/sh\nexit 0\n');idle.chmod(0o755);(src/'actor.py').write_text('changed');prefix=root/'failure'
 result=subprocess.run([sys.executable,str(src/'campaign_semantics_recalibrate_launch.py'),str(frozen),'--cap','90','--python','/nonexistent/child','--prefix',str(prefix)],env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']},capture_output=True,text=True)
 receipt=json.loads(Path(str(prefix)+'.occupancy.json').read_text());assert result.returncode==1 and receipt['exit_code']==1 and 'source changed' in receipt['error'];assert Path(str(prefix)+'.started.json').exists()
print(json.dumps(dict(selection_replay=True,prepared_recipe_rejection=True,immutable_freeze=True,cap_source_selection_checkpoint_cache_guards=True,durable_failure_receipt=True,no_model_import_or_inference=True,no_real_gpu_query=True)))
