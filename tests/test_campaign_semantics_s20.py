"""CPU-only S20 recipe, train/evaluation boundary, update replay and lifecycle tests."""
import copy,gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch import nn
from topoformer import campaign_semantics_s20 as run
from topoformer import campaign_semantics_s20_freeze as guard
from topoformer import campaign_semantics_s20_launch as launcher
from topoformer.campaign_semantics_s19_actor import TypedRecordActor
from topoformer.campaign_semantics_s19_codec import NODE,EOS,KINDS
from topoformer.thinking_language import ActorInput,state_hash


def config(job='profile'):
 return json.loads(Path(f'configs/campaign-s20-{job}-prepared-v1.json').read_text())


def test_all_nine_matrix_and_confirmation_is_absent_from_profile():
 for job in ('profile','main'):guard.validate(config(job))
 assert 'confirmation' not in config()['inputs']
 assert 'development' not in config('main')['inputs']
 assert [(s,a) for s in config('main')['seeds'] for a in config('main')['arms']]==[(s,a) for s in (701,702,703) for a in ('original','context','record')]
 for key,value in [('seeds',[702]),('updates',2048),('final_only',False),('schedule_seed',701),('record_warmup_updates',20),('baseline_learning_rate',3e-4),('record_initialization','parent'),('arms',['record'])]:
  with pytest.raises(ValueError):guard.validate({**config(),key:value})
 c=config();c['inputs']['confirmation']=config('main')['inputs']['confirmation']
 with pytest.raises(ValueError):guard.validate(c)
 c=config();c['parents']['701']['checkpoint']=copy.deepcopy(c['parents']['702']['checkpoint'])
 with pytest.raises(ValueError):guard.validate(c)


def test_stream_reset_eight_visits_identical_across_all_arms():
 hashes=[]
 for _ in range(3):
  generator,order,position=run.new_stream();visits=[0]*4096;h=hashlib.sha256()
  for _ in range(4096):
   ids,order,position=run.next_indices(order,position,generator,8);h.update(json.dumps(ids).encode())
   for i in ids:visits[i]+=1
  assert set(visits)=={8};hashes.append(h.hexdigest())
 assert hashes==[guard.STREAM]*3
 assert [run.presentation_seed(i) for i in (0,1,32767)]==[150150000,150150001,150182767]


def test_training_preparation_never_reads_evaluation(monkeypatch,tmp_path):
 c=config('main');rows=[{'index':i} for i in range(4096)];reads=[]
 audit=tmp_path/'audit';audit.write_text(json.dumps({'value_vocabulary':['x']}));selection=tmp_path/'selection';selection.write_text(json.dumps({'mixed':[]}))
 c['inputs']['vocabulary_audit']['path']=str(audit);c['inputs']['selection']['path']=str(selection)
 def load(path):
  reads.append(path);assert path==c['inputs']['train']['path'];return rows
 monkeypatch.setattr(run,'load_cache',load);monkeypatch.setattr(run,'selected_rows',lambda rows,sel:rows[:128])
 v,actual,panel=run.prepare_training(c);assert actual is rows and len(panel)==128 and len(reads)==1
 with pytest.raises(ValueError,match='before fixed endpoint'):run.load_final_evaluation(c,4095)
 assert len(reads)==1


def test_final_loader_retains_all_cells_and_rejects_duplicates(monkeypatch):
 rows=[dict(arity=a,facts=f,semantic_sha256=f'{a}/{f}/{i}') for a in (3,4) for f in (3,4) for i in range(512)];reads=[]
 def load(p):reads.append(p);return rows
 monkeypatch.setattr(run,'load_cache',load)
 assert len(run.load_final_evaluation(config(),20)[1])==64
 assert len(run.load_final_evaluation(config('main'),4096)[1])==2048
 assert reads==[config()['inputs']['development']['path'],config('main')['inputs']['confirmation']['path']]
 rows[-1]['semantic_sha256']=rows[0]['semantic_sha256']
 with pytest.raises(ValueError,match='duplicate'):run.load_final_evaluation(config('main'),4096)


def test_runner_never_evaluates_before_train_endpoint(monkeypatch,tmp_path):
 c=config('main');c['output_dir']=str(tmp_path/'out');c['cap_seconds']=1000;events=[]
 monkeypatch.setattr(run,'verify',lambda *args:{})
 monkeypatch.setattr(run,'prepare_training',lambda c:(['v'],['TRAIN'],['PANEL']))
 monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None);monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
 def train(c,seed,arm,vocab,rows,out):
  assert rows==['TRAIN'];events.append(('train',seed,arm));return object(),dict(seed=seed,arm=arm,added_update=4096,training_seconds=1)
 def load(c,updates):
  assert updates==4096 and events[-1][0]=='train';events.append(('load',));return 'confirmation',['CONFIRMATION']
 def evaluate(c,model,result,vocab,panel,rows,out,label):
  assert panel==['PANEL'] and rows==['CONFIRMATION'];events.append(('eval',));return {**result,'evaluation_seconds':1}
 monkeypatch.setattr(run,'train_endpoint',train);monkeypatch.setattr(run,'load_final_evaluation',load);monkeypatch.setattr(run,'evaluate_endpoint',evaluate)
 manifest=run.run(c);assert len(manifest['results'])==9
 assert [e[0] for e in events]==['train','load','eval']*9


def test_scratch_update_replays_exact_s19_adamw():
 torch.manual_seed(701);a=TypedRecordActor(value_count=4,width=16,heads=4);b=copy.deepcopy(a)
 oa=torch.optim.AdamW(a.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01);ob=torch.optim.AdamW(b.parameters(),lr=3e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
 batch=[dict(public=ActorInput('alice bob',()),records=[[NODE,KINDS.index('ident'),-1,0,-1],[NODE,KINDS.index('record'),0,-1,-1],[EOS,-1,-1,-1,-1]])]
 for update in (1,20,128,4096):
  loss=run.record_update(a,oa,batch,update)
  for group in ob.param_groups:group['lr']=run.learning_rate(update)
  ob.zero_grad();records=run.padded_records([x['records'] for x in batch],'cpu');out=b.teacher_forced([x['public'] for x in batch],records);expected,*_=run.field_losses(out,records);expected.backward();nn.utils.clip_grad_norm_(b.parameters(),1.);ob.step()
  assert torch.equal(loss,expected) and all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
  for x,y in zip(oa.state.values(),ob.state.values()):assert all(torch.equal(x[k],y[k]) for k in x)
 assert run.learning_rate(20)==3e-4*20/128
 from topoformer.campaign_semantics_s18 import optimizer_update
 assert run.optimizer_update is optimizer_update


def test_parent_and_adamw_identity_and_scratch_reset(monkeypatch):
 class Toy(nn.Module):
  def __init__(self,**kwargs):super().__init__();self.parts=nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(98)])
 parent=Toy();op=torch.optim.AdamW(parent.parameters(),lr=1e-5)
 for i,p in enumerate(parent.parameters()):op.state[p]=dict(step=torch.tensor(24576.),exp_avg=torch.ones_like(p)*i,exp_avg_sq=torch.ones_like(p)*(i+1))
 checkpoint=dict(model=copy.deepcopy(parent.state_dict()),optimizer=copy.deepcopy(op.state_dict()),update=24576,visits=[24]*8192)
 c=config();c['device']='cpu';c['parents']['701']['model_state_sha256']=state_hash(parent);monkeypatch.setattr(run,'S18Actor',Toy);monkeypatch.setattr(torch,'load',lambda *a,**k:copy.deepcopy(checkpoint))
 a,oa,ha=run.initialize(c,701,'original',['v']);b,ob,hb=run.initialize(c,701,'context',['v'])
 assert ha==hb and ha['inherited_presentations']==196608 and ha['initial_optimizer_states']==98
 assert state_hash(a)==state_hash(b)==state_hash(parent)
 for x,y in zip(oa.state.values(),op.state.values()):assert all(torch.equal(x[k],y[k]) for k in x)
 monkeypatch.setattr(run,'TypedRecordActor',Toy)
 a,oa,ha=run.initialize(c,701,'record',['v']);b,ob,hb=run.initialize(c,701,'record',['v']);d,_,_=run.initialize(c,702,'record',['v'])
 assert state_hash(a)==state_hash(b)!=state_hash(d) and len(oa.state)==0 and ha['inherited_presentations']==0
 checkpoint['update']=1
 with pytest.raises(ValueError,match='parent exposure'):run.initialize(c,701,'original',['v'])


def test_freeze_rejects_changed_source_cap_and_inputs(monkeypatch,tmp_path):
 c=config();source=tmp_path/'source';source.mkdir()
 for name in guard.SOURCES:(source/name).write_text(name)
 c['output_dir']=str(tmp_path/'unused');prepared=tmp_path/'prepared';prepared.write_text(json.dumps(c));real=guard.digest
 references={r['path']:r['sha256'] for r in c['inputs'].values()}
 for p in c['parents'].values():references.update({p[k]['path']:p[k]['sha256'] for k in ('checkpoint','manifest')})
 monkeypatch.setattr(guard,'digest',lambda path:references.get(str(path),None) or real(path))
 class Document:
  def __enter__(self):return self
  def __exit__(self,*args):pass
  def read(self):return json.dumps(dict(config=dict(seed=701),final_state_sha256=c['parents']['701']['model_state_sha256']))
 monkeypatch.setattr(gzip,'open',lambda *a,**k:Document())
 frozen=guard.freeze(prepared,tmp_path/'frozen',source,180);guard.verify(frozen,source,180)
 with pytest.raises(ValueError):guard.verify(frozen,source,181)
 with pytest.raises(FileExistsError):guard.freeze(prepared,tmp_path/'frozen',source,180)
 (source/guard.SOURCES[0]).write_text('corrupt')
 with pytest.raises(ValueError,match='source changed'):guard.verify(frozen,source,180)


@pytest.mark.parametrize('body,code',[('raise RuntimeError("failure")',1),('import time;time.sleep(2)',124),('import os;from pathlib import Path;assert Path(os.environ["PYTHONPATH"]).resolve()==Path(__file__).resolve().parent.parent',0)])
def test_launcher_own_source_and_durable_status(tmp_path,body,code):
 package=tmp_path/'src/topoformer';package.mkdir(parents=True);(package/'__init__.py').write_text('');(package/'campaign_semantics_s20_launch.py').write_text(Path(launcher.__file__).read_text());(package/'campaign_semantics_s20_freeze.py').write_text('def verify(*args):return {}');(package/'campaign_semantics_s20.py').write_text(body)
 binary=tmp_path/'bin';binary.mkdir();nv=binary/'nvidia-smi';nv.write_text('#!/bin/sh\nexit 0\n');nv.chmod(0o755);c=tmp_path/'config';c.write_text('{}');prefix=tmp_path/'job'
 p=subprocess.run([sys.executable,str(package/'campaign_semantics_s20_launch.py'),str(c),'--cap','.3','--python',sys.executable,'--prefix',str(prefix)],cwd='/',env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']},capture_output=True,text=True,timeout=5)
 assert p.returncode==code,p.stderr
 receipt=json.loads(Path(str(prefix)+'.occupancy.json').read_text());assert receipt['exit_code']==code and receipt['timed_out']==(code==124)
 assert Path(str(prefix)+'.started.json').exists() and not Path(str(prefix)+'.occupancy.json.tmp').exists()
