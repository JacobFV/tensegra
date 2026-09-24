"""No production actor forwards: small differential update and source guards."""
import copy,json
from pathlib import Path
import pytest
import torch
from topoformer.campaign_semantics_s18 import optimizer_update,selected_rows
from topoformer.campaign_semantics_s18_freeze import validate,freeze,verify,SOURCES
from topoformer.semantic_curriculum import SemanticCurriculumActor,curriculum_weights
from topoformer.campaign_semantics_s18_actor import S18Actor
from topoformer.semantic_text_acquisition import corrected_losses
from topoformer.thinking_language import ActorInput,ROLES


def test_exact_s15_update_order_and_adamw_differential():
 torch.manual_seed(201)
 old=SemanticCurriculumActor(value_count=4,width=16,capacity=8,workspace_rows=3,microsteps=2,edge_width=4)
 new=S18Actor(arm='original',value_count=4,width=16,capacity=8,workspace_rows=3,edge_width=4);new.load_state_dict(old.state_dict())
 ao=torch.optim.AdamW(old.parameters(),lr=1e-5);an=torch.optim.AdamW(new.parameters(),lr=1e-5)
 gold=dict(presence=torch.arange(8)<2,kind=torch.zeros(8,dtype=torch.long),value=torch.zeros(8,dtype=torch.long),copy=torch.zeros(8,dtype=torch.long),edges=torch.zeros(8,8,len(ROLES),dtype=torch.bool),slots=torch.full((8,8),-1,dtype=torch.long))
 gold['edges'][0,1,0]=True;gold['slots'][0,1]=0
 batch=[(ActorInput('alice bob',()),gold,{'hidden':'not passed to actor'})]*2
 queries=[torch.tensor([[0,1],[1,0],[2,3]])]*2
 for update in range(24576,24579):
  # The unchanged S15 block, copied verbatim apart from timer/GPU sync.
  ao.zero_grad();outputs=old.forward_batch([p for p,_,_ in batch],pairs=queries);weights=curriculum_weights(update*8,1000,2000)
  parts=[corrected_losses(o,g,q) for o,(_,g,_),q in zip(outputs,batch,queries)]
  expected=torch.stack([sum(weights[k]*v for k,v in item.items()) for item in parts]).mean()
  expected.backward();torch.nn.utils.clip_grad_norm_(old.parameters(),1.);ao.step()
  actual,_=optimizer_update(new,an,batch,queries,update)
  assert torch.equal(actual,expected)
  for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k])
  for i,state in ao.state_dict()['state'].items():
   for k,v in state.items():assert torch.equal(v,an.state_dict()['state'][i][k])


def test_fixed_recipe_changes_fail_closed(tmp_path):
 prepared=Path('configs/campaign-s18-profile-prepared-v1.json');c=json.loads(prepared.read_text());validate(c)
 for field,value in [('added_updates',10),('arms',['context']),('width',16),('control_microsteps',9),('calibration_policy','pick_best'),('batch_size',4)]:
  wrong=copy.deepcopy(c);wrong[field]=value
  with pytest.raises(ValueError):validate(wrong)
 source=tmp_path/'src';source.mkdir()
 for n in SOURCES:(source/n).write_text(n)
 frozen=freeze(prepared,tmp_path/'frozen.json',source,180)
 with pytest.raises(ValueError,match='cap mismatch'):verify(frozen,source,181)
 (source/SOURCES[0]).write_text('mutation')
 with pytest.raises(ValueError,match='source changed'):verify(frozen,source,180)
 for job in ('main','reference'):
  uncapped=json.loads(Path(f'configs/campaign-s18-{job}-prepared-v1.json').read_text());validate(uncapped)
  with pytest.raises(ValueError,match='positive prospective cap'):verify(uncapped,source)


def test_selection_uses_fixed_public_row_identity_and_mixture():
 rows=[dict(seed=i,semantic_sha256=str(i),alpha_sha256=str(i),arity=3 if i<64 else 4,facts=4 if i>=96 else 3) for i in range(128)]
 items=[dict(index=i,**{k:r[k] for k in ('seed','semantic_sha256','alpha_sha256')}) for i,r in enumerate(rows)]
 assert selected_rows(rows,items)==rows
 wrong=copy.deepcopy(items);wrong[0]['seed']=-1
 with pytest.raises(ValueError,match='selection mismatch'):selected_rows(rows,wrong)
 with pytest.raises(ValueError):selected_rows(rows,items[::-1])


@pytest.mark.parametrize('timeout',[False,True])
def test_atomic_failed_lifecycle_receipt_no_retry(monkeypatch,tmp_path,timeout):
 import sys,subprocess
 from types import SimpleNamespace
 from topoformer import campaign_semantics_s18_launch as launch
 config=tmp_path/'config.json';config.write_text(json.dumps({'output_dir':str(tmp_path/'new')}))
 monkeypatch.setitem(sys.modules,'campaign_semantics_s18_freeze',SimpleNamespace(verify=lambda *a:None))
 monkeypatch.setattr(sys,'argv',['launch',str(config),'--cap','180','--python','fake-python','--prefix',str(tmp_path/'attempt')])
 monkeypatch.setattr(launch.subprocess,'check_output',lambda command,**kw:'' if command[0]=='nvidia-smi' else 'mock full process snapshot')
 calls=[]
 def fake_run(command,**kw):
  calls.append(command)
  assert kw['env']['PYTHONPATH']==str(Path(launch.__file__).resolve().parent.parent)
  assert Path(kw['env']['PYTHONPATH']).is_absolute()
  if timeout:raise subprocess.TimeoutExpired(command,kw['timeout'])
  return SimpleNamespace(returncode=17)
 monkeypatch.setattr(launch.subprocess,'run',fake_run)
 with pytest.raises(SystemExit) as caught:launch.main()
 assert caught.value.code==(124 if timeout else 17)
 receipt=json.loads((tmp_path/'attempt.occupancy.json').read_text())
 assert receipt['timed_out']==timeout and receipt['exit_code']==caught.value.code
 assert receipt['full_process_state']=='mock full process snapshot'
 assert len(calls)==1 and calls[0][2]=='topoformer.campaign_semantics_s18'
 assert (tmp_path/'attempt.started.json').exists() and not list(tmp_path.glob('*.tmp'))


def test_replay_accepts_bijection_and_rejects_population_or_prediction_changes():
 from topoformer.campaign_semantics_s18 import verify_replay
 old=[dict(semantic_sha256=str(i),raw={'value':i},target={'value':i+1}) for i in range(4)]
 verify_replay(old[::-1],old)
 for wrong in (old[:-1],old+[old[0]],old[:-1]+[old[0]]):
  with pytest.raises(ValueError,match='population'):verify_replay(wrong,old)
 for field in ('raw','target'):
  wrong=copy.deepcopy(old[::-1]);wrong[0][field]={'value':-1}
  with pytest.raises(ValueError,match='replay mismatch'):verify_replay(wrong,old)
