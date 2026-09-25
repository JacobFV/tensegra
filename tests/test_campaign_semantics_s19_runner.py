import copy,importlib.util,json,math,tempfile,unittest,os,subprocess,sys,hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from tensegra.campaign_semantics_s19 import FIELDS,field_losses,learning_rate,padded_records,evaluate,decode_evaluation,worst_case_timing
from tensegra.campaign_semantics_s19_codec import NODE,EDGE,EOS,PAD,KINDS,ROLES,records_to_targets
from tensegra.campaign_semantics_s19_freeze import validate
from tensegra import campaign_semantics_s19_freeze as freeze_module
from tensegra.thinking_language import ActorInput
class S19Runner(unittest.TestCase):
 def logits(self,shape):
  return {k:torch.randn(*shape,c,requires_grad=True) for k,c in [('type',3),('kind',len(KINDS)),('value',3),('copy',5),('source',128),('target',128),('role',len(ROLES)),('slot',33)]}
 def test_exact_family_loss_and_masks(self):
  r=padded_records([[(NODE,KINDS.index('record'),1,-1,-1),(NODE,KINDS.index('ident'),-1,2,-1),(EDGE,0,1,0,-1),(EOS,-1,-1,-1,-1)],[(NODE,KINDS.index('record'),0,-1,-1),(EOS,-1,-1,-1,-1)]],'cpu');o=self.logits(r.shape[:2]);loss,sums,counts,correct=field_losses(o,r)
  self.assertEqual(counts,dict(type=6,kind=3,value=2,copy=1,source=1,target=1,role=1,slot=1))
  self.assertTrue(torch.allclose(loss,torch.stack([sums[k]/counts[k] for k in FIELDS]).mean()))
  self.assertTrue(torch.allclose(sums['slot'],F.cross_entropy(o['slot'][0,2][None],torch.tensor([0]),reduction='sum')))
  self.assertTrue(torch.allclose(sums['copy'],F.cross_entropy(o['copy'][0,1][None],torch.tensor([2]),reduction='sum')))
  changed={k:v.detach().clone().requires_grad_() for k,v in o.items()}
  with torch.no_grad():
   for v in changed.values():v[1,2:]=123
  self.assertTrue(torch.allclose(field_losses(changed,r)[0],loss))
  loss.backward();self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in o.values()))
 def test_absent_families_zero_with_fixed_denominator(self):
  r=torch.tensor([[[EOS,-1,-1,-1,-1]]]);o=self.logits((1,1));o['copy']=torch.tensor([[[0.,-torch.inf,-torch.inf,-torch.inf,-torch.inf]]],requires_grad=True);loss,sums,counts,_=field_losses(o,r)
  self.assertTrue(torch.allclose(loss,sums['type']/8));self.assertEqual(sum(counts.values()),1);loss.backward();self.assertTrue(torch.isfinite(o['copy'].grad).all())
 def test_warmup_cosine_boundaries(self):
  self.assertAlmostEqual(learning_rate(1),3e-4/128);self.assertEqual(learning_rate(128),3e-4);self.assertAlmostEqual(learning_rate(4096),3e-5)
  self.assertTrue(all(learning_rate(i+1)<=learning_rate(i) for i in range(128,4096)))
  for value in (0,4097):
   with self.assertRaises(ValueError):learning_rate(value)
 def test_exact_historical_construction_stream(self):
  from tensegra.campaign_semantics_continue import next_indices
  g=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=g).tolist();position=0;h=hashlib.sha256();visits=[0]*4096
  for _ in range(4096):
   indices,order,position=next_indices(order,position,g,8);h.update(json.dumps(indices).encode())
   for i in indices:visits[i]+=1
  self.assertEqual(set(visits),{8});self.assertEqual(h.hexdigest(),'85054bf25e3a3e2a5a9a932b441d6413fd0d7ad827df79e1583268aaddf5acd9')
 def test_forced_timing_is_public_only_and_160_steps(self):
  from tensegra.campaign_semantics_s19_actor import TypedRecordActor
  model=TypedRecordActor(value_count=3,width=16,heads=4,max_records=160);before={k:v.clone() for k,v in model.state_dict().items()}
  with patch('torch.cuda.synchronize'):result=worst_case_timing(model,[ActorInput('alice x',()),ActorInput('bob y',())])
  self.assertEqual(result['steps'],160);self.assertEqual([len(r) for r in result['records']],[160,160]);self.assertTrue(model.training);self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
 def test_recipe_rejection(self):
  c=json.loads(Path('configs/campaign-s19-main-prepared-v1.json').read_text());validate(c)
  for key,value in [('seed',1902),('updates',1024),('evaluation_batch_size',8),('loss_policy','sum'),('calibration','fit'),('schedule_seed',1901),('worst_case_profile',True)]:
   with self.assertRaises(ValueError):validate({**c,key:value})
 def test_immutable_source_cap_and_input_guards(self):
  c=json.loads(Path('configs/campaign-s19-profile-prepared-v1.json').read_text())
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source';source.mkdir()
   for name in freeze_module.SOURCES:(source/name).write_text(name)
   for k in c['inputs']:c['inputs'][k]['path']=str(root/('input-'+k))
   c['output_dir']=str(root/'unused');prepared=root/'prepared.json';prepared.write_text(json.dumps(c));frozen=root/'frozen.json';real_digest=freeze_module.digest
   def fake_input_digest(path):
    for k,r in c['inputs'].items():
     if str(path)==r['path']:return r['sha256']
    return real_digest(path)
   with patch.object(freeze_module,'digest',side_effect=fake_input_digest):
    config=freeze_module.freeze(prepared,frozen,source,180);freeze_module.verify(config,source,180)
    with self.assertRaises(FileExistsError):freeze_module.freeze(prepared,frozen,source,180)
    with self.assertRaises(ValueError):freeze_module.verify(config,source,181)
    with self.assertRaises(ValueError):freeze_module.verify(c,source,180)
    corrupt=copy.deepcopy(config);corrupt['inputs']['train']['sha256']='0'*64
    with self.assertRaises(ValueError):freeze_module.verify(corrupt,source,180)
    (source/freeze_module.SOURCES[0]).write_text('changed')
    with self.assertRaises(ValueError):freeze_module.verify(config,source,180)
 def test_launcher_failure_and_timeout_are_durable(self):
  import tensegra.campaign_semantics_s19_launch as launcher
  for body,expected in [('raise RuntimeError("synthetic child failure")',1),('import time;time.sleep(2)',124)]:
   with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);package=root/'src/topoformer';package.mkdir(parents=True);(package/'__init__.py').write_text('');(package/'campaign_semantics_s19_launch.py').write_text(Path(launcher.__file__).read_text());(package/'campaign_semantics_s19_freeze.py').write_text('def verify(*args): return {}');(package/'campaign_semantics_s19.py').write_text(body)
    binary=root/'bin';binary.mkdir();nv=binary/'nvidia-smi';nv.write_text('#!/bin/sh\nexit 0\n');nv.chmod(0o755);config=root/'c.json';config.write_text('{}');prefix=root/'job'
    command=[sys.executable,str(package/'campaign_semantics_s19_launch.py'),str(config),'--cap','0.3','--python',sys.executable,'--prefix',str(prefix)];result=subprocess.run(command,env={**os.environ,'PATH':str(binary)+os.pathsep+os.environ['PATH']},capture_output=True,text=True,timeout=5)
    self.assertEqual(result.returncode,expected,result.stderr);receipt=json.loads(Path(str(prefix)+'.occupancy.json').read_text());self.assertEqual(receipt['exit_code'],expected);self.assertEqual(receipt['timed_out'],expected==124);self.assertTrue(Path(str(prefix)+'.started.json').exists());self.assertFalse(Path(str(prefix)+'.occupancy.json.tmp').exists())
 def test_public_greedy_and_invalid_failure(self):
  records=[(NODE,KINDS.index('record'),1,-1,-1),(EOS,-1,-1,-1,-1)];gold=records_to_targets(records,token_count=1,vocab_size=2);public=ActorInput('x',())
  valid=decode_evaluation(public,dict(records=records,status='eos'),gold,2);self.assertTrue(valid['complete'])
  bad=decode_evaluation(public,dict(records=[(EDGE,0,0,0,-1),(EOS,-1,-1,-1,-1)],status='eos'),gold,2);self.assertFalse(bad['valid']);self.assertIsNone(bad['prediction']);self.assertFalse(any(bad['exact_components'].values()))
  owner=self
  class Spy(nn.Module):
   def __init__(self):super().__init__();self.features=nn.Linear(1,1);self.free_calls=0
   def teacher_forced(self,publics,target_records):
    owner.assertTrue(all(isinstance(p,ActorInput) for p in publics));return owner.logits(target_records.shape[:2])
   def greedy(self,publics):
    owner.assertTrue(all(type(p) is ActorInput for p in publics));self.free_calls+=1;return [dict(records=records,status='eos') for p in publics]
  example=dict(public=public,records=records,gold=gold,row=dict(seed=1,semantic_sha256='s',graph_sha256='g',arity=3,facts=3));model=Spy()
  with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.synchronize'):
   result=evaluate(model,[example],2,Path(tmp),'train',0,32);self.assertEqual(result['complete'],1);self.assertEqual(model.free_calls,1);self.assertTrue(model.training)
if __name__=='__main__':unittest.main()
