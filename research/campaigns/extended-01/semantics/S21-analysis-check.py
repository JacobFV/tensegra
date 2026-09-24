"""Synthetic prospective S21 full-loader guards; no cache/main-result reads."""
import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S21-analysis.py');s=importlib.util.spec_from_file_location('s21_analysis',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def fixture(self):
  exposure=dict(tokens=10,nodes=20,edges=30,records=40);c=dict(expected_construction_sequence_sha256='a'*64,expected_epoch_exposures={'original':exposure})
  r=dict(arm='original',seed=2101,parameters=62677315,inherited_presentations=0,added_presentations=32768,visits=[8]*4096,construction_sequence_sha256='a'*64,exposure={k:v*8 for k,v in exposure.items()},initial_state_sha256='b'*64,final_state_sha256='b'*64,worst_case_timing=None)
  r['curves']=[dict(update=u,presentations=u*8,exposure={k:v*(u//512) for k,v in exposure.items()},checkpoint=f'model-u{u}.pt',checkpoint_sha256='c'*64,model_state_sha256='b'*64) for u in a.STEPS]
  fields=('type','kind','value','copy','source','target','role','slot');r['losses']=[dict(arm='original',update=u,loss=1.,field_loss_sums=dict.fromkeys(fields,1.),field_counts=dict.fromkeys(fields,1)) for u in range(128,4097,128)];return r,c
 def test_arm(self):
  r,c=self.fixture();a.arm_guard(r,c,'original')
  for key,value in [('seed',2102),('visits',[8]*4095),('inherited_presentations',1),('construction_sequence_sha256','d'*64),('exposure',{}),('curves',r['curves'][:-1]),('losses',[])]:
   with self.assertRaises(ValueError):a.arm_guard({**r,key:value},c,'original')
 def test_curve(self):
  r,c=self.fixture()
  for key,value in [('presentations',9),('exposure',{}),('checkpoint_sha256','x'),('model_state_sha256','d'*64)]:
   z=copy.deepcopy(r);z['curves'][0][key]=value
   with self.assertRaises(ValueError):a.arm_guard(z,c,'original')
 def test_public_identity(self):
  e=dict(semantic_sha256='a'*64,seed=7,graph_sha256='b'*64,arity=3,facts=4);r=dict(semantic_sha256='a'*64,seed=7,graph_sha256='b'*64,cell='3x4',valid=True,complete=True,reason=None);a.row_guard(r,e)
  for key,value in [('seed',8),('graph_sha256','c'*64),('cell','4x3'),('valid',1),('complete',1),('valid',False),('reason','bad')]:
   with self.assertRaises(ValueError):a.row_guard({**r,key:value},e)
 def test_invalid(self):
  e=dict(semantic_sha256='a'*64,seed=7,graph_sha256='b'*64,arity=3,facts=4);r=dict(semantic_sha256='a'*64,seed=7,graph_sha256='b'*64,cell='3x4',valid=False,complete=False,reason='missing_eos');a.row_guard(r,e)
  for value in (None,'',False):
   with self.assertRaises(ValueError):a.row_guard({**r,'reason':value},e)
 def test_helper_pins(self):
  for name in a.PINS:a.helper(name)
if __name__=='__main__':unittest.main()
