"""Synthetic source-only loader checks. No confirmation/artifact/model access."""
import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S20-analysis-loader.py');s=importlib.util.spec_from_file_location('s20loader',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def fixture(self,arm):
  record=arm=='record';c={'parents':{'701':dict(checkpoint={'sha256':'b'*64},model_state_sha256='a'*64)}}
  r=dict(seed=701,arm=arm,added_update=4096,update=4096 if record else 28672,added_presentations=32768,visits=[8]*4096,construction_sequence_sha256=a.STREAM,pair_sequence_sha256=None if record else a.PAIRS,optimizer_tokens=1720320,optimizer_nodes=988008,optimizer_edges=2208616,optimizer_records=3229392 if record else None,parameters=62677315 if record else 57853781,inherited_presentations=0 if record else 196608,initial_optimizer_states=0 if record else 98,inherited_optimizer_steps=[] if record else [24576.],parent_checkpoint_sha256=None if record else 'b'*64,initial_state_sha256='a'*64,final_state_sha256='c'*64,checkpoint_sha256='d'*64,checkpoint=f'model-u{4096 if record else 28672}.pt',evaluation_population='confirmation',worst_case_timing=None,losses=[dict(added_update=u,seed=701,arm=arm,loss=1.) for u in range(128,4097,128)])
  return r,c
 def test_all_arms(self):
  for arm in ('original','context','record'):
   r,c=self.fixture(arm);a.state_guard(r,c)
   for key,value in [('added_update',20),('visits',[8]*4095),('pair_sequence_sha256','x'),('optimizer_tokens',1),('parameters',1),('inherited_optimizer_steps',[0.]),('checkpoint_sha256','x'),('evaluation_population','development'),('losses',[])]:
    with self.assertRaises(ValueError):a.state_guard({**r,key:value},c)
 def test_parent(self):
  r,c=self.fixture('context')
  with self.assertRaises(ValueError):a.state_guard({**r,'initial_state_sha256':'f'*64},c)
  with self.assertRaises(ValueError):a.state_guard({**r,'parent_checkpoint_sha256':'f'*64},c)
 def test_targets(self):
  expected={'e':dict(seed=1,graph_sha256='g')};rows=[dict(semantic_sha256='e',seed=1,graph_sha256='g',target={'x':1})];targets={};a.identity_guard(rows,expected,targets)
  with self.assertRaises(ValueError):a.identity_guard([{**rows[0],'target':{'x':2}}],expected,targets)
  with self.assertRaises(ValueError):a.identity_guard(rows*2,expected,{})
 def test_unfrozen(self):
  self.assertIsNone(a.MAIN_CONFIG_SHA)
  with self.assertRaises(ValueError):a.checked(Path(__file__),None)
if __name__=='__main__':unittest.main()
