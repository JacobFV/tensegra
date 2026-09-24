"""Synthetic prospective fixtures; reads no model results."""
import importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S19-analysis.py');s=importlib.util.spec_from_file_location('s19',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def test_linkage(self):
  m=dict(optimizer_records=3229392,optimizer_tokens=1720320,optimizer_nodes=988008,optimizer_edges=2208616,initial_state_sha256='a'*64)
  m['curves']=[dict(update=u,presentations=u*8,optimizer_records=m['optimizer_records']*u//4096,optimizer_tokens=m['optimizer_tokens']*u//4096,checkpoint_sha256='b'*64,model_state_sha256='a'*64) for u in a.STEPS]
  a.linkage_guard(m)
  import copy
  for key in ('optimizer_records','optimizer_tokens','optimizer_nodes','optimizer_edges'):
   bad=copy.deepcopy(m);bad[key]+=1
   with self.assertRaises(ValueError):a.linkage_guard(bad)
  for key,value in [('checkpoint_sha256','x'),('model_state_sha256','b'*64),('optimizer_records',1)]:
   bad=copy.deepcopy(m);bad['curves'][0][key]=value
   with self.assertRaises(ValueError):a.linkage_guard(bad)
 def test_entry(self):
  d=dict(rows=[dict(cell='3x3',complete=False)],teacher_forced_loss=1.0);e=dict(examples=1,teacher_forced_loss=1.0,cells={'3x3':dict(examples=1,complete=0)})
  a.entry_guard(e,d)
  for key,value in [('examples',2),('teacher_forced_loss',2.),('cells',{})]:
   with self.assertRaises(ValueError):a.entry_guard({**e,key:value},d)
 def test_promotion(self):
  final=dict.fromkeys(a.CELLS,52);early=dict.fromkeys(a.CELLS,0)
  self.assertTrue(a.decisions(64,early,final)['promotion'])
  self.assertFalse(a.decisions(63,early,final)['promotion'])
  for cell in ('3x3','4x3','4x4'):
   self.assertFalse(a.decisions(64,early,{**final,cell:51})['promotion'])
  self.assertTrue(a.decisions(64,early,{**final,'3x4':0})['promotion'])
  self.assertFalse(a.decisions(64,early,{**final,'3x4':51})['heldout_composition'])
 def test_extension(self):
  # Weighted numerator103/2048 first clears5%; gain62/2048 first clears3pp.
  z=dict.fromkeys(a.CELLS,0);f={**z,'3x3':40,'4x3':23};e={**z,'4x3':41}
  self.assertTrue(a.decisions(64,e,f)['extension_eligible'])
  self.assertFalse(a.decisions(64,{**e,'4x3':42},f)['extension_eligible'])
  self.assertFalse(a.decisions(64,z,{**f,'4x3':22})['extension_eligible'])
  self.assertFalse(a.decisions(63,e,f)['extension_eligible'])
  self.assertFalse(a.decisions(64,e,f)['automatic_extension'])
 def test_transition(self):self.assertEqual(a.paired([0,0,1,1],[0,1,0,1]),{'0->0':1,'0->1':1,'1->0':1,'1->1':1})
 def test_reject_counts(self):
  for train in (-1,129,64.0):
   with self.assertRaises(ValueError):a.decisions(train,dict.fromkeys(a.CELLS,0),dict.fromkeys(a.CELLS,0))
  with self.assertRaises(ValueError):a.decisions(64,{},dict.fromkeys(a.CELLS,0))
 def test_duplicate_events(self):
  with self.assertRaises(ValueError):a.indexed([{'semantic_sha256':'x'}]*2)
 def test_invalid_zero(self):
  row=dict(valid=False,complete=False,reason='missing EOS',exact_components=dict.fromkeys(a.COMPONENTS,False))
  summary=a.summarize([row]);self.assertEqual(summary['invalid_reasons'],{'missing EOS':1});self.assertEqual(summary['macro_graph_f1_invalid_zero'],dict.fromkeys(('node','typed_edge','ordered_edge'),0))
if __name__=='__main__':unittest.main()
