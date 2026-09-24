"""Synthetic prospective fixtures; reads no model results."""
import importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S19-analysis.py');s=importlib.util.spec_from_file_location('s19',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
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
