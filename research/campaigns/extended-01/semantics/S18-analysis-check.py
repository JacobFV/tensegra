"""Synthetic S18 analysis guards; imports no model or campaign outcomes."""
import copy,importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('S18-analysis.py');spec=importlib.util.spec_from_file_location('s18_analysis',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Checks(unittest.TestCase):
 def counts(self):return {'original':{'3x3':26,'3x4':26,'4x3':103,'4x4':51},'workspace_control':{'3x3':26,'3x4':26,'4x3':103,'4x4':51},'context':{'3x3':52,'3x4':52,'4x3':103,'4x4':0}}
 def test_gates_boundaries(self):
  c=self.counts();self.assertTrue(m.decisions(c)['recombination_criteria_met'])
  for arm,cell,value,gate in [('context','3x3',51,'acquisition'),('original','3x3',27,'acquisition'),('workspace_control','3x3',27,'acquisition'),('context','4x3',102,'retention'),('original','4x4',52,'retention'),('workspace_control','4x4',52,'retention'),('context','3x4',51,'recombination'),('original','3x4',27,'recombination'),('workspace_control','3x4',27,'recombination')]:
   d=copy.deepcopy(c);d[arm][cell]=value;self.assertFalse(m.decisions(d)[gate])
  self.assertFalse(m.decisions(c)['automatic_extension'])
 def test_missing_arms_and_cells(self):
  for arm in m.ARMS:
   c=self.counts();del c[arm]
   with self.assertRaises(ValueError):m.decisions(c)
  c=self.counts();del c['context']['3x4']
  with self.assertRaises(ValueError):m.decisions(c)
 def test_event_alignment(self):
  rows=[{'semantic_sha256':'a'},{'semantic_sha256':'b'}];self.assertEqual(set(m.index_rows(rows,{'a':0,'b':0})),{'a','b'})
  for bad in (rows[:1],rows+[rows[0]],[{'semantic_sha256':'x'},rows[1]]):
   with self.assertRaises(ValueError):m.index_rows(bad,{'a':0,'b':0})
 def test_stream_length_counts_and_hashes(self):
  s={'visits':[8]*4096,'inherited_optimizer_steps':[24576.],'order':'fixed'};m.stream_guard(s,{'order':'fixed'})
  for key,value in [('visits',[8]*4095),('visits',[8]*4095+[7]),('inherited_optimizer_steps',[24577.]),('order','changed')]:
   d={**s,key:value}
   with self.assertRaises(ValueError):m.stream_guard(d,{'order':'fixed'})
 def test_transitions_and_shared_bootstrap(self):
  self.assertEqual(m.transitions([0,0,1,1],[0,1,0,1]),{'0->0':1,'0->1':1,'1->0':1,'1->1':1})
  with self.assertRaises(ValueError):m.transitions([0],[0,1])
  draws=np.array([[0,1],[1,1],[0,0]]);r=m.bootstrap({'a':[1,-1],'b':[1,-1],'zero':[0,0]},draws)
  self.assertEqual(r['a'],r['b']);self.assertEqual(r['zero'],[0.,0.]);self.assertEqual(r['a'],[-95.,95.])
 def test_component_masks_and_policy_separation(self):
  a=m.helper();g={'presence':np.array([True,True,False]),'kind':np.array([1,2,0]),'value':np.array([-1,1,-1]),'copy':np.array([-1,0,-1]),'edges':np.zeros((3,3,1),bool),'slots':np.zeros((3,3),int)};g['edges'][0,1,0]=True
  p={k:v.copy() for k,v in g.items()};p['edges'][2,0,0]=True;self.assertTrue(all(a.components(p,g).values()))
  p['slots'][0,1]=1;self.assertFalse(a.components(p,g)['slots']);self.assertTrue(a.components(p,g)['edges'])
  p['edges'][0,1,0]=False;self.assertFalse(a.components(p,g)['edges'])
if __name__=='__main__':unittest.main()
