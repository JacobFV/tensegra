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
 def test_source_config_wrapper_metadata(self):
  c={'source_sha256':{name:'a'*64 for name in m.SOURCES}};r={'wrapper_sha256':'a'*64};m.metadata_guard(c,r,'b'*64,'b'*64)
  for changed in ({k:v for k,v in c['source_sha256'].items() if k!=m.SOURCES[0]},dict(c['source_sha256'],extra='a'*64)):
   with self.assertRaises(ValueError):m.metadata_guard({'source_sha256':changed},r,'b'*64,'b'*64)
  for receipt,actual,pin in [({'wrapper_sha256':'z'*64},'b'*64,'b'*64),(r,'c'*64,'b'*64),(r,'b'*64,None)]:
   with self.assertRaises(ValueError):m.metadata_guard(c,receipt,actual,pin)
 def test_final_state_metadata(self):
  r={'curves':[{'added_update':4096,'model_state_sha256':'a'}],'final_state_sha256':'a'};m.final_state_guard(r)
  for key,value in [('added_update',2048),('model_state_sha256','b')]:
   d=copy.deepcopy(r);d['curves'][-1][key]=value
   with self.assertRaises(ValueError):m.final_state_guard(d)
 def evaluation_fixture(self):
  metric={'semantic_equivalence':1.,'identity_copy_accuracy':1.,'typed_edge':{'true_positive':1,'predicted_count':1,'gold_count':1},'ordered_edge':{'true_positive':1,'predicted_count':1,'gold_count':1}}
  d={'update':28672,'calibration_data_artifact':'calibration-u28672.npz','calibration_data_sha256':'a'*64,'thresholds':[0.]*13,'calibration':[{'threshold':0.} for _ in range(13)],'rows':[{'raw_metrics':metric,'calibrated_metrics':metric}],'train_metrics':[{'raw':metric,'calibrated':metric}]}
  e={'artifact':'evaluation-u28672.json.gz','update':28672,**{k:{'examples':1,'exact':1.,'copy':1.,'typed_edge_f1':1.,'ordered_edge_f1':1.} for k in ('dev_raw','dev_calibrated','train_raw','train_calibrated')}}
  return d,e
 def test_evaluation_metadata_mutations(self):
  d,e=self.evaluation_fixture();m.evaluation_guard(d,e,e['artifact'])
  for key,value in [('update',1),('artifact','other')]:
   z=copy.deepcopy(e);z[key]=value
   with self.assertRaises(ValueError):m.evaluation_guard(d,z,e['artifact'])
  for key,value in [('calibration_data_artifact','other.npz'),('calibration_data_sha256','short'),('thresholds',[1.]*13)]:
   z=copy.deepcopy(d);z[key]=value
   with self.assertRaises(ValueError):m.evaluation_guard(z,e,e['artifact'])
  for name in ('exact','examples','copy','typed_edge_f1','ordered_edge_f1'):
   z=copy.deepcopy(e);z['dev_raw'][name]=0
   with self.assertRaises(ValueError):m.evaluation_guard(d,z,e['artifact'])
 def test_calibration_array_linkage(self):
  import io
  def arrays(**changes):
   buf=io.BytesIO();np.savez(buf,**dict({'scores':np.zeros((128,13)),'targets':np.zeros((128,13),bool),'pairs':np.zeros((128,2),int),'offsets':np.arange(129)},**changes));buf.seek(0);return np.load(buf,allow_pickle=False)
  d={'calibration_records':[{'start':i,'stop':i+1,'seed':i} for i in range(128)],'train_rows':[{'seed':i} for i in range(128)]}
  with arrays() as a:m.calibration_array_guard(d,a)
  for change in ({'targets':np.zeros((127,13),bool)},{'offsets':np.arange(128)},{'pairs':np.zeros((128,3),int)}):
   with arrays(**change) as a:
    with self.assertRaises(ValueError):m.calibration_array_guard(d,a)
  z=copy.deepcopy(d);z['calibration_records'][0]['seed']=999
  with arrays() as a:
   with self.assertRaises(ValueError):m.calibration_array_guard(z,a)
 def test_component_masks_and_policy_separation(self):
  a=m.helper();g={'presence':np.array([True,True,False]),'kind':np.array([1,2,0]),'value':np.array([-1,1,-1]),'copy':np.array([-1,0,-1]),'edges':np.zeros((3,3,1),bool),'slots':np.zeros((3,3),int)};g['edges'][0,1,0]=True
  p={k:v.copy() for k,v in g.items()};p['edges'][2,0,0]=True;self.assertTrue(all(a.components(p,g).values()))
  p['slots'][0,1]=1;self.assertFalse(a.components(p,g)['slots']);self.assertTrue(a.components(p,g)['edges'])
  p['edges'][0,1,0]=False;self.assertFalse(a.components(p,g)['edges'])
if __name__=='__main__':unittest.main()
