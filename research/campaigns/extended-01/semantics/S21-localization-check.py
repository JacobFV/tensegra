"""Small synthetic-only checks; never reads experiment outputs."""
import base64,copy,importlib.util,time,unittest
from pathlib import Path
p=Path(__file__).with_name('S21-localization.py');spec=importlib.util.spec_from_file_location('localize',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row():
 edges=bytearray(1);edges[0]|=1<<2
 return dict(semantic_sha256='x',cell='3x3',valid=True,reason=None,complete=True,records=[[1,7,1,-1,-1],[1,5,-1,0,-1],[2,0,1,0,0],[3,-1,-1,-1,-1]],target=dict(presence=[True,True],kind=[7,5],value=[1,-1],copy=[-1,0],slots=[[-1,0],[-1,-1]],edges=dict(shape=[2,2,2],bitorder='little',packed_b64=base64.b64encode(edges).decode())))
def matrix():
 c=dict(job='main',budget_status='frozen',updates=4096,checkpoints=list(m.STEPS),arms=list(m.ARMS),dev_per_cell=512,width=1024);r=dict(exit_code=0,timed_out=False,error=None,config_sha256='hash');results=[]
 for a in m.ARMS:results.append(dict(arm=a,seed=2101,inherited_presentations=0,added_presentations=32768,visits=[8]*4096,initial_state_sha256='same',final_state_sha256='final',curves=[dict(update=s,model_state_sha256='final') for s in m.STEPS]))
 return dict(config=c,results=results),r,c
def guarded_row():
 r=row();r.update(prediction={},metrics={'semantic_equivalence':1.},exact_components=dict.fromkeys(m.COMPONENTS,True))
 t=r['target'];t['presence'] += [False]*126;t['kind'] += [0]*126;t['value'] += [-1]*126;t['copy'] += [-1]*126;t['slots']=[[-1]*128 for _ in range(128)];t['slots'][0][1]=0;buf=bytearray(128*128*13//8);buf[13//8]|=1<<(13%8);t['edges']=dict(shape=[128,128,13],bitorder='little',packed_b64=base64.b64encode(buf).decode());return r
class Tests(unittest.TestCase):
 def describe(self,r):return m.describe(r,['x','x'],{'3x3':(7,5)})
 def test_correct(self):self.assertEqual(self.describe(row())['category'],'complete')
 def test_copy_occurrence(self):r=row();r['records'][1][3]=1;self.assertTrue(self.describe(r)['all_nodes_correct'])
 def test_wrong_kind(self):r=row();r['records'][1][1]=7;r['complete']=False;self.assertEqual(self.describe(r)['first_node_fields'],['kind'])
 def test_missing_node(self):r=row();del r['records'][1];r['complete']=False;self.assertEqual(self.describe(r)['first_node_fields'],['missing_node'])
 def test_slot_only(self):r=row();r['records'][2][-1]=1;r['complete']=False;d=self.describe(r);self.assertEqual(d['slot_only_missing_count'],1);self.assertEqual(d['category'],'edge_or_slot')
 def test_invalid_keeps_prefix(self):r=row();r.update(valid=False,complete=False,reason='duplicate_edge');r['records'].insert(3,r['records'][2][:]);d=self.describe(r);self.assertEqual(d['duplicate_edge_count'],1);self.assertTrue(d['all_nodes_correct']);self.assertEqual(d['category'],'invalid_termination_or_record')
 def test_empty_prefix(self):r=row();r.update(records=[],valid=False,complete=False,reason='missing_eos');self.assertEqual(self.describe(r)['first_node_index'],0)
 def test_eos_failure(self):r=row();r['records'].pop();r.update(valid=False,complete=False,reason='missing_eos');self.assertFalse(self.describe(r)['eos_present'])
 def test_complete_guard(self):a,b,c=matrix();m.matrix_guard(a,b,c,'hash')
 def test_missing_arm(self):a,b,c=matrix();a['results'].pop();self.assertRaises(ValueError,m.matrix_guard,a,b,c,'hash')
 def test_missing_checkpoint(self):a,b,c=matrix();a['results'][0]['curves'].pop();self.assertRaises(ValueError,m.matrix_guard,a,b,c,'hash')
 def test_timeout(self):a,b,c=matrix();b['timed_out']=True;self.assertRaises(ValueError,m.matrix_guard,a,b,c,'hash')
 def test_profile(self):a,b,c=matrix();c['job']='profile';self.assertRaises(ValueError,m.matrix_guard,a,b,c,'hash')
 def test_duplicate_identity(self):self.assertRaises(ValueError,m.indexed,[row(),row()])
 def test_bad_width(self):a,b,c=matrix();c['width']=32;self.assertRaises(ValueError,m.matrix_guard,a,b,c,'hash')
 def test_public_binding(self):
  r=row();r.update(seed=4,graph_sha256='g');p=dict(seed=4,graph_sha256='g',arity=3,facts=3);m.public_guard(r,p);p['seed']=5;self.assertRaises(ValueError,m.public_guard,r,p)
 def test_public_cell(self):
  r=row();r.update(seed=4,graph_sha256='g');self.assertRaises(ValueError,m.public_guard,r,dict(seed=4,graph_sha256='g',arity=4,facts=3))
 def test_boolean_flags(self):
  r=guarded_row();m.row_guard(r);r['valid']=1;self.assertRaises(ValueError,m.row_guard,r)
 def test_invalid_no_credit(self):
  r=guarded_row();r.update(valid=False,reason='bad');self.assertRaises(ValueError,m.row_guard,r)
 def test_exact_link(self):
  r=guarded_row();r['exact_components']['copy']=False;self.assertRaises(ValueError,m.row_guard,r)
 def test_packed_length(self):
  r=row();r['target']['edges']['packed_b64']=base64.b64encode(b'\x04\x00').decode();self.assertRaises(ValueError,m.target_edges,r['target'])
 def test_packed_padding(self):
  t=dict(slots=[[-1]],edges=dict(shape=[1,1,1],bitorder='little',packed_b64=base64.b64encode(b'\x02').decode()));self.assertRaises(ValueError,m.target_edges,t)
 def test_nonsquare(self):
  r=row();r['target']['edges']['shape']=[2,1,2];self.assertRaises(ValueError,m.target_edges,r['target'])
 def test_impossible_tag(self):
  r=row();r['records'][0][0]=0;self.assertRaises(ValueError,self.describe,r)
 def test_post_eos(self):
  r=row();r['records'].append([1,5,-1,0,-1]);self.assertRaises(ValueError,self.describe,r)
 def test_bad_eos_payload(self):
  r=row();r['records'][-1][1]=0;self.assertRaises(ValueError,self.describe,r)
 def test_nonleading_nodes_retained(self):
  r=row();r['records'].insert(3,[1,5,-1,0,-1]);r.update(valid=False,complete=False,reason='node_after_edge');d=self.describe(r);self.assertEqual(d['nonleading_node_records'],1);self.assertEqual(d['generated_leading_nodes'],2)
 def test_entry_counts(self):
  r=row();fields={k:dict(count=1,correct=1,loss_sum=.5,mean_loss=.5,accuracy=1.) for k in ('type','kind','value','copy','source','target','role','slot')};d=dict(rows=[r],teacher_forced=fields,teacher_forced_loss=.5);entry=dict(examples=1,teacher_forced=fields,teacher_forced_loss=.5,complete=1,valid=1,invalid_reasons={},cells={'3x3':dict(examples=1,complete=1)});m.entry_guard(entry,d);entry['cells']['3x3']['complete']=0;self.assertRaises(ValueError,m.entry_guard,entry,d)
if __name__=='__main__':
 start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));print({'cpu_wall_seconds':time.monotonic()-start,'tests':result.testsRun,'passed':result.wasSuccessful()});raise SystemExit(not result.wasSuccessful())
