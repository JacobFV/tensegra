import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('S22-analysis.py');s=importlib.util.spec_from_file_location('s22',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class Checks(unittest.TestCase):
 def fixture(self,policy):
  prefix=policy=='oracle_node_prefix';kinds=policy!='oracle_node_count'
  stats=dict(policy=policy,supplied_node_count=1,forced_node_tags=1,tag_replacements=0,forced_node_kinds=int(kinds),kind_replacements=0,supplied_prefix_records=int(prefix),prefix_value_replacements=0,prefix_copy_replacements=0,node_suppression_steps=1,node_suppression_replacements=0)
  return dict(valid=True,complete=True,reason=None,target=dict(presence=[True],kind=[5],value=[-1],copy=[0]),records=[[1,5,-1,0,-1],[3,-1,-1,-1,-1]],controller_stats=stats)
 def test_policies(self):
  for p in a.POLICIES:a.stats_guard(self.fixture(p),p)
 def test_counters(self):
  for p in a.POLICIES:
   for k in ('supplied_node_count','forced_node_tags','forced_node_kinds','supplied_prefix_records','node_suppression_steps'):
    r=self.fixture(p);r['controller_stats'][k]+=1
    with self.assertRaises(ValueError):a.stats_guard(r,p)
 def test_no_future_or_wrong_kind(self):
  for policy in a.POLICIES:
   r=self.fixture(policy);r['records'].append([2,0,0,0,-1]);r['controller_stats']['node_suppression_steps']+=1
   with self.assertRaises(ValueError):a.stats_guard(r,policy)
  r=self.fixture('oracle_node_kinds');r['records'][0][1]=6
  with self.assertRaises(ValueError):a.stats_guard(r,'oracle_node_kinds')
  r=self.fixture('oracle_node_prefix');r['records'][0][3]=1
  with self.assertRaises(ValueError):a.stats_guard(r,'oracle_node_prefix')
 def test_flags(self):
  r=self.fixture('oracle_node_count');a.flags_guard(r)
  for k,v in [('valid',1),('complete',1),('valid',False),('reason','bad')]:
   with self.assertRaises(ValueError):a.flags_guard({**r,k:v})
 def test_transitions(self):self.assertEqual(a.transitions([0,0,1,1],[0,1,0,1]),{'0->0':1,'0->1':1,'1->0':1,'1->1':1})
 def test_unpinned(self):
  self.assertIsNone(a.MAIN_CONFIG_SHA)
  with self.assertRaises(ValueError):a.checked(Path(__file__),None)
if __name__=='__main__':unittest.main()
