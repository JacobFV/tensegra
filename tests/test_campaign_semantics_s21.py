"""S21 source-only guards. No model, cache generation, or sealed confirmation access."""
import copy,json,unittest
from pathlib import Path
import importlib.util,sys,types
package=types.ModuleType('s21_guard_package');package.__path__=[str(Path('src/tensegra').resolve())];sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location('s21_guard_package.campaign_semantics_s21_freeze',Path('src/tensegra/campaign_semantics_s21_freeze.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class Guards(unittest.TestCase):
 def prepared(self):
  c=json.loads(Path('configs/campaign-s21-profile-prepared-v1.json').read_text())
  for r in c['inputs'].values():r['sha256']='a'*64
  c['expected_epoch_exposures']={arm:dict(tokens=1,nodes=1,edges=1,records=1) for arm in c['arms']};return c
 def test_fixed_recipe(self):
  c=self.prepared();f.validate(c)
  for key,value in [('seed',2102),('schedule_seed',1),('updates',21),('arms',['broad']),('dev_per_cell',512),('max_records',161),('learning_rate',1e-4)]:
   with self.assertRaises(ValueError):f.validate({**c,key:value})
 def test_sealed_input_rejected(self):
  c=self.prepared();c['inputs']['confirmation']={'path':'never-open','sha256':'b'*64}
  with self.assertRaises(ValueError):f.validate(c)
 def test_missing_data_blocks(self):
  c=json.loads(Path('configs/campaign-s21-profile-prepared-v1.json').read_text())
  c['inputs']['audit']['sha256']=None
  with self.assertRaises(ValueError):f.validate(c)
  with self.assertRaises(ValueError):f.verify(self.prepared(),Path('src/tensegra'),180)
 def test_dynamic_exposure(self):
  c=self.prepared();c['expected_epoch_exposures']['broad']['tokens']=0
  with self.assertRaises(ValueError):f.validate(c)
 def test_main_shape(self):
  c=self.prepared();c.update(job='main',updates=4096,checkpoints=[0,1024,2048,4096],dev_per_cell=512,worst_case_profile=False);f.validate(c)
 def test_sources_include_unchanged_actor(self):
  self.assertEqual(len(f.SOURCES),len(set(f.SOURCES)));self.assertIn('campaign_semantics_s19_actor.py',f.SOURCES);self.assertIn('campaign_semantics_s21.py',f.SOURCES)
if __name__=='__main__':unittest.main()
