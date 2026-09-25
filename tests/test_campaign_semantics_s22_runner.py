"""S22 immutable runner guards; no model or diagnostic outputs read."""
import copy,importlib.util,json,sys,types,unittest
from pathlib import Path
pkg=types.ModuleType('s22guard');pkg.__path__=[str(Path('src/tensegra').resolve())];sys.modules['s22guard']=pkg
spec=importlib.util.spec_from_file_location('s22guard.campaign_semantics_s22_freeze','src/tensegra/campaign_semantics_s22_freeze.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class Guards(unittest.TestCase):
 def config(self):return json.loads(Path('configs/campaign-s22-profile-prepared-v1.json').read_text())
 def test_prepared(self):f.validate(self.config())
 def test_policy_support(self):
  for key,value in [('policies',['oracle_node_prefix']),('dev_per_cell',8),('arms',['broad']),('batch_size',8),('no_optimizer_or_training',False),('checkpoint_update',2048),('calibration','train'),('privilege','public_only')]:
   with self.assertRaises(ValueError):f.validate({**self.config(),key:value})
 def test_no_confirmation(self):
  c=self.config();c['inputs']['confirmation']={'path':'not_opened','sha256':'a'*64}
  with self.assertRaises(ValueError):f.validate(c)
 def test_checkpoint_pin(self):
  for arm in ('original','broad'):
   for key in ('checkpoint','manifest','public_reference'):
    c=self.config();c['checkpoints'][arm][key]['sha256']='a'*64
    with self.assertRaises(ValueError):f.validate(c)
   c=self.config();c['checkpoints'][arm]['model_state_sha256']='a'*64
   with self.assertRaises(ValueError):f.validate(c)
 def test_prepared_cannot_launch(self):
  with self.assertRaises(ValueError):f.verify(self.config(),Path('src/tensegra'),180)
 def test_main(self):
  c=json.loads(Path('configs/campaign-s22-main-prepared-v1.json').read_text());f.validate(c);self.assertIsNone(c['cap_seconds'])
if __name__=='__main__':unittest.main()
