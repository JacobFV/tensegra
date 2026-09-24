import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('src/topoformer').resolve()))
spec=importlib.util.spec_from_file_location('main_freeze','src/topoformer/campaign_semantics_main_freeze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MainFreezeTests(unittest.TestCase):
    def test_exact_recipes_immutable_cap_source_and_primary_bindings(self):
        matrix=json.loads(Path('research/results/campaign-01/semantics/postmatrix-profile-staging.json').read_text())['all_six_matrix']
        for lane,prefix in [('s13','campaign-s13-english'),('s13','campaign-s13-mixed'),('s14','campaign-s14-motif'),('rename','campaign-s12-renamed')]:
            with self.subTest(lane=lane,prefix=prefix),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'src';source.mkdir()
                for name in m.SOURCES[lane]+('campaign_semantics_main_freeze.py','campaign_semantics_profile_freeze.py'):(source/name).write_text(name)
                original=Path(f'configs/{prefix}-main-prepared.json');before=original.read_bytes();output=root/'frozen.json'
                c=m.freeze(original,output,source,lane,matrix,600)
                self.assertEqual(original.read_bytes(),before)
                for key,value in json.loads(before).items():
                    if key!='budget_status':self.assertEqual(c[key],value)
                with patch.object(m,'require_complete_matrix',return_value=matrix):m.verify_main_source(c,source,600)
                with self.assertRaises(FileExistsError):m.freeze(original,output,source,lane,matrix,600)
                with self.assertRaises(ValueError):m.verify_main_source(c,source,601)
                with patch.object(m,'require_complete_matrix',return_value=matrix[:-1]),self.assertRaises(ValueError):m.verify_main_source(c,source,600)
                (source/'semantic_curriculum.py').write_text('changed')
                with self.assertRaises(ValueError):m.verify_main_source(c,source,600)
if __name__=='__main__':unittest.main()
