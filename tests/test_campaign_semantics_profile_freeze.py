"""CPU-only immutable profile/source/matrix guards, no actor imports."""
import gzip,importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('profile_freeze','src/topoformer/campaign_semantics_profile_freeze.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ProfileFreezeTests(unittest.TestCase):
    def test_freeze_is_exact_immutable_and_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'src';source.mkdir()
            for name in m.SOURCES['s13']+('campaign_semantics_profile_freeze.py',):(source/name).write_text(name)
            prepared=root/'prepared.json';original=dict(budget_status='prepared_not_released',job='profile',arm='english',added_updates=20,checkpoints=[0,20],batch_size=8,learning_rate=1e-5,parent_checkpoint_sha256='fixed')
            prepared.write_text(json.dumps(original));out=root/'frozen.json'
            c=m.freeze(prepared,out,source,'s13');m.verify_profile_source(c,source)
            for k,v in original.items():
                if k!='budget_status':self.assertEqual(c[k],v)
            self.assertEqual(json.loads(prepared.read_text()),original)
            with self.assertRaises(FileExistsError):m.freeze(prepared,out,source,'s13')
            (source/'semantic_curriculum.py').write_text('changed')
            with self.assertRaises(ValueError):m.verify_profile_source(c,source)
    def test_incomplete_or_modified_primary_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for seed in (701,702,703):
                for arm in ('constant','decay'):
                    p=root/f's12-{arm}-{seed}';p.mkdir();(p/'model-u24576.pt').write_bytes(b'fixed');(p/'evaluation.json').write_text('{}')
                    manifest=dict(config=dict(seed=seed,updates=24576,learning_rate={'constant':1e-4,'decay':1e-5}[arm]),curves=[dict(update=24576,checkpoint_sha256=m.digest(p/'model-u24576.pt'),artifact='evaluation.json',sha256=m.digest(p/'evaluation.json'))])
                    with gzip.open(p/'manifest.json.gz','wt') as f:json.dump(manifest,f)
                    (p/'completion.json').write_text('{"success":true}')
            self.assertEqual(len(m.require_complete_matrix(root)),6)
            p=root/'s12-decay-703'/'model-u24576.pt';p.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.require_complete_matrix(root)
            p.unlink()
            with self.assertRaises(FileNotFoundError):m.require_complete_matrix(root)
if __name__=='__main__':unittest.main()
