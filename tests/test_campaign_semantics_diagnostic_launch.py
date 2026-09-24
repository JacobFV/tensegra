"""Exercise receipt persistence without spawning any child or touching CUDA."""
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch


class DiagnosticReceiptTests(unittest.TestCase):
    def test_success_and_timeout_receipts(self):
        for module in ('campaign_semantics_motif_launch.py','campaign_semantics_confirmation_rename_launch.py'):
            for timeout in (False,True):
                with self.subTest(module=module,timeout=timeout),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);config=root/'config.json';config.write_text('{}');prefix=root/'job'
                    argv=[module,str(config),'--cap','1','--python','unused-python','--prefix',str(prefix)]
                    kwargs={'side_effect':subprocess.TimeoutExpired('fake',1)} if timeout else {'return_value':types.SimpleNamespace(returncode=0)}
                    with patch('sys.argv',argv),patch('subprocess.check_output',return_value=''),patch('subprocess.run',**kwargs),patch('builtins.print'),self.assertRaises(SystemExit) as exited:
                        runpy.run_path(str(Path('src/topoformer')/module),run_name='__main__')
                    self.assertEqual(exited.exception.code,124 if timeout else 0)
                    start=json.loads(Path(str(prefix)+'.started.json').read_text())
                    end=json.loads(Path(str(prefix)+'.occupancy.json').read_text())
                    self.assertEqual(start['started_utc'],end['started_utc'])
                    self.assertEqual(end['timed_out'],timeout)
                    self.assertEqual(start['config_sha256'],end['config_sha256'])
                    self.assertFalse(list(root.glob('*.tmp')))
                    # Immutable start/receipt guard rejects any repeated launch before a child runs.
                    with patch('sys.argv',argv),patch('subprocess.check_output',return_value=''),patch('subprocess.run') as child,self.assertRaises(RuntimeError):
                        runpy.run_path(str(Path('src/topoformer')/module),run_name='__main__')
                    child.assert_not_called()

if __name__=='__main__':unittest.main()
