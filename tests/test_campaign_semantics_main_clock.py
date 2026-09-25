"""No child/GPU: profile preflight must consume cap and retain failure receipts."""
import json,runpy,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch

class MainClockTests(unittest.TestCase):
    def test_main_preflight_clock_and_failure_accounting(self):
        for module in ('campaign_semantics_multisurface_main_launch.py','campaign_semantics_confirmation_rename_main_launch.py','campaign_semantics_motif_main_launch.py'):
            for fail in (False,True):
                with self.subTest(module=module,fail=fail),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);config=root/'config.json';config.write_text('{"job":"profile"}');prefix=root/'job'
                    def verify(*args):
                        self.assertTrue(Path(str(prefix)+'.started.json').exists())
                        if fail:raise ValueError('preflight rejected')
                        return [{'seed':701}]
                    helper=types.SimpleNamespace(verify_main_source=verify)
                    argv=[module,str(config),'--cap','10','--python','unused','--prefix',str(prefix)]
                    with patch('sys.argv',argv),patch.dict('sys.modules',{'campaign_semantics_main_freeze':helper}),patch('subprocess.check_output',return_value=''),patch('subprocess.run',return_value=types.SimpleNamespace(returncode=0)) as child,patch('time.monotonic',side_effect=[0,2] if fail else [0,4,6]),patch('builtins.print'),self.assertRaises(SystemExit) as exited:
                        runpy.run_path(str(Path('src/tensegra')/module),run_name='__main__')
                    receipt=json.loads(Path(str(prefix)+'.occupancy.json').read_text())
                    self.assertEqual(exited.exception.code,1 if fail else 0)
                    self.assertEqual(receipt['process_occupancy_seconds'],2 if fail else 6)
                    if fail:
                        child.assert_not_called();self.assertIn('preflight rejected',receipt['error'])
                    else:
                        self.assertEqual(child.call_args.kwargs['timeout'],6)
                        self.assertEqual(receipt['preflight_seconds'],4)
if __name__=='__main__':unittest.main()
