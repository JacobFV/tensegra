"""Translate an immutable registered profile config to the production runner."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
c=json.loads(Path(a.config).read_text());cmd=[sys.executable,'-m','topoformer.campaign02_training','--output',a.output,'--device','cuda']
for k,v in c.items():cmd += ['--world-json' if k=='world' else '--'+k.replace('_','-'),json.dumps(v) if k=='world' else str(v)]
print(json.dumps({'command':cmd,'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest()}),flush=True)
raise SystemExit(subprocess.call(cmd))
