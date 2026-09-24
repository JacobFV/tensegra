"""Serial-scheduler released R10 profile; externally wrap with GNU time + timeout."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

started=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
root=Path(sys.argv[1]);output=Path(sys.argv[2]);receipt=Path(sys.argv[3])
if output.exists() or receipt.exists():raise SystemExit('Immutable output/receipt already exists')
manifest=json.loads((root/'snapshot.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise SystemExit('Frozen source mismatch: '+name)
config=root/'configs/campaign-r10-profile.json';cfg=json.loads(config.read_text())
checkpoint=Path(cfg['checkpoint'])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=cfg['checkpoint_sha256']:raise SystemExit('Checkpoint mismatch')
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise SystemExit('GPU not idle: '+active)
cmd=[sys.executable,'-m','topoformer.campaign_returns_balanced_diversity','--config',str(config),'--output',str(output)]
child_start=time.monotonic();completed=subprocess.run(cmd,cwd=root,check=False)
record=dict(source_commit=manifest['source_commit'],snapshot_sha256=hashlib.sha256((root/'snapshot.json').read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),checkpoint_sha256=cfg['checkpoint_sha256'],
            started_utc=utc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            child_seconds=time.monotonic()-child_start,driver_before_receipt_seconds=time.monotonic()-started,exit_code=completed.returncode,
            accounting='GNU-time outer receipt includes driver startup/checks/shutdown; this record separates child runtime')
with receipt.open('x') as f:json.dump(record,f,indent=2)
raise SystemExit(completed.returncode)
