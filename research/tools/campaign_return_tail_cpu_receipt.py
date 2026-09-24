"""Reproduce final R10/R11 CPU analyses without rewriting immutable outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
if Path(a.output).exists():raise SystemExit('Immutable receipt exists')
records=[]
with tempfile.TemporaryDirectory(prefix='return-tail-audit-') as tmp:
    tmp=Path(tmp)
    for run,script in [('r10-development','campaign_return_diversity_summary.py'),('r11-development','campaign_return_continuation_summary.py')]:
        data=root/'research/results/campaign-01/returns'/run
        target=tmp/(run+'.json');source=root/'research/tools'/script
        command=[sys.executable,str(source),'--predictions',str(data/'predictions.json.gz'),'--output',str(target)]
        before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();subprocess.run(command,check=True);seconds=time.monotonic()-start
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        if target.read_bytes()!=(data/'summary.json').read_bytes():raise ValueError('Summary replay differs: '+run)
        records.append(dict(run=run,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                            exact_bytes=True,wall_seconds=seconds,cpu_user_seconds=after.ru_utime-before.ru_utime,cpu_system_seconds=after.ru_stime-before.ru_stime))
receipt=dict(scope='Final CPU analysis replay only; original exploratory command CPU usage was not separately instrumented',runs=records,
             wall_seconds=sum(r['wall_seconds']for r in records),cuda_used=False)
Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
