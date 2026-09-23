#!/usr/bin/env python3
"""Compact all-seed frozen probe tables; raw per-example records remain authoritative."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
rows=json.loads(Path(a.input).read_text()); groups=defaultdict(list)
for r in rows:
 if r['split']=='test':
  seed,encoding,availability=r['run'].split('-')
  groups[(encoding,availability,r['boundary'],r['family'])].append({'seed':int(seed),**r['metrics']})
out=[]
for key,values in sorted(groups.items()):
 out.append(dict(zip(('encoding','availability','boundary','family'),key),seeds=values,
  mean_accuracy=sum(v['correct']/v['total'] for v in values)/len(values),
  mean_mae=sum(v['mae'] for v in values)/len(values)))
Path(a.output).write_text(json.dumps(out,indent=2))
