"""Receipt accounting and launch timeline for A-HR / A-HR2 / A-PI-T / A-CF-T.

usage: python receipts_audit.py <data_root> <repo_root> <out.json>
"""
import datetime as dt
import glob
import json
import os
import subprocess
import sys

ROOT, REPO, OUT = sys.argv[1:4]


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


rec = []
for d in sorted(glob.glob(f'{ROOT}/*-process') + glob.glob(f'{ROOT}/dev/*-process')):
    name = os.path.basename(d)
    L = json.load(open(f'{d}/launch.json')) if os.path.exists(f'{d}/launch.json') else {}
    O = json.load(open(f'{d}/occupancy.json')) if os.path.exists(f'{d}/occupancy.json') else {}
    cmd = L.get('command', [])
    ctl = None
    if '--controller' in cmd:
        ctl = cmd[cmd.index('--controller') + 1]
    rec.append(dict(name=name, started=utc(L['started_unix']) if 'started_unix' in L else None,
                    started_unix=L.get('started_unix'), ended=utc(O['ended_unix']) if 'ended_unix' in O else None,
                    exit=O.get('exit_code'), stop=O.get('stop_reason'), cpu=O.get('cpu_core_seconds'),
                    wall=O.get('wall_seconds'), cwd=os.path.basename(L.get('cwd', '')), controller=ctl,
                    tool=next((c for c in cmd if c.startswith('research/tools/')), None),
                    sub=next((cmd[i + 1] for i, c in enumerate(cmd) if c.startswith('research/tools/') and i + 1 < len(cmd)), None)))


def fam(n):
    for p in ('hr2-', 'hr-full', 'hr-', 'apit-', 'acf-', 'dev/root', 'root-'):
        if n.startswith(p):
            return p
    return 'other'


groups = {}
for r in rec:
    k = r['name']
    g = ('hr2' if k.startswith('hr2-') else 'hr-full' if k.startswith('hr-full') else 'hr' if k.startswith('hr-') else
         'apit-nocontroller' if k.startswith('apit-') and 'nocontroller' in k else 'apit' if k.startswith('apit-') else
         'acf-nons' if k.startswith('acf-') and 'nons' in k else 'acf' if k.startswith('acf-') else 'root-dev')
    G = groups.setdefault(g, dict(n=0, exit_codes={}, cpu=0.0, first_start=None, last_end=None, sources=set(), controllers=set()))
    G['n'] += 1
    G['exit_codes'][str(r['exit'])] = G['exit_codes'].get(str(r['exit']), 0) + 1
    G['cpu'] += r['cpu'] or 0.0
    G['sources'].add(r['cwd'])
    if r['controller']:
        G['controllers'].add(r['controller'])
    if r['started'] and (G['first_start'] is None or r['started'] < G['first_start']):
        G['first_start'] = r['started']
    if r['ended'] and (G['last_end'] is None or r['ended'] > G['last_end']):
        G['last_end'] = r['ended']
for G in groups.values():
    G['sources'] = sorted(G['sources']); G['controllers'] = sorted(G['controllers'])


def git(*a):
    return subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True).stdout.strip()


commits = {}
for h in ('03fe1b22', '81d1ebee', '55a66cec', 'acf61800', '63d52c02', 'e6fa7e99', '7028e2c5', 'f04720a0', 'e29f06ba', 'd558ad58'):
    commits[h] = git('show', '-s', '--format=%cI %s', h)
ctl_first = git('log', '--diff-filter=A', '--format=%h %cI', '--all', '--', 'research/results/campaign-05/a-pi-t/controller.json')
res = dict(receipts=rec, groups=groups, commits=commits, controller_first_commit=ctl_first,
           registry_commits=git('log', '--format=%h %cI %s', '--', 'research/campaigns/extended-05/registry.json'))
json.dump(res, open(OUT, 'w'), indent=1)
for g, G in sorted(groups.items()):
    print(g, json.dumps(G))
print(json.dumps(commits, indent=0))
print('controller first commit', ctl_first)
print(res['registry_commits'])
