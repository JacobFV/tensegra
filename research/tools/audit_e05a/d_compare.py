"""Compare the independent r_mask / dep_reuse re-runs (d_spotcheck.json) with stored rows.
usage: python d_compare.py <data_root> <d_spotcheck.json> <out.json>"""
import json
import sys

from common import load_dir

ROOT, SPOT, OUT = sys.argv[1:4]
spot = json.load(open(SPOT))
stored = {}
for d in ('e05-apit/ev', 'e05-acf/ev'):
    for f in load_dir(f'{ROOT}/{d}'):
        for r in f['data']['rows']:
            stored[(d.split('/')[0], f['base'], f['cond'], r['policy'], r['seed'])] = r
for f in load_dir(f'{ROOT}/e05-hr/branch'):
    for e in f['data']['episodes']:
        stored[('e05-hr', f['base'], f['cond'], 'd', e['seed'])] = dict(utility=e['utility'], steps=e['steps'])
res = []
for j in spot['jobs']:
    exp = {'apit': 'e05-apit', 'acf': 'e05-acf', 'hr': 'e05-hr'}[j['label']]
    dm = [abs(x['utility'] - stored[(exp, j['base'], j['condition'], 'd', x['seed'])]['utility']) < 1e-9 and
          x['steps'] == stored[(exp, j['base'], j['condition'], 'd', x['seed'])]['steps'] for x in j['d']]
    tb = {'e05-apit': 'x1-r0', 'e05-acf': 'x1-r3'}.get(exp)
    tm = [abs(x['utility'] - stored[(exp, tb, j['condition'], 'teacher', x['seed'])]['utility']) < 1e-9 for x in j['teacher']]
    res.append(dict(label=j['label'], base=j['base'], condition=j['condition'], d_episodes=len(dm), d_identical=sum(dm),
                    teacher_episodes=len(tm), teacher_identical=sum(tm)))
json.dump(dict(results=res, remote_cpu_process_s=spot.get('cpu_process_s')), open(OUT, 'w'), indent=1)
for r in res:
    print(r)
