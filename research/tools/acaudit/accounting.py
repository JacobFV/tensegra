"""Sum job-wrapper receipts (occupancy.json) for A1, A2 training, A2 screening and Track C; flag non-zero exits / caps.
Usage: python accounting.py <data_root> <out.json>"""
import json, sys, os, glob, re

root, out = sys.argv[1], sys.argv[2]
GROUPS = [('A1', r'^a1-.*-process$'), ('A2-train', r'^a2-.*-x1-r2-process$'), ('A2-screen', r'^a2s-.*-process$'),
          ('C-labels-emptysha', r'^c-labels-.*-emptysha-process$'), ('C-labels', r'^c-labels-.*-[abc]-process$'),
          ('C-train', r'^c-train-.*-process$'), ('C-eval-tier1', r'^c-eval-.*-(iid_f0|iid_f2|events_train_kinds_p1|foreign4)-process$'),
          ('C-eval-causal', r'^c-eval-.*-causal-process$')]
res = {}
for d in sorted(os.listdir(root)):
    if not d.endswith('-process'):
        continue
    for g, pat in GROUPS:
        if re.match(pat, d):
            p = os.path.join(root, d, 'occupancy.json'); l = json.load(open(os.path.join(root, d, 'launch.json')))
            o = json.load(open(p)) if os.path.exists(p) else None
            G = res.setdefault(g, dict(jobs=0, cpu=0.0, wall_max=0.0, nonzero=[], capped=[], missing=[], caps=set()))
            G['jobs'] += 1
            if o is None:
                G['missing'].append(d); break
            G['cpu'] += o['cpu_core_seconds']; G['wall_max'] = max(G['wall_max'], o['wall_seconds'])
            G['caps'].add((l.get('wall_cap_seconds'), l.get('cpu_cap_seconds')))
            if o['exit_code'] != 0:
                G['nonzero'].append((d, o['exit_code']))
            if o.get('stop_reason'):
                G['capped'].append((d, o['stop_reason']))
            if l.get('cpu_cap_seconds') and o['cpu_core_seconds'] > 0.9 * l['cpu_cap_seconds']:
                G['capped'].append((d, 'near cpu cap %.0f/%.0f' % (o['cpu_core_seconds'], l['cpu_cap_seconds'])))
            break
for g, G in res.items():
    G['caps'] = sorted(G['caps'], key=str)
    print(f"{g:18s} jobs {G['jobs']:3d} cpu {G['cpu']:9.0f} core-s  max wall {G['wall_max']:7.0f}s nonzero {G['nonzero']} capped {G['capped']} missing {G['missing']} caps {G['caps']}")
tot = sum(G['cpu'] for G in res.values())
print('total', round(tot))
json.dump(dict(groups=res, total=tot), open(out, 'w'), indent=1, default=str)
