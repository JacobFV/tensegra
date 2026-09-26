"""Diff each A2 arm config against its P2a base (C1 for reh/imit/crit/dep, C0 for ent) and check that the
remote run's recorded a2 options match the config. Usage: python a2_config_diff.py <repo_root> <data_root>"""
import json, sys, os, hashlib

repo, data = sys.argv[1], sys.argv[2]
BASE = {'reh': 'p2a-c1', 'imit': 'p2a-c1', 'crit': 'p2a-c1', 'dep': 'p2a-c1', 'ent': 'p2a-c0'}


def flat(d, p=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f'{p}.{k}' if p else k))
    else:
        out[p] = d
    return out


report = {}
for arm, base in BASE.items():
    ap = os.path.join(repo, f'configs/campaign04/a2-{arm}-x1-r2.json')
    a = flat(json.load(open(ap)))
    b = flat(json.load(open(os.path.join(repo, f'configs/campaign03/{base}-x1-r2.json'))))
    diff = {k: (b.get(k, '<absent>'), a.get(k, '<absent>')) for k in sorted(set(a) | set(b)) if a.get(k, '<absent>') != b.get(k, '<absent>')}
    st = json.load(open(os.path.join(data, f'a2-{arm}-x1-r2/state.json')))
    report[arm] = dict(base=base, config_sha256=hashlib.sha256(open(ap, 'rb').read()).hexdigest(), diff=diff,
                       run_a2_options=st['a2']['options'], run_teacher=st['a2'].get('rehearsal_teacher'),
                       run_member_hparams=sorted({json.dumps({k: m[k] for k in ('kl_weight', 'entropy_weight', 'learning_rate')}) for m in st['members']}))
    print('==', arm, 'vs', base)
    for k, v in diff.items():
        print('  ', k, v)
    print('   run a2 options:', st['a2']['options'])
    print('   run member hparams:', report[arm]['run_member_hparams'])
json.dump(report, open(os.path.join(data, 'a2_config_diff.json'), 'w'), indent=1, default=str)
