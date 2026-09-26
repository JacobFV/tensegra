"""Independent A2 screening reconstruction (extended-04 A2/C audit). Pure Python, reads raw a2s rows.
Written without reading campaign04_a2_promote.py / campaign04_a1_analysis.py.
Usage: python a2_reconstruct.py <data_root> <out.json>"""
import gzip, json, sys, os, hashlib

root, out = sys.argv[1], sys.argv[2]
POL = {'boot': 'a2s-boot-x1-r2', 'reh': 'a2s-a2-reh-final', 'imit': 'a2s-a2-imit-final', 'crit': 'a2s-a2-crit-final',
       'ent': 'a2s-a2-ent-final', 'ent_dep': 'a2s-a2-ent-deployed', 'dep': 'a2s-a2-dep-final',
       'c1': 'a2s-c1-final', 'c0': 'a2s-c0-final'}
CONDS = ['iid_f0', 'iid_f2', 'events_train_kinds_p1', 'foreign4']
MODES = ['greedy', 'sampled', 'r_mask']
KEYS = ['success', 'utility', 'cost', 'np_episode_rate', 'np_step_rate', 'np_mean', 'work_per_success',
        'cost_per_success', 'trunc_rate', 'steps_mean']


def rows_in(d):
    fs = [f for f in os.listdir(d) if f.endswith('.jsonl.gz') and f != 'worlds.jsonl.gz']
    assert len(fs) == 1, (d, fs)
    return [json.loads(l) for l in gzip.open(os.path.join(d, fs[0]))]


def stats(rows):
    n = len(rows); o = [r['outcome'] for r in rows]
    succ = sum(bool(x['verified_success']) for x in o)
    npc = [r['progress']['no_progress'] for r in rows]
    steps = sum(x['steps'] for x in o)
    return dict(n=n, success=succ / n, utility=sum(x['utility'] for x in o) / n, cost=sum(x['cost'] for x in o) / n,
                np_episode_rate=sum(c > 0 for c in npc) / n, np_step_rate=sum(npc) / steps, np_mean=sum(npc) / n,
                work_per_success=(sum(x['work_units'] for x in o) / succ) if succ else None,
                cost_per_success=(sum(x['cost'] for x in o) / succ) if succ else None,
                trunc_rate=sum(bool(r.get('truncated')) for r in rows) / n, steps_mean=steps / n,
                modes=sorted({r.get('deployment', {}).get('mode', 'ref') for r in rows}),
                seeds=[r['seed'] for r in rows], spec=[r['spec_hash'] for r in rows])


def whash(s):
    return hashlib.sha256(json.dumps([s['seeds'], s['spec']]).encode()).hexdigest()


res, ident = {}, {}
for pol, d in POL.items():
    for m in MODES:
        for c in CONDS:
            s = stats(rows_in(os.path.join(root, d, f'{c}-{m}')))
            assert s['modes'] == [m], (pol, m, c, s['modes'])
            res[(pol, m, c)] = s
            ident.setdefault(c, set()).add(whash(s))
for c in CONDS:
    s = stats(rows_in(os.path.join(root, 'a2s-references', f'{c}-greedy')))
    res[('dep_reuse', 'greedy', c)] = s
    ident[c].add(whash(s))
seedcheck = {c: (min(res[('boot', 'greedy', c)]['seeds']), max(res[('boot', 'greedy', c)]['seeds']),
                 res[('boot', 'greedy', c)]['n']) for c in CONDS}


def iid(pol, m):
    a, b = res[(pol, m, 'iid_f0')], res[(pol, m, 'iid_f2')]
    return {k: (None if a[k] is None or b[k] is None else (a[k] + b[k]) / 2) for k in KEYS}


table = {f'{p}|{m}': iid(p, m) for (p, m, c) in res if c == 'iid_f0'}
other = {f'{p}|{m}|{c}': {k: res[(p, m, c)][k] for k in KEYS} for (p, m, c) in res if c not in ('iid_f0', 'iid_f2')}
# promotion rule (protocol-A2), applied from the registered text
REG = {'reh': 'greedy', 'imit': 'greedy', 'crit': 'greedy', 'ent': 'greedy', 'dep': 'r_mask'}
boot_best_mode = max(MODES, key=lambda m: table[f'boot|{m}']['utility'])
prom = {}
for arm, m in REG.items():
    A, B, Bb = table[f'{arm}|{m}'], table[f'boot|{m}'], table[f'boot|{boot_best_mode}']
    c1 = A['utility'] >= B['utility'] + .01 or (A['cost'] <= .9 * B['cost'] and A['success'] >= B['success'] - .02)
    c2 = A['utility'] >= Bb['utility'] + .005
    c3 = {k: A[k] <= B[k] + .02 for k in ('np_episode_rate', 'np_step_rate')}
    c4 = (A['utility'] >= table[f'imit|{m}']['utility'] + .01) if arm == 'reh' else None
    # also: best mode of the arm itself (for the 'any mode' localized claims)
    prom[arm] = dict(mode=m, utility=A['utility'], boot_same=B['utility'], boot_best_mode=boot_best_mode,
                     boot_best=Bb['utility'], c1=c1, c2=c2, c3_by_episode_rate=c3['np_episode_rate'],
                     c3_by_step_rate=c3['np_step_rate'], c4=c4,
                     promoted=bool(c1 and c2 and c3['np_episode_rate'] and (c4 is not False)),
                     arm_np_episode_rate=A['np_episode_rate'], boot_np_episode_rate=B['np_episode_rate'])
json.dump(dict(iid_table=table, other_conditions=other, promotion=prom,
               world_identity_distinct_hashes={c: len(v) for c, v in ident.items()}, seed_ranges=seedcheck),
          open(out, 'w'), indent=1)
print('boot best mode', boot_best_mode, '| distinct world lists per condition', {c: len(v) for c, v in ident.items()})
print('seed ranges', seedcheck)
print(f"{'policy|mode':18s} succ   util    cost    npEp   npStep  npMean work/s")
for k, v in table.items():
    print(f"{k:18s} {v['success']:.3f} {v['utility']:.4f} {v['cost']:.4f} {v['np_episode_rate']:.3f} "
          f"{v['np_step_rate']:.4f} {v['np_mean']:.3f} {v['work_per_success'] or 0:.0f}")
for a, v in prom.items():
    print(a, v)
