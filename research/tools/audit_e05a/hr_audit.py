"""Independent reconstruction of A-HR (G1a, G1b) and A-HR2 (G1a-multi, G1b-multi)
from raw branch rows, plus label-validity checks.

usage: python hr_audit.py <data_root> <out.json>
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from common import cluster_boot, load_dir

ROOT = sys.argv[1]
OUT = sys.argv[2]
TRIG = ('call', 'reuse_recompute', 'commit_revise')
EPS = 1e-9


def wid(f, seed):
    return f"{f['cond']}:{seed}"


# ------------------------------------------------------------------ load
hr = load_dir(f'{ROOT}/e05-hr/branch')
hr2 = load_dir(f'{ROOT}/e05-hr2/branch')
full = load_dir(f'{ROOT}/e05-hr/full')
res = {'files': {'hr': len(hr), 'hr2': len(hr2), 'full': len(full)}}


def episodes_points(files):
    eps, pts = {}, defaultdict(list)
    for f in files:
        for e in f['data']['episodes']:
            eps[(f['base'], f['cond'], e['seed'])] = e
        for p in f['data']['points']:
            pts[(f['base'], f['cond'], p['seed'])].append(p)
    return eps, pts


eps1, pts1 = episodes_points(hr)
eps2, pts2 = episodes_points(hr2)

# ------------------------------------------------------------------ world integrity
def seeds_by(files):
    s = defaultdict(set)
    for f in files:
        for e in f['data']['episodes']:
            s[(f['base'], f['cond'])].add(e['seed'])
    return s


s1, s2 = seeds_by(hr), seeds_by(hr2)
wi = {}
for k in sorted(s1):
    ss = sorted(s1[k])
    wi['/'.join(k)] = dict(n=len(ss), lo=ss[0], hi=ss[-1], same_as_hr2=(s1[k] == s2.get(k)))
bases = sorted({f['base'] for f in hr + hr2 + full})
conds = sorted({f['cond'] for f in hr})
same_worlds_across_bases = all(s1[(b, c)] == s1[('x1-r0', c)] for b in bases for c in conds)
cond_off = {'iid_f0': 0, 'iid_f2': 100000}
in_range = all(220000000 + cond_off[c] <= s < 220000000 + cond_off[c] + 256 for (b, c), ss in s1.items() for s in ss)
res['world_integrity'] = dict(per_base_condition=wi, bases=bases, conditions=conds,
                              same_worlds_across_bases=same_worlds_across_bases,
                              seeds_in_220M_registered_block=in_range,
                              base_sha={f['base']: f['meta']['base_sha256'] for f in hr},
                              protocol_worlds=all(f['meta'].get('protocol_worlds') for f in hr + hr2))

# ------------------------------------------------------------------ label validity
lv = Counter()
bad = []
for key, e in list(eps1.items()) + list(eps2.items()):
    # episode utility = success - cost (cost charged once over the episode)
    lv['episodes'] += 1
    if abs((1.0 if e['success'] else 0.0) - e['cost'] - e['utility']) > 1e-6:
        lv['episode_utility_ne_success_minus_cost'] += 1
for name, eps, pts in (('hr', eps1, pts1), ('hr2', eps2, pts2)):
    for key, plist in pts.items():
        e = eps[key]
        for p in plist:
            lv[f'{name}:points'] += 1
            # D-branch reproduces the main line: q_d == U_final - U_t
            if abs(p['q_d'] - (e['utility'] - p['U_t'])) > 1e-6:
                lv[f'{name}:q_d_ne_mainline_remaining'] += 1
                bad.append((name, key, p['step'], p['q_d'], e['utility'] - p['U_t']))
            if p['d_success'] != e['success']:
                lv[f'{name}:d_success_ne_mainline'] += 1
            if p['T'] != e['T']:
                lv[f'{name}:T_ne_mainline'] += 1
            # D action at the point equals the main-line action at that step
            if e['actions'][p['step']] != p['d_index']:
                lv[f'{name}:d_index_ne_mainline_action'] += 1
            dopts = [o for o in p['options'] if o['type'] == 'D']
            if len(dopts) != 1:
                lv[f'{name}:D_option_count_ne_1'] += 1
            for o in p['options']:
                lv[f'{name}:options'] += 1
                # remaining utility = terminal success bonus - remaining costs (costs charged once)
                if abs((1.0 if o['success'] else 0.0) - o['cost_rem'] - o['dU']) > 1e-6:
                    lv[f'{name}:dU_ne_success_minus_cost_rem'] += 1
                if o['type'] == 'D' and abs(o['dU'] - p['q_d']) > 1e-6:
                    lv[f'{name}:D_option_dU_ne_q_d'] += 1
                if o['type'] == 'D' and o['dsteps'] != e['T'] - p['step']:
                    lv[f'{name}:D_dsteps_ne_T_minus_step'] += 1
                if p['U_t'] + o['dU'] > 1 + 1e-9:
                    lv[f'{name}:final_utility_gt_1'] += 1
                if o['type'] == 'delegate':
                    lv['hr2:delegate_options'] += 1
                    lv[f"hr2:delegate_end:{o['delegate_end']}"] += 1
                    if o['teacher_steps'] > 12:
                        lv['hr2:teacher_steps_gt_12'] += 1
                    if o['delegate_end'] == 'max_steps' and o['teacher_steps'] != 12:
                        lv['hr2:max_steps_end_but_steps_ne_12'] += 1
                    if o['delegate_end'] != 'max_steps' and o['teacher_steps'] >= 12 and o['delegate_end'] != 'episode_end':
                        lv['hr2:non_max_end_at_12'] += 1
                    if o['dsteps'] < o['teacher_steps']:
                        lv['hr2:dsteps_lt_teacher_steps'] += 1
                    if o['delegate_end'] == 'episode_end' and o['dsteps'] != o['teacher_steps']:
                        lv['hr2:episode_end_but_D_continued'] += 1
                    if o['delegate_end'] == 'commit' and o['dsteps'] == o['teacher_steps'] and not o['success'] and not o['truncated']:
                        lv['hr2:commit_end_no_D_continuation_nonterminal_info'] += 1
                    if o.get('teacher_out_of_catalog'):
                        lv['hr2:teacher_out_of_catalog_steps'] += o['teacher_out_of_catalog']
                    # delegate whose teacher agrees with D on every teacher step must equal D exactly
                    if o['teacher_agree'] == o['teacher_steps'] and o['delegate_end'] != 'commit' and abs(o['dU'] - p['q_d']) > 1e-6:
                        lv['hr2:full_agreement_but_dU_ne_q_d(non-commit end)'] += 1
res['label_validity'] = dict(counts=dict(lv), first_mismatches=bad[:10],
                             meta_default_checks={n: [sum(f['meta']['stats']['default_checks'] for f in fs),
                                                      sum(f['meta']['stats']['default_check_matches'] for f in fs)]
                                                  for n, fs in (('hr', hr), ('hr2', hr2), ('full', full))})

# A-HR vs A-HR2 main lines identical (same worlds, same D) and the same public sampled points
ml = Counter()
for k in eps1:
    a, b = eps1[k], eps2.get(k)
    ml['episodes'] += 1
    if b is None:
        ml['missing_in_hr2'] += 1
        continue
    if a['actions'] != b['actions'] or abs(a['utility'] - b['utility']) > 1e-9:
        ml['mainline_differs'] += 1
    if a['sampled_steps'] != b['sampled_steps']:
        ml['sampled_steps_differ'] += 1
    p1 = {p['step']: p for p in pts1[k]}
    p2 = {p['step']: p for p in pts2.get(k, [])}
    if set(p1) != set(p2):
        ml['branched_steps_differ'] += 1
    for s in set(p1) & set(p2):
        if p1[s]['telemetry'] != p2[s]['telemetry']:
            ml['telemetry_differs_same_state'] += 1
        if abs(p1[s]['q_d'] - p2[s]['q_d']) > 1e-9:
            ml['q_d_differs_same_state'] += 1
res['hr_vs_hr2_mainline'] = dict(ml)

# sampled-point bookkeeping: sampled flag vs episode.sampled_steps, <= 8 per episode
sp = Counter()
for key, e in eps1.items():
    ss = set(e['sampled_steps'])
    psamp = {p['step'] for p in pts1[key] if p['sampled']}
    sp['episodes'] += 1
    if ss != psamp:
        sp['sampled_steps_ne_sampled_points'] += 1
    if len(ss) > 8:
        sp['gt_8_sampled'] += 1
    if not e['all_states'] and any(not p['sampled'] for p in pts1[key]):
        sp['non_all_states_has_unsampled_point'] += 1
    sp['all_states_episodes'] += int(e['all_states'])
    # sampled flags in the decision log agree
    dl = {d['step'] for d in e['decisions'] if d['sampled']}
    if dl != ss:
        sp['decision_log_sampled_ne'] += 1
res['sampling'] = dict(sp)


# ------------------------------------------------------------------ G1a (hindsight single deviation)
def adv_list(p, allow, include_abstain):
    out = []
    for o in p['options']:
        if o['type'] == 'D':
            continue
        if o['type'] == 'abstain' and not include_abstain:
            continue
        if allow is not None and o['type'] not in allow:
            continue
        out.append(o['dU'] - p['q_d'])
    return out


def g1a(eps, pts, allow=None, include_abstain=False, sampled_only=True, anchors=None):
    vals, cl, by = [], [], defaultdict(list)
    for key, e in eps.items():
        best = 0.0
        for p in pts.get(key, []):
            if sampled_only and not p['sampled']:
                continue
            if anchors and p['anchor'] not in anchors:
                continue
            a = adv_list(p, allow, include_abstain)
            if a:
                best = max(best, max(a))
        vals.append(best)
        cl.append(f'{key[1]}:{key[2]}')
        by['D_failure' if not e['success'] else 'D_success'].append(best)
    r = cluster_boot(vals, cl)
    r['by_D_outcome'] = {k: dict(n=len(v), mean=float(np.mean(v))) for k, v in by.items()}
    return r, vals


G = {}
G['G1a_excl_abstain'], _ = g1a(eps1, pts1)
G['G1a_incl_abstain'], _ = g1a(eps1, pts1, include_abstain=True)
G['G1a_excl_abstain_all_branched_points(not deployable on all-states worlds)'], _ = g1a(eps1, pts1, sampled_only=False)
per_base = {}
for b in bases:
    for c in conds:
        e = {k: v for k, v in eps1.items() if k[0] == b and k[1] == c}
        per_base[f'{b}/{c}'] = g1a(e, pts1)[0]['mean']
G['G1a_excl_abstain_per_base_condition'] = per_base
G['G1a_pass_(>=.02)'] = G['G1a_excl_abstain']['mean'] >= .02
# hr2 option T (delegate only), T plus O(I)
G['G1a_multi_T'], _ = g1a(eps2, pts2, allow={'delegate'})
merged = {}
for key in eps1:
    p1 = {p['step']: p for p in pts1[key]}
    lst = []
    for p in pts2.get(key, []):
        q = dict(p)
        if p['step'] in p1:
            q = dict(p, options=p['options'] + [o for o in p1[p['step']]['options'] if o['type'] != 'D'])
        lst.append(q)
    merged[key] = lst
G['G1a_multi_T_plus_OI_excl_abstain'], _ = g1a(eps1, merged)
G['G1a_multi_T_trigger_anchors_only'], _ = g1a(eps2, pts2, allow={'delegate'}, anchors=TRIG)
G['G1a_B_budget_from_AHR(budget+call_now)'], _ = g1a(eps1, pts1, allow={'budget', 'call_now'})
G['G1a_multi_pass_(>=.02)'] = G['G1a_multi_T']['mean'] >= .02
res['G1a'] = G


# ------------------------------------------------------------------ G1b (same-information cross-fitted)
def feats(p, o, keys):
    x = list(p['telemetry'])
    f = o['features']
    x += [float(f.get(k, 0.0)) for k in keys]
    x.append(1.0)
    return x


def build(eps, pts, types_fn, anchors=None):
    """rows: (world, key, step, type, x, adv)"""
    okeys = set()
    for plist in pts.values():
        for p in plist:
            for o in p['options']:
                okeys |= set(o['features'])
    okeys -= {'is_d'}
    okeys = sorted(okeys)
    rows = []
    for key, plist in pts.items():
        for p in plist:
            if not p['sampled']:
                continue
            if anchors and p['anchor'] not in anchors:
                continue
            for o in p['options']:
                t = types_fn(p, o)
                if t is None:
                    continue
                rows.append((f'{key[1]}:{key[2]}', key, p['step'], t, feats(p, o, okeys), o['dU'] - p['q_d']))
    return rows, okeys


def ridge_fit(X, y, lam):
    mu = X[:, :-1].mean(0)
    sd = X[:, :-1].std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.hstack([(X[:, :-1] - mu) / sd, np.ones((len(X), 1))])
    A = Z.T @ Z + lam * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0.0])
    w = np.linalg.solve(A, Z.T @ y)
    return mu, sd, w


def ridge_pred(m, X):
    mu, sd, w = m
    Z = np.hstack([(X[:, :-1] - mu) / sd, np.ones((len(X), 1))])
    return Z @ w


def fit_predict(train_rows, test_rows, lam):
    by = defaultdict(list)
    for r in train_rows:
        by[r[3]].append(r)
    models = {}
    for t, rr in by.items():
        if len(rr) < 10:
            continue
        X = np.array([r[4] for r in rr]); y = np.array([r[5] for r in rr])
        models[t] = ridge_fit(X, y, lam)
    pred = np.full(len(test_rows), -np.inf)
    byt = defaultdict(list)
    for i, r in enumerate(test_rows):
        byt[r[3]].append(i)
    for t, idx in byt.items():
        if t in models:
            X = np.array([test_rows[i][4] for i in idx])
            pred[idx] = ridge_pred(models[t], X)
    return pred


def episode_gain(rows, pred, m, eps_keys):
    """single deviation: first sampled point (step order) whose best predicted option gain > m."""
    bypt = defaultdict(list)
    for i, r in enumerate(rows):
        bypt[(r[1], r[2])].append(i)
    per_key = defaultdict(list)
    for (key, step), idx in bypt.items():
        per_key[key].append((step, idx))
    out = {}
    fired = {}
    for key in eps_keys:
        g = 0.0; ty = 'D'
        for step, idx in sorted(per_key.get(key, [])):
            j = max(idx, key=lambda i: pred[i])
            if pred[j] > m:
                g = rows[j][5]; ty = rows[j][3]
                break
        out[key] = g; fired[key] = ty
    return out, fired


def crossfit(eps, rows, lam=1.0, grid=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1), outer=5, inner=3, seed=0):
    worlds = sorted({r[0] for r in rows} | {f'{k[1]}:{k[2]}' for k in eps})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(worlds))
    fold = {w: int(perm[i] % outer) for i, w in enumerate(worlds)}
    gains, fixed = {}, {m: {} for m in grid}
    chosen = Counter(); types = Counter()
    for k in range(outer):
        tr = [r for r in rows if fold[r[0]] != k]
        te = [r for r in rows if fold[r[0]] == k]
        te_keys = [key for key in eps if fold[f'{key[1]}:{key[2]}'] == k]
        # inner selection of m on training folds only
        tr_worlds = sorted({r[0] for r in tr} | {f'{key[1]}:{key[2]}' for key in eps if fold[f'{key[1]}:{key[2]}'] != k})
        ifold = {w: i % inner for i, w in enumerate(rng.permutation(tr_worlds))}
        score = {m: 0.0 for m in grid}
        for j in range(inner):
            itr = [r for r in tr if ifold[r[0]] != j]
            ite = [r for r in tr if ifold[r[0]] == j]
            ikeys = [key for key in eps if fold[f'{key[1]}:{key[2]}'] != k and ifold[f'{key[1]}:{key[2]}'] == j]
            p = fit_predict(itr, ite, lam)
            for m in grid:
                score[m] += sum(episode_gain(ite, p, m, ikeys)[0].values())
        mstar = max(grid, key=lambda m: (score[m], m))
        chosen[mstar] += len(te_keys)
        p = fit_predict(tr, te, lam)
        g, f = episode_gain(te, p, mstar, te_keys)
        gains.update(g); types.update(f.values())
        for m in grid:
            fixed[m].update(episode_gain(te, p, m, te_keys)[0])
    keys = list(eps)
    vals = [gains[k] for k in keys]
    cl = [f'{k[1]}:{k[2]}' for k in keys]
    r = cluster_boot(vals, cl)
    r['deviation_rate'] = float(np.mean([t != 'D' for t in [None]])) if False else None
    r['deviation_types'] = dict(types)
    r['chosen_margins'] = {str(m): c for m, c in chosen.items()}
    r['fixed_margin'] = {str(m): cluster_boot([fixed[m][k] for k in keys], cl)['mean'] for m in grid}
    r['harmful_rate'] = float(np.mean([v < -EPS for v in vals]))
    r['deviation_rate'] = 1 - types['D'] / len(keys)
    return r


t_oi = lambda p, o: None if o['type'] in ('D', 'abstain') else o['type']
rows1, k1 = build(eps1, pts1, t_oi)
t_del = lambda p, o: p['anchor'] if o['type'] == 'delegate' else None
rows2, k2 = build(eps2, pts2, t_del)
rows2t, _ = build(eps2, pts2, t_del, anchors=TRIG)
rows2s, _ = build(eps2, pts2, lambda p, o: 'delegate' if o['type'] == 'delegate' else None)
B = {}
B['G1b_OI_excl_abstain'] = crossfit(eps1, rows1)
B['G1b_OI_excl_abstain']['pass_(>=.01_and_ci90_excl_0)'] = B['G1b_OI_excl_abstain']['mean'] >= .01 and B['G1b_OI_excl_abstain']['ci90'][0] > 0
B['G1b_multi_T_all_anchors'] = crossfit(eps2, rows2)
B['G1b_multi_T_trigger_anchors'] = crossfit(eps2, rows2t)
B['G1b_multi_T_single_type_all_anchors'] = crossfit(eps2, rows2s)
for kx in ('G1b_multi_T_all_anchors', 'G1b_multi_T_trigger_anchors', 'G1b_multi_T_single_type_all_anchors'):
    B[kx]['pass_(>=.01_and_ci90_excl_0)'] = B[kx]['mean'] >= .01 and B[kx]['ci90'][0] > 0
# robustness: other fold seeds / lambda
B['G1b_multi_T_all_anchors_seed1'] = {k: v for k, v in crossfit(eps2, rows2, seed=1).items() if k in ('mean', 'ci90')}
B['G1b_multi_T_all_anchors_lam100'] = {k: v for k, v in crossfit(eps2, rows2, lam=100.0).items() if k in ('mean', 'ci90')}
B['G1b_OI_rootgrid_(m<=.05)'] = {k: v for k, v in crossfit(eps1, rows1, grid=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)).items() if k in ('mean', 'ci90', 'deviation_rate', 'harmful_rate')}
B['G1b_multi_T_single_type_rootgrid_(m<=.05)'] = {k: v for k, v in crossfit(eps2, rows2s, grid=(0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)).items() if k in ('mean', 'ci90', 'deviation_rate', 'harmful_rate')}
B['G1b_OI_seed1'] = {k: v for k, v in crossfit(eps1, rows1, seed=1).items() if k in ('mean', 'ci90')}
B['feature_note'] = dict(telemetry_dims=len(next(iter(pts1.values()))[0]['telemetry']), option_keys_hr=k1, option_keys_hr2=k2)
res['G1b'] = B

# leakage probes: features must not change between the D-branch and the main line; option-feature keys
leak = Counter()
for plist in pts2.values():
    for p in plist:
        for o in p['options']:
            for kk in o['features']:
                if kk in ('success', 'dU', 'cost_rem', 'dsteps', 'teacher_steps', 'teacher_agree', 'delegate_end', 'truncated'):
                    leak['outcome_key_in_features'] += 1
res['leakage_probe'] = dict(leak)

json.dump(res, open(OUT, 'w'), indent=1, default=str)
print(json.dumps({k: res[k] for k in ('files', 'label_validity', 'hr_vs_hr2_mainline', 'sampling')}, indent=1, default=str)[:6000])
for k, v in res['G1a'].items():
    print('G1a', k, json.dumps(v)[:300])
for k, v in res['G1b'].items():
    print('G1b', k, json.dumps(v)[:400])
