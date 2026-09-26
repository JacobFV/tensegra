"""Independent Track C (protocol-C1) reconstruction from raw c-eval rows. Pure Python.
Written without reading campaign04_c1_score.py / campaign04_c_analysis.py.
Usage: python c_reconstruct.py <data_root> <out.json>"""
import gzip, json, sys, os, math, random, glob

root, out = sys.argv[1], sys.argv[2]
BASES = ['p1-rl-x1-r0', 'p1-rl-x1-r1', 'p1-rl-x1-r2', 'p1-boot-x1-r0', 'p1-boot-x1-r1', 'p1-boot-x1-r2']
LOOP, COMP = BASES[:3], BASES[3:]
CONDS = ['iid_f0', 'iid_f2', 'events_train_kinds_p1', 'foreign4']
FIXED = ['fixed-greedy', 'fixed-sampled', 'fixed-r_mask', 'fixed-r_sample']
SEEDED = ['learned', 'random_matched', 'threshold']
integrity = {'rows_checked': 0, 'utility_ne_success_minus_cost': 0, 'meta_charge_mismatch': 0, 'meta_forwards_ne_decisions': 0,
             'fixed_nonzero_meta': 0, 'compute_units_mismatch': 0, 'appraisal_vs_rmask_diff_worlds': 0,
             'appraisal_vs_rmask_max_abs_util_no_meta_diff': 0.0, 'appraisal_only_intervened': 0, 'seed_errors': []}


def load(path):
    return [json.loads(l) for l in gzip.open(path)]


def check_rows(rows, fixed):
    for r in rows:
        integrity['rows_checked'] += 1
        if abs(r['utility'] - ((1.0 if r['success'] else 0.0) - r['cost'])) > 1e-9:
            integrity['utility_ne_success_minus_cost'] += 1
        charge = r['meta_forwards'] * r['meta_charge_per_forward']
        if abs((r['utility_no_meta'] - r['utility']) - charge) > 1e-9:
            integrity['meta_charge_mismatch'] += 1
        if fixed:
            if r['meta_forwards'] or r['utility'] != r['utility_no_meta']:
                integrity['fixed_nonzero_meta'] += 1
        else:
            if r['meta_forwards'] != r['decisions']:
                integrity['meta_forwards_ne_decisions'] += 1
            # compute units = actor forwards (one per decision, neural_work_per_forward 1.0) + meta units
            meta_units = r['meta_charge_per_forward'] / 1e-4 * r['meta_forwards']  # compute price 1e-4 per unit
            if abs(r['compute_units'] - (r['decisions'] + meta_units)) > 1e-6:
                integrity['compute_units_mismatch'] += 1


def stats(rows):
    n = len(rows)
    dec = sum(r['decisions'] for r in rows)
    interv = sum(r['decisions'] - r['u_counts'].get('default', 0) for r in rows)
    uc = {}
    for r in rows:
        for k, v in r['u_counts'].items():
            uc[k] = uc.get(k, 0) + v
    return dict(n=n, success=sum(bool(r['success']) for r in rows) / n, utility=sum(r['utility'] for r in rows) / n,
                utility_no_meta=sum(r['utility_no_meta'] for r in rows) / n, cost=sum(r['cost'] for r in rows) / n,
                np_episode_rate=sum(r['no_progress'] > 0 for r in rows) / n, np_mean=sum(r['no_progress'] for r in rows) / n,
                intervention_rate=interv / dec, u_counts=uc)


def mean_dicts(ds):
    keys = [k for k in ds[0] if isinstance(ds[0][k], float)]
    return {k: sum(d[k] for d in ds) / len(ds) for k in keys}


def rank(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def spearman_ci(clusters, reps=1000, seed=7):
    """clusters: list of lists of (pred, actual); resample episodes."""
    pts = [p for c in clusters for p in c]
    rho = spearman([p[0] for p in pts], [p[1] for p in pts])
    rng = random.Random(seed); bs = []
    for _ in range(reps):
        s = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        q = [p for c in s for p in c]
        v = spearman([p[0] for p in q], [p[1] for p in q])
        if v is not None:
            bs.append(v)
    bs.sort()
    return dict(rho=rho, n=len(pts), ci95=(bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]) if bs else None)


def ece(pairs, bins=10, min_count=20):
    b = [[] for _ in range(bins)]
    for p, y in pairs:
        b[min(int(p * bins), bins - 1)].append((p, y))
    used = [x for x in b if len(x) >= min_count]
    tot = sum(len(x) for x in used)
    if not tot:
        return None
    return sum(len(x) * abs(sum(p for p, _ in x) / len(x) - sum(y for _, y in x) / len(x)) for x in used) / tot


U_ORDER = ['sample', 'mask_top', 'stop']
result = {}
for base in BASES:
    B = {'conds': {}, 'registration': {}}
    # registration re-check from training records
    for s in (0, 1, 2):
        t = json.load(open(os.path.join(root, f'c-train-{base}/main-s{s}.json')))['registration']
        mt = t['margin_table']; best = max(x['gain'] for x in mt)
        m_mine = max(x['margin'] for x in mt if x['gain'] == best)  # ties -> larger m
        tt = t['tau_table']; bt = max(x['gain'] for x in tt)
        tau_mine = min(x['tau'] for x in tt if x['gain'] == bt)  # ties -> smaller tau
        B['registration'][s] = dict(margin=t['margin'], margin_mine=m_mine, tau=t['tau'], tau_mine=tau_mine)
    rowsets = {}
    for ci, c in enumerate(CONDS):
        d = os.path.join(root, f'c-eval-{base}-{c}', c)
        C = {}
        for arm in FIXED:
            rows = load(os.path.join(d, arm + '.jsonl.gz')); check_rows(rows, True); C[arm] = stats(rows); rowsets[(c, arm)] = rows
            seeds = [r['seed'] for r in rows]
            if seeds != list(range(161000000 + 100000 * ci, 161000000 + 100000 * ci + 256)):
                integrity['seed_errors'].append((base, c, arm))
        rows = load(os.path.join(d, 'appraisal_only-s0.jsonl.gz')); check_rows(rows, False); C['appraisal_only'] = stats(rows)
        rowsets[(c, 'appraisal_only')] = rows
        rm = {r['seed']: r for r in rowsets[(c, 'fixed-r_mask')]}
        for r in rows:
            if r['decisions'] - r['u_counts'].get('default', 0):
                integrity['appraisal_only_intervened'] += 1
            f = rm[r['seed']]
            diff = abs(r['utility_no_meta'] - f['utility'])
            integrity['appraisal_vs_rmask_max_abs_util_no_meta_diff'] = max(integrity['appraisal_vs_rmask_max_abs_util_no_meta_diff'], diff)
            if diff > 1e-9 or r['classes'] != f['classes']:
                integrity['appraisal_vs_rmask_diff_worlds'] += 1
        for arm in SEEDED:
            per = []
            for s in (0, 1, 2):
                rows = load(os.path.join(d, f'{arm}-s{s}.jsonl.gz')); check_rows(rows, False)
                if [r['seed'] for r in rows] != [r['seed'] for r in rowsets[(c, 'fixed-r_mask')]]:
                    integrity['seed_errors'].append((base, c, arm, s))
                st = stats(rows); C[f'{arm}-s{s}'] = st; per.append(st); rowsets[(c, f'{arm}-s{s}')] = rows
            C[arm] = mean_dicts(per)
        B['conds'][c] = C
    # IID group
    def iid(arm):
        a, b = B['conds']['iid_f0'][arm], B['conds']['iid_f2'][arm]
        return {k: (a[k] + b[k]) / 2 for k in a if isinstance(a[k], float)}
    arms = FIXED + ['appraisal_only'] + SEEDED + [f'{a}-s{s}' for a in SEEDED for s in (0, 1, 2)]
    B['iid'] = {a: iid(a) for a in arms}
    # paired SE learned(seed-mean) - best fixed, per world
    best_fixed = max(FIXED, key=lambda a: B['iid'][a]['utility'])
    diffs = []
    for c in ('iid_f0', 'iid_f2'):
        L = [rowsets[(c, f'learned-s{s}')] for s in (0, 1, 2)]; F = rowsets[(c, best_fixed)]; D0 = rowsets[(c, 'fixed-r_mask')]
        for i in range(len(F)):
            diffs.append(sum(L[s][i]['utility'] for s in (0, 1, 2)) / 3 - F[i]['utility'])
    mu = sum(diffs) / len(diffs); sd = math.sqrt(sum((x - mu) ** 2 for x in diffs) / (len(diffs) - 1))
    B['learned_minus_best_fixed'] = dict(best_fixed=best_fixed, diff=mu, se=sd / math.sqrt(len(diffs)))
    # calibration (appraisal_only: behaviour = D0), IID
    step_pairs, first_pairs, branch_pairs = [], [], []
    for c in ('iid_f0', 'iid_f2'):
        for r in rowsets[(c, 'appraisal_only')]:
            y = 1.0 if r['success'] else 0.0
            step_pairs += [(p, y) for p in r['p']]
            first_pairs.append((r['p'][0], y))
            for bp in r.get('branch_points', []):
                branch_pairs.append((bp['pred']['p'], 1.0 if bp['success_default'] else 0.0))
    B['ece'] = dict(step=ece(step_pairs), first=ece(first_pairs), branch_default=ece(branch_pairs),
                    n_step=len(step_pairs), n_first=len(first_pairs), n_branch=len(branch_pairs))
    # predicted vs actual effects from branch points
    sp = {}
    for pool in ('appraisal_only', 'learned-s0', 'learned-s1', 'learned-s2', 'all'):
        clusters_all = {u: [] for u in U_ORDER + ['pooled']}
        srcs = ['appraisal_only', 'learned-s0', 'learned-s1', 'learned-s2'] if pool == 'all' else [pool]
        for src in srcs:
            for c in ('iid_f0', 'iid_f2'):
                for r in rowsets[(c, src)]:
                    per_u = {u: [] for u in U_ORDER}
                    for bp in r.get('branch_points', []):
                        for i, u in enumerate(U_ORDER):
                            if u in bp['actual_gain']:
                                per_u[u].append((bp['pred']['adv'][i], bp['actual_gain'][u]))
                    for u in U_ORDER:
                        if per_u[u]:
                            clusters_all[u].append(per_u[u])
                    pooled = [x for u in U_ORDER for x in per_u[u]]
                    if pooled:
                        clusters_all['pooled'].append(pooled)
        sp[pool] = {u: spearman_ci(cl, reps=400) for u, cl in clusters_all.items() if cl}
        # within-u rank (pooled after centering ranks within u) = mean of per-u rhos weighted by n
        ws = [(sp[pool][u]['rho'], sp[pool][u]['n']) for u in U_ORDER if u in sp[pool] and sp[pool][u]['rho'] is not None]
        sp[pool]['within_u_weighted_mean_rho'] = sum(r * n for r, n in ws) / sum(n for _, n in ws) if ws else None
    B['spearman'] = sp
    # causal: shuffled-target controller vs learned s0 vs default on common worlds (128 per IID condition)
    cz = {}
    for c in ('iid_f0', 'iid_f2'):
        sh = load(os.path.join(root, f'c-eval-{base}-causal', c, 'model-shuffled-s0.jsonl.gz'))
        idx = {r['seed']: r for r in rowsets[(c, 'learned-s0')]}
        d0 = {r['seed']: r for r in rowsets[(c, 'fixed-r_mask')]}
        for r in sh:
            cz.setdefault('n', 0); cz['n'] += 1
            cz['shuffled'] = cz.get('shuffled', 0) + r['utility']
            cz['learned_s0'] = cz.get('learned_s0', 0) + idx[r['seed']]['utility']
            cz['default'] = cz.get('default', 0) + d0[r['seed']]['utility']
            cz['seed_max'] = max(cz.get('seed_max', 0), r['seed'])
    for k in ('shuffled', 'learned_s0', 'default'):
        cz[k] /= cz['n']
    gain = cz['learned_s0'] - cz['default']; cz['learned_gain'] = gain
    cz['shuffled_gain'] = cz['shuffled'] - cz['default']
    cz['fraction_lost'] = (gain - cz['shuffled_gain']) / gain if gain > 0 else None
    B['causal_shuffled'] = cz
    result[base] = B
    print(base, 'best fixed', best_fixed, {a: round(B['iid'][a]['utility'], 4) for a in FIXED + ['appraisal_only', 'learned', 'random_matched', 'threshold']})

# decision rules
dec = {}
def lu(b, a='learned'): return result[b]['iid'][a]
h1 = {}
for b in LOOP:
    mx = max(lu(b, a)['utility'] for a in FIXED)
    h1[b] = dict(learned=lu(b)['utility'], max_fixed=mx, margin=lu(b)['utility'] - mx,
                 success=lu(b)['success'], default_success=lu(b, 'fixed-r_mask')['success'],
                 ok=lu(b)['utility'] >= mx + .02 and lu(b)['success'] >= lu(b, 'fixed-r_mask')['success'] - .02,
                 per_seed=[lu(b, f'learned-s{s}')['utility'] for s in (0, 1, 2)])
dec['C-H1'] = dict(per_base=h1, supported=sum(v['ok'] for v in h1.values()) >= 2)
h2 = {}
for b in COMP:
    L, D = lu(b), lu(b, 'fixed-r_mask'); bf = max(lu(b, a)['utility'] for a in FIXED)
    h2[b] = dict(learned=L['utility'], default=D['utility'], best_fixed=bf, cost_l=L['cost'], cost_d=D['cost'],
                 np_l=L['np_episode_rate'], np_d=D['np_episode_rate'], np_mean_l=L['np_mean'], np_mean_d=D['np_mean'],
                 intervention_rate=L['intervention_rate'],
                 ok_protocol=L['utility'] >= D['utility'] - .01 and (L['cost'] < D['cost'] or L['np_episode_rate'] < D['np_episode_rate']),
                 ok_trackc_wording=L['utility'] >= bf - .01 and (L['cost'] < D['cost'] or L['np_episode_rate'] < D['np_episode_rate']),
                 per_seed_rate=[lu(b, f'learned-s{s}')['intervention_rate'] for s in (0, 1, 2)])
dec['C-H2'] = dict(per_base=h2, supported_protocol=sum(v['ok_protocol'] for v in h2.values()) >= 2,
                   supported_trackc_wording=sum(v['ok_trackc_wording'] for v in h2.values()) >= 2)
h3 = {}
for grp, bs in (('loop_prone', LOOP), ('competent', COMP)):
    per = {b: dict(learned=lu(b)['utility'], random=lu(b, 'random_matched')['utility'], threshold=lu(b, 'threshold')['utility'],
                   ok=lu(b)['utility'] > lu(b, 'random_matched')['utility'] and lu(b)['utility'] > lu(b, 'threshold')['utility'],
                   rate_learned=lu(b)['intervention_rate'], rate_random=lu(b, 'random_matched')['intervention_rate'],
                   rate_threshold=lu(b, 'threshold')['intervention_rate']) for b in bs}
    h3[grp] = dict(per_base=per, supported=sum(v['ok'] for v in per.values()) >= 2)
dec['C-H3'] = h3
h4 = {}
for grp, bs in (('loop_prone', LOOP), ('competent', COMP)):
    h4[grp] = {b: dict(ece=result[b]['ece'], spearman_all_pooled=result[b]['spearman']['all']['pooled'],
                       spearman_by_u={u: result[b]['spearman']['all'].get(u) for u in U_ORDER},
                       spearman_appraisal_pooled=result[b]['spearman']['appraisal_only'].get('pooled')) for b in bs}
dec['C-H4'] = h4
dec['C-H5'] = {b: result[b]['causal_shuffled'] for b in BASES}
# label worlds
lab = {}
for b in BASES:
    seeds = []
    for f in sorted(glob.glob(os.path.join(root, f'c-labels-{b}', 'labels-c*.json'))):
        s = json.load(open(f))['seeds']; seeds.append(tuple(s))
    lab[b] = dict(chunks=len(seeds), lo=min(s[0] for s in seeds), hi=max(s[1] for s in seeds),
                  contiguous=sorted(seeds) == [(150000000 + 32 * i, 150000032 + 32 * i) for i in range(len(seeds))])
json.dump(dict(bases=result, decisions=dec, integrity=integrity, label_worlds=lab), open(out, 'w'), indent=1, default=str)
print(json.dumps(integrity, default=str))
print(json.dumps(lab))
for k in ('C-H1', 'C-H2'):
    print(k, json.dumps(dec[k], indent=0, default=str)[:3000])
print('C-H3', json.dumps(dec['C-H3'], default=str))
for g in h4:
    for b, v in h4[g].items():
        print('C-H4', b, 'ECE', {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v['ece'].items()},
              'pooled', v['spearman_all_pooled'], 'by u', {u: (round(x['rho'], 3), [round(y, 3) for y in x['ci95']], x['n']) if x else None for u, x in v['spearman_by_u'].items()})
for b in BASES:
    print('C-H5', b, dec['C-H5'][b], 'reg', result[b]['registration'])
