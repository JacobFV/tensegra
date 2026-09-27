"""Independent re-derivation for the extended-07 P2/P3 audit (written from the definitions, not from
campaign07_frontier.py / campaign07_diagscore.py code).  Read-only on eval logs; writes one JSON to argv[1]."""
import gzip, json, os, sys, time
from collections import defaultdict
import numpy as np

R = os.path.expanduser('~/structured-latent-dynamics-campaign07/results/')
OUT = sys.argv[1]
FAM = 'SCE'
SUBS = ['S', 'C', 'E', 'SC', 'SE', 'CE']
T0 = time.process_time()


def rd(path):
    with gzip.open(path, 'rt') as f:
        f.readline()
        for line in f:
            if line.strip():
                yield json.loads(line)


def cf_counts(path):
    """per octet per stratum counters; strata = all, xq2 (all but q2_after_notH), and each type."""
    units = defaultdict(dict)
    flags = {}
    for r in rd(path):
        if r.get('kind') != 'decision':
            continue
        c = r['cf']
        u = (c['octet'], c['type'])
        units[u][c['member']] = (r['a'], tuple(r['opt']), c)
    C = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for (o, t), mem in units.items():
        strata = ['all', t] + ([] if t == 'q2_after_notH' else ['xq2'])
        def add(k, v=1.0):
            for s in strata:
                C[s][o][k] += float(v)
        if FAM in mem:
            a, opt, c = mem[FAM]
            if c['flip'] and c['unique_full']:
                add('flip_n'); add('flip_ok', a in opt)
                if c['has_near_miss'] and 'near_miss' in mem:
                    add('fnm_n'); add('fnm_ok', a in opt)
            else:
                add('inv_n'); add('inv_ok', a in opt)
        if 'near_miss' in mem:
            a, opt, _ = mem['near_miss']
            add('nm_n'); add('nm_ok', a in opt)
        if FAM in mem:
            af, of, _ = mem[FAM]
            if len(of) == 1:
                for m in SUBS:
                    if m not in mem:
                        continue
                    am, om, _ = mem[m]
                    if len(om) != 1:
                        continue
                    ch = af != am
                    if of != om:
                        add('pos'); add('hit', ch)
                    else:
                        add('neg'); add('fa', ch)
    return C


def pool_acc(path):
    n = ok = 0
    for r in rd(path):
        if r.get('kind') == 'decision':
            n += 1; ok += bool(r['ok'])
    return ok / n, n


EP = {'flip': [('flip_ok', 'flip_n')], 'nm': [('nm_ok', 'nm_n')], 'inv': [('inv_ok', 'inv_n')],
      'BA_nm': [('fnm_ok', 'fnm_n'), ('nm_ok', 'nm_n')], 'H': [('hit', 'pos')], 'FA': [('fa', 'neg')],
      'J': [('hit', 'pos'), ('fa', 'neg')], 'bias': [('chg', 'all')]}


def matrix(Cs, stratum, octets, keys):
    M = {k: np.zeros((len(Cs), len(octets))) for k in keys}
    oi = {o: j for j, o in enumerate(octets)}
    for s, C in enumerate(Cs):
        for o, cnt in C[stratum].items():
            for k in keys:
                M[k][s, oi[o]] = cnt.get(k, 0.0)
    return M


def stat(M, e, W=None):
    parts = []
    for n, d in EP[e]:
        if W is None:
            num, den = M[n].sum(1), M[d].sum(1)
        else:
            num, den = W @ M[n].T, W @ M[d].T
        with np.errstate(invalid='ignore', divide='ignore'):
            parts.append(num / np.where(den > 0, den, np.nan))
    if e == 'J':
        return parts[0] - parts[1]
    if e == 'BA_nm':
        return (parts[0] + parts[1]) / 2
    return parts[0]


RNG = np.random.default_rng(424242)  # auditor's own draws (differ from the campaign's seed-7 draws)
NB = 20000


def draws(S, C):
    W = RNG.multinomial(C, np.full(C, 1.0 / C), size=NB).astype(float)
    P = RNG.integers(0, S, size=(NB, S))
    return W, P


def contrast(CA, CB, stratum, e, octets, WP):
    keys = sorted({k for pr in EP[e] for k in pr})
    MA, MB = matrix(CA, stratum, octets, keys), matrix(CB, stratum, octets, keys)
    per = stat(MA, e) - stat(MB, e)
    W, P = WP
    v = []
    for b0 in range(0, NB, 5000):
        d = stat(MA, e, W[b0:b0 + 5000]) - stat(MB, e, W[b0:b0 + 5000])
        v.append(np.nanmean(np.take_along_axis(d, P[b0:b0 + 5000], 1), 1))
    v = np.concatenate(v); v = v[~np.isnan(v)]
    return {'mean': float(np.nanmean(per)), 'per_seed': [None if np.isnan(x) else round(float(x), 6) for x in per],
            'pos': int((per > 0).sum()), 'ci': [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
            'half_ci': [[float(np.quantile(h, .025)), float(np.quantile(h, .975))] for h in (v[:len(v) // 2], v[len(v) // 2:])]}


def point(C, stratum, e, octets):
    keys = sorted({k for pr in EP[e] for k in pr})
    return float(np.nanmean(stat(matrix(C, stratum, octets, keys), e)))


res = {'p3': {}, 'p2': {}}
# ------------------------------------------------------------------ P3
P3 = {}
for contract in ('mix', 'exact'):
    for arch in ('lin', 'mlp', 'bil', 'gate'):
        for inp in ('pred', 'exact'):
            nm = f'CONS3-{arch}-{contract}-{inp}'
            if contract == 'exact' and inp == 'exact':
                continue
            P3[nm] = [cf_counts(f'{R}e07-p3-eval-{contract}-s4{i}/diag-v1-cf-SCE-{nm}-s4{i}.jsonl.gz') for i in range(5)]
octets = sorted({o for Cs in P3.values() for C in Cs for o in C['all']})
res['p3']['octets'] = len(octets)
WP = draws(5, len(octets))
TYPES = ['first', 'after_probe_failed', 'after_b1_timeout', 'q2_after_H', 'q2_after_notH']
arms = {}
for nm, Cs in P3.items():
    arms[nm] = {s: {e: point(Cs, s, e, octets) for e in ('flip', 'nm', 'inv', 'BA_nm', 'H', 'FA', 'J')}
                for s in ['all', 'xq2'] + TYPES}
res['p3']['arms'] = arms
pairs = [('CONS3-bil-mix-pred', 'CONS3-mlp-mix-pred'), ('CONS3-bil-exact-pred', 'CONS3-mlp-exact-pred'),
         ('CONS3-gate-mix-pred', 'CONS3-mlp-mix-pred'), ('CONS3-bil-mix-pred', 'CONS3-gate-mix-pred'),
         ('CONS3-mlp-mix-pred', 'CONS3-lin-mix-pred'), ('CONS3-bil-mix-pred', 'CONS3-lin-mix-pred'),
         ('CONS3-mlp-mix-exact', 'CONS3-mlp-mix-pred'), ('CONS3-bil-mix-exact', 'CONS3-mlp-mix-exact')]
res['p3']['contrasts'] = {}
for A, B in pairs:
    d = {}
    for s in ['all', 'xq2'] + TYPES:
        es = ('J', 'H', 'FA', 'BA_nm', 'nm', 'flip') if s in ('all', 'xq2') else ('J', 'H', 'FA')
        d[s] = {e: contrast(P3[A], P3[B], s, e, octets, WP) for e in es}
    res['p3']['contrasts'][f'{A} - {B}'] = d
# support of the sub-pair sets per type (seed 0 bil)
res['p3']['support'] = {s: {k: float(sum(P3['CONS3-bil-mix-pred'][0][s][o].get(k, 0) for o in octets)) for k in ('pos', 'neg', 'flip_n', 'nm_n')}
                        for s in ['all', 'xq2'] + TYPES}
# pool accuracy
res['p3']['pool'] = {}
for nm in ('CONS3-lin-mix-pred', 'CONS3-mlp-mix-pred', 'CONS3-bil-mix-pred', 'CONS3-gate-mix-pred'):
    res['p3']['pool'][nm] = [pool_acc(f'{R}e07-p3-eval-mix-s4{i}/diag-v1-B-{nm}-s4{i}.jsonl.gz')[0] for i in range(5)]
res['cpu_p3'] = time.process_time() - T0
# ------------------------------------------------------------------ P2
P2 = {}
for nm in ('S0R0-bank', 'S1R0-bank', 'S0R1-bank', 'S1R1-bank', 'CONS-exact-pred', 'CONS-mix-pred', 'CONS-oof-pred'):
    P2[nm] = (f'{R}e07-p2-eval-s4{{i}}/diag-v1-%s-{nm}-s4{{i}}.jsonl.gz')
for nm in ('SEP-bank', 'CONS-exact-predLRN', 'CONS-mix-predLRN', 'CONS-oof-predLRN'):
    P2[nm] = (f'{R}e07-p2-evalX-s4{{i}}/diag-v1-%s-' + (nm + '-s4{i}' if nm == 'SEP-bank' else nm + '3{j}-s4{i}') + '.jsonl.gz')
p2 = {}
for nm, pat in P2.items():
    Cs, pools = [], []
    for i in range(5):
        Cs.append(cf_counts((pat % 'cf-SCE').format(i=i, j=5 + i)))
        pools.append(pool_acc((pat % 'B').format(i=i, j=5 + i))[0])
    p2[nm] = (Cs, pools)
oct2 = sorted({o for Cs, _ in p2.values() for C in Cs for o in C['all']})
res['p2']['octets'] = len(oct2)
res['p2']['arms'] = {nm: {'pool': float(np.mean(pl)), 'pool_per_seed': pl,
                          **{e: point(Cs, 'all', e, oct2) for e in ('flip', 'nm', 'inv', 'J', 'BA_nm')}}
                     for nm, (Cs, pl) in p2.items()}
ref = 'S1R1-bank' if res['p2']['arms']['S1R1-bank']['pool'] >= res['p2']['arms']['S1R0-bank']['pool'] else 'S1R0-bank'
res['p2']['reference'] = ref
RA = res['p2']['arms'][ref]
res['p2']['candidates'] = {}
for nm, A in res['p2']['arms'].items():
    if nm == ref:
        continue
    g = {'flip_gain': A['flip'] - RA['flip'], 'nm_diff': A['nm'] - RA['nm'], 'inv_diff': A['inv'] - RA['inv'],
         'pool': A['pool'], 'pool_ok': A['pool'] >= RA['pool'] - .02 and A['pool'] >= .70}
    g['qualifies'] = bool(g['flip_gain'] >= .03 and g['nm_diff'] >= -.05 and g['inv_diff'] >= -.03 and g['pool_ok'])
    res['p2']['candidates'][nm] = g
res['cpu_total'] = time.process_time() - T0
json.dump(res, open(OUT, 'w'), indent=1)
print('done', res['cpu_total'])
