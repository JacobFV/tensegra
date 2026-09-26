"""Paired (same-world) utility differences with SEs for the A2 localized-reason claims (IID group, 512 worlds).
Usage: python a2_paired.py <data_root>"""
import gzip, json, sys, os, math

root = sys.argv[1]
POL = {'boot': 'a2s-boot-x1-r2', 'reh': 'a2s-a2-reh-final', 'imit': 'a2s-a2-imit-final', 'crit': 'a2s-a2-crit-final',
       'ent': 'a2s-a2-ent-final', 'dep': 'a2s-a2-dep-final', 'c1': 'a2s-c1-final', 'c0': 'a2s-c0-final'}


def load(pol, mode):
    out = {}
    for c in ('iid_f0', 'iid_f2'):
        d = os.path.join(root, POL[pol], f'{c}-{mode}')
        f = [x for x in os.listdir(d) if x.endswith('.jsonl.gz') and x != 'worlds.jsonl.gz'][0]
        for l in gzip.open(os.path.join(d, f)):
            r = json.loads(l)
            out[(c, r['seed'])] = (r['outcome']['utility'], r['outcome']['work_units'], bool(r['outcome']['verified_success']))
    return out


cache = {}


def get(p, m):
    if (p, m) not in cache:
        cache[(p, m)] = load(p, m)
    return cache[(p, m)]


def paired(a, b):
    A, B = get(*a), get(*b)
    assert A.keys() == B.keys()
    d = [A[k][0] - B[k][0] for k in A]
    n = len(d); mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return dict(a='|'.join(a), b='|'.join(b), n=n, diff=mu, se=sd / math.sqrt(n), z=mu / (sd / math.sqrt(n)) if sd else None)


def boot_ratio(a, b, reps=2000, seed=1):
    """bootstrap CI for work-per-success ratio a/b (worlds resampled jointly)."""
    import random
    A, B = get(*a), get(*b); keys = list(A); rng = random.Random(seed); rs = []
    def wps(D, ks):
        s = sum(D[k][2] for k in ks); return sum(D[k][1] for k in ks) / s
    for _ in range(reps):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        rs.append(wps(A, ks) / wps(B, ks))
    rs.sort()
    return dict(a='|'.join(a), b='|'.join(b), ratio=wps(A, keys) / wps(B, keys), ci95=(rs[int(.025 * reps)], rs[int(.975 * reps)]))


comps = [(('reh', 'greedy'), ('imit', 'greedy')), (('reh', 'sampled'), ('imit', 'sampled')), (('reh', 'r_mask'), ('imit', 'r_mask')),
         (('imit', 'sampled'), ('boot', 'sampled')), (('reh', 'sampled'), ('boot', 'sampled')),
         (('crit', 'sampled'), ('boot', 'sampled')), (('ent', 'sampled'), ('boot', 'sampled')), (('dep', 'sampled'), ('boot', 'sampled')),
         (('c1', 'sampled'), ('boot', 'sampled')), (('c0', 'sampled'), ('boot', 'sampled')),
         (('reh', 'sampled'), ('boot', 'greedy')), (('imit', 'sampled'), ('boot', 'greedy')),
         (('dep', 'greedy'), ('boot', 'greedy')), (('dep', 'r_mask'), ('boot', 'r_mask')), (('crit', 'r_mask'), ('boot', 'r_mask')),
         (('imit', 'greedy'), ('boot', 'greedy')), (('ent', 'greedy'), ('c1', 'greedy')), (('ent', 'r_mask'), ('c1', 'r_mask')),
         (('boot', 'r_mask'), ('boot', 'greedy'))]
res = [paired(a, b) for a, b in comps]
for r in res:
    print(f"{r['a']:14s} - {r['b']:14s} {r['diff']:+.4f} (SE {r['se']:.4f}, z {r['z']:+.1f})")
ratios = [boot_ratio(('ent', 'greedy'), ('boot', 'greedy')), boot_ratio(('ent', 'greedy'), ('c1', 'greedy')),
          boot_ratio(('c1', 'greedy'), ('boot', 'greedy')), boot_ratio(('reh', 'greedy'), ('imit', 'greedy')),
          boot_ratio(('ent', 'sampled'), ('c1', 'sampled'))]
for r in ratios:
    print(r)
json.dump(dict(paired=res, work_per_success_ratios=ratios), open(os.path.join(root, 'a2_paired.json'), 'w'), indent=1)
