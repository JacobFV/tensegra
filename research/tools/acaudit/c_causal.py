"""Descriptive: all causal arms vs learned-s0 and default, PAIRED on the common 128+128 IID worlds.
Usage: python c_causal.py <data_root> <out.json>"""
import gzip, json, sys, os

root, out = sys.argv[1], sys.argv[2]
BASES = ['p1-rl-x1-r0', 'p1-rl-x1-r1', 'p1-rl-x1-r2', 'p1-boot-x1-r0', 'p1-boot-x1-r1', 'p1-boot-x1-r2']


def load(p):
    return {json.loads(l)['seed']: json.loads(l) for l in gzip.open(p)}


res = {}
for b in BASES:
    arms = sorted(f[:-9] for f in os.listdir(os.path.join(root, f'c-eval-{b}-causal', 'iid_f0')))
    ref = {}
    for c in ('iid_f0', 'iid_f2'):
        ref[c] = {a: load(os.path.join(root, f'c-eval-{b}-{c}', c, f'{a}.jsonl.gz')) for a in ('learned-s0', 'fixed-r_mask')}
    R = {}
    for a in arms:
        u, l, d, stops, n = 0, 0, 0, 0, 0
        for c in ('iid_f0', 'iid_f2'):
            rows = load(os.path.join(root, f'c-eval-{b}-causal', c, a + '.jsonl.gz'))
            for s, r in rows.items():
                u += r['utility']; l += ref[c]['learned-s0'][s]['utility']; d += ref[c]['fixed-r_mask'][s]['utility']; n += 1
                stops += r['u_counts'].get('stop', 0)
        R[a] = dict(n=n, utility=u / n, minus_learned_s0=(u - l) / n, minus_default=(u - d) / n, stops=stops)
    R['_learned_s0_minus_default'] = (l - d) / n
    res[b] = R
    print('==', b, 'learned_s0 - default on these worlds: %+.4f' % R['_learned_s0_minus_default'])
    for a in arms:
        print('   %-26s u %.4f  vs learned %+.4f  vs default %+.4f  stops %d' % (a, R[a]['utility'], R[a]['minus_learned_s0'], R[a]['minus_default'], R[a]['stops']))
json.dump(res, open(out, 'w'), indent=1)
