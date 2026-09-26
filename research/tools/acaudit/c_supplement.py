"""Supplementary Track C checks on top of c_reconstruct.py output: charge bookkeeping per arm, paired SEs for
small margins (C-H2/C-H3/C-H5 evaluability), Spearman pools, transfer floors.
Usage: python c_supplement.py <data_root> <c_recon.json>"""
import gzip, json, sys, os, math

root, rec = sys.argv[1], json.load(open(sys.argv[2]))
BASES = list(rec['bases'])


def load(b, c, arm):
    return [json.loads(l) for l in gzip.open(os.path.join(root, f'c-eval-{b}-{c}', c, arm + '.jsonl.gz'))]


# 1. meta forwards per arm
mf = {}
for arm in ['appraisal_only-s0', 'learned-s0', 'random_matched-s0', 'threshold-s0']:
    rows = load('p1-rl-x1-r2', 'iid_f0', arm)
    mf[arm] = dict(meta_forwards=sum(r['meta_forwards'] for r in rows), decisions=sum(r['decisions'] for r in rows),
                   charged=sum(r['utility_no_meta'] - r['utility'] for r in rows) / len(rows))
print('meta forwards per arm (rl-r2 iid_f0):', mf)


def paired(b, a1, a2, seeds1=None):
    d = []
    for c in ('iid_f0', 'iid_f2'):
        A = [load(b, c, f'{a1}-s{s}') for s in (0, 1, 2)] if seeds1 else [load(b, c, a1)]
        B = load(b, c, a2)
        for i in range(len(B)):
            d.append(sum(x[i]['utility'] for x in A) / len(A) - B[i]['utility'])
    mu = sum(d) / len(d); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
    return dict(diff=mu, se=sd / math.sqrt(len(d)))


pr = {}
for b in BASES:
    pr[b] = dict(learned_vs_default=paired(b, 'learned', 'fixed-r_mask', True),
                 learned_vs_threshold_s_mean=None)
    # threshold seed-mean vs learned seed-mean
    d = []
    for c in ('iid_f0', 'iid_f2'):
        L = [load(b, c, f'learned-s{s}') for s in (0, 1, 2)]; T = [load(b, c, f'threshold-s{s}') for s in (0, 1, 2)]
        for i in range(256):
            d.append(sum(x[i]['utility'] for x in L) / 3 - sum(x[i]['utility'] for x in T) / 3)
    mu = sum(d) / len(d); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
    pr[b]['learned_vs_threshold'] = dict(diff=mu, se=sd / math.sqrt(len(d)))
    iid = rec['bases'][b]['iid']
    pr[b]['cost'] = {a: iid[a]['cost'] for a in ('fixed-r_mask', 'fixed-r_sample', 'learned', 'threshold')}
    pr[b]['success'] = {a: iid[a]['success'] for a in ('fixed-r_mask', 'learned')}
    pr[b]['u_mix_learned'] = {s: {c: rec['bases'][b]['conds'][c][f'learned-s{s}']['u_counts'] for c in ('iid_f0', 'iid_f2')} for s in (0, 1, 2)}
    sp = rec['bases'][b]['spearman']
    pr[b]['spearman'] = {pool: {k: (v if not isinstance(v, dict) else (round(v['rho'], 3), [round(x, 3) for x in v['ci95']], v['n']))
                                for k, v in sp[pool].items()} for pool in sp}
    # transfer floor (secondary): learned success >= .8 x default success on events/foreign4
    tf = {}
    for c in ('events_train_kinds_p1', 'foreign4'):
        C = rec['bases'][b]['conds'][c]
        ls = sum(C[f'learned-s{s}']['success'] for s in (0, 1, 2)) / 3
        tf[c] = dict(learned_success=ls, default_success=C['fixed-r_mask']['success'], floor_ok=ls >= .8 * C['fixed-r_mask']['success'],
                     learned_utility=sum(C[f'learned-s{s}']['utility'] for s in (0, 1, 2)) / 3, default_utility=C['fixed-r_mask']['utility'],
                     best_fixed_utility=max(C[a]['utility'] for a in ('fixed-greedy', 'fixed-sampled', 'fixed-r_mask', 'fixed-r_sample')))
    pr[b]['transfer'] = tf
    print('==', b)
    for k, v in pr[b].items():
        print('  ', k, v)
json.dump(dict(meta_forwards=mf, per_base=pr), open(os.path.join(root, 'c_supplement.json'), 'w'), indent=1, default=str)
