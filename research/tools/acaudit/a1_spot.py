"""A1 spot check from raw rows: IID-group success/utility/no-progress per policy x mode; world identity; seeds.
Usage: python a1_spot.py <data_root> <out.json>"""
import gzip, json, sys, os, hashlib

root, out = sys.argv[1], sys.argv[2]
POLS = ['p1-rl-x1-r0', 'p1-rl-x1-r1', 'p1-rl-x1-r2', 'p1-boot-x1-r0', 'p1-boot-x1-r1', 'p1-boot-x1-r2', 'p2a-c1-final-x1-r2']
MODES = ['greedy', 'sampled', 'r_mask', 'r_sample']
CONDS = ['iid_f0', 'iid_f2', 'events_train_kinds_p1', 'foreign4']
res, ident, seeds = {}, {}, {}


def rows_in(d):
    f = [x for x in os.listdir(d) if x.endswith('.jsonl.gz') and x != 'worlds.jsonl.gz'][0]
    return [json.loads(l) for l in gzip.open(os.path.join(d, f))]


for p in POLS:
    for m in MODES:
        acc = []
        for c in CONDS:
            rows = rows_in(os.path.join(root, f'a1-{p}', f'{c}-{m}'))
            h = hashlib.sha256(json.dumps([[r['seed'], r['spec_hash']] for r in rows]).encode()).hexdigest()
            ident.setdefault(c, set()).add(h)
            seeds[c] = (rows[0]['seed'], rows[-1]['seed'], len(rows))
            n = len(rows); o = [r['outcome'] for r in rows]
            st = dict(success=sum(bool(x['verified_success']) for x in o) / n, utility=sum(x['utility'] for x in o) / n,
                      np_step=sum(r['progress']['no_progress'] for r in rows) / sum(x['steps'] for x in o),
                      np_ep=sum(r['progress']['no_progress'] > 0 for r in rows) / n)
            res[f'{p}|{m}|{c}'] = st
        res[f'{p}|{m}|IID'] = {k: (res[f'{p}|{m}|iid_f0'][k] + res[f'{p}|{m}|iid_f2'][k]) / 2 for k in res[f'{p}|{m}|iid_f0']}
        v = res[f'{p}|{m}|IID']
        print(f'{p:20s} {m:9s} succ {v["success"]:.3f} util {v["utility"]:.4f} npStep {v["np_step"]:.4f} npEp {v["np_ep"]:.3f}')
for c in CONDS:
    rows = rows_in(os.path.join(root, 'a1-references', f'{c}-greedy'))
    print('ref', c, len(rows), sum(r['outcome']['utility'] for r in rows) / len(rows))
print('distinct world lists per condition', {c: len(v) for c, v in ident.items()}, seeds)
json.dump(dict(cells=res, distinct_world_lists={c: len(v) for c, v in ident.items()}, seeds=seeds), open(out, 'w'), indent=1)
