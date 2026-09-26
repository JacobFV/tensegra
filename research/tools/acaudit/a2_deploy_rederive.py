"""Re-derive the P2a deployment rule for each A2 run from raw development rows (not from state summaries),
and compare with configs/campaign04/a2s-deployment-*.json. Also reports the training-batch entropy trace.
Usage: python a2_deploy_rederive.py <repo_root> <data_root>"""
import gzip, json, sys, os, hashlib

repo, data = sys.argv[1], sys.argv[2]
boot = json.load(open(os.path.join(data, 'tensegra-campaign03/results/p1-boot-x1-r2/state.json')))


def dev_means(path):
    rows = [json.loads(l) for l in gzip.open(path)]
    succ = [bool(r['outcome']['verified_success']) if 'outcome' in r else bool(r['verified_success']) for r in rows]
    util = [r['outcome']['utility'] if 'outcome' in r else r['utility'] for r in rows]
    return len(rows), sum(succ) / len(rows), sum(util) / len(rows)


# bootstrap development: the checkpoint referenced by the deployment files
bdev = None
for a in boot['allocations']:
    if a['checkpoint'].endswith('round-0-slot-5-attempt-5.pt'):
        bdev = (a['success'], a['utility'])
out = {'bootstrap_dev_from_ext03_state': bdev}
for arm in ['reh', 'imit', 'crit', 'ent', 'dep']:
    st = json.load(open(os.path.join(data, f'a2-{arm}-x1-r2/state.json')))
    dep = json.load(open(os.path.join(repo, f'configs/campaign04/a2s-deployment-{arm}.json')))
    ts, tu = bdev[0] - .02, bdev[1] - .02
    cands, mism = [], 0
    for a in st['allocations']:
        dp = os.path.join(data, f'a2-{arm}-x1-r2', a['development_predictions'])
        h = hashlib.sha256(open(dp, 'rb').read()).hexdigest()
        n, s, u = dev_means(dp)
        if abs(s - a['success']) > 1e-12 or abs(u - a['utility']) > 1e-9 or h != a['development_sha256']:
            mism += 1
        cands.append(dict(label=os.path.basename(a['checkpoint'])[:-3], n=n, s=s, u=u, q=(s >= ts - 1e-12 and u >= tu - 1e-12),
                          sha=a['checkpoint_sha256']))
    q = [c for c in cands if c['q']]
    deployed = q[-1]['label'] if q else 'rollback'
    rec = dict(n_qualify=len(q), n=len(cands), deployed=deployed, recorded=dep['deployed']['label'] if dep.get('deployed') else None,
               recorded_sha=dep['deployed']['sha256'] if dep.get('deployed') else None,
               my_sha=q[-1]['sha'] if q else None, final_label=cands[-1]['label'], final_sha=cands[-1]['sha'],
               final_dev=(cands[-1]['s'], cands[-1]['u']), dev_row_vs_state_mismatches=mism,
               entropy_trace=[st['allocations'][i]['training_timing'].get('last', {}).get('objective_parts', {}).get('entropy')
                              for i in range(len(st['allocations']))])
    et = [e for e in rec['entropy_trace'] if e is not None]
    rec['entropy_trace_mean'] = sum(et) / len(et) if et else None
    rec['entropy_trace_last10_mean'] = sum(et[-10:]) / len(et[-10:]) if et else None
    out[arm] = rec
    print(arm, {k: v for k, v in rec.items() if k != 'entropy_trace'})
json.dump(out, open(os.path.join(data, 'a2_deploy.json'), 'w'), indent=1)
