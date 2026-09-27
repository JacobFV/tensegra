"""Independent reconstruction of A-PI-T (screen) and A-CF-T (confirmation) from raw
evaluation rows: PI-T-1 / PI-T-2 per lineage, comparator definitions, pairing and
world integrity, controller hash.

usage: python eval_audit.py <data_root> <out.json>
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from common import cluster_boot, load_dir

ROOT, OUT = sys.argv[1], sys.argv[2]
POL = ('d', 'pit', 'r1', 'r2', 'random', 'teacher')
IID = ('iid_f0', 'iid_f2')
COND_I = {'iid_f0': 0, 'iid_f2': 1, 'events_train_kinds_p1': 2, 'foreign4': 3}
CTRL_SHA = 'dea16e83488d2d74c4cbbef6dbbc13dbda4625e01ba601552fcb367854cd6db0'


def analyse(files, block):
    out = {}
    rows = defaultdict(dict)  # (base, cond, policy) -> seed -> row
    integ = Counter()
    metas = []
    for f in files:
        m = f['meta']
        metas.append(m)
        integ['files'] += 1
        integ[f"controller_sha_ok"] += int(m['controller']['file_sha256'] == CTRL_SHA)
        integ['margin_.075'] += int(m['controller']['margin'] == 0.075 and m['controller']['margin_override'] is None)
        integ['protocol_worlds'] += int(bool(m.get('protocol_worlds')))
        lo = block + 100000 * COND_I[f['cond']]
        for r in f['data']['rows']:
            assert r['condition'] == f['cond']
            if not (lo <= r['seed'] < lo + 512):
                integ['seed_outside_registered_block'] += 1
            if r['seed'] in rows[(f['base'], f['cond'], r['policy'])]:
                integ['duplicate_rows'] += 1
            rows[(f['base'], f['cond'], r['policy'])][r['seed']] = r
        # chunk-level matched random rate == pit's realized per-eligible rate in the same chunk
        s = m['summary']
        integ['chunks_random_rate_eq_pit_rate'] += int(abs(m['matched_random_rate'] - s['pit']['delegation_rate_per_eligible']) < 1e-12)
    bases = sorted({k[0] for k in rows})
    # teacher (dep_reuse) does not use the actor: record which bases evaluated it, then share
    tb = sorted({k[0] for k in rows if k[2] == 'teacher'})
    out['teacher_evaluated_in_bases'] = tb
    out['policies_by_base'] = {b: sorted({k[2] for k in rows if k[0] == b}) for b in bases}
    for b in bases:
        for c in {k[1] for k in rows}:
            if (b, c, 'teacher') not in rows:
                rows[(b, c, 'teacher')] = rows[(tb[0], c, 'teacher')]
    conds = sorted({k[1] for k in rows}, key=COND_I.get)
    # pairing: identical seed sets for every policy and base within a condition
    pair = {}
    for c in conds:
        ref = set(rows[(bases[0], c, 'd')])
        pair[c] = dict(n=len(ref), identical_across_policies_and_bases=all(set(rows[(b, c, p)]) == ref for b in bases for p in POL))
    out['pairing'] = pair
    out['integrity'] = dict(integ)
    out['bases'] = bases
    out['controller_sha_in_meta'] = sorted({m['controller']['file_sha256'] for m in metas})
    out['controller_path_in_meta'] = sorted({m['controller']['path'] for m in metas})
    out['actor_sha_by_base'] = {b: sorted({m['base_sha256'] for m, f in zip(metas, files) if f['base'] == b}) for b in bases}
    # teacher rows identical across bases (teacher does not use the actor)
    tid = Counter()
    for c in conds:
        for s in rows[(bases[0], c, 'teacher')]:
            u = {round(rows[(b, c, 'teacher')][s]['utility'], 12) for b in bases}
            tid['identical' if len(u) == 1 else 'differs'] += 1
    out['teacher_rows_identical_across_bases'] = dict(tid)

    # comparator definition checks at row level
    cd = Counter()
    for (b, c, p), rr in rows.items():
        for s, r in rr.items():
            elig = r.get('eligible_by_anchor') or {}
            dele = r.get('delegations_by_anchor') or {}
            if p == 'd' and (r['delegations'] or r['teacher_steps']):
                cd['d_has_delegation'] += 1
            if p == 'r1':
                if any(k != 'call' and v for k, v in dele.items()):
                    cd['r1_delegates_outside_call'] += 1
                if dele.get('call', 0) != elig.get('call', 0):
                    cd['r1_not_all_call_anchors'] += 1
            if p == 'r2':
                if any(k != 'commit_revise' and v for k, v in dele.items()):
                    cd['r2_delegates_outside_commit_revise'] += 1
                if dele.get('commit_revise', 0) != elig.get('commit_revise', 0):
                    cd['r2_not_all_commit_revise_anchors'] += 1
            if p == 'teacher' and r['teacher_steps'] != r['steps']:
                cd['teacher_steps_ne_steps'] += 1
            if p == 'pit' and any(k not in ('call', 'reuse_recompute', 'commit_revise') and v for k, v in dele.items()):
                cd['pit_delegates_outside_trigger_set'] += 1
            if abs((1.0 if r['success'] else 0.0) - r['cost'] - r['utility']) > 1e-6:
                cd[f'{p}_utility_ne_success_minus_cost'] += 1
            if r.get('truncated'):
                cd[f'{p}_truncated'] += 1
            if r.get('delegation_out_of_catalog'):
                cd[f'{p}_delegation_out_of_catalog'] += r['delegation_out_of_catalog']
            for k, v in (r.get('delegation_ends') or {}).items():
                cd[f'{p}_end_{k}'] += v
            if r['teacher_steps'] > 0 and p in ('pit', 'r1', 'r2', 'random') and r['teacher_steps'] > 12 * max(1, r['delegations']):
                cd[f'{p}_teacher_steps_gt_12_per_delegation'] += 1
    out['comparator_checks'] = dict(cd)

    lin = {}
    for b in bases:
        L = {'conditions': {}}
        for grp, cs in [(c, (c,)) for c in conds] + [('iid_group', IID)]:
            seeds = [(c, s) for c in cs for s in sorted(rows[(b, c, 'd')])]
            cl = [f'{c}:{s}' for c, s in seeds]
            pol = {}
            for p in POL:
                rr = [rows[(b, c, p)][s] for c, s in seeds]
                ts = sum(r['teacher_steps'] for r in rr); st = sum(r['steps'] for r in rr)
                el = sum(sum((r.get('eligible_by_anchor') or {}).values()) for r in rr)
                dl = sum(r['delegations'] for r in rr)
                pol[p] = dict(n=len(rr), utility=float(np.mean([r['utility'] for r in rr])),
                              success=float(np.mean([r['success'] for r in rr])),
                              cost=float(np.mean([r['cost'] for r in rr])),
                              delegated_step_fraction_pooled=ts / st if st else 0.0,
                              delegated_step_fraction_mean_episode=float(np.mean([r['teacher_steps'] / r['steps'] for r in rr])),
                              delegation_rate_per_eligible=dl / el if el else None,
                              delegations_per_episode=dl / len(rr))
            diffs = {}
            for q in POL:
                if q == 'pit':
                    continue
                du = [rows[(b, c, 'pit')][s]['utility'] - rows[(b, c, q)][s]['utility'] for c, s in seeds]
                ds = [float(rows[(b, c, 'pit')][s]['success']) - float(rows[(b, c, q)][s]['success']) for c, s in seeds]
                diffs[q] = dict(utility=cluster_boot(du, cl), success_mean=float(np.mean(ds)))
            # per-world outcome agreement of pi_T with teacher / D
            agree_t = float(np.mean([rows[(b, c, 'pit')][s]['success'] == rows[(b, c, 'teacher')][s]['success'] for c, s in seeds]))
            d_fail = [(c, s) for c, s in seeds if not rows[(b, c, 'd')][s]['success']]
            rescued = sum(rows[(b, c, 'pit')][s]['success'] for c, s in d_fail)
            t_rescued = sum(rows[(b, c, 'teacher')][s]['success'] for c, s in d_fail)
            broke = sum(1 for c, s in seeds if rows[(b, c, 'd')][s]['success'] and not rows[(b, c, 'pit')][s]['success'])
            entry = dict(policies=pol, pit_minus=diffs, pit_teacher_success_agreement=agree_t,
                         D_failures=len(d_fail), pit_rescues=int(rescued), teacher_rescues=int(t_rescued),
                         pit_breaks_D_success=broke)
            if grp == 'iid_group':
                best = max(('teacher', 'r1', 'r2', 'random'), key=lambda q: pol[q]['utility'])
                L['PI-T-1'] = dict(utility_diff=pol['pit']['utility'] - pol['d']['utility'],
                                   success_diff=pol['pit']['success'] - pol['d']['success'],
                                   ci=diffs['d']['utility'],
                                   pass_=bool(pol['pit']['utility'] - pol['d']['utility'] >= .015 and pol['pit']['success'] - pol['d']['success'] >= -.01))
                L['PI-T-2'] = dict(best_comparator=best, diff_vs_best=pol['pit']['utility'] - pol[best]['utility'],
                                   ci=diffs[best]['utility'],
                                   dsf_pit=pol['pit']['delegated_step_fraction_pooled'], dsf_teacher=pol['teacher']['delegated_step_fraction_pooled'],
                                   pass_=bool(pol['pit']['utility'] - pol[best]['utility'] >= .005 and pol['pit']['delegated_step_fraction_pooled'] < pol['teacher']['delegated_step_fraction_pooled']))
                L['iid_group'] = entry
            else:
                entry['floor_success_ge_.8x_teacher'] = bool(pol['pit']['success'] >= .8 * pol['teacher']['success'])
                L['conditions'][grp] = entry
        lin[b] = L
    out['lineages'] = lin
    # pooled over lineages (world clusters), pi_T minus each simple rule; teacher-vs-D gap vs pi_T-vs-D gap
    pooled = {}
    for q in ('r1', 'r2', 'teacher', 'random', 'd'):
        du, cl = [], []
        for b in bases:
            for c in IID:
                for s in sorted(rows[(b, c, 'd')]):
                    du.append(rows[(b, c, 'pit')][s]['utility'] - rows[(b, c, q)][s]['utility']); cl.append(f'{c}:{s}')
        pooled[q] = cluster_boot(du, cl)
    out['pooled_iid_pit_minus'] = pooled
    out['gap_comparison'] = {b: dict(pit_minus_d=lin[b]['iid_group']['policies']['pit']['utility'] - lin[b]['iid_group']['policies']['d']['utility'],
                                     teacher_minus_d=lin[b]['iid_group']['policies']['teacher']['utility'] - lin[b]['iid_group']['policies']['d']['utility'],
                                     r2_minus_pit=lin[b]['iid_group']['policies']['r2']['utility'] - lin[b]['iid_group']['policies']['pit']['utility'],
                                     dsf_r2=lin[b]['iid_group']['policies']['r2']['delegated_step_fraction_pooled'],
                                     dsf_pit=lin[b]['iid_group']['policies']['pit']['delegated_step_fraction_pooled'])
                             for b in bases}
    out['summary'] = dict(PI_T_1_pass=sum(L['PI-T-1']['pass_'] for L in lin.values()),
                          PI_T_2_pass=sum(L['PI-T-2']['pass_'] for L in lin.values()), lineages=len(lin))
    return out, rows


res = {}
allrows = {}
for name, d, block in (('A-PI-T', 'e05-apit/ev', 240000000), ('A-CF-T', 'e05-acf/ev', 260000000)):
    res[name], allrows[name] = analyse(load_dir(f'{ROOT}/{d}'), block)
# cross-experiment world disjointness
sets = {n: {(k[1], s) for k, rr in allrows[n].items() for s in rr} for n in allrows}
res['disjoint_240M_vs_260M'] = not ({s for _, s in sets['A-PI-T']} & {s for _, s in sets['A-CF-T']})
json.dump(res, open(OUT, 'w'), indent=1, default=str)
for n in res:
    if not isinstance(res[n], dict):
        print(n, res[n]); continue
    print('==', n, json.dumps(res[n]['summary']), json.dumps(res[n]['pairing']), json.dumps(res[n]['integrity']))
    print(json.dumps(res[n]['comparator_checks']))
    print('pooled', {q: (round(v['mean'], 5), [round(x, 5) for x in v['ci90']]) for q, v in res[n]['pooled_iid_pit_minus'].items()})
    print('teacher identical across bases', res[n]['teacher_rows_identical_across_bases'], res[n]['controller_sha_in_meta'], res[n]['actor_sha_by_base'])
    for b, L in res[n]['lineages'].items():
        P = L['iid_group']['policies']
        print(b, 'util', {p: round(P[p]['utility'], 5) for p in P}, 'succ', {p: round(P[p]['success'], 4) for p in P})
        print('   dsf', {p: (round(P[p]['delegated_step_fraction_pooled'], 4), round(P[p]['delegated_step_fraction_mean_episode'], 4)) for p in P},
              'rate', {p: P[p]['delegation_rate_per_eligible'] for p in ('pit', 'random')})
        print('   PI-T-1', json.dumps({k: v for k, v in L['PI-T-1'].items() if k != 'ci'}), L['PI-T-1']['ci']['ci90'])
        print('   PI-T-2', json.dumps({k: v for k, v in L['PI-T-2'].items() if k != 'ci'}), L['PI-T-2']['ci']['ci90'])
        g = L['iid_group']
        print('   agreement pit/teacher', g['pit_teacher_success_agreement'], 'Dfail', g['D_failures'], 'pit rescues', g['pit_rescues'], 'teacher rescues', g['teacher_rescues'], 'pit breaks', g['pit_breaks_D_success'])
        print('   gap', json.dumps(res[n]['gap_comparison'][b]))
        print('   floors', {c: (v['floor_success_ge_.8x_teacher'], round(v['policies']['pit']['success'], 4), round(v['policies']['pit']['utility'] - v['policies']['d']['utility'], 4)) for c, v in L['conditions'].items()})
