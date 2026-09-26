"""Telemetry hidden-state spot check (auditor-written; run from a source snapshot with PYTHONPATH=src).

For N recorded worlds (A2 screening boot rows, which carry the world spec and the action history) we build
counterfactual worlds that differ ONLY in hidden, not-yet-observed spec content:
  (a) the event argument (capacity/edge/slot change) -- hidden until the event fires;
  (b) the price and weight of the last item in the inventory -- hidden until that item is inspected;
then replay the identical recorded action sequence in both worlds with the real ProgressTracker and
TelemetryRecorder (policy stats fixed, since the actor sees only the observation encoding) and check that
telemetry d_t is identical at every step t at which the public observation history up to t is identical,
i.e. telemetry never diverges before the observations do.
Usage: python telemetry_spotcheck.py <rows.jsonl.gz> <worlds.jsonl.gz> <n>"""
import gzip, json, sys, dataclasses
from tensegra.campaign03_depworld import DepSpec, DepItem, DepWorkshop, depworld_executor
from tensegra.campaign02_world import Action
from tensegra.campaign04_progress import ProgressTracker
from tensegra.campaign04_telemetry import TelemetryRecorder, PolicyStats



def tup(v):
    return tuple(tup(x) for x in v) if isinstance(v, list) else v


def build_spec(d):
    d = dict(d)
    items = tuple(DepItem(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in it.items()}) for it in d.pop('items'))
    fields = {f.name for f in dataclasses.fields(DepSpec)}
    kw = {k: tup(v) for k, v in d.items() if k in fields}
    return DepSpec(items=items, **kw)


STATS = PolicyStats(p_top1=.5, entropy=1.0, margin=.1, candidates=5, p_default=.5, default_not_greedy=False, masked=0, flagged_state=False)


def run(spec, addr, actions):
    env = DepWorkshop(spec, depworld_executor, address_seed=addr)
    o = env.observe(); tr = ProgressTracker(o); rec = TelemetryRecorder(o)
    obs, tel = [], []
    for a in actions:
        obs.append(json.dumps(o.to_dict(), sort_keys=True, default=str))
        tel.append(rec.features(o, tr, STATS).values)
        after = env.step(a)
        sc = tr.update(after, a)
        rec.update(after, a, sc, 'default', 1.0)
        o = after
        if o.done:
            break
    return obs, tel


def main():
    rows_p, worlds_p, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rows = [json.loads(l) for l in gzip.open(rows_p)]
    worlds = {w['seed']: w for w in (json.loads(l) for l in gzip.open(worlds_p))}
    report = dict(worlds=0, variants=0, pairs_with_obs_divergence=0, telemetry_diverged_before_obs=0,
                  identical_prefix_steps=0, errors=[])
    for r in rows[:n]:
        w = worlds[r['seed']]
        base = build_spec(w['spec'])
        actions = [Action(h['action']['kind'], h['action'].get('arguments', {})) for h in r['outcome']['history']]
        variants = []
        if base.event is not None:
            kind, when, arg = base.event
            if isinstance(arg, int):
                variants.append(('event_arg', dataclasses.replace(base, event=(kind, when, arg + 1))))
            else:
                variants.append(('event_kind_when', dataclasses.replace(base, event=(kind, when + 1, arg))))
        last = base.items[-1]
        variants.append(('hidden_item', dataclasses.replace(base, items=base.items[:-1] + (dataclasses.replace(last, price=last.price + 1, weight=last.weight + 1),))))
        try:
            o0, t0 = run(base, w['address_seed'], actions)
        except Exception as e:  # noqa
            report['errors'].append((r['seed'], 'base', repr(e))); continue
        report['worlds'] += 1
        for name, sp in variants:
            try:
                o1, t1 = run(sp, w['address_seed'], actions)
            except Exception as e:  # noqa
                report['errors'].append((r['seed'], name, repr(e))); continue
            report['variants'] += 1
            m = min(len(o0), len(o1))
            first_obs = next((i for i in range(m) if o0[i] != o1[i]), m)
            first_tel = next((i for i in range(m) if t0[i] != t1[i]), m)
            report['identical_prefix_steps'] += first_obs
            if first_obs < m:
                report['pairs_with_obs_divergence'] += 1
            if first_tel < first_obs:
                report['telemetry_diverged_before_obs'] += 1
    report['n_errors'] = len(report['errors']); report['errors'] = [e[:2] + (e[2][:200],) for e in report['errors'][:3]]
    print(json.dumps(report))



if __name__ == '__main__':
    main()
