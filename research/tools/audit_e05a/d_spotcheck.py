"""Independent spot check (remote, CPU, 1 thread): re-run D with the UNCHANGED extended-04
deployment code (campaign04_deploy.deploy_episodes, mode='r_mask') and the dep_reuse teacher
(campaign02_references.run_episode(DepReference('reuse'), tariff=1)) on protocol worlds, and
print per-seed outcomes for comparison with the stored evaluation rows. World construction,
checkpoint remap and hash verification come from campaign05_hr (plumbing only).

run from a campaign-05 source snapshot:  PYTHONPATH=src python d_spotcheck.py <out.json>
"""
import importlib.util
import json
import sys
import time
from functools import partial

spec = importlib.util.spec_from_file_location('hr', 'research/tools/campaign05_hr.py')
hr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hr)

import torch  # noqa: E402

torch.set_num_threads(1)
from tensegra.campaign02_protocol import BoundedSolver  # noqa: E402
from tensegra.campaign02_references import make_reference, run_episode  # noqa: E402
from tensegra.campaign03_depworld import depworld_executor  # noqa: E402
from tensegra.campaign04_branch import SolverCache  # noqa: E402
from tensegra.campaign04_deploy import deploy_episodes  # noqa: E402

# (label, base, condition, world block, namespace, n)
JOBS = [
    ('apit', 'x1-r0', 'iid_f0', 240_000_000, 'e05apit', 16),
    ('apit', 'x1-r2', 'iid_f2', 240_100_000, 'e05apit', 16),
    ('acf', 'x1-r3', 'iid_f0', 260_000_000, 'e05acf', 16),
    ('acf', 'x1-r5', 'iid_f2', 260_100_000, 'e05acf', 16),
    ('hr', 'x1-r1', 'iid_f0', 220_000_000, 'e05hr', 16),
]


def main():
    out = {'jobs': []}
    B = hr.bases()
    t0 = time.process_time()
    for label, base, cond, block, ns, n in JOBS:
        actor, cfg, info = hr.load_actor(B[base], verify=True)
        seeds = list(range(block, block + n))
        with BoundedSolver() as solver:
            cache = SolverCache(partial(depworld_executor, execute_call=solver.execute))
            envs = [hr.world(s, cond, cache, ns) for s in seeds]
            rows = deploy_episodes(actor, envs, mode='r_mask', max_steps=cfg.max_steps,
                                   neural_work_per_forward=cfg.neural_work_per_forward)
            d = [dict(seed=s, utility=r['outcome'].get('utility'), success=r['outcome'].get('success', r['outcome'].get('verified')), steps=len(r['trace']), keys=sorted(k for k in r['outcome'] if not isinstance(r['outcome'][k], (list, dict))))
                 for s, r in zip(seeds, rows)]
            t = []
            if label != 'hr':
                for s in seeds[:8]:
                    r = run_episode(hr.world(s, cond, cache, ns), make_reference('dep_reuse'),
                                    model_compute_tariff=cfg.neural_work_per_forward)
                    t.append(dict(seed=s, utility=r.get('utility'), success=r.get('success', r.get('verified')), keys=sorted(k for k in r if not isinstance(r[k], (list, dict)))))
        out['jobs'].append(dict(label=label, base=base, condition=cond, namespace=ns, actor_sha=info['sha256'],
                                max_steps=cfg.max_steps, d=d, teacher=t))
        print(label, base, cond, 'D mean utility', sum(x['utility'] for x in d) / n, flush=True)
    out['cpu_process_s'] = time.process_time() - t0
    json.dump(out, open(sys.argv[1], 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
