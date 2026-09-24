"""Generate the registered E08 population comparison configs.

Three independent replicate processes r in {0,1,2}. Within a replicate, PBT,
multistart and single share the initial bank (init seeds, per-slot hyperparameters),
training-stream starts, development panel and world mixture: a paired design.
Across replicates every seed range differs. Development episodes select only;
sealed E09 seeds (70_000_000+) are never referenced here.
"""
import argparse, json
from pathlib import Path

HYPER = [{"learning_rate": 3e-4, "entropy_weight": 0.01}, {"learning_rate": 1e-4, "entropy_weight": 0.003},
         {"learning_rate": 1e-3, "entropy_weight": 0.03}, {"learning_rate": 5e-5, "entropy_weight": 0.001},
         {"learning_rate": 2e-3, "entropy_weight": 0.1}, {"learning_rate": 6e-4, "entropy_weight": 0.3}]


def build(base, mode, replicate, *, rounds, updates, methods, dev_examples, tag):
    c = dict(base)
    c.update(mode=mode, population_seed=8100 + replicate,
             initialization_seeds=[81000 + 100*replicate + i for i in range(6)],
             rounds=rounds, updates_per_slot=updates, methods=methods,
             training_seed_start=500_000_000 + 20_000_000*replicate, training_seed_stride=1_000_000,
             development_seed_start=590_000_000 + 1_000_000*replicate, development_examples=dev_examples,
             address_namespace=f"e08-{tag}-r{replicate}", member_hyperparameters=HYPER, optimizer_policy="inherit")
    return c


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--supervised-rounds", type=int, default=2)
    p.add_argument("--dev-examples", type=int, default=128)
    a = p.parse_args()
    base = json.load(open(a.base))
    methods = ["supervised"]*a.supervised_rounds + ["actor_critic"]*(a.rounds - a.supervised_rounds)
    for r in range(3):
        for mode in ("pbt", "multistart", "single"):
            out = Path(f"configs/campaign02/e08-{a.tag}-{mode}-r{r}.json")
            out.write_text(json.dumps(build(base, mode, r, rounds=a.rounds, updates=a.updates, methods=methods,
                                            dev_examples=a.dev_examples, tag=a.tag), indent=2) + "\n")
            print(out)
