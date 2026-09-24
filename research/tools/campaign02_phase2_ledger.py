"""Fold phase-2 remote job receipts into budget.json (run locally; reads via ssh).

GPU charge convention (unchanged from phase 1): whole experiment-process wall
occupancy for CUDA jobs. Concurrent jobs sharing the GPU are each charged their
full wall time, which over-counts device occupancy (conservative). CPU is the
wrapper's inclusive parent+reaped-descendant user+system time.
"""
import json, subprocess, sys
from pathlib import Path

BUDGET = Path(__file__).resolve().parents[1] / "campaigns/extended-02/budget.json"
REMOTE = "gb10-direct"


def main(job_ids, gpu_jobs):
    script = ("import json,os,sys\nout={}\nfor j in sys.argv[1:]:\n p=os.path.expanduser('~/topoformer-campaign02/results/'+j+'-process/occupancy.json')\n"
              " out[j]=json.load(open(p)) if os.path.exists(p) else None\nprint(json.dumps(out))")
    raw = subprocess.run(["ssh", REMOTE, "python3", "-", *job_ids], input=script, capture_output=True, text=True, check=True).stdout
    receipts = json.loads(raw)
    budget = json.loads(BUDGET.read_text())
    phase2 = {row["id"]: row for row in budget.get("phase2_jobs", [])}
    for job, r in receipts.items():
        if r is None:
            continue
        phase2[job] = {"id": job, "status": "completed" if r["exit_code"] == 0 else f"exit_{r['exit_code']}",
                       "cpu_core_seconds": r["cpu_core_seconds"], "wall_seconds": r["wall_seconds"],
                       "gpu_seconds": r["wall_seconds"] if job in gpu_jobs else 0.0,
                       "receipt": f"research/results/campaign-02/{job}-process/occupancy.json"}
    budget["phase2_jobs"] = sorted(phase2.values(), key=lambda x: x["id"])
    budget["phase2_charged_gpu_seconds"] = sum(x["gpu_seconds"] for x in phase2.values())
    budget["phase2_charged_cpu_core_seconds"] = sum(x["cpu_core_seconds"] for x in phase2.values())
    budget["total_charged_gpu_seconds"] = budget["charged_gpu_seconds"] + budget["phase2_charged_gpu_seconds"]
    budget["total_charged_cpu_core_seconds"] = budget["charged_cpu_core_seconds"] + budget["phase2_charged_cpu_core_seconds"]
    budget["status"] = "phase2_active"
    BUDGET.write_text(json.dumps(budget, indent=2) + "\n")
    print({k: budget[k] for k in ("phase2_charged_gpu_seconds", "phase2_charged_cpu_core_seconds",
                                   "total_charged_gpu_seconds", "total_charged_cpu_core_seconds")})


if __name__ == "__main__":
    # usage: ledger.py job1 job2 ... ; jobs prefixed with 'gpu:' are CUDA jobs
    args = sys.argv[1:]
    main([a.removeprefix("gpu:") for a in args], {a[4:] for a in args if a.startswith("gpu:")})
