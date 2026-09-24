"""Fold phase-2 remote job receipts into budget.json (run locally; reads via ssh).

GPU charge (phase 2): device occupancy = measure of the union of CUDA-job wall
intervals [launch started_unix, receipt ended_unix] on the single GB10. The
phase-1 convention (sum of per-process wall) is also reported as an upper bound;
it over-counts when several small jobs share the device. CPU is the wrapper's
inclusive parent+reaped-descendant user+system time (additive across jobs).
"""
import json, subprocess, sys
from pathlib import Path

BUDGET = Path(__file__).resolve().parents[1] / "campaigns/extended-02/budget.json"
REMOTE = "gb10-direct"


def main(job_ids, gpu_jobs):
    script = ("import json,os,sys\nout={}\nfor j in sys.argv[1:]:\n p=os.path.expanduser('~/topoformer-campaign02/results/'+j+'-process/occupancy.json')\n"
              " l=p.replace('occupancy.json','launch.json')\n"
              " out[j]=dict(json.load(open(p)), started_unix=json.load(open(l))['started_unix']) if os.path.exists(p) else None\nprint(json.dumps(out))")
    raw = subprocess.run(["ssh", REMOTE, "python3", "-", *job_ids], input=script, capture_output=True, text=True, check=True).stdout
    receipts = json.loads(raw)
    budget = json.loads(BUDGET.read_text())
    phase2 = {row["id"]: row for row in budget.get("phase2_jobs", [])}
    for job, r in receipts.items():
        if r is None:
            continue
        phase2[job] = {"id": job, "status": "completed" if r["exit_code"] == 0 else f"exit_{r['exit_code']}",
                       "cpu_core_seconds": r["cpu_core_seconds"], "wall_seconds": r["wall_seconds"],
                       "gpu_process_wall_seconds": r["wall_seconds"] if job in gpu_jobs else 0.0,
                       "interval_unix": [r["started_unix"], r["ended_unix"]], "uses_gpu": job in gpu_jobs,
                       "receipt": f"research/results/campaign-02/{job}-process/occupancy.json"}
    budget["phase2_jobs"] = sorted(phase2.values(), key=lambda x: x["id"])
    spans = sorted(x["interval_unix"] for x in phase2.values() if x["uses_gpu"])
    union, current = 0.0, None
    for a, b in spans:
        if current is None or a > current[1]:
            union += 0 if current is None else current[1]-current[0]
            current = [a, b]
        else:
            current[1] = max(current[1], b)
    union += 0 if current is None else current[1]-current[0]
    budget["phase2_charged_gpu_seconds"] = union
    budget["phase2_gpu_process_wall_sum_upper_bound"] = sum(x["gpu_process_wall_seconds"] for x in phase2.values())
    budget["phase2_gpu_accounting"] = "device occupancy: union of CUDA-job wall intervals; per-process sum reported as upper bound"
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
