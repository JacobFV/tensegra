"""Recompute the extended-03 budget ledger from fetched job receipts.

CPU = sum of process-tree core-seconds; GPU = device occupancy, the union of
job wall intervals (every campaign job holds the GPU device). Reads
research/results/campaign-03/*-process/{launch,occupancy}.json and rewrites
the jobs/charged fields of research/campaigns/extended-03/budget.json.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "research/results/campaign-03"
BUDGET = REPO / "research/campaigns/extended-03/budget.json"


def main():
    b = json.loads(BUDGET.read_text())
    notes = {j["job"]: j.get("note") for j in b.get("jobs", [])}
    jobs, spans = [], []
    for d in sorted(RES.glob("*-process")):
        occ, lau = d / "occupancy.json", d / "launch.json"
        if not occ.exists():
            continue
        o, l = json.loads(occ.read_text()), json.loads(lau.read_text())
        name = d.name[: -len("-process")]
        start, end = l["started_unix"], o["ended_unix"]
        spans.append((start, end))
        j = {"job": name, "host": "pro6000", "started_unix": start, "ended_unix": end,
             "wall_seconds": round(o["wall_seconds"], 2), "cpu_core_seconds": round(o["cpu_core_seconds"], 2),
             "exit_code": o["exit_code"], "stop_reason": o.get("stop_reason")}
        if notes.get(name):
            j["note"] = notes[name]
        jobs.append(j)
    union, cur = 0.0, None
    for s, e in sorted(spans):
        if cur is None or s > cur[1]:
            if cur:
                union += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        union += cur[1] - cur[0]
    b["jobs"] = jobs
    b["charged_cpu_core_seconds"] = round(sum(j["cpu_core_seconds"] for j in jobs), 2)
    b["charged_gpu_seconds"] = round(union, 2)
    BUDGET.write_text(json.dumps(b, indent=2) + "\n")
    c = b["ceilings"]
    print(f"cpu {b['charged_cpu_core_seconds']:.0f}/{c['cpu_core_seconds']} core-s, "
          f"gpu {b['charged_gpu_seconds']:.0f}/{c['gpu_device_seconds']} s, jobs {len(jobs)}")


if __name__ == "__main__":
    main()
