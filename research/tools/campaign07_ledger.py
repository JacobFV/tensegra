"""Extended-07 budget ledger, computed from ALL job-wrapper receipts on the pro6000 plus declared local work.

Main jobs: ~/structured-latent-dynamics-campaign07/results/<job>-process; dev/test/audit work:
~/structured-latent-dynamics-campaign07/results/dev/<tag>-<utc>-<pid>-process (bin/metered.sh).
CPU = sum of process-tree core-seconds over every receipt (failed/capped included) + local entries in budget.json.
GPU (charged) = union of wall intervals of jobs that actually used a GPU: job names / dev tags containing '-gpu-'
  (CPU-only jobs run with CUDA_VISIBLE_DEVICES empty and are never counted as GPU use).
Legacy occupancy (extended-06 convention: union of all main-job wall intervals) is reported separately, not charged.
"""
import json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUDGET = REPO / "research/campaigns/extended-07/budget.json"
OUT = REPO / "research/results/campaign-07/receipts.json"
SCRIPT = r'''
import json, glob, os
rows = []
for occ in glob.glob(os.path.expanduser("~/structured-latent-dynamics-campaign07/results/**/*-process/occupancy.json"), recursive=True):
    d = os.path.dirname(occ)
    try:
        o = json.load(open(occ)); l = json.load(open(os.path.join(d, "launch.json")))
    except Exception:
        continue
    rows.append({"dir": d.split("/results/", 1)[1], "dev": "/dev/" in d, "started_unix": l["started_unix"],
                 "ended_unix": o["ended_unix"], "wall_seconds": o["wall_seconds"], "cpu_core_seconds": o["cpu_core_seconds"],
                 "exit_code": o["exit_code"], "stop_reason": o.get("stop_reason")})
running = [os.path.dirname(p) for p in glob.glob(os.path.expanduser("~/structured-latent-dynamics-campaign07/results/**/*-process/launch.json"), recursive=True)
           if not os.path.exists(os.path.join(os.path.dirname(p), "occupancy.json"))]
print(json.dumps({"rows": rows, "running": [r.split("/results/", 1)[1] for r in running]}))
'''


def union(spans):
    tot, cur = 0.0, None
    for s, e in sorted(spans):
        if cur is None or s > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur: tot += cur[1] - cur[0]
    return tot


def main():
    out = subprocess.run(["ssh", "pro6000", "wsl -d Ubuntu -- python3 -"], input=SCRIPT.encode(), capture_output=True, check=True)
    data = json.loads(out.stdout.decode().replace("\0", ""))
    rows = sorted(data["rows"], key=lambda r: r["started_unix"])
    gpu = union((r["started_unix"], r["ended_unix"]) for r in rows if "-gpu-" in r["dir"].split("/")[-1])
    legacy = union((r["started_unix"], r["ended_unix"]) for r in rows if not r["dev"])
    remote_cpu = sum(r["cpu_core_seconds"] for r in rows)
    dev_cpu = sum(r["cpu_core_seconds"] for r in rows if r["dev"])
    b = json.loads(BUDGET.read_text())
    local = sum(x["core_seconds"] for x in b.get("local_cpu", []))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "running": data["running"]}, indent=1) + "\n")
    b["charged_cpu_core_seconds"] = round(remote_cpu + local, 1)
    b["charged_gpu_seconds"] = round(gpu, 1)
    b["legacy_occupancy_seconds_not_charged"] = round(legacy, 1)
    b["charged_breakdown"] = {"remote_main_cpu": round(remote_cpu - dev_cpu, 1), "remote_dev_cpu": round(dev_cpu, 1),
                              "local_cpu": round(local, 1), "receipts": len(rows), "running_unreceipted": data["running"]}
    BUDGET.write_text(json.dumps(b, indent=2) + "\n")
    c = b["ceilings"]
    print(f"cpu {remote_cpu + local:.0f}/{c['cpu_core_seconds']} (remote dev {dev_cpu:.0f}, local {local:.0f}), "
          f"gpu {gpu:.0f}/{c['gpu_device_seconds']} (legacy occupancy {legacy:.0f}, not charged), "
          f"receipts {len(rows)}, running {len(data['running'])}")


if __name__ == "__main__":
    main()
