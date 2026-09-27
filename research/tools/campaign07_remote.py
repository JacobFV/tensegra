"""Root-only helper for extended-07 jobs on the pro6000 (Windows host, WSL2 Ubuntu).

Subcommands:
  snapshot                       push the clean worktree HEAD as source-<sha> (refuses conflict markers)
  launch-cmd JOB SHA [opts] -- CMD...
                                 launch one command (cwd = source snapshot) as a detached job-wrapped systemd unit
      --wall-cap S --cpu-cap S   caps (job.py)
      --needs PATH               (repeatable) refuse unless PATH exists on the remote (checkpoints, labels, ...)
      --mkdir DIR                (repeatable) create DIR before launching (output dirs; must be under ROOT/results)
      --max-concurrent N         refuse if >= N e07 units are active (default 4; never above 8)
      --min-free-gb G            refuse if MemAvailable < G GiB (default 16)
  status                         list campaign units and finished receipts
Guards: job names must start with 'e07-'; the snapshot must exist; output paths outside ROOT/results are refused.
"""
import argparse, re, subprocess, sys
from pathlib import Path

ROOT = "/home/brand/structured-latent-dynamics-campaign07"
PY = "/home/brand/structured-latent-dynamics-campaign03/env/bin/python"
DETACH = "/home/brand/structured-latent-dynamics-campaign03/bin/detach.sh"
REPO = Path(__file__).resolve().parents[2]
READ_ONLY_ROOTS = ("/home/brand/structured-latent-dynamics-campaign06/",)  # historical artifacts: may be read, never written
MAX_CONCURRENT_CEILING = 8


def wsl(script: str) -> str:
    out = subprocess.run(["ssh", "pro6000", "wsl -d Ubuntu -- bash -s"], input=script.encode(), capture_output=True, check=True)
    return out.stdout.decode(errors="replace").replace("\0", "")


def head_sha() -> str:
    if subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip():
        sys.exit("worktree not clean; commit before snapshotting")
    markers = subprocess.run(["git", "grep", "-l", "-E", "^(<<<<<<<|>>>>>>>) "], cwd=REPO, capture_output=True, text=True).stdout.strip()
    if markers:
        sys.exit(f"refusing to snapshot: unresolved conflict markers in {markers.split()}")
    return subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def snapshot(_):
    sha = head_sha()
    tar = subprocess.Popen(["tar", "--exclude=.git", "--exclude=research/results", "-czf", "-", "."], cwd=REPO, stdout=subprocess.PIPE)
    dest = f"{ROOT}/source-{sha}"
    subprocess.run(["ssh", "pro6000", f'wsl -d Ubuntu -- bash -c "mkdir -p {dest} && tar -xzf - -C {dest}"'], stdin=tar.stdout, check=True)
    tar.wait()
    print(sha)


def parse_launch(argv):
    """launch-cmd JOB SHA [opts] -- CMD..."""
    if "--" not in argv:
        sys.exit("launch-cmd needs '--' before the command")
    i = argv.index("--")
    head, cmd = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="launch-cmd")
    p.add_argument("job"); p.add_argument("sha")
    p.add_argument("--wall-cap", type=float, default=7200); p.add_argument("--cpu-cap", type=float, default=14400)
    p.add_argument("--needs", action="append", default=[]); p.add_argument("--mkdir", action="append", default=[])
    p.add_argument("--max-concurrent", type=int, default=4); p.add_argument("--min-free-gb", type=float, default=16)
    a = p.parse_args(head)
    a.cmd = cmd
    return a


def check_paths_in_cmd(cmd):
    """Refuse writes to historical roots: any --out/--output/--*-out argument must be under ROOT/results."""
    for k, tok in enumerate(cmd[:-1]):
        if re.fullmatch(r"--(out|output|md|[a-z-]*-out)", tok):
            path = cmd[k + 1]
            if not path.startswith(f"{ROOT}/results/"):
                sys.exit(f"refusing to launch: output path {path!r} for {tok} is not under {ROOT}/results/")


def launch_cmd(argv):
    a = parse_launch(argv)
    if not a.job.startswith("e07-"):
        sys.exit(f"refusing to launch: job names must start with 'e07-' (got {a.job!r})")
    if not re.fullmatch(r"[0-9a-f]{7,40}", a.sha):
        sys.exit(f"refusing to launch: bad snapshot sha {a.sha!r}")
    if a.max_concurrent > MAX_CONCURRENT_CEILING:
        sys.exit(f"refusing: --max-concurrent above {MAX_CONCURRENT_CEILING}")
    if not a.cmd:
        sys.exit("refusing to launch: empty command")
    check_paths_in_cmd(a.cmd)
    for d in a.mkdir:
        if not d.startswith(f"{ROOT}/results/"):
            sys.exit(f"refusing: --mkdir {d!r} not under {ROOT}/results/")
    checks = [f"test -d {ROOT}/source-{a.sha} || {{ echo 'MISSING source-{a.sha}'; exit 3; }}",
              f"test ! -e {ROOT}/results/{a.job}-process || {{ echo 'EXISTS {a.job}-process'; exit 3; }}"]
    for n in a.needs:
        checks.append(f"test -e {n} || {{ echo 'MISSING {n}'; exit 3; }}")
    checks.append(f"n=$(systemctl --user list-units --type=service --state=active --no-legend --plain 'e07-*' | wc -l); "
                  f"[ $n -lt {a.max_concurrent} ] || {{ echo \"CONCURRENCY $n >= {a.max_concurrent}\"; exit 4; }}")
    checks.append(f"m=$(awk '/MemAvailable/ {{print int($2/1048576)}}' /proc/meminfo); "
                  f"[ $m -ge {int(a.min_free_gb)} ] || {{ echo \"MEMORY $m GiB < {a.min_free_gb}\"; exit 4; }}")
    for d in a.mkdir:
        checks.append(f"mkdir -p {d}")
    script = "set -e\n" + "\n".join(checks) + "\n" + (
        f"{DETACH} {a.job} {ROOT}/source-{a.sha} {PY} {ROOT}/bin/job.py --output {ROOT}/results/{a.job}-process "
        f"--wall-cap {a.wall_cap} --cpu-cap {a.cpu_cap} -- {' '.join(a.cmd)}\necho LAUNCHED {a.job}\n")
    out = subprocess.run(["ssh", "pro6000", "wsl -d Ubuntu -- bash -s"], input=script.encode(), capture_output=True)
    txt = out.stdout.decode(errors="replace").replace("\0", "") + out.stderr.decode(errors="replace").replace("\0", "")
    print(txt.strip())
    if out.returncode != 0 or "LAUNCHED" not in txt:
        sys.exit(f"launch refused or failed (rc {out.returncode})")


def status(_):
    print(wsl(f"""systemctl --user list-units --type=service --no-legend --plain 'e07-*' | awk '{{print $1, $3, $4}}'
cd {ROOT}/results 2>/dev/null || exit 0
for f in *-process/occupancy.json; do [ -e "$f" ] || continue
  python3 -c "import json,sys;o=json.load(open('$f'));print('$f'.split('/')[0], o['exit_code'], round(o['wall_seconds'],1), round(o['cpu_core_seconds'],1))"
done"""))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "launch-cmd":
        return launch_cmd(sys.argv[2:])
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("snapshot").set_defaults(f=snapshot)
    s.add_parser("status").set_defaults(f=status)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
