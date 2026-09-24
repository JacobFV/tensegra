"""Run one immutable experiment job with process-tree CPU and wall receipts.

No imports from learning code; wrapper overhead is charged. RLIMIT_CPU applies
per child process; aggregate caps additionally monitor Linux /proc descendants.
Experiment children must not daemonize; the session is killed on cap/timeout.
"""
from __future__ import annotations
import argparse, hashlib, json, os, resource, signal, subprocess, time
from pathlib import Path

def process_group_cpu(pgid: int) -> float:
    ticks = os.sysconf('SC_CLK_TCK')
    total = 0
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                # own + reaped-child ticks: children still live have separate rows.
                total += sum(int(fields[i]) for i in (11, 12, 13, 14))
        except (OSError, ValueError, IndexError):
            continue
    return total / ticks

def run(command, output, wall_cap, cpu_cap, memory_gib=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'occupancy.json'
    if receipt_path.exists() or (output / 'launch.json').exists():
        raise FileExistsError('immutable job directory already launched')
    launch = {'command':command,'cwd':os.getcwd(),'wall_cap_seconds':wall_cap,
              'cpu_cap_seconds':cpu_cap,'started_unix':time.time(),
              'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']}}
    (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    start = time.monotonic(); own = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(cpu_cap)),max(2,int(cpu_cap)+1)))
        resource.setrlimit(resource.RLIMIT_NOFILE,(1024,1024))
        if memory_gib:
            size=int(memory_gib*1024**3);resource.setrlimit(resource.RLIMIT_AS,(size,size))
    reason=None; peak_sampled_cpu=0.
    with (output/'process.log').open('wb') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
        while proc.poll() is None:
            elapsed=time.monotonic()-start
            peak_sampled_cpu=max(peak_sampled_cpu,process_group_cpu(proc.pid))
            if elapsed>wall_cap or peak_sampled_cpu>cpu_cap:
                reason='wall_cap' if elapsed>wall_cap else 'cpu_cap'
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(.2)
        exit_code=proc.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)+time.process_time()-own
    receipt={'exit_code':exit_code,'stop_reason':reason,'wall_seconds':time.monotonic()-start,
             'cpu_core_seconds':cpu,'peak_sampled_group_cpu_seconds':peak_sampled_cpu,
             'max_child_rss_kib':after.ru_maxrss,'ended_unix':time.time(),
             'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest(),
             'scope':'Inclusive parent+reaped descendants user/system CPU; wall includes startup/export. Do not sum nested receipts.'}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return exit_code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wall-cap',type=float,required=True);p.add_argument('--cpu-cap',type=float,required=True);p.add_argument('--memory-gib',type=float);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:p.error('command required')
    raise SystemExit(run(cmd,a.output,a.wall_cap,a.cpu_cap,a.memory_gib))
