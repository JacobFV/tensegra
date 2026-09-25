"""Bounded S18 lifecycle wrapper; includes source preflight in its timer/cap."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def persist(path,record):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def main():
    tick=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--cap',type=float,required=True)
    p.add_argument('--python',required=True);p.add_argument('--prefix',required=True);args=p.parse_args()
    prefix=Path(args.prefix);receipt=Path(str(prefix)+'.occupancy.json');log=Path(str(prefix)+'.log')
    started_path=Path(str(prefix)+'.started.json')
    if any(p.exists() for p in (receipt,log,started_path)):raise RuntimeError('immutable launch prefix exists')
    command=[args.python,'-m','topoformer.campaign_semantics_s18',args.config]
    common=dict(started_utc=started,wrapper_pid=os.getpid(),cap_seconds=args.cap,command=command,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
    persist(started_path,{**common,'stage':'before idle/source/preflight checks'})
    timed_out=False;error=None;code=1
    with log.open('x') as stream:
        try:
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                raise RuntimeError('GPU occupied; root must release idle slot')
            sys.path.insert(0,str(Path(__file__).parent))
            from campaign_semantics_s18_freeze import verify
            config=json.loads(Path(args.config).read_text());verify(config,Path(__file__).parent,args.cap)
            if Path(config['output_dir']).exists():raise RuntimeError('immutable output exists')
            env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(Path(__file__).resolve().parent.parent)}
            remaining=args.cap-(time.monotonic()-tick)
            if remaining<=0:raise subprocess.TimeoutExpired(command,args.cap)
            code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=remaining).returncode
        except subprocess.TimeoutExpired:code=124;timed_out=True
        except Exception as exc:error=repr(exc);traceback.print_exc(file=stream)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True)
    record={**common,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'process_occupancy_seconds':time.monotonic()-tick,'exit_code':code,'timed_out':timed_out,
        'error':error,'gpu_processes':gpu,'full_process_state':processes}
    persist(receipt,record);print(json.dumps(record),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
