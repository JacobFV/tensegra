"""Queue-authorized semantic process wrapper with complete occupancy accounting."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def persist(path,record):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(record,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--cap',type=float,required=True);p.add_argument('--python',required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; coordinator must release an idle slot')
receipt=Path(a.prefix+'.occupancy.json');log=Path(a.prefix+'.log');started_receipt=Path(a.prefix+'.started.json')
if receipt.exists() or log.exists() or started_receipt.exists():raise RuntimeError('immutable launch prefix already exists')
env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':'src'}
command=[a.python,'-m','topoformer.campaign_semantics_motif_inference',a.config]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic();timed_out=False
persist(started_receipt,dict(started_utc=started,wrapper_pid=os.getpid(),cap_seconds=a.cap,command=command,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
with log.open('x') as stream:
    try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=a.cap).returncode
    except subprocess.TimeoutExpired:code=124;timed_out=True
record=dict(started_utc=started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process_occupancy_seconds=time.monotonic()-tick,cap_seconds=a.cap,exit_code=code,timed_out=timed_out,command=command,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
persist(receipt,record);print(json.dumps(record),flush=True)
raise SystemExit(code)
