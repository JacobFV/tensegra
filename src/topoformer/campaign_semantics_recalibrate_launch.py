"""S17 serial wrapper including CPU provenance checks in whole occupancy and cap."""
import argparse,datetime,hashlib,json,os,subprocess,time,sys,traceback
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
command=[a.python,'-m','topoformer.campaign_semantics_recalibrate',a.config]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic();timed_out=False;matrix=[];error=None;preflight_seconds=None
common=dict(started_utc=started,wrapper_pid=os.getpid(),cap_seconds=a.cap,command=command,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
persist(started_receipt,{**common,'stage':'before CPU main preflight'})
with log.open('x') as stream:
    try:
        config=json.loads(Path(a.config).read_text())
        if True:
            sys.path.insert(0,str(Path(__file__).parent))
            from campaign_semantics_recalibrate_freeze import verify
            matrix=verify(config,Path(__file__).parent,a.cap)
        preflight_seconds=time.monotonic()-tick;remaining=a.cap-preflight_seconds
        if remaining<=0:
            code=124;timed_out=True;error='CPU preflight exhausted whole-process cap; child not launched'
        else:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=remaining).returncode
    except subprocess.TimeoutExpired:code=124;timed_out=True
    except Exception as exc:
        code=1;error=type(exc).__name__+': '+str(exc);traceback.print_exc(file=stream)
record={**common,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'process_occupancy_seconds':time.monotonic()-tick,'preflight_seconds':preflight_seconds,'exit_code':code,'timed_out':timed_out,'primary_matrix':matrix,'error':error}
persist(receipt,record);print(json.dumps(record),flush=True)
raise SystemExit(code)
