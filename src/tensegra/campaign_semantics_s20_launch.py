"""S20 bounded immutable launcher; outer GNUtime/timeout also binds imports."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

def persist(path,record):
 temporary=path.with_name(path.name+'.tmp')
 with temporary.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def main():
 tick=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--cap',type=float,required=True);p.add_argument('--python',required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
 receipt=Path(a.prefix+'.occupancy.json');log=Path(a.prefix+'.log');start_receipt=Path(a.prefix+'.started.json')
 if any(p.exists() for p in (receipt,log,start_receipt)):raise RuntimeError('immutable prefix already used')
 command=[a.python,'-m','topoformer.campaign_semantics_s20',a.config];common=dict(started_utc=started,wrapper_pid=os.getpid(),cap_seconds=a.cap,command=command,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());persist(start_receipt,common)
 env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(Path(__file__).resolve().parent.parent)};error=None;timed_out=False;code=1;preflight=None
 with log.open('x') as stream:
  try:
   if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; root idle release required')
   sys.path.insert(0,str(Path(__file__).resolve().parent.parent));from topoformer.campaign_semantics_s20_freeze import verify
   c=json.loads(Path(a.config).read_text());verify(c,Path(__file__).parent,a.cap);preflight=time.monotonic()-tick;remaining=a.cap-preflight
   if remaining<=0:raise subprocess.TimeoutExpired(command,a.cap)
   code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=remaining).returncode
  except subprocess.TimeoutExpired:code=124;timed_out=True
  except Exception as exc:error=repr(exc);traceback.print_exc(file=stream)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 record={**common,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'process_occupancy_seconds':time.monotonic()-tick,'preflight_seconds':preflight,'exit_code':code,'timed_out':timed_out,'error':error,'gpu_processes':gpu}
 persist(receipt,record);print(json.dumps(record),flush=True);raise SystemExit(code)
if __name__=='__main__':main()
