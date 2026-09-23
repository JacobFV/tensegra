"""Verify every tracked Stage10-baseline byte without importing model code."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
BASE='e10b80a'
def audit(root):
 records=[];errors=[]
 for line in subprocess.check_output(['git','ls-tree','-r','--full-tree',BASE],cwd=root,text=True).splitlines():
  meta,name=line.split('\t',1);mode,kind,blob=meta.split()
  if kind!='blob':continue
  path=root/name
  data=os.readlink(path).encode()if mode=='120000'and path.is_symlink()else path.read_bytes()if path.is_file()else None
  actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()if data is not None else None
  if actual!=blob:errors.append(name)
  records.append(dict(path=name,mode=mode,git_blob=blob))
 return dict(baseline=BASE,files=len(records),errors=errors,passed=not errors,records=records)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.root.resolve());a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='records'}));raise SystemExit(0 if r['passed']else 1)
