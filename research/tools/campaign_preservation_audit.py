"""Verify every baseline tracked file retains its exact Git blob bytes."""
import argparse,hashlib,json,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',default='123299a7c5312e1f75019fa491618943f54d0de2');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];rows=subprocess.check_output(['git','ls-tree','-rz',a.baseline],cwd=root).split(b'\0');bad=[];count=0
for row in rows:
 if not row:continue
 meta,path=row.split(b'\t',1);mode,kind,want=meta.split();assert kind==b'blob';f=root/path.decode();count+=1
 if not f.is_file() and not f.is_symlink():bad.append(dict(path=path.decode(),reason='missing'));continue
 data=os.readlink(f).encode() if mode==b'120000' else f.read_bytes();got=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 if got!=want.decode():bad.append(dict(path=path.decode(),expected=want.decode(),actual=got))
out=dict(baseline=a.baseline,files=count,errors=bad,passed=not bad);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out);assert not bad
