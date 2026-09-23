"""Stage9 historical diagnosis; standard library only, no model inference/training.
Reuses independently written Stage8 count auditors and adds paired event outcomes.
"""
import argparse, gzip, hashlib, importlib.util, json, subprocess, os
from pathlib import Path

BASE='0f44e13'
FIELDS=('value','type','operation','argument0','argument1','provenance')
def module(path):
    spec=importlib.util.spec_from_file_location(path.stem,path); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def pairs(root):
    results=[]
    for file in sorted(root.glob('*.jsonl.gz')):
        with gzip.open(file,'rt') as f: rows=[json.loads(line) for line in f]
        clean={(r['split'],r['seed'],r['distractors'],r['steps']):r for r in rows if r['phase']=='eval' and r['intervention']=='none'}
        for (split,seed,d,early),a in clean.items():
            for late in (1,4,16,32):
                if late<=early or early not in (0,1,4,16):continue
                b=clean[split,seed,d,late]
                assert a['targets']==b['targets'], 'paired event targets differ'
                n=len(a['targets']['value'])
                # Frozen runner regenerates data(seed, size, distractors) independently
                # at each delay; paired position is therefore the same underlying event.
                for name,fields in [('scalar',FIELDS[:1]),('non_value_joint',FIELDS[1:]),('full_joint',FIELDS)]:
                    ca=[all(a['predictions'][k][i]==a['targets'][k][i] for k in fields) for i in range(n)]
                    cb=[all(b['predictions'][k][i]==b['targets'][k][i] for k in fields) for i in range(n)]
                    counts={key:0 for key in ('correct_correct','correct_wrong','wrong_correct','wrong_wrong')}
                    for x,y in zip(ca,cb):counts[('correct' if x else 'wrong')+'_'+('correct' if y else 'wrong')]+=1
                    results.append(dict(run=file.name.split('.jsonl')[0],split=split,data_seed=seed,distractors=d,early=early,late=late,field=name,total=n,counts=counts,paired_target_sha256=hashlib.sha256(json.dumps(a['targets'],sort_keys=True).encode()).hexdigest()))
    return results

def preservation(repo):
    records=[];errors=[]
    tree=subprocess.check_output(['git','ls-tree','-r','--full-tree',BASE],cwd=repo,text=True)
    for line in tree.splitlines():
        meta,name=line.split('\t',1);mode,kind,blob=meta.split()
        if kind!='blob':continue
        p=repo/name
        data=os.readlink(p).encode() if mode=='120000' and p.is_symlink() else p.read_bytes() if p.is_file() else None
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() if data is not None else None
        if actual!=blob:errors.append(name)
        records.append(dict(path=name,mode=mode,git_blob=blob))
    return dict(baseline=BASE,files=len(records),errors=errors,passed=not errors,records=records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--semantic',action='store_true');a=p.parse_args();repo=a.root.resolve();out=repo/'research/results/stage9/audits';out.mkdir(parents=True,exist_ok=True)
    def write(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n')
    write('baseline-preservation.json',preservation(repo))
    for track,dir in [('beliefs','belief-idmatched'),('returns','return-main')]:
        audit=module(repo/f'scripts/audit_stage8_{track}.py').audit(repo/f'research/results/stage8/{dir}')
        write(f'historical-{track}-reconstruction.json',audit);assert not audit['errors'];print(track,audit.get('cells',audit.get('rows')))
    write('historical-return-paired-outcomes.json',pairs(repo/'research/results/stage8/return-main'))
    if a.semantic:
        files=sorted((repo/'research/results/stage8/semantic-main-n1000').glob('*.jsonl.gz'))+sorted((repo/'research/results/stage8/semantic-main-n10000').glob('*.jsonl.gz'))
        audit=module(repo/'scripts/audit_stage8_semantics.py').audit(files);write('historical-semantics-reconstruction.json',audit);assert not audit['errors'];print('semantics',audit['rows'])
if __name__=='__main__':main()
