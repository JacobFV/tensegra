"""CPU/std-library analysis of frozen S10 records, without modifying decoding."""
import argparse, gzip, hashlib, json
from pathlib import Path


def load(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def compare(old_path,new_path,old_name,new_name,old_archive,new_archive):
    old,new=load(old_path),load(new_path)
    oa,na=load(old_archive),load(new_archive)
    for key in ('train_rows','rows'):
        x={r['seed']:r for r in oa[key]};y={r['seed']:r for r in na[key]}
        assert x.keys()==y.keys(),'different paired populations'
        assert all(x[k]['target']==y[k]['target'] for k in x),'target mismatch'
    key=lambda r:(r['split'],r['policy'],r['variant'],r['seed'])
    x={key(r):r for r in old['rows'] if r['checkpoint']==old_name};y={key(r):r for r in new['rows'] if r['checkpoint']==new_name}
    assert x.keys()==y.keys()
    cells=[]
    for cell in sorted({k[:3] for k in x}):
        kk=[k for k in x if k[:3]==cell];counts={'correct_correct':0,'correct_wrong':0,'wrong_correct':0,'wrong_wrong':0}
        for k in kk:
            a=bool(x[k]['metrics']['semantic_equivalence']);b=bool(y[k]['metrics']['semantic_equivalence'])
            counts[('correct' if a else 'wrong')+'_'+('correct' if b else 'wrong')]+=1
        cells.append(dict(zip(('split','policy','variant'),cell),examples=len(kk),**counts))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(old=old_name,new=new_name,cells=cells,target_equality_verified=True,inputs_sha256={str(p):digest(p) for p in (old_path,new_path,old_archive,new_archive)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('old_path');p.add_argument('new_path');p.add_argument('old_name');p.add_argument('new_name');p.add_argument('old_archive');p.add_argument('new_archive');p.add_argument('output');a=p.parse_args()
    Path(a.output).write_text(json.dumps(compare(a.old_path,a.new_path,a.old_name,a.new_name,a.old_archive,a.new_archive),indent=2)+'\n')
