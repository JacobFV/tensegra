"""All-six endpoint byte preflight only; no torch/model imports or inference."""
import argparse,gzip,hashlib,json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
c=json.loads(Path(a.config).read_text())
assert sorted((e['seed'],e['arm']) for e in c['runs'])==[(s,a) for s in (701,702,703) for a in ('constant','decay')]
assert digest(c['cache'])==c['cache_sha256']
rows=[json.loads(line) for line in gzip.open(c['cache'],'rt')]
identities=[r['semantic_sha256'] for r in rows];assert len(identities)==len(set(identities))==1024
records=[]
for entry in c['runs']:
    root=Path(entry['directory']);required=[root/'manifest.json.gz',root/'model-u24576.pt',root/'evaluation-u24576.json.gz']
    missing=[str(f) for f in required if not f.exists()]
    if missing:records.append(dict(**entry,ready=False,missing=missing));continue
    m=json.load(gzip.open(required[0],'rt'));end=next(v for v in m['curves'] if v['update']==24576)
    assert m['config']['seed']==entry['seed'] and m['config']['learning_rate']=={'constant':1e-4,'decay':1e-5}[entry['arm']]
    assert digest(c['data_audit'])==m['data_audit_sha256']
    assert digest(required[1])==end['checkpoint_sha256']
    evaluation=root/end['artifact'];assert digest(evaluation)==end['sha256']
    for name,sha in m['source_sha256'].items():assert digest(Path('src/topoformer')/name)==sha
    primary=json.load(gzip.open(evaluation,'rt'));assert [r['semantic_sha256'] for r in primary['rows']]==identities
    records.append(dict(**entry,ready=True,checkpoint_sha256=end['checkpoint_sha256'],primary_sha256=end['sha256'],thresholds=primary['thresholds']))
print(json.dumps(dict(all_six_ready=all(r['ready'] for r in records),rows=records,inference_performed=False,cache_sha256=c['cache_sha256'],config_sha256=digest(a.config),preflight_source_sha256=digest(__file__)),indent=2))
