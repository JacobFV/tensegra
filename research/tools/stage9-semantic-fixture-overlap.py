"""Canonical overlap of fixed acquisition fixtures; no generalization claim."""
import argparse,gzip,json,sqlite3
from pathlib import Path
from topoformer.semantic_scaling import semantic_key,build_tcn_example

p=argparse.ArgumentParser();p.add_argument('old_results');p.add_argument('new_results');p.add_argument('historical_database');a=p.parse_args()
def read(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()) if str(path).endswith('.gz') else Path(path).read_bytes())
def key(audit):
    return semantic_key(build_tcn_example(audit['lesson'],audit['seed'],difficulty=.5).privileged.graph)
old={key(audit) for audit in read(a.old_results)['graph_audits']}
db=sqlite3.connect(f'file:{a.historical_database}?mode=ro',uri=True)
history={row[0] for row in db.execute('select semantic from examples')}
print(json.dumps([dict(seed=v['seed'],lesson=v['lesson'],overlap_previous_fixed=key(v) in old,overlap_stage8_pool=key(v) in history) for v in read(a.new_results)['graph_audits']],indent=2))
