"""CPU-only paired summaries from immutable S16 exported predictions."""
import argparse,gzip,hashlib,json
from pathlib import Path


def summarize(metrics):
    n=len(metrics)
    result={'examples':n,'exact':sum(int(r['semantic_equivalence']) for r in metrics)}
    for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence'):
        result[k]=sum(r[k] for r in metrics)/n
    for k in ('node','typed_edge','ordered_edge'):
        counts={f:sum(r[k][f] for r in metrics) for f in ('true_positive','predicted_count','gold_count')}
        result[k]={**counts,'micro_f1':2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])}
    return result


def analyze(root):
    manifest=json.load(gzip.open(root/'manifest.json.gz')); endpoints=[]; population=None
    assert len(manifest['artifacts'])==6
    for artifact in manifest['artifacts']:
        path=root/artifact['artifact'];assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
        data=json.load(gzip.open(path));rows=data['rows'];assert len(rows)==1024
        ids=[r['semantic_sha256'] for r in rows]
        if population is None:population=ids
        assert ids==population
        assert [r['index'] for r in rows]==list(range(1024))
        result={'seed':artifact['seed'],'arm':artifact['arm'],'artifact':path.name,'policies':{}}
        for policy in ('raw','calibrated'):
            before=[r['original_metrics'][policy] for r in rows];after=[r['metrics'][policy] for r in rows]
            repairs=sum(bool(b['semantic_equivalence']) and not bool(a['semantic_equivalence']) for a,b in zip(before,after))
            regressions=sum(bool(a['semantic_equivalence']) and not bool(b['semantic_equivalence']) for a,b in zip(before,after))
            assert repairs==sum(r['transitions'][policy]['repair'] for r in rows)
            assert regressions==sum(r['transitions'][policy]['regression'] for r in rows)
            result['policies'][policy]={'original':summarize(before),'normalized':summarize(after),'repairs':repairs,'regressions':regressions,'net_exact_change':repairs-regressions}
        endpoints.append(result)
    return {'endpoints':endpoints,'distinct_semantic_instances':len(population),'model_instance_cells':6144,
        'scope':'Same1024 instances repeated over six fixed models; do not treat6144 cells as independent examples. Original outputs are frozen references. Normalized-renamed implications come from programmed fullinput equality, not independent model forwards or learned rename invariance.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as f:json.dump(analyze(a.root),f,indent=2);f.write('\n')
