"""CPU-only summaries from immutable compact predictions; no model/target refit."""
import collections,gzip,hashlib,json
from pathlib import Path
root=Path('research/results/campaign-01/semantics/s14-motif-main')
load=lambda p:json.load(gzip.open(p,'rt'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=load(root/'manifest.json.gz');target_file=load(root/'targets.json.gz');targets={r['semantic_sha256']:r['target'] for r in target_file['rows']}
assert len(targets)==1024

def aggregate(rows,policy):
    metrics=[r[policy+'_metrics'] for r in rows];out=dict(exact=sum(m['semantic_equivalence'] for m in metrics),copy=sum(m['identity_copy_accuracy'] for m in metrics)/len(rows),node_type=sum(m['node_type_accuracy'] for m in metrics)/len(rows))
    for key in ('node','typed_edge','ordered_edge'):
        tp=sum(m[key]['true_positive'] for m in metrics);pred=sum(m[key]['predicted_count'] for m in metrics);gold=sum(m[key]['gold_count'] for m in metrics)
        out[key]=dict(true_positive=tp,predicted_count=pred,gold_count=gold,f1=2*tp/(pred+gold) if pred+gold else 1.)
    return out

results=[]
for artifact in manifest['artifacts']:
    path=root/artifact['artifact'];assert sha(path)==artifact['sha256'];data=load(path);rows=data['rows'];assert len(rows)==1024 and {r['semantic_sha256'] for r in rows}==set(targets)
    seed,arm=artifact['seed'],artifact['arm'];primary=load(root.parent/f's12-{arm}-{seed}/evaluation-u24576.json.gz');assert data['endpoint']['thresholds']==primary['thresholds']
    exact_presence=exact_types=exact_copy=0;predicted_counts=collections.Counter()
    for row in rows:
        g=targets[row['semantic_sha256']];p=row['raw'];predicted_counts[len(p['present'])]+=1
        exact_presence+=p['present']==g['present'];exact_types+=all(p['kind'][i]==g['kind'][i] for i in g['present']);exact_copy+=all(p['copy'][i]==v for i,v in enumerate(g['copy']) if v>=0)
    sizes={str(n):dict(examples=sum(r['node_count']==n for r in rows),raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in rows if r['node_count']==n),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows if r['node_count']==n)) for n in sorted({r['node_count'] for r in rows})}
    motifs={shape:dict(examples=sum(r['tree_shape_sha256']==shape for r in rows),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows if r['tree_shape_sha256']==shape)) for shape in sorted({r['tree_shape_sha256'] for r in rows})}
    results.append(dict(seed=seed,arm=arm,artifact_sha256=sha(path),raw=aggregate(rows,'raw'),calibrated=aggregate(rows,'calibrated'),exact_presence=exact_presence,exact_node_types=exact_types,exact_copy=exact_copy,predicted_node_counts=dict(predicted_counts),by_node_count=sizes,by_motif=motifs))
frequency=load(root/'frequency.json.gz');out=dict(source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json.gz'),targets_sha256=sha(root/'targets.json.gz'),results=results,frequency_exact=frequency['exact'],scope='All6 fixed endpoints; original TRAIN thresholds; no fitting/model calls; exact failure counts are descriptive')
(root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in results:print(r['seed'],r['arm'],'presence/type/copy exact',r['exact_presence'],r['exact_node_types'],r['exact_copy'],'copy',r['calibrated']['copy'],'typed',r['calibrated']['typed_edge']['f1'],'ordered',r['calibrated']['ordered_edge']['f1'])
