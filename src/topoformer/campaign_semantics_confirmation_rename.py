"""S12 optional fixed lexical rename: frozen model and primary TRAIN thresholds."""
import argparse,gzip,json,resource,time
from pathlib import Path
import torch
from .campaign_semantics import Dataset,digest,write_gzip,decode
from .campaign_semantics_data import load_cache
from .semantic_curriculum import SemanticCurriculumActor,pack_graph
from . import semantic_scaling as base


def run(config):
    start=time.monotonic();torch.set_num_threads(2)
    if sorted((x['seed'],x['arm']) for x in config['runs'])!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('six complete fixed endpoints required')
    if digest(config['cache'])!=config['cache_sha256']:raise ValueError('renamed cache changed')
    rows=load_cache(config['cache']);audit=json.loads(Path(config['data_audit']).read_text());vocab=audit['value_vocabulary']
    if len(rows)!=1024:raise ValueError('wrong renamed support')
    dataset=Dataset(rows,vocab);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False);records=[]
    for entry in config['runs']:
        root=Path(entry['directory']);manifest=json.load(gzip.open(root/'manifest.json.gz','rt'));endpoint=next(c for c in manifest['curves'] if c['update']==24576)
        if manifest['config']['seed']!=entry['seed'] or manifest['config']['learning_rate']!={'constant':1e-4,'decay':1e-5}[entry['arm']]:raise ValueError('wrong paired endpoint')
        checkpoint=root/'model-u24576.pt';evaluation=root/endpoint['artifact']
        if digest(checkpoint)!=endpoint['checkpoint_sha256'] or digest(evaluation)!=endpoint['sha256']:raise ValueError('endpoint bytes changed')
        for name,sha in manifest['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('inherited source mismatch')
        primary=json.load(gzip.open(evaluation,'rt'))
        if len(primary['rows'])!=1024 or [r['semantic_sha256'] for r in primary['rows']]!=[r['semantic_sha256'] for r in rows]:raise ValueError('unpaired primary/renamed semantic identities')
        thresholds=torch.tensor(primary['thresholds']);model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval();predictions=[]
        with torch.no_grad():
            for index in range(len(dataset)):
                public,gold,row=dataset[index];raw,cal=decode(model(public),public,thresholds)
                predictions.append(dict(seed=row['seed'],semantic_sha256=row['semantic_sha256'],graph_sha256=row['graph_sha256'],raw=pack_graph(raw),calibrated_edges=pack_graph(cal)['edges'],target=pack_graph(gold),raw_metrics=base.metrics(raw,gold),calibrated_metrics=base.metrics(cal,gold)))
        result=dict(seed=entry['seed'],arm=entry['arm'],thresholds=primary['thresholds'],primary_sha256=digest(evaluation),checkpoint_sha256=digest(checkpoint),renamed_cache_sha256=config['cache_sha256'],rows=predictions)
        path=out/f"renamed-{entry['arm']}-{entry['seed']}.json.gz";write_gzip(path,result);records.append(dict(seed=entry['seed'],arm=entry['arm'],artifact=path.name,sha256=digest(path)))
        del model
    manifest=dict(config=config,source_sha256=digest(__file__),artifacts=records,process_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_gzip(out/'manifest.json.gz',manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();print(json.dumps(run(json.loads(Path(a.config).read_text()))))
