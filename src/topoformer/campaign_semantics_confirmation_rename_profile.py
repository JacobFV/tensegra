"""Mechanical fixed-prefix profile; frozen rename main evaluator is unchanged."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from .campaign_semantics import Dataset,digest,decode,write_gzip
from .campaign_semantics_data import load_cache
from .campaign_semantics_profile_freeze import verify_profile_source,require_complete_matrix
from .semantic_curriculum import SemanticCurriculumActor,pack_graph
from . import semantic_scaling as base


def run(config):
    start=time.monotonic();torch.set_num_threads(2)
    verify_profile_source(config,Path(__file__).parent);matrix=require_complete_matrix()
    if config['profile_lane']!='rename':raise ValueError('wrong profile lane')
    if digest(config['cache'])!=config['cache_sha256'] or config['cache_sha256']!='a3f3bc5ed3d6fb188f5fb787d079af532cc67ec170a6b89937ea1d0f7777a4fb':raise ValueError('fixed renamed population changed')
    rows=load_cache(config['cache']);audit=json.loads(Path(config['data_audit']).read_text())
    if len(rows)!=1024:raise ValueError('full frozen population required')
    if sorted((e['seed'],e['arm']) for e in config['runs'])!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('all six endpoints required')
    preflight=[]
    for entry in config['runs']:
        root=Path(entry['directory']);m=json.load(gzip.open(root/'manifest.json.gz','rt'));end=next(e for e in m['curves'] if e['update']==24576)
        if m['config']['seed']!=entry['seed'] or m['config']['learning_rate']!={'constant':1e-4,'decay':1e-5}[entry['arm']]:raise ValueError('wrong endpoint')
        if digest(config['data_audit'])!=m['data_audit_sha256']:raise ValueError('audit changed')
        if digest(root/'model-u24576.pt')!=end['checkpoint_sha256'] or digest(root/end['artifact'])!=end['sha256']:raise ValueError('endpoint changed')
        for name,sha in m['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('inherited source changed')
        primary=json.load(gzip.open(root/end['artifact'],'rt'))
        if [r['semantic_sha256'] for r in primary['rows']]!=[r['semantic_sha256'] for r in rows]:raise ValueError('unpaired rename population')
        preflight.append((entry,root,end,primary['thresholds']))
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False);dataset=Dataset(rows,audit['value_vocabulary']);records=[]
    for entry,root,end,thresholds in preflight:
        tick=time.monotonic();model=SemanticCurriculumActor(value_count=len(audit['value_vocabulary']),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
        model.load_state_dict(torch.load(root/'model-u24576.pt',map_location='cpu',weights_only=True)['model']);model.eval();output_digest=hashlib.sha256()
        with torch.no_grad():
            for index in range(8):
                public,gold,row=dataset[index];raw,cal=decode(model(public),public,torch.tensor(thresholds))
                payload=dict(raw=pack_graph(raw),calibrated_edges=pack_graph(cal)['edges'],raw_metrics=base.metrics(raw,gold),calibrated_metrics=base.metrics(cal,gold))
                output_digest.update(json.dumps(payload,sort_keys=True).encode())
        records.append(dict(**entry,examples=8,seconds=time.monotonic()-tick,checkpoint_sha256=end['checkpoint_sha256'],output_sha256=output_digest.hexdigest()));del model
    result=dict(config=config,primary_matrix=matrix,source_sha256=digest(__file__),rows=records,examples=48,
                scope='Fixed first8 per all6 endpoints; timing only, no outcome selection, not main diagnostic',process_seconds=time.monotonic()-start)
    write_gzip(out/'profile.json.gz',result);(out/'completion.json').write_text(json.dumps(dict(success=True,process_seconds=time.monotonic()-start))+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
