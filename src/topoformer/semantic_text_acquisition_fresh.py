"""Frozen public-text transfer evaluation; no optimization or recalibration."""
import argparse,gzip,hashlib,json,time,resource
from pathlib import Path
import torch
from . import semantic_scaling as base
from .semantic_curriculum import SemanticCurriculumActor,pack_graph


def run(config):
    torch.set_num_threads(2)
    archive=json.load(gzip.open(config['training_archive'],'rt'));audit=json.loads(Path(config['audit']).read_text())
    assert audit['accepted']==config['examples']==512
    assert [r['seed'] for r in archive['runs']]==config['fixed_seeds']==[30,31,32]
    assert all(r['curves'][-1]['update']==4000 and r['curves'][-1]['calibrated_exact']==8 for r in archive['runs'])
    dependencies={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in archive['source_sha256']}
    assert dependencies==archive['source_sha256']
    vocab=archive['value_vocabulary'];items=[]
    for row in audit['rows']:
        example=base.build_tcn_example('unification',row['seed'],difficulty=.5)
        assert example.privileged.graph.digest()==row['graph_sha256']
        public,_=base.surface_input(example,'english')
        assert hashlib.sha256(public.text.encode()).hexdigest()==row['public_sha256']
        gold=base.targets(example.privileged.graph,public,128,vocab,'english')
        assert not gold['value'].eq(0).any()
        items.append((public,gold,row))
    output=Path(config['output']);assert not output.exists();runs=[]
    for old in archive['runs']:
        checkpoint=Path(config['checkpoint_dir'])/f"model-seed{old['seed']}-u4000.pt"
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==old['curves'][-1]['checkpoint_sha256']
        model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
        model.load_state_dict(torch.load(checkpoint,map_location=config['device'],weights_only=True));model.eval()
        thresholds=torch.tensor([x['threshold'] for x in old['curves'][-1]['calibration']],device=config['device'])
        tick=time.monotonic();rows=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for public,gold,identity in items:
                outputs=model(public);pred=base.decode({k:v.cpu() for k,v in outputs.items()},public)
                cal={**pred,'edges':outputs['edges'].gt(thresholds).cpu()}
                rows.append(dict(graph_seed=identity['seed'],graph_sha256=identity['graph_sha256'],raw=pack_graph(pred),calibrated=pack_graph(cal),target=pack_graph(gold),raw_metrics=base.metrics(pred,gold),calibrated_metrics=base.metrics(cal,gold)))
        result=dict(seed=old['seed'],checkpoint_sha256=digest,thresholds=thresholds.cpu().tolist(),rows=rows,wall_seconds=time.monotonic()-tick,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        runs.append(result)
        print(json.dumps(dict(seed=old['seed'],raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in rows),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows),examples=len(rows),wall_seconds=result['wall_seconds'])),flush=True)
        manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),dependency_sha256=dependencies,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_archive_sha256=hashlib.sha256(Path(config['training_archive']).read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(Path(config['audit']).read_bytes()).hexdigest(),value_vocabulary=vocab,scope='Frozen inference; thresholds inherited only from TRAIN8, no optimizer or fresh calibration',runs=runs)
        with gzip.GzipFile(filename=str(output),mode='wb',mtime=0) as f:f.write(json.dumps(manifest,separators=(',',':')).encode())
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
