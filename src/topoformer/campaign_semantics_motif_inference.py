"""S14 fixed all-six endpoint diagnostic; inherited thresholds, no fitting."""
import argparse,gzip,json,time,resource
from pathlib import Path
import torch
from .campaign_semantics import Dataset,digest,write_gzip,decode
from .campaign_semantics_data import load_cache
from .semantic_curriculum import SemanticCurriculumActor,unpack_graph
from . import semantic_scaling as base

CACHE_SHA='8b75c555e33ea69e3cc15a1ed7a922acd5b631294a0b1c57ccd9429711eea4d5'
BASELINE_SHA='600bfeeaf6d4f11a9129a7f6cf96b94ee8a2966e4395a99a8617c757bdad845e'


def validate(config):
    if config.get('budget_status')!='frozen':raise ValueError('separate coordinator freeze required')
    if sorted((e['seed'],e['arm']) for e in config['runs'])!=[(s,a) for s in (701,702,703) for a in ('constant','decay')]:raise ValueError('all six fixed endpoints required')
    if config['cache_sha256']!=CACHE_SHA or config['baseline_sha256']!=BASELINE_SHA:raise ValueError('registered population/control changed')
    if config['job'] not in ('profile','main'):raise ValueError('unregistered job')


def compact(graph):
    """Metric-sufficient public prediction; retain all node labels, active edges only."""
    active=graph['presence'].bool();edges=graph['edges'].bool()&active[:,None,None]&active[None,:,None]
    triples=torch.nonzero(edges).tolist();pairs=sorted({(i,j) for i,j,r in triples})
    return dict(capacity=len(active),present=torch.nonzero(active).flatten().tolist(),
                **{k:graph[k].tolist() for k in ('kind','value','copy')},edges=triples,
                slots=[[i,j,int(graph['slots'][i,j])] for i,j in pairs])


def expand(record):
    n=record['capacity'];result=dict(presence=torch.zeros(n,dtype=torch.bool),
        edges=torch.zeros(n,n,len(base.ROLES),dtype=torch.bool),slots=torch.full((n,n),-1,dtype=torch.long),
        **{k:torch.tensor(record[k],dtype=torch.long) for k in ('kind','value','copy')})
    result['presence'][record['present']]=True
    for i,j,r in record['edges']:result['edges'][i,j,r]=True
    for i,j,s in record['slots']:result['slots'][i,j]=s
    return result


def expand_policies(row):
    raw=expand(row['raw'])
    record={**row['raw'],'edges':row['calibrated_edges'],
            'slots':row['raw']['slots']+row['calibrated_extra_slots']}
    return raw,expand(record)


def summarize(rows):
    groups={}
    for row in rows:
        group=groups.setdefault(str(row['node_count']),dict(examples=0,raw_exact=0,calibrated_exact=0))
        group['examples']+=1
        for policy in ('raw','calibrated'):group[policy+'_exact']+=int(row[policy+'_metrics']['semantic_equivalence'])
    return dict(examples=len(rows),raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in rows),
                calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows),by_node_count=groups)


def preflight(config):
    """Validate every endpoint before the first model construction/forward."""
    validate(config)
    if digest(config['cache'])!=CACHE_SHA or digest(config['baseline'])!=BASELINE_SHA:raise ValueError('cache/control bytes changed')
    audit=json.loads(Path(config['data_audit']).read_text())
    if audit['value_vocabulary']!=['<unknown>','"parent"','"unify"','null']:raise ValueError('inherited vocabulary changed')
    rows=load_cache(config['cache'])
    if len(rows)!=1024 or len({r['semantic_sha256'] for r in rows})!=1024:raise ValueError('population changed')
    inherited=[]
    for entry in config['runs']:
        root=Path(entry['directory']);manifest_path=root/'manifest.json.gz';manifest=json.load(gzip.open(manifest_path,'rt'))
        endpoint=next(v for v in manifest['curves'] if v['update']==24576)
        if manifest['config']['seed']!=entry['seed'] or manifest['config']['learning_rate']!={'constant':1e-4,'decay':1e-5}[entry['arm']]:raise ValueError('wrong seed/arm')
        if digest(config['data_audit'])!=manifest['data_audit_sha256']:raise ValueError('inherited audit changed')
        checkpoint=root/'model-u24576.pt';primary_path=root/endpoint['artifact']
        if digest(checkpoint)!=endpoint['checkpoint_sha256'] or digest(primary_path)!=endpoint['sha256']:raise ValueError('endpoint bytes changed')
        for name,sha in manifest['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('inherited source changed')
        primary=json.load(gzip.open(primary_path,'rt'))
        if primary['update']!=24576 or len(primary['rows'])!=1024:raise ValueError('wrong primary endpoint')
        inherited.append(dict(**entry,checkpoint=str(checkpoint),checkpoint_sha256=endpoint['checkpoint_sha256'],
            primary_sha256=endpoint['sha256'],parent_manifest_sha256=digest(manifest_path),thresholds=primary['thresholds']))
    baseline=json.load(gzip.open(config['baseline'],'rt'))
    if baseline['train_examples']!=8192 or baseline['training_cache_sha256']!=audit['cache_sha256']['train']:raise ValueError('control training changed')
    return rows,audit,inherited,baseline


def run(config):
    tick=time.monotonic();torch.set_num_threads(2)
    rows,audit,inherited,baseline=preflight(config)
    if config['job']=='profile':rows=rows[:8]  # fixed mechanical prefix, never a selected population
    dataset=Dataset(rows,audit['value_vocabulary']);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    targets=[];frequency=[];fixed_pred=unpack_graph(baseline['prediction'])
    for index,row in enumerate(rows):
        _,gold,_=dataset[index];packed=compact(gold)
        if base.metrics(expand(packed),gold)!=base.metrics(gold,gold):raise ValueError('target compact roundtrip changed metrics')
        targets.append(dict(semantic_sha256=row['semantic_sha256'],target=packed))
        metrics=base.metrics(fixed_pred,gold)
        if base.metrics(expand(compact(fixed_pred)),gold)!=metrics:raise ValueError('control compact roundtrip changed metrics')
        frequency.append(dict(semantic_sha256=row['semantic_sha256'],metrics=metrics))
    write_gzip(out/'targets.json.gz',dict(cache_sha256=CACHE_SHA,rows=targets))
    write_gzip(out/'frequency.json.gz',dict(baseline_sha256=BASELINE_SHA,prediction=compact(fixed_pred),rows=frequency,
        exact=sum(r['metrics']['semantic_equivalence'] for r in frequency),fit='inherited full-TRAIN coordinate frequency; no refit'))
    artifacts=[];torch.cuda.reset_peak_memory_stats()
    for entry in inherited:
        start=time.monotonic();model=SemanticCurriculumActor(value_count=len(audit['value_vocabulary']),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to(config['device'])
        model.load_state_dict(torch.load(entry['checkpoint'],map_location='cpu',weights_only=True)['model']);model.eval();thresholds=torch.tensor(entry['thresholds']);predictions=[]
        with torch.no_grad():
            for index,row in enumerate(rows):
                public,gold,_=dataset[index];raw,cal=decode(model(public),public,thresholds)
                raw_metrics=base.metrics(raw,gold);cal_metrics=base.metrics(cal,gold);raw_pack=compact(raw);cal_pack=compact(cal)
                if base.metrics(expand(raw_pack),gold)!=raw_metrics or base.metrics(expand(cal_pack),gold)!=cal_metrics:raise ValueError('compact prediction changes metrics')
                # Attributes and pair slots are shared across policies; store calibrated edges/extra slots only.
                raw_pairs={(i,j) for i,j,s in raw_pack['slots']}
                predictions.append(dict(seed=row['seed'],semantic_sha256=row['semantic_sha256'],node_count=len(row['nodes']),tree_shape_sha256=row['tree_shape_sha256'],
                    raw=raw_pack,calibrated_edges=cal_pack['edges'],calibrated_extra_slots=[v for v in cal_pack['slots'] if tuple(v[:2]) not in raw_pairs],raw_metrics=raw_metrics,calibrated_metrics=cal_metrics))
        path=out/f"motif-{entry['arm']}-{entry['seed']}.json.gz";result=dict(endpoint=entry,cache_sha256=CACHE_SHA,targets_sha256=digest(out/'targets.json.gz'),rows=predictions,summary=summarize(predictions),seconds=time.monotonic()-start)
        write_gzip(path,result);record=dict(seed=entry['seed'],arm=entry['arm'],artifact=path.name,sha256=digest(path),summary=result['summary'],seconds=result['seconds']);artifacts.append(record)
        print(json.dumps(dict(event='endpoint',**record)),flush=True);del model
    manifest=dict(config=config,source_sha256=digest(__file__),artifacts=artifacts,targets_sha256=digest(out/'targets.json.gz'),frequency_sha256=digest(out/'frequency.json.gz'),
        process_seconds=time.monotonic()-tick,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_gzip(out/'manifest.json.gz',manifest);(out/'completion.json').write_text(json.dumps(dict(success=True,process_seconds=time.monotonic()-tick))+'\n');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
