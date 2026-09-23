"""Frozen S10 secondary policies; no inference, fitting, or reserved-data loading."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from topoformer import campaign_semantics_contract_decode as decoder
from topoformer.campaign_semantics import digest,write_gzip
from topoformer.semantic_curriculum import unpack_graph
from topoformer.semantic_scaling import metrics

POLICIES={'baseline':(False,False),'schema':(True,False),'bookkeeping':(False,True),'combined':(True,True)}
CONFIRMATION_SHA='fc85da89fe4e1a522d1e0256fc1e58f8afb279494cb1af7ee4ac7235403a43f8'


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def target_hash(target):
    return hashlib.sha256(json.dumps(target,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_archive(path,seed):
    archive=load(path);manifest_path=Path(path).parent/'manifest.json.gz';manifest=load(manifest_path)
    if archive['update']!=24576 or len(archive['rows'])!=1024 or len(archive['train_rows'])!=128:raise ValueError('endpoint/support mismatch')
    if manifest['config']['seed']!=seed or manifest['confirmation_sha256']!=CONFIRMATION_SHA:raise ValueError('producer identity mismatch')
    if not any(r['update']==24576 and r['evaluation_split']=='reserved_confirmation' for r in manifest['curves']):raise ValueError('producer split not reserved confirmation')
    identities=[r['semantic_sha256'] for r in archive['rows']]
    if len(set(identities))!=1024:raise ValueError('repeated semantic construction')
    return archive,dict(archive_sha256=digest(path),manifest_sha256=digest(manifest_path),archive_path=str(path),manifest_path=str(manifest_path))


def endpoint(config,seed,arm):
    torch.set_num_threads(2);tick=time.monotonic()
    if digest(decoder.__file__)!=config['decoder_source_sha256']:raise ValueError('frozen decoder changed')
    pair=next(x for x in config['pairs'] if x['seed']==seed);archive,provenance=validate_archive(pair[arm if arm!='frequency' else 'constant'],seed)
    frequency=None
    if arm=='frequency':
        if digest(config['frequency_baseline'])!=config['frequency_sha256']:raise ValueError('frequency prediction changed')
        baseline=load(config['frequency_baseline']);frequency=baseline['prediction'];provenance['frequency_sha256']=digest(config['frequency_baseline'])
        if baseline['train_examples']!=8192:raise ValueError('frequency fit population changed')
    rows=[]
    for split,key in [('train','train_rows'),('confirmation','rows')]:
        for event in archive[key]:
            gold=unpack_graph(event['target'])
            for mode in ('raw','calibrated'):
                packed=frequency if frequency is not None else event['raw'] if mode=='raw' else {**event['raw'],'edges':event['calibrated_edges']}
                pred=unpack_graph(packed);base=metrics(pred,gold)
                if frequency is None and base!=event[mode+'_metrics']:raise ValueError('primary actor metric replay failed')
                active=pred['presence'][:,None,None]&pred['presence'][None,:,None];old_error=(pred['edges']&active).ne(gold['edges'])
                for variant,(mask,book) in POLICIES.items():
                    if time.monotonic()-tick>config['cpu_seconds']:raise TimeoutError('CPU analysis cap')
                    out,info=decoder.decode(pred,mask=mask,bookkeeping=book);m=metrics(out,gold);error=(out['edges']&active).ne(gold['edges'])
                    rows.append(dict(split=split,mode=mode,variant=variant,event_seed=event['seed'],semantic_sha256=event['semantic_sha256'],target_sha256=target_hash(event['target']),metrics=m,info=info,repair=bool(m['semantic_equivalence'] and not base['semantic_equivalence']),regression=bool(base['semantic_equivalence'] and not m['semantic_equivalence']),removed_errors=int((old_error&~error).sum()),introduced_errors=int((~old_error&error).sum()),edge_sha256=hashlib.sha256(out['edges'].numpy().tobytes()).hexdigest(),slot_sha256=hashlib.sha256(out['slots'].numpy().tobytes()).hexdigest()))
    output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True);path=output/(f'{seed}-{arm}.json.gz' if arm!='frequency' else 'frequency.json.gz')
    if path.exists():raise FileExistsError(path)
    write_gzip(path,dict(seed=seed,arm=arm,decoder_sha256=digest(decoder.__file__),analysis_sha256=digest(__file__),config=config,provenance=provenance,rows=rows,cpu_seconds=time.monotonic()-tick))
    return str(path)


def paired_interval(delta,repetitions,seed):
    delta=np.asarray(delta,dtype=np.float64)
    if delta.shape!=(3,1024):raise ValueError('all three paired event matrices required')
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,repetitions,128):
        indices=rng.integers(0,1024,size=(min(128,repetitions-start),1024));samples.append(delta[:,indices].mean(-1).T)
    samples=np.concatenate(samples)
    return dict(per_seed_mean=delta.mean(1).tolist(),per_seed_interval=np.quantile(samples,[.025,.975],axis=0).T.tolist(),mean_gain=float(delta.mean()),shared_event_interval=np.quantile(samples.mean(1),[.025,.975]).tolist(),scope='Event resampling conditional on three fixed seeds, not seed-population uncertainty')


def summarize(config):
    tick=time.monotonic();root=Path(config['output_dir']);datasets={};hashes={}
    for seed in (701,702,703):
        for arm in ('constant','decay'):
            p=root/f'{seed}-{arm}.json.gz';datasets[(seed,arm)]=load(p);hashes[p.name]=digest(p)
    p=root/'frequency.json.gz';frequency=load(p);hashes[p.name]=digest(p)
    aggregates=[]
    for (seed,arm),data in list(datasets.items())+[((None,'frequency'),frequency)]:
        for split in ('train','confirmation'):
            for mode in ('raw','calibrated'):
                for variant in POLICIES:
                    rows=[r for r in data['rows'] if (r['split'],r['mode'],r['variant'])==(split,mode,variant)];s=dict(seed=seed,arm=arm,split=split,mode=mode,variant=variant,examples=len(rows),exact=sum(r['metrics']['semantic_equivalence'] for r in rows))
                    for metric in ('typed_edge','ordered_edge'):
                        counts={k:sum(r['metrics'][metric][k] for r in rows) for k in ('true_positive','predicted_count','gold_count')};s[metric]={**counts,'f1':2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])}
                    for f in ('repair','regression','removed_errors','introduced_errors'):s[f]=sum(r[f] for r in rows)
                    aggregates.append(s)
    pairs={}
    for mode in ('raw','calibrated'):
        for variant in POLICIES:
            differences=[];counts=[];shared=None
            for seed in (701,702,703):
                rr={arm:[r for r in datasets[(seed,arm)]['rows'] if (r['split'],r['mode'],r['variant'])==('confirmation',mode,variant)] for arm in ('constant','decay')}
                keys={arm:[(r['event_seed'],r['semantic_sha256'],r['target_sha256']) for r in rows] for arm,rows in rr.items()}
                if keys['constant']!=keys['decay'] or len(set(keys['constant']))!=1024:raise ValueError('unpaired event targets')
                if shared is not None and shared!=keys['constant']:raise ValueError('event order differs across seeds')
                shared=keys['constant'];c=np.array([r['metrics']['semantic_equivalence'] for r in rr['constant']],dtype=bool);d=np.array([r['metrics']['semantic_equivalence'] for r in rr['decay']],dtype=bool)
                differences.append(d.astype(int)-c.astype(int));counts.append(dict(seed=seed,both_correct=int((c&d).sum()),constant_only=int((c&~d).sum()),decay_only=int((~c&d).sum()),neither=int((~c&~d).sum())))
            fr=[r for r in frequency['rows'] if (r['split'],r['mode'],r['variant'])==('confirmation',mode,variant)]
            if [(r['event_seed'],r['semantic_sha256'],r['target_sha256']) for r in fr]!=shared:raise ValueError('frequency population differs')
            pairs[mode+'/'+variant]=dict(counts=counts,interval=paired_interval(differences,config['bootstrap_repetitions'],config['bootstrap_seed']))
    result=dict(aggregates=aggregates,paired=pairs,input_sha256=hashes,config=config,analysis_sha256=digest(__file__),cpu_seconds=time.monotonic()-tick,interpretation='All engineered policies secondary; primary learned promotion criteria unchanged')
    p=root/'summary.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n');return str(p)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');parser.add_argument('mode',choices=['endpoint','frequency','summary']);parser.add_argument('--seed',type=int,choices=[701,702,703],default=701);parser.add_argument('--arm',choices=['constant','decay'],default='constant');args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if sorted(p['seed'] for p in config['pairs'])!=[701,702,703]:raise ValueError('configuration missing prescribed seeds')
    print(summarize(config) if args.mode=='summary' else endpoint(config,args.seed,'frequency' if args.mode=='frequency' else args.arm))
