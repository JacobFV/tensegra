"""Prespecified paired confirmation intervals, conditional on three fixed seeds."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from .campaign_semantics import digest


def paired_interval(differences,repetitions=10000,seed=12012):
    delta=np.asarray(differences,dtype=np.float64)
    if delta.ndim!=2 or delta.shape[0]!=3 or delta.shape[1]!=1024:raise ValueError('three complete paired seed matrices required')
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,repetitions,128):
        index=rng.integers(0,delta.shape[1],size=(min(128,repetitions-start),delta.shape[1]))
        samples.append(delta[:,index].mean(axis=-1).T)
    samples=np.concatenate(samples)
    return dict(per_seed_mean=delta.mean(1).tolist(),per_seed_interval=np.quantile(samples,[.025,.975],axis=0).T.tolist(),mean_paired_gain=float(delta.mean()),shared_event_mean_interval=np.quantile(samples.mean(1),[.025,.975]).tolist(),scope='Example-resampling uncertainty conditional on three fixed seeds; not a seed-population interval',repetitions=repetitions,bootstrap_seed=seed)


def run(config):
    if sorted(x['seed'] for x in config['pairs'])!=[701,702,703]:raise ValueError('incomplete seed matrix')
    outputs={};source={}
    for policy in ('raw_metrics','calibrated_metrics'):
        differences=[];pairs=[];shared_identities=None
        for item in sorted(config['pairs'],key=lambda x:x['seed']):
            arms={arm:json.load(gzip.open(item[arm],'rt')) for arm in ('constant','decay')}
            for arm,a in arms.items():
                if a['update']!=24576 or len(a['rows'])!=1024:raise ValueError('wrong endpoint/support')
                source[item[arm]]=digest(item[arm])
            identities=[[r['semantic_sha256'] for r in arms[arm]['rows']] for arm in ('constant','decay')]
            if identities[0]!=identities[1] or len(set(identities[0]))!=1024:raise ValueError('unpaired or repeated semantic examples')
            if shared_identities is not None and identities[0]!=shared_identities:raise ValueError('seed event ordering differs')
            shared_identities=identities[0]
            c=np.asarray([r[policy]['semantic_equivalence'] for r in arms['constant']['rows']],dtype=bool)
            d=np.asarray([r[policy]['semantic_equivalence'] for r in arms['decay']['rows']],dtype=bool)
            differences.append(d.astype(int)-c.astype(int));pairs.append(dict(seed=item['seed'],constant=int(c.sum()),decay=int(d.sum()),both_correct=int((c&d).sum()),constant_only=int((c&~d).sum()),decay_only=int((~c&d).sum()),neither=int((~c&~d).sum()),support=1024))
        interval=paired_interval(differences)
        outputs[policy]=dict(pairs=pairs,interval=interval,all_decayed_competent=all(x['decay']>=103 for x in pairs),replicated_directional_advantage=all(x>0 for x in interval['per_seed_mean']) and interval['shared_event_mean_interval'][0]>0)
    result=dict(config=config,source_sha256=source,results=outputs,primary='calibrated_metrics',secondary='raw_metrics; supplied-schema outcomes separate')
    Path(config['output']).write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();print(json.dumps(run(json.loads(Path(a.config).read_text()))))
