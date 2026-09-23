"""Frozen full-TRAIN coordinate-frequency reference on reserved constructions."""
import argparse,gzip,json,time,hashlib
from pathlib import Path
import torch
from .campaign_semantics import digest,write_gzip
from .campaign_semantics_data import load_cache,target
from .semantic_curriculum import unpack_graph
from . import semantic_scaling as base


def run(config):
    torch.set_num_threads(2);tick=time.monotonic();data=Path(config['data_dir'])
    if digest(config['baseline'])!=config['baseline_sha256']:raise ValueError('frozen frequency model changed')
    if digest(data/'reserved_confirmation.jsonl.gz')!=config['confirmation_sha256']:raise ValueError('reserved cache changed')
    archived=json.load(gzip.open(config['baseline'],'rt'));audit=json.loads((data/'audit.json').read_text())
    if archived['train_examples']!=8192 or archived['training_cache_sha256']!=audit['cache_sha256']['train']:raise ValueError('frequency training mismatch')
    pred=unpack_graph(archived['prediction']);rows=load_cache(data/'reserved_confirmation.jsonl.gz');results=[]
    for row in rows:
        gold=target(row,audit['value_vocabulary']);results.append(dict(seed=row['seed'],semantic_sha256=row['semantic_sha256'],metrics=base.metrics(pred,gold)))
    result=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256=digest(__file__),scope=archived['scope'],prediction=archived['prediction'],rows=results,examples=len(results),exact=sum(r['metrics']['semantic_equivalence'] for r in results),cpu_seconds=time.monotonic()-tick)
    write_gzip(config['output'],result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();r=run(json.loads(Path(a.config).read_text()));print(json.dumps({k:r[k] for k in ('examples','exact','cpu_seconds')}))
