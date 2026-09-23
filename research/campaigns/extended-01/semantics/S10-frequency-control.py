"""Replay the immutable matched frequency prediction, then frozen S10 policies."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics import digest,write_gzip
from topoformer.campaign_semantics_data import load_cache,target
from topoformer.semantic_curriculum import unpack_graph,pack_graph
from topoformer.semantic_scaling import metrics
from topoformer.campaign_semantics_contract_decode import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=json.loads(Path(a.config).read_text());tick=time.monotonic();torch.set_num_threads(2)
    baseline=json.load(gzip.open(c['baseline'],'rt'));data=Path(c['data_dir']);assert digest(data/'train.jsonl.gz')==baseline['training_cache_sha256'];assert baseline['train_examples']==8192
    vocab=json.loads((data/'audit.json').read_text())['value_vocabulary'];pred=unpack_graph(baseline['prediction']);old={r['seed']:r['metrics'] for r in baseline['rows']};archive={}
    for split,key,n in [('train','train_rows',128),('development','rows',512)]:
        rows=load_cache(data/(split+'.jsonl.gz'))[:n];archive[key]=[]
        for row in rows:
            gold=target(row,vocab);m=metrics(pred,gold)
            if split=='development':assert m==old[row['seed']],(row['seed'],'historical frequency mismatch')
            archive[key].append(dict(seed=row['seed'],target=pack_graph(gold),raw=baseline['prediction'],calibrated_edges=baseline['prediction']['edges']))
    path=Path(c['archive_output']);path.parent.mkdir(parents=True,exist_ok=True);write_gzip(path,archive)
    config=dict(arms=[dict(name='TRAIN8192_frequency',archive=str(path))],data_dir=c['data_dir'],output_dir=c['output_dir'],cpu_seconds=max(1,c['cpu_seconds']-(time.monotonic()-tick)))
    run(config)
    receipt=dict(wall_internal_seconds=time.monotonic()-tick,baseline_sha256=digest(c['baseline']),baseline_scope=baseline['scope'],baseline_dev_replay=512,script_sha256=digest(__file__),config_sha256=digest(a.config),supplied_policy_source_sha256=digest(Path(__file__).parents[4]/'src/topoformer/campaign_semantics_contract_decode.py') if False else hashlib.sha256(__import__('inspect').getsource(__import__('topoformer.campaign_semantics_contract_decode',fromlist=[''])).encode()).hexdigest())
    (Path(c['output_dir'])/'baseline-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
