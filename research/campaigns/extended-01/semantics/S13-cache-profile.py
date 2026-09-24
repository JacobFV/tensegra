"""Fixed TRAIN prefix CPU profile; no model or reserved-data access."""
import gzip,itertools,json,time
from pathlib import Path
import torch
from topoformer.campaign_semantics import digest
from topoformer.campaign_semantics_multisurface_data import regenerate_allowed_train,write_rows

def run(config):
    torch.set_num_threads(2);tick=time.monotonic();source=Path(config['source_data_dir']);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    if digest(source/'train.jsonl.gz')!=config['train_sha256']:raise ValueError('TRAIN changed')
    with gzip.open(source/'train.jsonl.gz','rt') as stream:rows=[json.loads(line) for line in itertools.islice(stream,config['count'])]
    vocab=json.loads((source/'audit.json').read_text())['value_vocabulary'];start=time.monotonic();records=regenerate_allowed_train(rows,vocab,{},{});generate_seconds=time.monotonic()-start
    start=time.monotonic();write_rows(out/'train-prefix.jsonl.gz',records);export_seconds=time.monotonic()-start
    result=dict(config=config,source_sha256=digest(__file__),count=len(records),generation_seconds=generate_seconds,export_seconds=export_seconds,internal_seconds=time.monotonic()-tick,projected_8704_record_seconds=(generate_seconds+export_seconds)*8704/len(records),estimate_scope='Linear conservative same-cost-per-record estimate; full metadata load and candidate duplicate rejection add overhead',prefix_sha256=digest(out/'train-prefix.jsonl.gz'),english_regeneration_exact=True)
    (out/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    import sys;run(json.loads(Path(sys.argv[1]).read_text()))
