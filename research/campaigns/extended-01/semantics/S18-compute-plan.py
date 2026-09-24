"""CPU-only public-text length/schedule audit; no actor or model forward."""
import argparse,collections,gzip,hashlib,json
from pathlib import Path
import torch
from topoformer.semantic_curriculum import encode_text
from topoformer.campaign_semantics_continue import next_indices
from topoformer.campaign_semantics_s18_compute import plan

CACHE_SHA='8ddbd15bea881c13bfd24554ada9293082fcef7b616f6eeb3dad78a4bfa662d2'

def run(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==CACHE_SHA
    with gzip.open(path,'rt') as stream:
        # Only public text determines lengths; no graph, labels or metadata.
        texts=[json.loads(line)['text'] for line in stream]
    assert len(texts)==4096
    lengths=[encode_text(t)[1] for t in texts]
    rng=torch.Generator().manual_seed(15115);order=torch.randperm(4096,generator=rng).tolist();position=0
    maxima=[];sequence=hashlib.sha256();seen=collections.Counter()
    for _ in range(4096):
        indices,order,position=next_indices(order,position,rng,8)
        sequence.update(json.dumps(indices).encode());seen.update(indices)
        maxima.append(max(lengths[i] for i in indices))
    assert len(seen)==4096 and set(seen.values())=={8}
    return dict(cache_sha256=CACHE_SHA,torch_version=torch.__version__,schedule_seed=15115,
        updates=4096,batch_size=8,unique_examples=4096,presentations=32768,
        public_length_counts=dict(sorted(collections.Counter(lengths).items())),
        padded_batch_length_counts=dict(sorted(collections.Counter(maxima).items())),
        sequence_sha256=sequence.hexdigest(),**plan(maxima))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as stream:json.dump(run(a.cache),stream,indent=2);stream.write('\n')
