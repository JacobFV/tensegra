"""Losslessly convert legacy COO edge arrays; run beside installed source package."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from topoformer.semantic_curriculum import pack_graph, unpack_graph
from topoformer.thinking_language import ROLES


def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): value.update(block)
    return value.hexdigest()


def repack(source,target):
    records=0
    with gzip.open(source,'rt') as incoming, gzip.open(target,'wt',compresslevel=6) as outgoing:
        for line in incoming:
            row=json.loads(line)
            for role in ('prediction','target'):
                old=row[role]; capacity=len(old['presence'])
                graph={key:torch.tensor(value) for key,value in old.items() if key!='edges'}
                edges=torch.zeros(capacity,capacity,len(ROLES),dtype=torch.bool)
                if old['edges']:
                    coordinates=torch.tensor(old['edges']); edges[tuple(coordinates.T)]=True
                graph['edges']=edges
                compact=pack_graph(graph)
                recovered=unpack_graph(compact)
                assert torch.nonzero(recovered['edges']).tolist()==old['edges']
                assert all(recovered[key].tolist()==old[key] for key in old if key!='edges')
                row[role]=compact
            outgoing.write(json.dumps(row)+'\n'); records+=1
    return dict(source_sha256=digest(source),target_sha256=digest(target),records=records,
                source_bytes=Path(source).stat().st_size,target_bytes=Path(target).stat().st_size,
                verification='all decoded tensor fields exactly match every legacy record, including ordered COO edges')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('target'); parser.add_argument('audit')
    args=parser.parse_args(); torch.set_num_threads(2)
    Path(args.audit).write_text(json.dumps(repack(args.source,args.target),indent=2)+'\n')
