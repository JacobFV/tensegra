"""CPU-only S12 lexical-bijection contract audit; never runs an actor."""
import argparse,dataclasses,gzip,hashlib,json,re,time
from pathlib import Path
import torch
from .campaign_semantics import digest
from .campaign_semantics_data import load_cache,target
from . import semantic_scaling as base
from .semantic_graph import SemanticGraph
from .thinking_language import ActorInput,lexical_bits


def rename(row,mapping):
    text=re.sub(r'\w+',lambda m:mapping.get(m.group(),m.group()),row['text'])
    nodes=[[kind,mapping[value] if kind in ('ident','entity') else value] for kind,value in row['nodes']]
    return {**row,'text':text,'nodes':nodes,'public_sha256':hashlib.sha256(text.encode()).hexdigest()}


def audit(config):
    torch.set_num_threads(2);tick=time.monotonic();data=Path(config['data_dir']);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    train=load_cache(data/'train.jsonl.gz');rows=load_cache(data/'reserved_confirmation.jsonl.gz');meta=json.loads((data/'audit.json').read_text())
    if digest(data/'reserved_confirmation.jsonl.gz')!=config['confirmation_sha256']:raise ValueError('reserved cache mismatch')
    vocab=meta['value_vocabulary'];names=sorted({v for r in train+rows for k,v in r['nodes'] if k in ('ident','entity')})
    mapping={name:'nuv'+chr(97+i//26)+chr(97+i%26)+'z' for i,name in enumerate(names)}
    mapping={name:(new.upper() if name.isupper() else new) for name,new in mapping.items()}
    old_tokens={t for r in train for t in base.tokens(ActorInput(r['text'],()))}
    if set(mapping.values())&old_tokens:raise ValueError('renamed token already TRAIN-visible')
    bits={}
    for token in old_tokens|set(mapping.values()):
        key=tuple(lexical_bits(token))
        if key in bits and bits[key]!=token:raise ValueError('lexical hash collision')
        bits[key]=token
    renamed=[]
    for row in rows:
        new=rename(row,mapping);oldtarget=target(row,vocab);newtarget=target(new,vocab)
        if any(not torch.equal(oldtarget[k],newtarget[k]) for k in oldtarget):raise ValueError('copy/typed target changed')
        if len(base.tokens(ActorInput(new['text'],())))!=row['tokens']:raise ValueError('token count changed')
        graph=base.build_tcn_example('unification',row['seed'],difficulty=.5).privileged.graph
        id_map={n.id:'entity:'+hashlib.sha256(mapping[n.value].encode()).hexdigest() for n in graph.nodes if n.kind=='entity'}
        newgraph=SemanticGraph(tuple(dataclasses.replace(n,id=id_map.get(n.id,n.id),value=mapping[n.value] if n.kind in ('ident','entity') else n.value) for n in graph.nodes),tuple(dataclasses.replace(e,source=id_map.get(e.source,e.source),target=id_map.get(e.target,e.target)) for e in graph.edges),graph.roots,graph.compiler_version)
        if base.semantic_key(newgraph)!=base.semantic_key(graph):raise ValueError('alpha semantics changed')
        new['graph_sha256']=newgraph.digest();new['original_graph_sha256']=row['graph_sha256'];renamed.append(new)
    path=out/'renamed_confirmation.jsonl.gz'
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
        for row in renamed:f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
    report=dict(config=config,mapping=mapping,examples=len(rows),unchanged_tensor_targets=len(rows),alpha_equivalent_graphs=len(rows),new_identifier_tokens=len(mapping),training_lexicon_overlap=0,lexical_hash_collisions=0,renamed_cache_sha256=digest(path),original_cache_sha256=config['confirmation_sha256'],cpu_seconds=time.monotonic()-tick,scope='Same1024semanticgraphs with consistently renamed visible identifier tokens and canonical entity IDs; no model forward, no new renderer')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();print(json.dumps(audit(json.loads(Path(a.config).read_text()))))
