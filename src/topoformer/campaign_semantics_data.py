"""Versioned compact procedural semantic cache; public text stays the actor input."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,time
from pathlib import Path
import torch
from . import semantic_scaling as base
from .semantic_contracts import single_slot_labels
from .semantic_curriculum import encode_text
from .thinking_language import ActorInput,KINDS,ROLES,lexical_bits


def compact_example(example,public):
    graph=example.privileged.graph;positions={n.id:i for i,n in enumerate(graph.nodes)}
    return dict(seed=example.audit['seed'],text=public.text,graph_sha256=graph.digest(),
        semantic_sha256=hashlib.sha256(base.semantic_key(graph).encode()).hexdigest(),
        public_sha256=hashlib.sha256(public.text.encode()).hexdigest(),
        nodes=[[n.kind,n.value] for n in graph.nodes],
        edges=[[positions[e.source],positions[e.target],e.role,e.slot] for e in graph.edges],
        tokens=len(base.tokens(public)))


def target(row,vocab,capacity=128):
    n=len(row['nodes']);assert n<=capacity
    public=ActorInput(row['text'],());tokens=base.tokens(public)
    out=dict(presence=torch.arange(capacity)<n,kind=torch.zeros(capacity,dtype=torch.long),value=torch.full((capacity,),-1,dtype=torch.long),copy=torch.full((capacity,),-1,dtype=torch.long),edges=torch.zeros(capacity,capacity,len(ROLES),dtype=torch.bool),slots=torch.full((capacity,capacity),-1,dtype=torch.long))
    for i,(kind,value) in enumerate(row['nodes']):
        out['kind'][i]=KINDS.index(kind)
        if kind in ('ident','entity'):
            positions=[j for j,t in enumerate(tokens) if t in base.identifier_forms(str(value),'english')]
            if not positions:raise ValueError('uncopyable identity')
            out['copy'][i]=positions[0]
        else:
            serialized=json.dumps(value,sort_keys=True)
            if serialized not in vocab:raise ValueError('unknown finite value')
            out['value'][i]=vocab.index(serialized)
    for i,j,role,slot in row['edges']:
        out['edges'][i,j,ROLES.index(role)]=True
        if slot is not None:out['slots'][i,j]=slot
    return out


def audit_and_cache(config):
    torch.set_num_threads(2);tick=time.monotonic();out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    requested=sum(config['splits'].values());seen=set();public_targets={};hashes={};rows=[];duplicates=0;collisions=[]
    for attempt in range(config['max_attempts']):
        example=base.build_tcn_example('unification',config['data_seed']+attempt,difficulty=.5)
        graph=example.privileged.graph;key=base.semantic_key(graph)
        if key in seen:duplicates+=1;continue
        public,_=base.surface_input(example,'english');tok=base.tokens(public)
        if public.text in public_targets and public_targets[public.text]!=key:raise ValueError('identical public input with incompatible targets')
        single_slot_labels(graph)
        if len(graph.nodes)>config['capacity'] or max((e.slot for e in graph.edges if e.slot is not None),default=-1)>=base.MAX_SLOT:raise ValueError('capacity overflow')
        features,length=encode_text(public.text)
        if length!=len(tok) or features.shape[1]!=len(tok):raise ValueError('token truncation')
        for t in tok:
            bits=tuple(lexical_bits(t))
            if bits in hashes and hashes[bits]!=t:raise ValueError('lexical hash collision')
            hashes[bits]=t
        row=compact_example(example,public)
        seen.add(key);public_targets[public.text]=key;rows.append(row)
        if len(rows)==requested:break
    if len(rows)!=requested:raise ValueError(f'insufficient canonical diversity: {len(rows)}/{requested}')
    train_count=config['splits']['train'];vocab=['<unknown>']+sorted({json.dumps(value,sort_keys=True) for r in rows[:train_count] for kind,value in r['nodes'] if kind not in ('ident','entity')})
    # Validate compact targets against the inherited compiler path on every row.
    equality_checks=0
    for row in rows:
        gold=target(row,vocab,config['capacity'])
        example=base.build_tcn_example('unification',row['seed'],difficulty=.5)
        original=base.targets(example.privileged.graph,ActorInput(row['text'],()),config['capacity'],vocab,'english')
        if not all(torch.equal(gold[k],original[k]) for k in gold):raise ValueError('compact cache target mismatch')
        equality_checks+=1
    offset=0;split_records={};artifacts={}
    for name,count in config['splits'].items():
        selected=rows[offset:offset+count];offset+=count;path=out/f'{name}.jsonl.gz'
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
            for row in selected:f.write((json.dumps(row,separators=(',',':'))+'\n').encode())
        split_records[name]=dict(count=count,semantic_sha256=hashlib.sha256(''.join(r['semantic_sha256'] for r in selected).encode()).hexdigest(),nodes=dict(collections.Counter(len(r['nodes']) for r in selected)),tokens=dict(collections.Counter(r['tokens'] for r in selected)),lesson='unification',renderer='english')
        artifacts[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    prefix={}
    for n in (128,1024,8192):
        selected=rows[:min(n,train_count)];prefix[str(n)]=dict(actual_unique=len(selected),nodes=dict(collections.Counter(len(r['nodes']) for r in selected)),tokens=sum(r['tokens'] for r in selected),representational_values=sorted({json.dumps(v,sort_keys=True) for r in selected for k,v in r['nodes'] if k not in ('ident','entity')}))
    record=dict(config=config,attempts=attempt+1,accepted=len(rows),duplicates=duplicates,rejected_contract_violations=collisions,complete_alpha_disjoint=True,scope='Full canonical alpha-key equality; no generated model predictions or test selection',value_vocabulary=vocab,split_records=split_records,train_prefixes=prefix,cache_sha256=artifacts,compact_target_equalities=equality_checks,token_types=len(hashes),cpu_seconds=time.monotonic()-tick,pinned_generator_audit=example.audit,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(record,indent=2)+'\n');return record


def load_cache(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def audit_cached_target_identifiability(data_dir,output):
    """Additional full canonical target collision check, including alpha aliases."""
    tick=time.monotonic();torch.set_num_threads(2);root=Path(data_dir);audit=json.loads((root/'audit.json').read_text());seen={};examples=0
    for split in audit['split_records']:
        for row in load_cache(root/f'{split}.jsonl.gz'):
            gold=target(row,audit['value_vocabulary']);n=len(row['nodes'])
            signature=json.dumps(dict(nodes=[[int(gold[k][i]) for k in ('kind','value','copy')] for i in range(n)],edges=sorted(row['edges'],key=lambda e:(e[0],e[1],e[2],-1 if e[3] is None else e[3]))),sort_keys=True,separators=(',',':'))
            if row['text'] in seen and seen[row['text']]!=signature:raise ValueError('same public text has incompatible canonical targets')
            seen[row['text']]=signature;examples+=1
    record=dict(examples=examples,distinct_public=len(seen),incompatible_canonical_targets=0,criterion='Full canonical node kind/value/copy-position sequence and all indexed typed ordered edges, not alpha key alone',cache_sha256=audit['cache_sha256'],cpu_seconds=time.monotonic()-tick)
    Path(output).write_text(json.dumps(record,indent=2)+'\n');return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();print(json.dumps(audit_and_cache(json.loads(Path(a.config).read_text())),indent=2))
