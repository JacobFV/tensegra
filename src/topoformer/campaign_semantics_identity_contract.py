"""Scoped programmed refers-to relation from learned public copy identities."""
import argparse,gzip,hashlib,json,time
import dataclasses
import re
import operator
from pathlib import Path
import torch
from .campaign_semantics import digest,write_gzip
from .campaign_semantics_data import load_cache
from .semantic_curriculum import unpack_graph,pack_graph
from .semantic_scaling import metrics,tokens,identifier_forms
from .thinking_language import KINDS,ROLES,ActorInput


# S16 supplied renderer/lexical contract; no model, target or graph determines it.
S16_VARIABLE_INVENTORY = ('A','B','C','D','E')
S16_NAME_INVENTORY = ('alice','bob','carol','dave','erin','frank')
S16_RESERVED = frozenset('You know these facts The pattern is What the unify parent and'.split())


@dataclasses.dataclass(frozen=True)
class PublicIdentityNormalization:
    public: ActorInput
    original_tokens: tuple
    canonical_tokens: tuple
    identity_positions: tuple
    original_to_canonical: tuple
    arity: int
    facts: int


def canonicalize_public_identifiers(public):
    """S16 first-occurrence alpha normalization for one closed English renderer.

    The supplied parser locates identifier argument spans in public text only.
    Case supplies variable/name type; it is not inferred from privileged nodes.
    Reject unsupported grammar, arity, namespace/candidate collisions or capacity.
    This does not infer the semantic graph or choose a unifying fact.
    """
    if not isinstance(public,ActorInput) or public.options:
        raise ValueError('S16 requires text-only public ActorInput')
    matches=list(re.finditer(r'\w+|[^\w\s]',public.text,re.UNICODE))
    tok=[m.group() for m in matches]
    if not tok: raise ValueError('empty public renderer input')
    cursor=0; positions=[]

    def expect(*literal):
        nonlocal cursor
        if tok[cursor:cursor+len(literal)]!=list(literal):
            raise ValueError('unsupported S16 public renderer grammar')
        cursor+=len(literal)

    def identifier():
        nonlocal cursor
        if cursor>=len(tok): raise ValueError('missing public identifier')
        value=tok[cursor]
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',value)
                or not (value.isupper() or value.islower()) or value in S16_RESERVED):
            raise ValueError('unsupported or reserved opaque identifier')
        positions.append(cursor);cursor+=1

    def parent():
        if cursor<len(tok) and tok[cursor]=='parent':
            expect('parent',':');identifier();arity=1
            while cursor<len(tok) and tok[cursor]==',':
                expect(',');identifier();arity+=1
            expect('and');identifier();arity+=1
            if not 3<=arity<=5:raise ValueError('parent list arity outside3–5')
        else:
            identifier();expect('parent');identifier();arity=2
        return arity

    expect('You','know','these','facts',':')
    arity=parent();fact_count=1
    while cursor<len(tok) and tok[cursor]==';':
        expect(';')
        if parent()!=arity:raise ValueError('inconsistent public predicate arity')
        fact_count+=1
    if not 1<=fact_count<=4:raise ValueError('fact count exceeds S16 renderer scope')
    expect('.','The','pattern','is')
    if parent()!=arity:raise ValueError('pattern/fact public arity mismatch')
    expect('.','What','is','the','unify');identifier();expect('?')
    if cursor!=len(tok):raise ValueError('trailing public text outside S16 grammar')

    mapping={};used={'variable':0,'name':0}
    for position in positions:
        value=tok[position]
        if value in mapping:continue
        namespace='variable' if value.isupper() else 'name'
        inventory=S16_VARIABLE_INVENTORY if namespace=='variable' else S16_NAME_INVENTORY
        if used[namespace]>=len(inventory):raise ValueError('public identity inventory capacity exceeded')
        mapping[value]=inventory[used[namespace]];used[namespace]+=1
    canonical=tok.copy()
    for position in positions:canonical[position]=mapping[tok[position]]
    # Protect the actor's existing all-token first-occurrence copy normalization.
    # An identity may not alias any nonidentity token before or after rewriting.
    other={value for i,value in enumerate(tok) if i not in set(positions)}
    if other & (set(mapping)|set(mapping.values())):
        raise ValueError('identity collides with public structural token')
    pieces=[];previous=0
    for i,match in enumerate(matches):
        pieces.extend((public.text[previous:match.start()],canonical[i]));previous=match.end()
    pieces.append(public.text[previous:])
    normalized=ActorInput(''.join(pieces),())
    if tokens(normalized)!=canonical:raise ValueError('canonicalization changed token boundaries')
    return PublicIdentityNormalization(normalized,tuple(tok),tuple(canonical),tuple(positions),
        tuple(mapping.items()),arity,fact_count)


def restore_public_copy_tokens(normalization,copy_positions):
    """Read predicted indices against original public inventory; never repair.

    Nonidentity token choices remain nonidentity strings (errors remain errors).
    -1 remains absent; out-of-range indices fail rather than using gold labels.
    All token indices, including the actor's first-occurrence reduction, survive
    normalization unchanged because its substitution is globally injective.
    """
    output=[]
    for position in copy_positions:
        index=operator.index(position)
        if index==-1:output.append(None)
        elif 0<=index<len(normalization.original_tokens):output.append(normalization.original_tokens[index])
        else:raise ValueError('predicted copy pointer outside public inventory')
    return tuple(output)


def derive_refers_to(pred):
    source=pred['presence']&pred['kind'].eq(KINDS.index('ident'))&pred['copy'].ge(0)
    target=pred['presence']&pred['kind'].eq(KINDS.index('entity'))&pred['copy'].ge(0)
    return source[:,None]&target[None,:]&pred['copy'][:,None].eq(pred['copy'][None,:])


def audit_contract(data):
    counts={};examples=edges_count=0
    for split in ('train','development'):
        rows=load_cache(Path(data)/(split+'.jsonl.gz'));counts[split]=len(rows)
        for row in rows:
            tok=tokens(ActorInput(row['text'],()));entities={};copies={}
            for i,(kind,value) in enumerate(row['nodes']):
                if kind=='entity':
                    if value in entities:raise ValueError('duplicate exact entity')
                    entities[value]=i
                if kind in ('ident','entity'):
                    visible=[j for j,t in enumerate(tok) if t in identifier_forms(value,'english')]
                    if not visible:raise ValueError('identity not publicly copyable')
                    copies[value]=visible[0]
            if len(set(copies.values()))!=len(copies):raise ValueError('visible identity collision')
            expected={(i,entities[v]) for i,(k,v) in enumerate(row['nodes']) if k=='ident'}
            actual={(i,j) for i,j,r,slot in row['edges'] if r=='refers_to'}
            if expected!=actual:raise ValueError('compiler equality contract mismatch')
            if any(slot is not None for i,j,r,slot in row['edges'] if r=='refers_to'):raise ValueError('unexpected ordered reference')
            examples+=1;edges_count+=len(actual)
    return dict(examples=examples,split_counts=counts,reference_edges=edges_count,failures=0,data_sha256={s:digest(Path(data)/(s+'.jsonl.gz')) for s in counts})


def count_relation(pred,gold):
    tp=int((pred&gold).sum());pp=int(pred.sum());gg=int(gold.sum())
    return dict(true_positive=tp,predicted=pp,gold=gg,exact=bool(pred.eq(gold).all()))


def run(config):
    torch.set_num_threads(2);start=time.monotonic();out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False);audit=audit_contract(config['data_dir']);archive=json.load(gzip.open(config['archive'],'rt'));r=ROLES.index('refers_to');rows=[]
    for split,items in [('train',archive['train_rows']),('development',archive['rows'])]:
        for item in items:
            gold=unpack_graph(item['target'])
            for mode in ('raw','calibrated'):
                pred=unpack_graph(item['raw'])
                if mode=='calibrated':pred['edges']=unpack_graph({**item['raw'],'edges':item['calibrated_edges']})['edges']
                relation=derive_refers_to(pred);active=pred['presence'][:,None]&pred['presence'][None,:];source=pred['presence']&pred['kind'].eq(KINDS.index('ident'))
                record=dict(split=split,mode=mode,seed=item['seed'],missing_predicted_entity=int((relation.sum(-1).eq(0)&source).sum()),multiple_predicted_entities=int((relation.sum(-1).gt(1)&source).sum()),arms={})
                for arm,ref in [('learned',pred['edges'][:,:,r]),('programmed',relation),('gold_reference',gold['edges'][:,:,r])]:
                    edges=pred['edges'].clone();edges[:,:,r]=ref;candidate={**pred,'edges':edges}
                    record['arms'][arm]=dict(metrics=metrics(candidate,gold),reference=count_relation(ref&active,gold['edges'][:,:,r]))
                record['programmed_ref_indices']=relation.nonzero().tolist();rows.append(record)
    result=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),dependencies_sha256={n:digest(Path(__file__).with_name(n)) for n in ('semantic_graph.py','semantic_curriculum.py','semantic_scaling.py','thinking_language.py','campaign_semantics_data.py')},audit=audit,archive_sha256=digest(config['archive']),source_sha256=digest(__file__),rows=rows,cpu_seconds=time.monotonic()-start)
    write_gzip(out/'predictions.json.gz',result)
    summary={}
    for split in ('train','development'):
        summary[split]={}
        for mode in ('raw','calibrated'):
            selected=[x for x in rows if x['split']==split and x['mode']==mode];summary[split][mode]={}
            for arm in ('learned','programmed','gold_reference'):
                rr=[x['arms'][arm] for x in selected];tp=sum(x['reference']['true_positive'] for x in rr);pp=sum(x['reference']['predicted'] for x in rr);gg=sum(x['reference']['gold'] for x in rr)
                summary[split][mode][arm]=dict(examples=len(rr),complete_graphs=sum(x['metrics']['semantic_equivalence'] for x in rr),exact_reference_sets=sum(x['reference']['exact'] for x in rr),true_positive=tp,predicted=pp,gold=gg,reference_F1=2*tp/max(1,pp+gg))
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,audit=audit,cpu_seconds=time.monotonic()-start),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
