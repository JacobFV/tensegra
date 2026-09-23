"""Versioned CPU-only public compiler-contract decoding of frozen predictions."""
import argparse, collections, gzip, json, time
from pathlib import Path
import torch
from .campaign_semantics import digest, write_gzip
from .campaign_semantics_identity_contract import derive_refers_to
from .semantic_curriculum import unpack_graph
from .semantic_scaling import metrics
from .semantic_graph import compile_term
from .thinking_language import KINDS, ROLES

TERMS = set(KINDS) - {'scope', 'entity'}
STRUCTURAL = {'argument','item','binds','binding_scope'} | {r for r in ROLES if r.startswith('field:')}


def allowed(source, target, role):
    if role == 'contains': return source == 'scope' and target in TERMS
    if role == 'declares': return source == 'scope' and target == 'entity'
    if role == 'refers_to': return source == 'ident' and target == 'entity'
    if role == 'argument': return source in {'pred','rel','node'} and target in TERMS
    if role == 'item': return source in {'tuple','list'} and target in TERMS
    if role.startswith('field:'): return source in {'record','app'} and target in TERMS
    if role == 'binds': return source in TERMS and target in TERMS
    if role == 'binding_scope': return source == 'pred' and target == 'scope'
    raise ValueError(role)


def decode(pred, *, mask=False, bookkeeping=False):
    """Accept prediction only: no target, public gold count, or gold scope argument."""
    out={k:v.clone() for k,v in pred.items()}
    active=pred['presence']; kinds=pred['kind']; scopes=(active & kinds.eq(KINDS.index('scope'))).nonzero().flatten()
    info=dict(predicted_nodes=int(active.sum()),predicted_scopes=len(scopes),bookkeeping_abstained=bookkeeping and len(scopes)!=1)
    if mask:
        for r,role in enumerate(ROLES):
            table=torch.tensor([[allowed(a,b,role) for b in KINDS] for a in KINDS])
            out['edges'][:,:,r] &= table[kinds[:,None],kinds[None,:]] & active[:,None] & active[None,:]
    if bookkeeping:
        if len(scopes)==1:
            s=int(scopes[0]);terms=active & ~kinds.eq(KINDS.index('scope')) & ~kinds.eq(KINDS.index('entity'))
            for role,targets in [('contains',terms),('declares',active & kinds.eq(KINDS.index('entity')))]:
                r=ROLES.index(role);out['edges'][:,:,r]=False;out['edges'][s,:,r]=targets
        ref=derive_refers_to(pred);out['edges'][:,:,ROLES.index('refers_to')]=ref
        sources=active & kinds.eq(KINDS.index('ident'))
        info.update(missing_reference=int((sources & ref.sum(-1).eq(0)).sum()),multiple_references=int((sources & ref.sum(-1).gt(1)).sum()))
    structural=out['edges'][:,:,[ROLES.index(r) for r in STRUCTURAL]].any(-1)
    book=out['edges'][:,:,[ROLES.index(r) for r in ('contains','declares','refers_to')]].any(-1)
    info['shared_slot_conflicts']=int((book & structural).sum())
    if mask or bookkeeping:out['slots'][book & ~structural]=-1
    return out,info


def audit(data_dir):
    fixtures=[{'t':k,'v':v} for k,v in [('token','t'),('str','x'),('num',1),('ident','x'),('nil',None)]]
    fixtures += [{'t':k,'head':'bind' if k=='pred' else 'f','args':[{'t':'ident','v':'x'},{'t':'num','v':2}]} for k in ('pred','rel','node')]
    fixtures += [{'t':k,'items':fixtures[:5]} for k in ('tuple','list')]
    fixtures += [{'t':'record','fields':{'query':fixtures[0]}},{'t':'app','fn':'f','args':{'fact':fixtures[1]}}]
    count=edges=0;observed=collections.Counter()
    def check(nodes,relations):
        nonlocal count,edges
        scopes=[i for i,(k,v) in enumerate(nodes) if k=='scope'];assert len(scopes)==1
        s=scopes[0]
        assert {(i,j) for i,j,r,sl in relations if r=='contains'}=={(s,j) for j,(k,v) in enumerate(nodes) if k in TERMS}
        assert {(i,j) for i,j,r,sl in relations if r=='declares'}=={(s,j) for j,(k,v) in enumerate(nodes) if k=='entity'}
        for i,j,r,sl in relations:
            assert allowed(nodes[i][0],nodes[j][0],r),(nodes[i],nodes[j],r)
            if r in ('contains','declares','refers_to'):assert sl is None
            observed[r]+=1;edges+=1
        count+=1
    for term in fixtures:
        g=compile_term(term);ids={n.id:i for i,n in enumerate(g.nodes)}
        check([(n.kind,n.value) for n in g.nodes],[(ids[e.source],ids[e.target],e.role,e.slot) for e in g.edges])
    hashes={}
    for split in ('train','development'):
        p=Path(data_dir)/(split+'.jsonl.gz');hashes[split]=digest(p)
        with gzip.open(p,'rt') as stream:
            for line in stream:
                row=json.loads(line);check(row['nodes'],row['edges'])
    return dict(graphs=count,fixtures=len(fixtures),edges=edges,roles=dict(observed),data_sha256=hashes)


def run(config):
    torch.set_num_threads(2);start=time.monotonic();contract=audit(config['data_dir']);rows=[];hashes={}
    variants={'baseline':(False,False),'schema':(True,False),'bookkeeping':(False,True),'combined':(True,True)}
    for arm in config['arms']:
        hashes[arm['name']]=digest(arm['archive']);archive=json.load(gzip.open(arm['archive'],'rt'))
        for split,key in [('train','train_rows'),('development','rows')]:
            for item in archive[key]:
                gold=unpack_graph(item['target'])
                for policy in ('raw','calibrated'):
                    pred=unpack_graph(item['raw'] if policy=='raw' else {**item['raw'],'edges':item['calibrated_edges']})
                    original=metrics(pred,gold);base_edges=pred['edges'] & pred['presence'][:,None,None] & pred['presence'][None,:,None]
                    for variant,(mask,book) in variants.items():
                        if time.monotonic()-start>config['cpu_seconds']:raise TimeoutError('S10 CPU budget')
                        out,info=decode(pred,mask=mask,bookkeeping=book);m=metrics(out,gold)
                        actual=out['edges'] & out['presence'][:,None,None] & out['presence'][None,:,None]
                        old_error=base_edges.ne(gold['edges']);new_error=actual.ne(gold['edges'])
                        rows.append(dict(checkpoint=arm['name'],split=split,policy=policy,variant=variant,seed=item['seed'],metrics=m,info=info,repair=bool(m['semantic_equivalence'] and not original['semantic_equivalence']),regression=bool(original['semantic_equivalence'] and not m['semantic_equivalence']),removed_errors=int((old_error&~new_error).sum()),introduced_errors=int((~old_error&new_error).sum()),edge_flips=out['edges'].ne(pred['edges']).nonzero().tolist(),slot_changes=[(i,j,int(out['slots'][i,j])) for i,j in out['slots'].ne(pred['slots']).nonzero().tolist()]))
    summary=[]
    for key in sorted({(r['checkpoint'],r['split'],r['policy'],r['variant']) for r in rows}):
        rr=[r for r in rows if (r['checkpoint'],r['split'],r['policy'],r['variant'])==key]
        s=dict(zip(('checkpoint','split','policy','variant'),key));s.update(examples=len(rr),exact=sum(r['metrics']['semantic_equivalence'] for r in rr))
        for metric in ('typed_edge','ordered_edge'):
            tp=sum(r['metrics'][metric]['true_positive'] for r in rr);den=sum(r['metrics'][metric]['predicted_count']+r['metrics'][metric]['gold_count'] for r in rr);s[metric+'_f1']=2*tp/max(1,den)
        for f in ('repair','regression','removed_errors','introduced_errors'):s[f]=sum(r[f] for r in rr)
        s['info_totals']={f:sum(r['info'].get(f,0) for r in rr) for f in rr[0]['info']};summary.append(s)
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    write_gzip(out/'predictions.json.gz',dict(config=config,source_sha256=digest(__file__),archive_sha256=hashes,audit=contract,rows=rows))
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,cpu_seconds=time.monotonic()-start,audit=contract),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
