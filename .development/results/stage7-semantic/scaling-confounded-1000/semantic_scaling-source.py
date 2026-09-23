"""Stage7 independent semantic acquisition; no runtime or execution coupling.

Matching is exact under the pinned compiler's traversal, not arbitrary graph
isomorphism or denotational equivalence. Alpha-normalization preserves repeated
identity, all values, edge roles and argument slots. Public token pointers are
learned: gold spans never enter forward(). SQLite stores construction descriptors,
not rendered corpora. Repeated surfaces/renamings never increase graph cardinality.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3
import time

import torch
from torch import nn
from torch.nn import functional as F
from .tcn_data import build_tcn_example, verify_vendor_manifest, LESSONS
from .thinking import ThinkingModel, ThinkingConfig
from .thinking_language import ActorInput, encode_public, KINDS, ROLES, FEATURE_DIM, state_hash

MAX_SLOT=32


def semantic_key(graph):
    ids={n.id:i for i,n in enumerate(graph.nodes)}; identities={}
    nodes=[]
    for n in graph.nodes:
        value=n.value
        if n.kind in ('ident','entity'):
            identities.setdefault(value,len(identities)); value=['identity',identities[value]]
        nodes.append((n.kind,value))
    return json.dumps([nodes,sorted((ids[e.source],ids[e.target],e.role,e.slot) for e in graph.edges),
                       [ids[r] for r in graph.roots]],sort_keys=True,separators=(',',':'))


class CorpusIndex:
    def __init__(self,path,*,requested,seed,max_attempts,difficulty=.5):
        path=Path(path)
        if path.exists(): raise ValueError('index already exists; use a fresh output directory')
        if requested<1 or max_attempts<1: raise ValueError('positive corpus budgets required')
        self.db=sqlite3.connect(path); self.difficulty=difficulty
        self.db.execute('create table examples (id integer primary key, lesson text, seed integer, semantic text unique)')
        start=time.monotonic(); duplicates=0; count=0; digest=hashlib.sha256()
        for attempt in range(max_attempts):
            lesson=LESSONS[attempt%len(LESSONS)]; example=build_tcn_example(lesson,seed+attempt,difficulty=difficulty)
            key=semantic_key(example.privileged.graph)
            cursor=self.db.execute('insert or ignore into examples(lesson,seed,semantic) values (?,?,?)',(lesson,seed+attempt,key))
            if cursor.rowcount:
                count+=1; digest.update(key.encode()); digest.update(f'{lesson}:{seed+attempt}'.encode())
            else: duplicates+=1
            if count==requested: break
        self.db.commit()
        self.audit=dict(requested_unique_graphs=requested,actual_unique_graphs=count,attempts=attempt+1,
            duplicates=duplicates,candidate_budget_exhausted=count<requested,
            finite_space_exhaustion_proven=False,dataset_sha256=digest.hexdigest(),
            generation_seconds=time.monotonic()-start,difficulty=difficulty,
            dedup='complete alpha-normalized typed ordered graph JSON; collision-free equality')
    def __len__(self): return self.audit['actual_unique_graphs']
    def __getitem__(self,index):
        row=self.db.execute('select lesson,seed from examples where id=?',(index+1,)).fetchone()
        if row is None: raise IndexError(index)
        return build_tcn_example(*row,difficulty=self.difficulty)


def surface_input(example,language,renamed=False):
    if renamed:
        mapping={v:f'heldoutentity{i}' for i,v in enumerate(example.public[0].options)}
        if example.audit['lesson']!='set_operations': mapping.update({v:f'heldoutvar{i}' for i,v in enumerate('ABCDE')})
        example=build_tcn_example(example.audit['lesson'],example.audit['seed'],difficulty=example.audit['difficulty'],identifier_renaming=mapping)
    surface=next(s for s in example.public if s.language==language)
    return ActorInput(surface.text,()),example.privileged.graph


def tokens(public): return re.findall(r'\w+|[^\w\s]',public.text,re.UNICODE) or ['']


def value_vocabulary(examples):
    return ['<unknown>']+sorted({json.dumps(n.value,sort_keys=True) for e in examples for n in e.privileged.graph.nodes if n.kind not in ('ident','entity')})


@lru_cache(maxsize=512)
def identifier_forms(value,language):
    """Supervision only: exact forms from the unchanged pinned renderer pack."""
    from ._vendor.tcn_language.languages import get_language
    from ._vendor.tcn_language._structure import Ident
    from ._vendor.tcn_language.languages.lexicon import load_vocabulary
    forms={get_language(language).render(Ident(value))}
    if language!='symbols':
        lexical=load_vocabulary(language)[0]
        adjective=lexical.adjectives.get(value); noun=lexical.nouns.get(value)
        if adjective is not None: forms.update(getattr(adjective,key) for key in ('base','ms','fs','mp','fp'))
        if noun is not None: forms.update((noun.lemma,noun.plural))
    return frozenset(forms-{''})


def targets(graph,public,capacity,vocab,language="english"):
    if isinstance(public,tuple): public=public[0]
    if len(graph.nodes)>capacity: raise ValueError('graph exceeds prespecified node capacity')
    tok=tokens(public); indices={n.id:i for i,n in enumerate(graph.nodes)}
    out=dict(presence=torch.zeros(capacity,dtype=torch.bool),kind=torch.zeros(capacity,dtype=torch.long),
        value=torch.full((capacity,),-1,dtype=torch.long),copy=torch.full((capacity,),-1,dtype=torch.long),
        edges=torch.zeros(capacity,capacity,len(ROLES),dtype=torch.bool),slots=torch.full((capacity,capacity),-1,dtype=torch.long))
    for i,n in enumerate(graph.nodes):
        out['presence'][i]=True; out['kind'][i]=KINDS.index(n.kind)
        if n.kind in ('ident','entity'):
            visible=[j for j,token in enumerate(tok) if token in identifier_forms(str(n.value),language)]
            if not visible: raise ValueError(f'identifier is not visibly copyable: {n.value}')
            out['copy'][i]=visible[0]
        else:
            value=json.dumps(n.value,sort_keys=True); out['value'][i]=vocab.index(value) if value in vocab else 0
    for e in graph.edges:
        i,j=indices[e.source],indices[e.target]; out['edges'][i,j,ROLES.index(e.role)]=True
        if e.slot is not None:
            if e.slot>=MAX_SLOT: raise ValueError('ordered slot exceeds fixed schema capacity')
            out['slots'][i,j]=e.slot
    return out


class SemanticActor(nn.Module):
    def __init__(self,*,width,capacity,workspace_rows,microsteps,value_count,no_input=False):
        super().__init__(); self.capacity=capacity; self.microsteps=microsteps; self.no_input=no_input
        self.cell=ThinkingModel(ThinkingConfig(feature_dim=FEATURE_DIM,width=width,workspace_rows=workspace_rows,min_microsteps=1,max_microsteps=microsteps))
        self.queries=nn.Parameter(torch.randn(capacity,width)*.02)
        self.decode=nn.MultiheadAttention(width,4,batch_first=True)
        self.presence=nn.Linear(width,1); self.kind=nn.Linear(width,len(KINDS)); self.value=nn.Linear(width,value_count)
        self.copy_query=nn.Linear(width,width); self.copy_key=nn.Linear(FEATURE_DIM,width)
        self.source=nn.Linear(width,width*len(ROLES)); self.target=nn.Linear(width,width)
        self.slot_source=nn.Linear(width,MAX_SLOT+1); self.slot_target=nn.Linear(width,MAX_SLOT+1)
    def forward(self,public):
        if isinstance(public,tuple): public=public[0]
        if not isinstance(public,ActorInput): raise TypeError('only public text may enter the actor')
        context,_=encode_public(ActorInput(public.text,('dummy',)))
        if self.no_input: context=torch.zeros_like(context[:,:1])
        workspace=self.cell.initialize({'context':context})
        for step in range(self.microsteps): workspace=self.cell.step(workspace,context,context,microstep=step+1)['workspace']
        nodes=self.decode(self.queries[None],workspace,workspace,need_weights=False)[0][0]+self.queries
        edge=torch.einsum('nrw,mw->nmr',self.source(nodes).reshape(self.capacity,len(ROLES),-1),self.target(nodes))/math.sqrt(nodes.shape[-1])
        copy=self.copy_query(nodes)@self.copy_key(context[0]).T/math.sqrt(nodes.shape[-1])
        if self.no_input: copy=copy.expand(-1,len(tokens(public)))
        return dict(presence=self.presence(nodes)[:,0],kind=self.kind(nodes),value=self.value(nodes),copy=copy,edges=edge,
                    slots=self.slot_source(nodes)[:,None,:]+self.slot_target(nodes)[None,:,:])


def losses(out,gold):
    present=gold['presence']; positive=gold['edges']; raw=F.binary_cross_entropy_with_logits(out['edges'],positive.float(),reduction='none')
    active=present[:,None]&present[None,:]
    edge_negative=(~positive)&active[:,:,None]
    slot_raw=F.cross_entropy(out['slots'].flatten(0,1),(gold['slots']+1).flatten(),reduction='none').reshape_as(gold['slots'])
    ordered=gold['slots'].ge(0)
    slot_loss=.5*(slot_raw[ordered].mean()+slot_raw[active&~ordered].mean())
    result=dict(presence=F.binary_cross_entropy_with_logits(out['presence'],present.float()),
        kind=F.cross_entropy(out['kind'][present],gold['kind'][present]),edges=.5*(raw[positive].mean()+raw[edge_negative].mean()),slots=slot_loss)
    for key in ('value','copy'):
        mask=gold[key].ge(0); result[key]=F.cross_entropy(out[key][mask],gold[key][mask]) if mask.any() else out[key].sum()*0
    return result


def decode(out,public=None):
    result = {k:(v.gt(0) if k in ('presence','edges') else v.argmax(-1)-(1 if k=='slots' else 0)) for k,v in out.items()}
    if public is not None:
        tok=tokens(public); first=torch.tensor([tok.index(t) for t in tok])
        result['copy']=first[result['copy']]
    return result


def metrics(pred,gold):
    mask=gold['presence']; p=pred['presence']; edges=pred['edges'] & p[:,None,None] & p[None,:,None]
    def counts(a,b):
        tp=int((a&b).sum()); np=int(a.sum()); ng=int(b.sum())
        return dict(true_positive=tp,predicted_count=np,gold_count=ng,precision=tp/np if np else 0.,recall=tp/ng if ng else 0.,f1=2*tp/(np+ng) if np+ng else 1.)
    ordered_gold=gold['edges'] & gold['slots'].ge(0)[:,:,None]
    ordered_pred=edges & pred['slots'].ge(0)[:,:,None]
    ordered=counts(ordered_pred,ordered_gold)
    ordered['true_positive']=int((ordered_pred&ordered_gold&pred['slots'].eq(gold['slots'])[:,:,None]).sum())
    tp=ordered['true_positive']; np=ordered['predicted_count']; ng=ordered['gold_count']
    ordered.update(precision=tp/np if np else 0.,recall=tp/ng if ng else 0.,f1=2*tp/(np+ng) if np+ng else 1.)
    correct={k:pred[k].eq(gold[k]) for k in ('kind','value','copy')}
    copied=gold['copy'].ge(0)
    # Equality is measured separately from exact visible identifier copying.
    pair=copied[:,None]&copied[None,:]
    eq=pred['copy'][:,None].eq(pred['copy'][None,:]).eq(gold['copy'][:,None].eq(gold['copy'][None,:]))
    semantic=bool(not gold['value'].eq(0).any() and p.eq(mask).all() and correct['kind'][mask].all() and edges.eq(gold['edges']).all()
        and correct['value'][gold['value'].ge(0)].all() and correct['copy'][copied].all()
        and pred['slots'][gold['edges'].any(-1)].eq(gold['slots'][gold['edges'].any(-1)]).all())
    return dict(node=counts(p,mask),typed_edge=counts(edges,gold['edges']),ordered_edge=ordered,
        node_type_accuracy=float(correct['kind'][mask].float().mean()),
        identity_copy_accuracy=float(correct['copy'][copied].float().mean()) if copied.any() else 1.,
        entity_equivalence=float(eq[pair].float().mean()) if pair.any() else 1.,semantic_equivalence=float(semantic))


def evaluate(model,examples,language,vocab,capacity,renamed=False,raw_path=None,context=None):
    rows=[]
    with torch.no_grad():
        for e in examples:
            public,g=surface_input(e,language,renamed); gold=targets(g,public,capacity,vocab,language=language)
            pred=model if isinstance(model,dict) else decode(model(public),public)
            rows.append(metrics(pred,gold))
            if raw_path is not None:
                def compact(graph):
                    return {k:(torch.nonzero(v).tolist() if k=='edges' else v.tolist()) for k,v in graph.items()}
                with gzip.open(raw_path,'at',compresslevel=1) as stream:
                    stream.write(json.dumps(dict(context=context,language=language,renamed=renamed,public_text=public.text,graph_sha256=g.digest(),prediction=compact(pred),target=compact(gold),metrics=rows[-1]))+'\n')
    result={k:sum(r[k] for r in rows)/len(rows) for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence','semantic_equivalence')}
    for k in ('node','typed_edge','ordered_edge'):
        counts={f:sum(r[k][f] for r in rows) for f in ('true_positive','predicted_count','gold_count')}
        result[k]={**counts,'f1':2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])}
    result['examples']=len(rows); return result


def split_indices(train_count,eval_count):
    return dict(eval=range(eval_count),train=range(eval_count,eval_count+train_count))


def run(config):
    if not verify_vendor_manifest(): raise ValueError('pinned vendor manifest mismatch')
    torch.set_num_threads(min(2,config.get('threads',2)))
    outdir=Path(config['output_dir']); outdir.mkdir(parents=True,exist_ok=False)
    corpus=CorpusIndex(outdir/'constructions.db',requested=config['train_count']+config['eval_count'],seed=config['data_seed'],max_attempts=config['max_attempts'],difficulty=config.get('difficulty',.5))
    (outdir/'corpus.json').write_text(json.dumps(corpus.audit,indent=2))
    if len(corpus)<config['train_count']+config['eval_count']:
        raise ValueError('requested cardinality unavailable within candidate budget; actual counts saved; training aborted')
    train_count=config['train_count']; splits=split_indices(train_count,config['eval_count'])
    heldout=[corpus[i] for i in splits['eval']]
    def train_example(i): return corpus[config['eval_count']+i]
    vocab=value_vocabulary(train_example(i) for i in range(train_count))
    capacity=config['node_capacity']; source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('tcn_data.py'),Path(__file__).with_name('semantic_graph.py'),Path(__file__).with_name('thinking.py'),Path(__file__).with_name('thinking_language.py')]}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=dict(config=config,config_sha256=config_hash,source_sha256=source,corpus=corpus.audit,value_vocabulary=vocab,
        pinned_audit=corpus[0].audit,runs=[],matching='fixed compiler traversal, exact values and identities and explicit ordered slots; no general isomorphism',
        gate_f={'passed':False,'reason':'requires paired cross-budget heldout typed-edge and semantic trend above baselines; no automatic single-run pass'},
        priors=['canonical decoder slot order','fixed node capacity and role/type ontology','first visible occurrence pointer supervision'],
        train_renderers=['english','spanish'],heldout_renderer='symbols',
        limitations=['alpha-equivalent graphs counted once; not logical/denotational equivalence','heldout lexicon is consistent entity/variable renaming','no claim of exhaustive finite-family enumeration'])
    manifest['split_semantic_sha256']={name:hashlib.sha256(''.join(semantic_key(corpus[i].privileged.graph) for i in indices).encode()).hexdigest() for name,indices in splits.items()}
    manifest['split_policy']='fixed eval prefix reserved before nested train prefixes; canonical disjoint by SQLite uniqueness'
    manifest['loss_recipe']='v2: balance ordered/nonordered active pairs; edge negatives on gold-present pairs only'
    (outdir/'semantic_scaling-source.py').write_bytes(Path(__file__).read_bytes())
    # Elementwise frequency decoder is fitted solely on training labels; pointers
    # are positional frequencies, without any evaluation graph access.
    counters={}
    for i in range(train_count):
        e=train_example(i); public,g=surface_input(e,'english'); target=targets(g,public,capacity,vocab)
        for key,value in target.items():
            size={'presence':2,'kind':len(KINDS),'value':len(vocab)+1,'copy':config.get('max_public_tokens',2048)+1,'edges':2,'slots':MAX_SLOT+1}[key]
            labels=value.long()+(1 if key in ('value','copy','slots') else 0)
            if labels.max()>=size: raise ValueError('frequency pointer exceeds prespecified public token capacity')
            if key not in counters: counters[key]=torch.zeros((*value.shape,size),dtype=torch.int32)
            counters[key].scatter_add_(-1,labels[...,None],torch.ones_like(labels[...,None],dtype=torch.int32))
    frequency={key:(cs.argmax(-1)-(1 if key in ('value','copy','slots') else 0)).to(target[key].dtype) for key,cs in counters.items()}
    del counters
    records=[]
    def record(model,seed,arm,step,start,seen,tokcount):
        row=dict(seed=seed,arm=arm,step=step,optimizer_examples=step*config['batch_size'] if arm!='frequency' else 0,
            actual_unique_graphs_seen=train_count if arm=='frequency' else len(seen),fit_label_examples=train_count if arm=='frequency' else step*config['batch_size'],public_tokens_seen=tokcount,elapsed_seconds=time.monotonic()-start,
            dataset_sha256=corpus.audit['dataset_sha256'],config_sha256=config_hash,
            evaluation={lang:evaluate(model,heldout,lang,vocab,capacity,raw_path=outdir/'raw-evaluation.jsonl.gz',context=dict(seed=seed,arm=arm,step=step)) for lang in ('english','spanish','symbols')})
        row['evaluation']['heldout_lexicon']=evaluate(model,heldout,'english',vocab,capacity,True,raw_path=outdir/'raw-evaluation.jsonl.gz',context=dict(seed=seed,arm=arm,step=step))
        row['fixed_training_set']=evaluate(model,[train_example(i) for i in range(min(train_count,config.get('acquisition_count',8)))],'english',vocab,capacity)
        records.append(row)
        with (outdir/'curves.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
    controls=[]
    for e in heldout:
        partner=next((other for other in heldout if other.audit['lesson']==e.audit['lesson'] and semantic_key(other.privileged.graph)!=semantic_key(e.privileged.graph)),None)
        if partner is not None:
            p,g=surface_input(e,'english'); q,h=surface_input(partner,'english')
            controls.append(dict(gold=e.audit['semantic_digest'],wrong=partner.audit['semantic_digest'],token_jaccard=len(set(tokens(p))&set(tokens(q)))/len(set(tokens(p))|set(tokens(q))),semantic_equivalence=float(semantic_key(g)==semantic_key(h))))
    manifest['different_graph_similar_wording_controls']=controls
    record(frequency,None,'frequency',0,time.monotonic(),set(),0)
    for seed in config['seeds']:
        for arm in ('semantic','no_input'):
            torch.manual_seed(seed); model=SemanticActor(width=config['width'],capacity=capacity,workspace_rows=config['workspace_rows'],microsteps=config['microsteps'],value_count=len(vocab),no_input=arm=='no_input')
            initial=state_hash(model); optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']); start=time.monotonic(); seen=set(); tokcount=0
            for step in range(config['updates']+1):
                if step==0 or step in config['eval_steps'] or step==config['updates']: record(model,seed,arm,step,start,seen,tokcount)
                if step==config['updates']: break
                optimizer.zero_grad(); parts=[]; component_log=[]
                for j in range(config['batch_size']):
                    index=(step*config['batch_size']+j)%train_count; e=train_example(index)
                    language=('english','spanish')[(step//max(1,math.ceil(train_count/config['batch_size'])))%2]
                    public,g=surface_input(e,language)
                    seen.add(index); tokcount+=len(tokens(public)); component=losses(model(public),targets(g,public,capacity,vocab,language=language)); parts.append(sum(component.values())); component_log.append({k:float(v.detach()) for k,v in component.items()})
                loss=torch.stack(parts).mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                with (outdir/'losses.jsonl').open('a') as stream: stream.write(json.dumps(dict(seed=seed,arm=arm,step=step+1,loss=float(loss.detach()),parts={k:sum(row[k] for row in component_log)/len(component_log) for k in component}))+'\n')
            path=outdir/f'{arm}-{seed}.pt'; torch.save(model.state_dict(),path)
            manifest['runs'].append(dict(seed=seed,arm=arm,initial_sha256=initial,final_state_sha256=state_hash(model),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start))
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
