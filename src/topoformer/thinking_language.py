"""Independent Stage 6 TCN semantic track. Privileged graphs are loss targets only.

Run: python -m topoformer.thinking_language --config configs/stage6-language-pilot.json
Graph scores use canonical compiler traversal alignment, not graph isomorphism.
Output slots are decoder positions, never labelled recurrent workspace positions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from .thinking import ThinkingConfig, ThinkingModel
from .tcn_data import build_tcn_corpus, build_tcn_example, verify_vendor_manifest, LESSONS

KINDS = ('scope','entity','token','str','num','ident','nil','pred','rel','node','tuple','list','record','app')
ROLES = ('contains','declares','refers_to','argument','item','binds','binding_scope',
         'field:query','field:substitution','field:pattern','field:fact','field:facts','field:scene')
HASH_BITS = 64
FEATURE_DIM = HASH_BITS + 4


def lexical_bits(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(raw).digest()[:HASH_BITS//8]
    return [float((byte >> bit) & 1) for byte in digest for bit in range(8)]


@dataclass(frozen=True)
class ActorInput:
    """Only this public projection enters model.forward; no metadata or targets."""
    text: str
    options: tuple[str, ...]


def public_input(example, language, shuffle_seed):
    surface = next(s for s in example.public if s.language == language)
    options = list(surface.options)
    random.Random(shuffle_seed).shuffle(options)
    return ActorInput(surface.text, tuple(options))


def rendered_answer(example, language):
    surface = next(s for s in example.public if s.language == language)
    return surface.options[example.privileged.answer_index]


def encode_public(public, device='cpu'):
    # Unicode lexical chunks and punctuation, no TCN parser or dictionary.
    tokens = re.findall(r'\w+|[^\w\s]', public.text, re.UNICODE) or ['']
    rows = [lexical_bits(token) + [i/max(1,len(tokens)), math.sin(i), math.cos(i), 0.]
            for i,token in enumerate(tokens)]
    options = [lexical_bits(option) + [0.,0.,0.,1.] for option in public.options]
    return torch.tensor(rows,device=device)[None], torch.tensor(options,device=device)[None]


class LanguageActor(nn.Module):
    def __init__(self, *, width=32, workspace_rows=8, node_capacity=128, microsteps=3):
        super().__init__()
        self.capacity, self.microsteps = node_capacity, microsteps
        self.cell = ThinkingModel(ThinkingConfig(feature_dim=FEATURE_DIM,width=width,
            workspace_rows=workspace_rows,min_microsteps=1,max_microsteps=microsteps))
        self.output_queries = nn.Parameter(torch.randn(node_capacity,width)*.02)
        self.decode = nn.MultiheadAttention(width,4,batch_first=True)
        self.presence = nn.Linear(width,1)
        self.kind = nn.Linear(width,len(KINDS))
        self.lexical = nn.Linear(width,HASH_BITS)
        self.edge_source = nn.Linear(width,width*len(ROLES))
        self.edge_target = nn.Linear(width,width)
        self.option_query = nn.Linear(width,width)
        self.option_key = nn.Linear(FEATURE_DIM,width)

    def forward(self, public: ActorInput):
        if not isinstance(public,ActorInput):
            raise TypeError('forward accepts only ActorInput')
        context,options = encode_public(public,next(self.parameters()).device)
        workspace = self.cell.initialize({'context':context})
        # All phases share this cell. Only the latest workspace survives each step.
        for step in range(self.microsteps):
            workspace = self.cell.step(workspace,context,options,microstep=step+1)['workspace']
        nodes,_ = self.decode(self.output_queries[None],workspace,workspace,need_weights=False)
        nodes = nodes + self.output_queries[None]
        width = nodes.shape[-1]
        sources = self.edge_source(nodes).reshape(1,self.capacity,len(ROLES),width)
        edges = torch.einsum('bnrw,bmw->bnmr',sources,self.edge_target(nodes))/math.sqrt(width)
        pooled = workspace.mean(1)
        choices = torch.einsum('bw,bow->bo',self.option_query(pooled),self.option_key(options))/math.sqrt(width)
        return dict(choice=choices,presence=self.presence(nodes).squeeze(-1),kind=self.kind(nodes),
                    lexical=self.lexical(nodes),edges=edges,latent=pooled)


def graph_targets(graph, capacity, device='cpu'):
    if len(graph.nodes)>capacity:
        raise ValueError(f'graph has {len(graph.nodes)} nodes; fixed capacity is {capacity}')
    presence = torch.zeros(1,capacity,device=device)
    kind = torch.zeros(1,capacity,dtype=torch.long,device=device)
    lexical = torch.zeros(1,capacity,HASH_BITS,device=device)
    edges = torch.zeros(1,capacity,capacity,len(ROLES),device=device)
    indices = {node.id:i for i,node in enumerate(graph.nodes)}
    for i,node in enumerate(graph.nodes):
        presence[0,i] = 1
        kind[0,i] = KINDS.index(node.kind)
        lexical[0,i] = torch.tensor(lexical_bits(node.value),device=device)
    for edge in graph.edges:
        edges[0,indices[edge.source],indices[edge.target],ROLES.index(edge.role)] = 1
    return dict(presence=presence,kind=kind,lexical=lexical,edges=edges)


def graph_losses(output,target):
    mask = target['presence'].bool()
    # Positive and negative edge terms balanced to avoid the all-empty optimum.
    edge_target = target['edges']
    raw = F.binary_cross_entropy_with_logits(output['edges'],edge_target,reduction='none')
    positive = edge_target.bool()
    edge_loss = .5*(raw[positive].mean()+raw[~positive].mean())
    return dict(presence=F.binary_cross_entropy_with_logits(output['presence'],target['presence']),
        kind=F.cross_entropy(output['kind'][mask],target['kind'][mask]),
        lexical=F.binary_cross_entropy_with_logits(output['lexical'][mask],target['lexical'][mask]),
        edges=edge_loss)


def graph_metrics(output,target):
    predicted = output['presence'].gt(0)
    gold = target['presence'].bool()
    def pr(p,g):
        tp = (p & g).sum().item()
        return dict(precision=tp/max(1,p.sum().item()),recall=tp/max(1,g.sum().item()))
    edges = output['edges'].gt(0) & predicted[:,:,None,None] & predicted[:,None,:,None]
    gold_edges = target['edges'].bool()
    lexical = output['lexical'].gt(0).eq(target['lexical'].bool()).all(-1)
    types = output['kind'].argmax(-1).eq(target['kind'])
    return dict(node=pr(predicted,gold),typed_edge=pr(edges,gold_edges),
        node_type_accuracy=types[gold].float().mean().item(),
        lexical_hash_fidelity=lexical[gold].float().mean().item(),
        exact_canonical_graph=float(predicted.eq(gold).all() and edges.eq(gold_edges).all()
            and types[gold].all() and lexical[gold].all()))


def build_splits(config):
    required = config['train_count']+config['eval_count']
    seen,unique,collisions = set(),[],0
    for offset in range(10):
        corpus = build_tcn_corpus(count=required*2,seed=config['data_seed']+offset*required*2,
                                  difficulty=config.get('difficulty',0.5))
        for example in corpus:
            digest = example.audit['semantic_digest']
            if digest in seen:
                collisions += 1
                continue
            graph_targets(example.privileged.graph,config['node_capacity'])
            if example.audit['distinct_surface_texts'] != 3:
                raise ValueError('renderer did not change text across all three languages')
            seen.add(digest)
            unique.append(example)
            if len(unique)==required:
                train,heldout = unique[:config['train_count']],unique[config['train_count']:]
                return train,heldout,collisions
    raise ValueError('insufficient identity-disjoint constructions')


def _mean(values):
    return sum(values)/max(1,len(values))


@torch.no_grad()
def evaluate(model,examples,language,seed):
    model.eval()
    rows=[]
    failures=[]
    for i,example in enumerate(examples):
        public=public_input(example,language,seed+i)
        output=model(public)
        target=graph_targets(example.privileged.graph,model.capacity,next(model.parameters()).device)
        choice=public.options[int(output['choice'].argmax())]
        correct=choice==rendered_answer(example,language)
        metrics=graph_metrics(output,target)
        rows.append(dict(correct=float(correct),lesson=example.audit['lesson'],metrics=metrics,
                         chance=1/len(public.options),answer_index=public.options.index(rendered_answer(example,language))))
        if not correct and len(failures)<5:
            failures.append(dict(semantic_digest=example.audit['semantic_digest'],predicted=choice,
                                 expected=rendered_answer(example,language)))
    return dict(task_choice_accuracy=_mean([r['correct'] for r in rows]),
        node_precision=_mean([r['metrics']['node']['precision'] for r in rows]),
        node_recall=_mean([r['metrics']['node']['recall'] for r in rows]),
        typed_edge_precision=_mean([r['metrics']['typed_edge']['precision'] for r in rows]),
        typed_edge_recall=_mean([r['metrics']['typed_edge']['recall'] for r in rows]),
        **{key:_mean([r['metrics'][key] for r in rows]) for key in
           ('node_type_accuracy','lexical_hash_fidelity','exact_canonical_graph')},
        random_choice_baseline=_mean([r['chance'] for r in rows]),
        oracle_majority_position_baseline=max(Counter(r['answer_index'] for r in rows).values(),default=0)/max(1,len(rows)),
        task_counts=dict(Counter(r['lesson'] for r in rows)),
        per_task_accuracy={lesson:_mean([r['correct'] for r in rows if r['lesson']==lesson]) for lesson in LESSONS},
        failures=failures)


def run(config):
    if not verify_vendor_manifest():
        raise ValueError('TCN vendor manifest verification failed')
    torch.set_num_threads(config.get('threads',1))
    train,heldout,collisions=build_splits(config)
    renamed=[]
    for example in heldout:
        mapping={name:f'novelentity{i}' for i,name in enumerate(example.public[0].options)}
        renamed.append(build_tcn_example(example.audit['lesson'],example.audit['seed'],
            difficulty=config.get('difficulty',0.5),identifier_renaming=mapping))
    train_languages=config.get('train_languages',['english'])
    paired_language=config.get('paired_language','spanish')
    output_dir=Path(config['output_dir']); output_dir.mkdir(parents=True,exist_ok=True)
    audit=dict(config=config,semantic_train=[e.audit['semantic_digest'] for e in train],
        semantic_eval=[e.audit['semantic_digest'] for e in heldout],dedup_collisions=collisions,
        examples=[e.audit for e in train+heldout],renamed_examples=[e.audit for e in renamed],max_nodes=max(len(e.privileged.graph.nodes) for e in train+heldout),
        execution_scope='independent_tcn_semantic_track',
        limitations=['canonical traversal alignment; no isomorphism matching',
          'edge slot order implicit in canonical node positions; roles explicitly decoded',
          '64-bit lexical hash fidelity, not lexical string generation',
          'unseen_lexical is controlled entity alpha-renaming on heldout constructions; no heldout-motif claim',
          'Spanish paired exposure for consistency arm; symbols withheld from all arms',
          'consistency arm also receives paired task supervision and extra surface exposure; not an isolated consistency causal effect'])
    lexemes={token for e in train+heldout+renamed for surface in e.public for token in re.findall(r'\w+|[^\w\s]',surface.text,re.UNICODE)}
    lexical_signatures={tuple(lexical_bits(token)) for token in lexemes}
    audit['public_lexical_hash_collisions']=len(lexemes)-len(lexical_signatures)
    audit['task_counts']={'train':dict(Counter(e.audit['lesson'] for e in train)),
                          'eval':dict(Counter(e.audit['lesson'] for e in heldout))}
    audit['arm_surface_exposure']={'single_pass':['english'],'recurrent':['english'],
        'semantic_supervision':['english'],'multisurface_consistency':['english',paired_language]}
    source_paths=[Path(__file__),Path(__file__).with_name('thinking.py'),Path(__file__).with_name('tcn_data.py'),Path(__file__).with_name('semantic_graph.py')]
    audit['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    audit['data_hash']=hashlib.sha256(json.dumps(audit['examples'],sort_keys=True).encode()).hexdigest()
    (output_dir/'data-audit.json').write_text(json.dumps(audit,indent=2))
    records=[]
    for seed in config['seeds']:
        for arm in ('single_pass','recurrent','semantic_supervision','multisurface_consistency'):
            torch.manual_seed(seed)
            model=LanguageActor(width=config['width'],workspace_rows=config['workspace_rows'],
                 node_capacity=config['node_capacity'],microsteps=1 if arm=='single_pass' else config['microsteps']).to(config.get('device','cpu'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
            start=time.monotonic()
            for step in range(config['updates']+1):
                if step in config['eval_steps'] or step==config['updates']:
                    record=dict(seed=seed,arm=arm,step=step,elapsed_seconds=time.monotonic()-start,
                        evaluation={language:evaluate(model,heldout,language,seed+100000) for language in ('english','spanish','symbols')})
                    record['evaluation']['unseen_lexical']=evaluate(model,renamed,'english',seed+100000)
                    records.append(record)
                    with (output_dir/'curves.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                    print(json.dumps(dict(seed=seed,arm=arm,step=step)),flush=True)
                if step==config['updates']: break
                model.train(); optimizer.zero_grad()
                losses=[]; loss_log={}
                for j in range(config.get('batch_size',2)):
                    example=train[(step*config.get('batch_size',2)+j)%len(train)]
                    public=public_input(example,train_languages[step%len(train_languages)],seed+step*100+j)
                    output=model(public)
                    parts={'task':F.cross_entropy(output['choice'],torch.tensor([public.options.index(rendered_answer(example,train_languages[step%len(train_languages)]))],device=output['choice'].device))}
                    if arm in ('semantic_supervision','multisurface_consistency'):
                        parts.update({f'graph_{key}':value*config['graph_weight'] for key,value in graph_losses(output,
                            graph_targets(example.privileged.graph,model.capacity,output['choice'].device)).items()})
                    if arm=='multisurface_consistency':
                        paired=model(public_input(example,paired_language,seed+step*100+j))
                        parts['paired_task']=F.cross_entropy(paired['choice'],torch.tensor([public.options.index(rendered_answer(example,train_languages[step%len(train_languages)]))],device=output['choice'].device))
                        parts['consistency']=config['consistency_weight']*(F.mse_loss(F.normalize(output['latent'],dim=-1),F.normalize(paired['latent'],dim=-1))
                            + F.mse_loss(output['choice'].softmax(-1),paired['choice'].softmax(-1))
                            + F.mse_loss(output['presence'].sigmoid(),paired['presence'].sigmoid()))
                    losses.append(sum(parts.values()))
                    for key,value in parts.items(): loss_log.setdefault(key,[]).append(float(value.detach()))
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                with (output_dir/'losses.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(seed=seed,arm=arm,step=step+1,loss=float(loss.detach()),parts={k:_mean(v) for k,v in loss_log.items()}))+'\n')
            torch.save(model.state_dict(),output_dir/f'{arm}-seed{seed}.pt')
    (output_dir/'results.json').write_text(json.dumps(records,indent=2))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
