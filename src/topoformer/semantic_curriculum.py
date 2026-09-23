"""Stage 8: separately controlled graph diversity and optimizer exposure.

Only public text enters forward. Canonical node order and renderer-derived copy
labels are privileged training priors, not inferred graph isomorphism. Width 1024
is the default; tests must explicitly request smaller widths.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import semantic_scaling as base
from .thinking import ThinkingConfig, _WorkspaceBlock
from .thinking_language import ActorInput, encode_public, FEATURE_DIM, KINDS, ROLES, state_hash


def exposure_schedule(presentation, train_count):
    """Both languages per construction, independently of corpus size."""
    return (presentation // 2) % train_count, ('english', 'spanish')[presentation % 2]


def curriculum_weights(presentation, node_only=1000, identity_only=2000):
    return dict(presence=1., kind=1., value=float(presentation >= node_only),
                copy=float(presentation >= node_only),
                edges=float(presentation >= identity_only), slots=float(presentation >= identity_only))


class SemanticCurriculumActor(nn.Module):
    def __init__(self, *, value_count, width=1024, capacity=128, workspace_rows=8,
                 microsteps=2, no_input=False, edge_width=128):
        super().__init__()
        self.capacity, self.microsteps, self.no_input = capacity, microsteps, no_input
        self.width = width
        config = ThinkingConfig(feature_dim=FEATURE_DIM,width=width,structural_heads=0)
        self.features = nn.Linear(FEATURE_DIM,width)
        self.initial = nn.Parameter(torch.randn(1,workspace_rows,width)*.02)
        self.blocks = nn.ModuleList(_WorkspaceBlock(config) for _ in range(4))
        self.queries = nn.Parameter(torch.randn(1,capacity,width)*.02)
        self.decode_nodes = nn.MultiheadAttention(width,4,batch_first=True)
        self.presence = nn.Linear(width,1)
        self.kind = nn.Linear(width,len(KINDS))
        self.value = nn.Linear(width,value_count)
        self.copy_query = nn.Linear(width,width)
        self.copy_key = nn.Linear(FEATURE_DIM,width)
        self.edge_source = nn.Linear(width,len(ROLES)*edge_width)
        self.edge_target = nn.Linear(width,edge_width)
        self.slot_source = nn.Linear(width,base.MAX_SLOT+1)
        self.slot_target = nn.Linear(width,base.MAX_SLOT+1)
        self.edge_width = edge_width

    def forward(self, public, *, pairs=None):
        if not isinstance(public,ActorInput): raise TypeError('public ActorInput required')
        features,_ = encode_public(ActorInput(public.text,('dummy',)))
        features = features.to(self.initial.device)
        if self.no_input: features = torch.zeros_like(features[:,:1])
        memory = self.features(features)
        state = self.initial
        mask = torch.ones(memory.shape[:2],dtype=torch.bool,device=memory.device)
        for _ in range(self.microsteps):
            for block in self.blocks: state,_ = block(state,memory,mask,0.)
        nodes = self.decode_nodes(self.queries,state,state,need_weights=False)[0][0]+self.queries[0]
        source = self.edge_source(nodes).reshape(self.capacity,len(ROLES),self.edge_width)
        target = self.edge_target(nodes)
        if pairs is None:
            edge = torch.einsum('nrd,md->nmr',source,target)/math.sqrt(self.edge_width)
            slots = self.slot_source(nodes)[:,None,:]+self.slot_target(nodes)[None,:,:]
        else:
            i,j = pairs.to(nodes.device).unbind(-1)
            edge = (source[i]*target[j,None,:]).sum(-1)/math.sqrt(self.edge_width)
            slots = self.slot_source(nodes)[i]+self.slot_target(nodes)[j]
        copy = self.copy_query(nodes)@self.copy_key(features[0]).T/math.sqrt(self.width)
        if self.no_input: copy = copy.expand(-1,len(base.tokens(public)))
        return dict(presence=self.presence(nodes)[:,0],kind=self.kind(nodes),value=self.value(nodes),
                    copy=copy,edges=edge,slots=slots)


def sampled_pairs(gold, generator, negative_count=128):
    positive = gold['edges'].any(-1)
    active = gold['presence'][:,None]&gold['presence'][None,:]
    pos = torch.nonzero(positive)
    neg = torch.nonzero(active&~positive)
    if len(neg): neg = neg[torch.randperm(len(neg),generator=generator)[:negative_count]]
    return torch.cat((pos,neg))


def sampled_losses(out,gold,pairs):
    device = out['presence'].device
    gold = {k:v.to(device) for k,v in gold.items()}
    pairs = pairs.to(device)
    present = gold['presence']
    parts = dict(presence=F.binary_cross_entropy_with_logits(out['presence'],present.float()),
                 kind=F.cross_entropy(out['kind'][present],gold['kind'][present]))
    for key in ('value','copy'):
        mask = gold[key].ge(0)
        parts[key] = F.cross_entropy(out[key][mask],gold[key][mask]) if mask.any() else out[key].sum()*0
    i,j = pairs.unbind(-1)
    truth = gold['edges'][i,j]
    raw = F.binary_cross_entropy_with_logits(out['edges'],truth.float(),reduction='none')
    def balanced(values,positive):
        terms = [values[m].mean() for m in (positive,~positive) if m.any()]
        return sum(terms)/len(terms) if terms else values.sum()*0
    parts['edges'] = balanced(raw,truth)
    slots = gold['slots'][i,j]
    raw = F.cross_entropy(out['slots'],slots+1,reduction='none')
    parts['slots'] = balanced(raw,slots.ge(0))
    return parts


def frequency_fit(corpus,indices,vocab,capacity):
    """Sparse elementwise counts; no huge node-pair×vocabulary tensor."""
    counters = {}
    shapes = {}
    for index in indices:
        e=corpus[index]
        for language in ('english','spanish'):
            public,g = base.surface_input(e,language)
            gold = base.targets(g,public,capacity,vocab,language)
            for key,value in gold.items():
                shapes[key]=(value.shape,value.dtype)
                if key not in counters: counters[key]={}
                for position,label in enumerate(value.flatten().tolist()):
                    counters[key].setdefault(position,Counter())[label]+=1
    return {key:torch.tensor([counts[i].most_common(1)[0][0] for i in range(math.prod(shapes[key][0]))],
                             dtype=shapes[key][1]).reshape(shapes[key][0]) for key,counts in counters.items()}


def run(config):
    if not base.verify_vendor_manifest(): raise ValueError('vendor checksum mismatch')
    torch.set_num_threads(min(2,config.get('threads',2)))
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=False)
    train_count=config['train_count']; eval_count=config.get('eval_count',64)
    vocab_pool_count=config.get('vocab_pool_count',train_count)
    if vocab_pool_count<train_count: raise ValueError('vocabulary pool must include training prefix')
    presentations=config['presentations']; batch_size=config.get('batch_size',2)
    if presentations % batch_size: raise ValueError('presentations must divide batch size exactly')
    if eval_count<1: raise ValueError('nonempty heldout required')
    corpus=base.CorpusIndex(output/'corpus.db',requested=vocab_pool_count+eval_count,seed=config.get('data_seed',700000),
                           max_attempts=config.get('max_attempts',20*(vocab_pool_count+eval_count)))
    (output/'corpus.json').write_text(json.dumps(corpus.audit,indent=2))
    if len(corpus)<vocab_pool_count+eval_count: raise ValueError('actual corpus insufficient; no training performed')
    heldout=[corpus[i] for i in range(eval_count)]
    train_indices=range(eval_count,eval_count+train_count)
    vocab=base.value_vocabulary(corpus[i] for i in range(eval_count,eval_count+vocab_pool_count))
    capacity=config.get('node_capacity',128)
    files=[Path(__file__),Path(base.__file__),Path(__file__).with_name('thinking.py'),Path(__file__).with_name('thinking_language.py'),Path(__file__).with_name('tcn_data.py'),Path(__file__).with_name('semantic_graph.py')]
    manifest=dict(config=config,corpus=corpus.audit,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  value_vocabulary=vocab,vocabulary_pool_unique_graphs=vocab_pool_count,vocabulary_sha256=hashlib.sha256(json.dumps(vocab).encode()).hexdigest(),pinned_generator=corpus[0].audit,runs=[],
                  split_policy='same heldout prefix, nested unique training prefixes',
                  heldout_digest=hashlib.sha256(''.join(base.semantic_key(e.privileged.graph) for e in heldout).encode()).hexdigest(),
                  priors=['canonical compiler node order','privileged visible-copy and typed-edge labels','fixed ontology and node capacity','gold-edge training loss sampling only; no effect on hidden state','no_input control retains public pointer candidate count and token normalization'],
                  limitations=['exact decoder-slot equivalence is not general graph isomorphism','value ontology is fitted on declared shared training-only vocabulary pool, including unoptimized examples in smaller corpus arms','curriculum, width, edge sampling and actor differ from stage7; not a single-variable comparison'],
                  composition_allowed=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[]
    def record(model,arm,seed,seen,exposure,tokens,start):
        row=dict(arm=arm,seed=seed,optimizer_presentations=exposure,actual_unique_graphs_seen=len(seen),
                 available_unique_graphs=train_count,public_tokens_seen=tokens,elapsed_seconds=time.monotonic()-start,
                 renderer_exposure={'english':(exposure+1)//2,'spanish':exposure//2},evaluation={})
        if arm=='frequency': row.update(fit_label_presentations=2*train_count,actual_unique_graphs_seen=train_count,renderer_exposure={'english':train_count,'spanish':train_count})
        evaluation_model = model if isinstance(model,dict) else lambda public: {k:v.cpu() for k,v in model(public).items()}
        for language in ('english','spanish','symbols'):
            row['evaluation'][language]=base.evaluate(evaluation_model,heldout,language,vocab,capacity,raw_path=output/'predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure))
        row['evaluation']['heldout_lexicon']=base.evaluate(evaluation_model,heldout,'english',vocab,capacity,True,raw_path=output/'predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure))
        records.append(row)
        with (output/'curves.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    if config.get('frequency_control',True):
        start=time.monotonic(); frequency=frequency_fit(corpus,train_indices,vocab,capacity)
        record(frequency,'frequency',None,set(),0,0,start)
    for seed in config.get('seeds',[0,1,2]):
        for arm in config.get('arms',['semantic','no_input']):
            if arm not in ('semantic','no_input'): raise ValueError('unknown arm')
            torch.manual_seed(seed)
            model=SemanticCurriculumActor(value_count=len(vocab),width=config.get('width',1024),capacity=capacity,
                        workspace_rows=config.get('workspace_rows',8),microsteps=config.get('microsteps',2),no_input=arm=='no_input').to(config.get('device','cpu'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0001))
            generator=torch.Generator().manual_seed(seed+1729)
            seen=set(); tokens=0; start=time.monotonic(); initial_hash=state_hash(model)
            checkpoints=set(config.get('eval_presentations',[0,presentations]))|{0,presentations}
            for exposure in range(0,presentations+1,batch_size):
                if exposure in checkpoints: record(model,arm,seed,seen,exposure,tokens,start)
                if exposure==presentations: break
                optimizer.zero_grad(); logs=[]
                for offset in range(batch_size):
                    index,language=exposure_schedule(exposure+offset,train_count)
                    public,g=base.surface_input(corpus[eval_count+index],language)
                    gold=base.targets(g,public,capacity,vocab,language)
                    pairs=sampled_pairs(gold,generator,config.get('negative_pairs',128))
                    parts=sampled_losses(model(public,pairs=pairs),gold,pairs)
                    weights=curriculum_weights(exposure+offset,config.get('node_presentations',1000),config.get('identity_presentations',2000))
                    loss=sum(weights[k]*v for k,v in parts.items())/batch_size
                    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                    loss.backward(); logs.append({k:float(v.detach()) for k,v in parts.items()})
                    seen.add(index); tokens+=len(base.tokens(public))
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                with (output/'losses.jsonl').open('a') as f: f.write(json.dumps(dict(arm=arm,seed=seed,presentations=exposure+batch_size,parts={k:sum(row[k] for row in logs)/batch_size for k in logs[0]},weights=weights))+'\n')
            path=output/f'{arm}-{seed}.pt'; torch.save(model.state_dict(),path)
            manifest['runs'].append(dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in model.parameters()),width=model.width,
                           initial_state_sha256=initial_hash,final_state_sha256=state_hash(model),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
