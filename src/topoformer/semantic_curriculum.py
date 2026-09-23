"""Stage 8: separately controlled graph diversity and optimizer exposure.

Only public text enters forward. Canonical node order and renderer-derived copy
labels are privileged training priors, not inferred graph isomorphism. Width 1024
is the default; tests must explicitly request smaller widths.
"""
from __future__ import annotations
import argparse
import base64
import gzip
import numpy as np
import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache
import time
import torch
from torch import nn
from torch.nn import functional as F
from . import semantic_scaling as base
from .thinking import ThinkingConfig, _WorkspaceBlock
from .thinking_language import ActorInput, encode_public, FEATURE_DIM, KINDS, ROLES, state_hash


@lru_cache(maxsize=32768)
def encode_text(text):
    """CPU public features only; no labels or learned state enter this cache."""
    features,_=encode_public(ActorInput(text,('dummy',)))
    return features,len(base.tokens(ActorInput(text,())))


def exposure_schedule(presentation, train_count):
    """Both languages per construction, independently of corpus size."""
    return (presentation // 2) % train_count, ('english', 'spanish')[presentation % 2]


def curriculum_weights(presentation, node_only=1000, identity_only=2000):
    return dict(presence=1., kind=1., value=float(presentation >= node_only),
                copy=float(presentation >= node_only),
                edges=float(presentation >= identity_only), slots=float(presentation >= identity_only))


class SemanticCurriculumActor(nn.Module):
    def __init__(self, *, value_count, width=1024, capacity=128, workspace_rows=8,
                 microsteps=2, no_input=False, edge_width=128, autocast_dtype=None):
        super().__init__()
        self.capacity, self.microsteps, self.no_input = capacity, microsteps, no_input
        self.width = width
        if autocast_dtype not in (None,'bfloat16'): raise ValueError('only optional bfloat16 autocast supported')
        self.autocast_dtype=autocast_dtype
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
        return self.forward_batch([public],pairs=None if pairs is None else [pairs])[0]

    def forward_batch(self, publics, *, pairs=None):
        with torch.autocast(self.initial.device.type,dtype=torch.bfloat16,enabled=self.autocast_dtype=='bfloat16'):
            outputs=self._forward_batch(publics,pairs=pairs)
        return [{k:v.float() for k,v in row.items()} for row in outputs]

    def _forward_batch(self, publics, *, pairs=None):
        if not publics or any(not isinstance(p,ActorInput) for p in publics):
            raise TypeError('nonempty public ActorInput batch required')
        encoded=[]; lengths=[]
        for public in publics:
            features,length=encode_text(public.text)
            lengths.append(length)
            encoded.append(torch.zeros_like(features[0,:1]) if self.no_input else features[0])
        features=nn.utils.rnn.pad_sequence(encoded,batch_first=True).to(self.initial.device)
        mask=torch.arange(features.shape[1],device=features.device)[None,:]<torch.tensor([len(e) for e in encoded],device=features.device)[:,None]
        memory=self.features(features)
        state=self.initial.expand(len(publics),-1,-1)
        for _ in range(self.microsteps):
            for block in self.blocks: state,_=block(state,memory,mask,0.)
        queries=self.queries.expand(len(publics),-1,-1)
        nodes=self.decode_nodes(queries,state,state,need_weights=False)[0]+queries
        source=self.edge_source(nodes).reshape(len(publics),self.capacity,len(ROLES),self.edge_width)
        target=self.edge_target(nodes)
        copy=self.copy_query(nodes)@self.copy_key(features).transpose(-1,-2)/math.sqrt(self.width)
        presence=self.presence(nodes)[...,0]; kind=self.kind(nodes); value=self.value(nodes)
        slot_source=self.slot_source(nodes); slot_target=self.slot_target(nodes)
        outputs=[]
        for row in range(len(publics)):
            if pairs is None:
                edge=torch.einsum('nrd,md->nmr',source[row],target[row])/math.sqrt(self.edge_width)
                slots=slot_source[row,:,None,:]+slot_target[row,None,:,:]
            else:
                i,j=pairs[row].to(nodes.device).unbind(-1)
                edge=(source[row,i]*target[row,j,None,:]).sum(-1)/math.sqrt(self.edge_width)
                slots=slot_source[row,i]+slot_target[row,j]
            copying=copy[row,:,:lengths[row]]
            if self.no_input: copying=copy[row].expand(-1,lengths[row])
            outputs.append(dict(presence=presence[row],kind=kind[row],value=value[row],copy=copying,edges=edge,slots=slots))
        return outputs


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
    """Vectorized categorical counts; pair tables have small fixed vocabularies."""
    counters = {}; shapes = {}
    for index in indices:
        e=corpus[index]
        for language in ('english','spanish'):
            public,g = base.surface_input(e,language)
            gold = base.targets(g,public,capacity,vocab,language)
            for key,value in gold.items():
                shapes[key]=(value.shape,value.dtype)
                size={'presence':2,'kind':len(KINDS),'value':len(vocab)+1,
                      'copy':2049,'edges':2,'slots':base.MAX_SLOT+1}[key]
                labels=value.long()+(1 if key in ('value','copy','slots') else 0)
                if labels.max()>=size: raise ValueError('frequency pointer exceeds fixed token capacity')
                if key not in counters: counters[key]=torch.zeros((*value.shape,size),dtype=torch.int32)
                counters[key].scatter_add_(-1,labels[...,None],torch.ones_like(labels[...,None],dtype=torch.int32))
    return {key:(counts.argmax(-1)-(1 if key in ('value','copy','slots') else 0)).to(shapes[key][1])
            for key,counts in counters.items()}


def pack_graph(graph):
    result={k:v.tolist() for k,v in graph.items() if k!='edges'}
    edges=graph['edges'].detach().cpu().numpy()
    result['edges']=dict(shape=list(edges.shape),bitorder='little',
                         packed_b64=base64.b64encode(np.packbits(edges,bitorder='little').tobytes()).decode())
    return result


def unpack_graph(graph):
    result={k:torch.tensor(v) for k,v in graph.items() if k!='edges'}
    encoded=graph['edges']; packed=np.frombuffer(base64.b64decode(encoded['packed_b64']),dtype=np.uint8)
    edges=np.unpackbits(packed,bitorder=encoded['bitorder'],count=math.prod(encoded['shape'])).reshape(encoded['shape'])
    result['edges']=torch.from_numpy(edges.astype(bool))
    return result


def evaluate(model,examples,language,vocab,capacity,renamed=False,raw_path=None,context=None):
    """Same Stage7 metrics, lossless bit-packed raw edge tensors."""
    rows=[]
    with torch.no_grad():
        for example in examples:
            public,graph=base.surface_input(example,language,renamed)
            gold=base.targets(graph,public,capacity,vocab,language=language)
            pred=model if isinstance(model,dict) else base.decode(model(public),public)
            rows.append(base.metrics(pred,gold))
            if raw_path is not None:
                with gzip.open(raw_path,'at',compresslevel=6) as stream:
                    stream.write(json.dumps(dict(context=context,language=language,renamed=renamed,public_text=public.text,
                      graph_sha256=graph.digest(),prediction=pack_graph(pred),target=pack_graph(gold),metrics=rows[-1]))+'\n')
    result={k:sum(r[k] for r in rows)/len(rows) for k in ('node_type_accuracy','identity_copy_accuracy','entity_equivalence','semantic_equivalence')}
    for key in ('node','typed_edge','ordered_edge'):
        counts={field:sum(r[key][field] for r in rows) for field in ('true_positive','predicted_count','gold_count')}
        result[key]={**counts,'f1':2*counts['true_positive']/max(1,counts['predicted_count']+counts['gold_count'])}
    return {**result,'examples':len(rows)}


def calibrate_edge_thresholds(model,examples,vocab,capacity,grid=(-6.,-4.,-2.,0.,2.,4.,6.,8.,12.)):
    """Per-relation decoder thresholds fitted exclusively on training graphs."""
    counts=torch.zeros(len(grid),len(ROLES),3,dtype=torch.long)
    with torch.no_grad():
        for example in examples:
            for language in ('english','spanish'):
                public,graph=base.surface_input(example,language)
                gold=base.targets(graph,public,capacity,vocab,language)['edges']
                out=model(public); present=out['presence'].gt(0)
                for index,threshold in enumerate(grid):
                    pred=out['edges'].gt(threshold)&present[:,None,None]&present[None,:,None]
                    counts[index,:,0]+=(pred&gold).sum((0,1))
                    counts[index,:,1]+=pred.sum((0,1))
                    counts[index,:,2]+=gold.sum((0,1))
    f1=2*counts[:,:,0]/(counts[:,:,1]+counts[:,:,2]).clamp_min(1)
    best=f1.argmax(0)
    thresholds=torch.tensor(grid)[best]
    return thresholds,dict(grid=list(grid),thresholds=thresholds.tolist(),fit_graphs=len(examples),fit_label_presentations=2*len(examples),
                          training_f1_by_relation=f1[best,torch.arange(len(ROLES))].tolist(),counts_by_grid_relation=counts.tolist(),count_order=['true_positive','predicted_count','gold_count'],relation_order=list(ROLES),
                          policy='fixed first training graphs, both languages; no evaluation labels; per-relation grid F1; lowest threshold wins ties')


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
    manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),corpus=corpus.audit,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  value_vocabulary=vocab,vocabulary_pool_unique_graphs=vocab_pool_count,vocabulary_sha256=hashlib.sha256(json.dumps(vocab).encode()).hexdigest(),pinned_generator=corpus[0].audit,runs=[],
                  split_policy='same heldout prefix, nested unique training prefixes',
                  training_digest=hashlib.sha256(''.join(base.semantic_key(corpus[i].privileged.graph) for i in train_indices).encode()).hexdigest(),
                  heldout_digest=hashlib.sha256(''.join(base.semantic_key(e.privileged.graph) for e in heldout).encode()).hexdigest(),
                  priors=['canonical compiler node order','privileged visible-copy and typed-edge labels','fixed ontology and node capacity','gold-edge training loss sampling only; no effect on hidden state','no_input control retains public pointer candidate count and token normalization'],
                  limitations=['exact decoder-slot equivalence is not general graph isomorphism','value ontology is fitted on declared shared training-only vocabulary pool, including unoptimized examples in smaller corpus arms','curriculum, width, edge sampling and actor differ from stage7; not a single-variable comparison'],
                  composition_allowed=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    @lru_cache(maxsize=config.get('cache_surfaces',20000))
    def training_item(index,language):
        public,graph=base.surface_input(corpus[eval_count+index],language)
        return public,base.targets(graph,public,capacity,vocab,language)
    records=[]
    def record(model,arm,seed,seen,exposure,tokens,start):
        row=dict(arm=arm,seed=seed,optimizer_updates=exposure//batch_size,optimizer_presentations=exposure,actual_unique_graphs_seen=len(seen),
                 available_unique_graphs=train_count,public_tokens_seen=tokens,feature_tokens_consumed=exposure if arm=='no_input' else tokens,elapsed_seconds=time.monotonic()-start,
                 renderer_exposure={'english':(exposure+1)//2,'spanish':exposure//2},evaluation={})
        if arm=='frequency': row.update(fit_label_presentations=2*train_count,actual_unique_graphs_seen=train_count,renderer_exposure={'english':train_count,'spanish':train_count})
        evaluation_start=time.monotonic()
        cache={}
        def cached_scores(public):
            if public.text not in cache: cache[public.text]={k:v.cpu() for k,v in model(public).items()}
            return cache[public.text]
        evaluation_model=model if isinstance(model,dict) else cached_scores
        for language in ('english','spanish','symbols'):
            row['evaluation'][language]=evaluate(evaluation_model,heldout,language,vocab,capacity,raw_path=output/'predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure))
        row['evaluation']['heldout_lexicon']=evaluate(evaluation_model,heldout,'english',vocab,capacity,True,raw_path=output/'predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure))
        acquisition=[corpus[eval_count+i] for i in range(min(train_count,config.get('acquisition_count',8)))]
        if not isinstance(model,dict) and config.get('calibrate_edges',True):
            thresholds,calibration=calibrate_edge_thresholds(evaluation_model,acquisition,vocab,capacity)
            row['edge_calibration']=calibration
            def calibrated(public):
                scores=evaluation_model(public)
                return {**scores,'edges':scores['edges']-thresholds}
            row['calibrated_evaluation']={lang:evaluate(calibrated,heldout,lang,vocab,capacity,raw_path=output/'calibrated-predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure)) for lang in ('english','spanish','symbols')}
            row['calibrated_evaluation']['heldout_lexicon']=evaluate(calibrated,heldout,'english',vocab,capacity,True,raw_path=output/'calibrated-predictions.jsonl.gz',context=dict(arm=arm,seed=seed,presentations=exposure))
        row['training_acquisition']={lang:evaluate(evaluation_model,acquisition,lang,vocab,capacity) for lang in ('english','spanish')}
        row['evaluation_seconds']=time.monotonic()-evaluation_start
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
                        workspace_rows=config.get('workspace_rows',8),microsteps=config.get('microsteps',2),no_input=arm=='no_input',autocast_dtype=config.get('autocast_dtype')).to(config.get('device','cpu'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',.0001))
            generator=torch.Generator().manual_seed(seed+1729)
            seen=set(); tokens=0; start=time.monotonic(); initial_hash=state_hash(model); training_seconds=0.
            cuda=model.initial.is_cuda
            if cuda: torch.cuda.reset_peak_memory_stats(model.initial.device)
            requested_checkpoints=set(config.get('eval_presentations',[0,presentations]))|{0,presentations}
            checkpoints={min(presentations,math.ceil(point/batch_size)*batch_size) for point in requested_checkpoints}
            for exposure in range(0,presentations+1,batch_size):
                if exposure in checkpoints: record(model,arm,seed,seen,exposure,tokens,start)
                if exposure==presentations: break
                if cuda: torch.cuda.synchronize(model.initial.device)
                train_start=time.monotonic()
                optimizer.zero_grad(); logs=[]; publics=[]; golds=[]; pair_queries=[]
                for offset in range(batch_size):
                    index,language=exposure_schedule(exposure+offset,train_count)
                    public,gold=training_item(index,language)
                    publics.append(public); golds.append(gold)
                    pair_queries.append(sampled_pairs(gold,generator,config.get('negative_pairs',128)))
                    seen.add(index); tokens+=len(base.tokens(public))
                outputs=model.forward_batch(publics,pairs=pair_queries)
                objectives=[]
                for offset,(prediction,gold,pairs) in enumerate(zip(outputs,golds,pair_queries)):
                    parts=sampled_losses(prediction,gold,pairs)
                    weights=curriculum_weights(exposure+offset,config.get('node_presentations',1000),config.get('identity_presentations',2000))
                    objectives.append(sum(weights[k]*v for k,v in parts.items()))
                    logs.append({k:v.detach() for k,v in parts.items()})
                loss=torch.stack(objectives).mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                if cuda: torch.cuda.synchronize(model.initial.device)
                step_seconds=time.monotonic()-train_start
                training_seconds+=step_seconds
                component_keys=list(logs[0])
                component_values=torch.stack([torch.stack([row[k] for k in component_keys]) for row in logs]).mean(0).cpu().tolist()
                with (output/'losses.jsonl').open('a') as f: f.write(json.dumps(dict(arm=arm,seed=seed,presentations=exposure+batch_size,training_step_seconds=step_seconds,parts=dict(zip(component_keys,component_values)),weights=weights))+'\n')
            path=output/f'{arm}-{seed}.pt'; torch.save(model.state_dict(),path)
            manifest['runs'].append(dict(arm=arm,seed=seed,parameters=sum(p.numel() for p in model.parameters()),width=model.width,autocast_dtype=model.autocast_dtype,parameter_dtype='float32',training_cache=training_item.cache_info()._asdict(),public_feature_cache=encode_text.cache_info()._asdict(),
                           initial_state_sha256=initial_hash,final_state_sha256=state_hash(model),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start,training_seconds=training_seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(model.initial.device) if cuda else 0))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
