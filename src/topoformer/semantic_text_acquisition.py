"""Stage 11 small public-text graph acquisition; privileged graphs are targets."""
from __future__ import annotations
import argparse,gzip,hashlib,json,resource,time
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from . import semantic_scaling as base
from .semantic_curriculum import (SemanticCurriculumActor,sampled_pairs,sampled_losses,
    curriculum_weights,pack_graph,encode_text)
from .semantic_contracts import slot_objective,single_slot_labels,train_relation_thresholds
from .thinking_language import ActorInput,lexical_bits,state_hash,ROLES


def fixed_corpus(count=8,data_seed=11000000,lesson='unification'):
    examples=[];seen=set();attempt=0
    while len(examples)<count:
        if attempt>=10000:raise ValueError('fixed corpus budget exhausted')
        example=base.build_tcn_example(lesson,data_seed+attempt,difficulty=.5);attempt+=1
        key=base.semantic_key(example.privileged.graph)
        if key not in seen:examples.append(example);seen.add(key)
    return examples


def prepare(examples,capacity=128,language='english'):
    vocab=base.value_vocabulary(examples);items=[];texts={};hashes={};maximum=Counter()
    for example in examples:
        graph=example.privileged.graph;single_slot_labels(graph)
        public,_=base.surface_input(example,language);tokens=base.tokens(public)
        gold=base.targets(graph,public,capacity,vocab,language)
        signature=base.semantic_key(graph)
        if public.text in texts and texts[public.text]!=signature:raise ValueError('incompatible identical public text')
        texts[public.text]=signature
        for token in tokens:
            bits=tuple(lexical_bits(token))
            if bits in hashes and hashes[bits]!=token:raise ValueError('public lexical hash collision')
            hashes[bits]=token
        encoded,length=encode_text(public.text)
        if encoded.shape[1]!=len(tokens) or length!=len(tokens):raise ValueError('public token truncation or alignment mismatch')
        if gold['value'].eq(0).any():raise ValueError('unknown categorical target')
        maximum['nodes']=max(maximum['nodes'],len(graph.nodes));maximum['tokens']=max(maximum['tokens'],len(tokens))
        maximum['slot']=max(maximum['slot'],max((e.slot for e in graph.edges if e.slot is not None),default=-1))
        items.append((public,gold,example))
    return items,vocab,dict(examples=len(examples),unique_semantics=len(texts),lesson_counts=dict(Counter(e.audit['lesson'] for e in examples)),maxima=dict(maximum),language=language,
        observable='Only raw public text and its token copy inventory enter actor; graph, active node count, labels and spans do not.',
        checks=['distinct alpha-normalized constructions','identical-text target consistency','lexical hash uniqueness in support','visible-copy target resolution','known value vocabulary','node/slot capacity','no token truncation','pair-slot collision rejection'],
        limitations=['finite fixed-set audit, not proof of language identifiability','compiler order and ontology supplied as target conventions','no heldout semantics or renderer evaluated'])


def corrected_losses(output,gold,pairs):
    parts=sampled_losses(output,gold,pairs)
    i,j=pairs.unbind(-1);truth=gold['edges'][i,j].any(-1).to(output['slots'].device)
    labels=(gold['slots'][i,j]+1).to(output['slots'].device)
    parts['slots']=slot_objective(output['slots'],labels,truth,edge_conditional=True)
    return parts


def evaluate(model,items):
    """Inference receives public text only. Gold selects calibration loss, never actor inputs."""
    model.eval();cache=[];scores=[];labels=[]
    with torch.no_grad():
        for public,gold,example in items:
            output=model(public);pred=base.decode(output,public)
            active=pred['presence'][:,None]&pred['presence'][None,:]
            scores.append(output['edges'][active]);labels.append(gold['edges'].to(active.device)[active])
            cache.append((public,gold,example,output,pred))
        joined=torch.cat(scores);truth=torch.cat(labels)
        if len(joined):thresholds,calibration=train_relation_thresholds(joined,truth)
        else:
            thresholds=torch.zeros(len(ROLES),device=next(model.parameters()).device)
            calibration=[dict(positive=0,negative=0,train_errors=0,threshold=0.,empty_support=True) for _ in ROLES]
        rows=[]
        for public,gold,example,output,pred in cache:
            pred={k:v.cpu() for k,v in pred.items()};cal={**pred,'edges':output['edges'].gt(thresholds).cpu()}
            raw_metrics=base.metrics(pred,gold);cal_metrics=base.metrics(cal,gold)
            rows.append(dict(graph_seed=example.audit['seed'],graph_sha256=example.privileged.graph.digest(),public_text=public.text,raw=pack_graph(pred),calibrated=pack_graph(cal),target=pack_graph(gold),raw_metrics=raw_metrics,calibrated_metrics=cal_metrics))
    model.train()
    return dict(rows=rows,calibration=calibration,calibration_policy='final/checkpoint TRAIN only; one threshold/relation minimizes entry errors over predicted-present pairs, lowest tie; no predicted pairs => threshold0 fallback',raw_exact=sum(r['raw_metrics']['semantic_equivalence'] for r in rows),calibrated_exact=sum(r['calibrated_metrics']['semantic_equivalence'] for r in rows))


def run(config):
    torch.set_num_threads(2);out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    examples=fixed_corpus(config['graphs'],config['data_seed'],config['lesson'])
    items,vocab,audit=prepare(examples,config['node_capacity'],config['language']);runs=[]
    files=[Path(__file__),Path(__file__).with_name('semantic_curriculum.py'),Path(__file__).with_name('semantic_scaling.py'),Path(__file__).with_name('semantic_contracts.py')]
    manifest=dict(config=config,config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},graph_audits=[e.audit for e in examples],public_observability=audit,value_vocabulary=vocab,runs=runs)
    # Same labeled fixed mixture, no text features; repeated labels cannot alter frequencies.
    class Corpus:
        def __getitem__(self,index):return examples[index]
    from .semantic_curriculum import frequency_fit
    frequency=frequency_fit(Corpus(),list(range(len(examples))),vocab,config['node_capacity'])
    manifest['frequency']=[dict(graph_seed=e.audit['seed'],metrics=base.metrics(frequency,gold)) for _,gold,e in items]
    for seed in config['seeds']:
        torch.manual_seed(seed)
        model=SemanticCurriculumActor(value_count=len(vocab),width=config['width'],capacity=config['node_capacity'],workspace_rows=8,microsteps=2,autocast_dtype=config.get('autocast_dtype')).to(config['device'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate']);generator=torch.Generator().manual_seed(seed+1729)
        start=time.monotonic();curves=[];losses=[];train_time=0.;initial=state_hash(model)
        if config['device']=='cuda':torch.cuda.reset_peak_memory_stats()
        for update in range(config['updates']+1):
            if update in config['checkpoints']:
                result=evaluate(model,items);curves.append(dict(update=update,**result))
                path=out/f'model-seed{seed}-u{update}.pt';torch.save(model.state_dict(),path)
                curves[-1]['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            if update==config['updates']:break
            if config['device']=='cuda':torch.cuda.synchronize()
            tick=time.monotonic();optimizer.zero_grad()
            queries=[sampled_pairs(gold,generator,config['negative_pairs']) for _,gold,_ in items]
            outputs=model.forward_batch([public for public,_,_ in items],pairs=queries)
            weights=curriculum_weights(update*len(items),config['node_presentations'],config['identity_presentations'])
            parts=[corrected_losses(output,gold,pairs) for output,(_,gold,_),pairs in zip(outputs,items,queries)]
            loss=torch.stack([sum(weights[k]*v for k,v in row.items()) for row in parts]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if config['device']=='cuda':torch.cuda.synchronize()
            train_time+=time.monotonic()-tick
            component=torch.stack([torch.stack([row[k].detach() for k in parts[0]]) for row in parts]).mean(0).cpu().tolist()
            losses.append(dict(update=update+1,parts=dict(zip(parts[0],component)),weights=weights))
        runs.append(dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),width=model.width,workspace_rows=8,context_tokens=[len(base.tokens(p)) for p,_,_ in items],optimizer_presentations=config['updates']*len(items),actual_unique_graphs=len(items),training_seconds=train_time,wall_seconds=time.monotonic()-start,peak_cuda_allocated=torch.cuda.max_memory_allocated() if config['device']=='cuda' else None,process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,initial_state_sha256=initial,final_state_sha256=state_hash(model),curves=curves,losses=losses))
        with gzip.GzipFile(filename=str(out/'results.json.gz'),mode='wb',mtime=0) as stream:stream.write(json.dumps(manifest,separators=(',',':')).encode())
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
