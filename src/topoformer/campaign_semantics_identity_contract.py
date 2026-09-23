"""Scoped programmed refers-to relation from learned public copy identities."""
import argparse,gzip,json,time
from pathlib import Path
import torch
from .campaign_semantics import digest,write_gzip
from .campaign_semantics_data import load_cache
from .semantic_curriculum import unpack_graph,pack_graph
from .semantic_scaling import metrics,tokens,identifier_forms
from .thinking_language import KINDS,ROLES,ActorInput


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
    result=dict(config=config,audit=audit,archive_sha256=digest(config['archive']),source_sha256=digest(__file__),rows=rows,cpu_seconds=time.monotonic()-start)
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
