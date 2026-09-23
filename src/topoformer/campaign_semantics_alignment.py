"""Frozen public-only copy replay; occurrence targets enter scoring only."""
import argparse,gzip,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from .campaign_semantics import Dataset,digest,write_gzip
from .campaign_semantics_data import load_cache
from .campaign_semantics_occurrence import occurrence_targets
from .semantic_curriculum import SemanticCurriculumActor,pack_graph
from .semantic_scaling import decode,tokens
from .thinking_language import KINDS,ROLES


def copy_statistics(logits,first,occurrence,token_ids):
    probabilities=logits.float().softmax(-1);argmax=int(logits.argmax())
    identity=token_ids.eq(token_ids[first])
    return dict(raw_argmax=argmax,first_target=int(first),occurrence_target=int(occurrence),first_correct=argmax==first,occurrence_correct=argmax==occurrence,canonical_correct=bool(identity[argmax]),first_mass=float(probabilities[first]),occurrence_mass=float(probabilities[occurrence]),identity_mass=float(probabilities[identity].sum()),entropy=float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()))


def run(config):
    torch.set_num_threads(2);tick=time.monotonic();out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    data=Path(config['data_dir']);audit=json.loads((data/'audit.json').read_text());vocab=audit['value_vocabulary'];records=[];arrays=[];offsets=[0];models=[]
    for arm in config['arms']:
        root=Path(arm['root']);manifest=json.load(gzip.open(root/'manifest.json.gz','rt'));checkpoint=root/'model-u8192.pt';archive_path=root/'evaluation-u8192.json.gz';archive=json.load(gzip.open(archive_path,'rt'))
        if digest(checkpoint)!=arm['checkpoint_sha256']:raise ValueError('frozen checkpoint mismatch')
        if digest(data/'audit.json')!=manifest['data_audit_sha256']:raise ValueError('dataset audit changed')
        for name,sha in manifest['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('historical source changed: '+name)
        model=SemanticCurriculumActor(value_count=len(vocab),width=1024,capacity=128,workspace_rows=8,microsteps=2,autocast_dtype='bfloat16').to('cuda')
        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model']);model.eval();thresholds=torch.tensor(archive['thresholds'])
        for split,key,limit in [('train','train_rows',128),('development','rows',512)]:
            rows=load_cache(data/(split+'.jsonl.gz'))[:limit];dataset=Dataset(rows,vocab)
            for index,row in enumerate(rows):
                public,gold,_=dataset[index]
                with torch.no_grad():output={k:v.cpu() for k,v in model(public).items()}
                predicted=decode(output,public);old=archive[key][index]
                if row['seed']!=old['seed'] or pack_graph(predicted)!=old['raw']:raise ValueError('frozen raw prediction replay mismatch')
                aligned=occurrence_targets(row,gold);tok=tokens(public);ids=torch.tensor([tok.index(t) for t in tok]);node_rows=[]
                edges=output['edges'].gt(thresholds);active=predicted['presence'][:,None,None]&predicted['presence'][None,:,None];edges=edges&active
                for node in gold['copy'].ge(0).nonzero().flatten().tolist():
                    stats=copy_statistics(output['copy'][node],int(gold['copy'][node]),int(aligned['copy'][node]),ids)
                    required=(gold['edges'][:,node,:]&gold['slots'][:,node,None].ge(0)).nonzero()
                    correct=sum(bool(edges[i,node,r] and predicted['slots'][i,node]==gold['slots'][i,node]) for i,r in required.tolist())
                    node_rows.append(dict(node=node,kind=row['nodes'][node][0],**stats,required_incoming_ordered_edges=len(required),correct_incoming_ordered_edges=correct))
                flat=output['copy'].float().numpy().ravel();arrays.append(flat);offsets.append(offsets[-1]+len(flat))
                records.append(dict(arm=arm['name'],split=split,seed=row['seed'],semantic_sha256=row['semantic_sha256'],token_identity=ids.tolist(),copy_shape=list(output['copy'].shape),logit_start=offsets[-2],logit_stop=offsets[-1],raw_replay_exact=True,nodes=node_rows))
        models.append(dict(arm=arm['name'],checkpoint_sha256=digest(checkpoint),archive_sha256=digest(archive_path),manifest_sha256=digest(root/'manifest.json.gz'),historical_source_sha256=manifest['source_sha256']))
        del model
    np.savez_compressed(out/'copy-logits.npz',logits=np.concatenate(arrays),offsets=np.array(offsets,dtype=np.int64))
    write_gzip(out/'predictions.json.gz',dict(config=config,models=models,records=records,logits_sha256=digest(out/'copy-logits.npz'),source_sha256=digest(__file__),config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),data_sha256={s:digest(data/(s+'.jsonl.gz')) for s in ('train','development')}))
    summary={}
    for arm in config['arms']:
        summary[arm['name']]={}
        for split in ('train','development'):
            selected=[r for r in records if r['arm']==arm['name'] and r['split']==split];groups={}
            for kind in ('ident','entity'):
                nodes=[n for r in selected for n in r['nodes'] if n['kind']==kind]
                groups[kind]=dict(nodes=len(nodes),**{k:sum(n[k] for n in nodes)/len(nodes) for k in ('first_correct','occurrence_correct','canonical_correct','first_mass','occurrence_mass','identity_mass','entropy')})
                groups[kind]['ordered_edge_association']={str(value):dict(nodes=sum(n['occurrence_correct']==value for n in nodes),correct=sum(n['correct_incoming_ordered_edges'] for n in nodes if n['occurrence_correct']==value),required=sum(n['required_incoming_ordered_edges'] for n in nodes if n['occurrence_correct']==value)) for value in (False,True)}
            summary[arm['name']][split]=dict(graphs=len(selected),groups=groups)
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,process_seconds=time.monotonic()-tick,peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen development diagnostic, not new acquisition or causal proof'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();run(json.loads(Path(a.config).read_text()))
