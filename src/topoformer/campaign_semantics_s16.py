"""S16 frozen six-model engineering normalization diagnostic; no optimization."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from .campaign_semantics import digest,decode,write_gzip
from .campaign_semantics_data import load_cache,target
from .campaign_semantics_identity_contract import canonicalize_public_identifiers,restore_public_copy_tokens
from .campaign_semantics_motif_inference import compact,expand
from .campaign_semantics_s16_freeze import verify
from .semantic_curriculum import SemanticCurriculumActor,encode_text
from .thinking_language import ActorInput
from . import semantic_scaling as base


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        cpu=value.detach().cpu().contiguous()
        h.update(json.dumps([name,str(cpu.dtype),list(cpu.shape)]).encode())
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def predict_public(model,normalization,thresholds):
    """Forward gets normalized ActorInput only; no row, gold or semantic metadata."""
    output=model(normalization.public)
    raw,cal=decode(output,normalization.public,thresholds)
    copied=restore_public_copy_tokens(normalization,raw['copy'].tolist())
    return raw,cal,copied


def preflight(config):
    """Validate all public inputs and all six inherited endpoints before a model."""
    verify(config,Path(__file__).parent)
    for key in ('cache','renamed_cache','data_audit','normalizer_audit'):
        if digest(config[key])!=config[key+'_sha256']:raise ValueError('artifact changed: '+key)
    proof=json.loads(Path(config['normalizer_audit']).read_text())
    if proof['status']!='pass' or proof['accepted']!=1024 or proof['rejected'] or proof['lexical_collisions']:
        raise ValueError('normalizer public proof incomplete')
    if proof['identity_contract_sha256']!=digest(Path(__file__).with_name('campaign_semantics_identity_contract.py')):
        raise ValueError('normalizer differs from audited source')
    rows=load_cache(config['cache']);renamed=load_cache(config['renamed_cache'])
    audit=json.loads(Path(config['data_audit']).read_text())
    if len(rows)!=1024 or len(renamed)!=1024:raise ValueError('full public population required')
    normalized=[]
    for index,(row,other) in enumerate(zip(rows,renamed)):
        a=canonicalize_public_identifiers(ActorInput(row['text'],()))
        b=canonicalize_public_identifiers(ActorInput(other['text'],()))
        af,al=encode_text(a.public.text);bf,bl=encode_text(b.public.text)
        if al!=bl or not torch.equal(af,bf):raise ValueError('normalized full inputs differ')
        if hashlib.sha256(af.numpy().tobytes()).hexdigest()!=proof['rows'][index]['full_fp32_feature_sha256']:
            raise ValueError('normalized feature proof mismatch')
        normalized.append(a)
    inherited=[]
    for entry,binding in zip(config['runs'],config['primary_matrix']):
        root=Path(entry['directory']);manifest_path=root/'manifest.json.gz'
        manifest=json.load(gzip.open(manifest_path,'rt'))
        if digest(manifest_path)!=binding['manifest_sha256']:raise ValueError('frozen manifest changed')
        if not json.loads((root/'completion.json').read_text())['success']:raise ValueError('unfinished S12 endpoint')
        if manifest['config']['seed']!=entry['seed'] or manifest['config']['learning_rate']!={'constant':1e-4,'decay':1e-5}[entry['arm']]:
            raise ValueError('wrong inherited endpoint')
        if digest(config['data_audit'])!=manifest['data_audit_sha256']:raise ValueError('inherited audit changed')
        for name,sha in manifest['source_sha256'].items():
            if digest(Path(__file__).with_name(name))!=sha:raise ValueError('inherited model source changed: '+name)
        end=next(x for x in manifest['curves'] if x['update']==24576)
        checkpoint=root/'model-u24576.pt';primary_path=root/end['artifact']
        if digest(checkpoint)!=binding['checkpoint_sha256'] or binding['checkpoint_sha256']!=end['checkpoint_sha256']:
            raise ValueError('frozen checkpoint changed')
        if digest(primary_path)!=end['sha256']:raise ValueError('primary prediction reference changed')
        primary=json.load(gzip.open(primary_path,'rt'))
        if primary['update']!=24576 or len(primary['rows'])!=1024 or [r['semantic_sha256'] for r in primary['rows']]!=[r['semantic_sha256'] for r in rows]:
            raise ValueError('unpaired original prediction references')
        inherited.append((entry,str(checkpoint),primary,str(primary_path),end['sha256']))
    return rows,audit,normalized,inherited


def run(config):
    start=time.monotonic();torch.set_num_threads(2)
    rows,audit,normalized,inherited=preflight(config)
    preflight_seconds=time.monotonic()-start
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=False)
    profile=config['job']=='profile';count=config['examples'];artifacts=[]
    torch.cuda.reset_peak_memory_stats()
    for entry,checkpoint,primary,primary_path,primary_sha in inherited:
        tick=time.monotonic()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
        expected=tensor_hash(state)
        checkpoint_read_hash_seconds=time.monotonic()-tick
        setup_tick=time.monotonic()
        model=SemanticCurriculumActor(value_count=len(audit['value_vocabulary']),width=1024,capacity=128,
            workspace_rows=8,microsteps=2,autocast_dtype='bfloat16')
        model.load_state_dict(state);del state
        model.to(config['device']).eval().requires_grad_(False)
        before=tensor_hash(model.state_dict())
        if before!=expected:raise ValueError('loaded tensor bytes differ from checkpoint')
        thresholds=torch.tensor(primary['thresholds']);threshold_before=tensor_hash({'thresholds':thresholds})
        model_setup_hash_seconds=time.monotonic()-setup_tick
        forward_decode_seconds=packing_hash_seconds=scoring_seconds=0.
        predictions=[];output_hash=hashlib.sha256()
        with torch.no_grad():
            for index in range(count):
                stage_tick=time.monotonic()
                raw,cal,copied=predict_public(model,normalized[index],thresholds)
                # decode copies every output to CPU, synchronizing the CUDA work.
                forward_decode_seconds+=time.monotonic()-stage_tick
                stage_tick=time.monotonic()
                raw_pack=compact(raw);cal_pack=compact(cal)
                raw_pairs={(i,j) for i,j,s in raw_pack['slots']}
                packed=dict(raw=raw_pack,calibrated_edges=cal_pack['edges'],
                    calibrated_extra_slots=[v for v in cal_pack['slots'] if tuple(v[:2]) not in raw_pairs],
                    copied_original_public_tokens=copied)
                output_hash.update(json.dumps(packed,sort_keys=True).encode())
                packing_hash_seconds+=time.monotonic()-stage_tick
                if not profile:
                    stage_tick=time.monotonic()
                    # Scoring occurs only after public-only forward has completed.
                    row=rows[index];gold=target(row,audit['value_vocabulary'])
                    metrics={policy:base.metrics(graph,gold) for policy,graph in [('raw',raw),('calibrated',cal)]}
                    if base.metrics(expand(raw_pack),gold)!=metrics['raw'] or base.metrics(expand(cal_pack),gold)!=metrics['calibrated']:
                        raise ValueError('compact graph changed metrics')
                    original={policy:primary['rows'][index][policy+'_metrics'] for policy in ('raw','calibrated')}
                    predictions.append(dict(index=index,semantic_sha256=row['semantic_sha256'],**packed,
                        metrics=metrics,original_metrics=original,
                        transitions={p:dict(repair=bool(metrics[p]['semantic_equivalence'] and not original[p]['semantic_equivalence']),
                            regression=bool(original[p]['semantic_equivalence'] and not metrics[p]['semantic_equivalence'])) for p in metrics}))
                    scoring_seconds+=time.monotonic()-stage_tick
        stage_tick=time.monotonic()
        torch.cuda.synchronize();after=tensor_hash(model.state_dict())
        if after!=before or tensor_hash({'thresholds':thresholds})!=threshold_before:raise RuntimeError('frozen tensor mutation')
        post_inference_tensor_hash_seconds=time.monotonic()-stage_tick
        stage_tick=time.monotonic()
        record=dict(**entry,examples=count,checkpoint_sha256=digest(checkpoint),primary_sha256=primary_sha,
            thresholds=primary['thresholds'],tensor_before_sha256=before,tensor_after_sha256=after,
            threshold_tensor_sha256=threshold_before,output_sha256=output_hash.hexdigest(),seconds=time.monotonic()-tick)
        checkpoint_rehash_seconds=time.monotonic()-stage_tick
        stage_tick=time.monotonic()
        if not profile:
            result=dict(endpoint=record,rows=predictions,original_reference=primary_path,
                normalized_renamed='same full actor inputs by public proof; not independently forwarded')
            path=out/f"normalized-{entry['arm']}-{entry['seed']}.json.gz";write_gzip(path,result)
            record.update(artifact=path.name,sha256=digest(path),summary={p:dict(
                exact=sum(r['metrics'][p]['semantic_equivalence'] for r in predictions),
                repairs=sum(r['transitions'][p]['repair'] for r in predictions),
                regressions=sum(r['transitions'][p]['regression'] for r in predictions),
                copy=sum(r['metrics'][p]['identity_copy_accuracy'] for r in predictions)/count) for p in ('raw','calibrated')})
        export_seconds=time.monotonic()-stage_tick
        record['timing']=dict(checkpoint_read_hash_seconds=checkpoint_read_hash_seconds,
            model_setup_hash_seconds=model_setup_hash_seconds,forward_decode_seconds=forward_decode_seconds,
            packing_hash_seconds=packing_hash_seconds,scoring_seconds=scoring_seconds,
            post_inference_tensor_hash_seconds=post_inference_tensor_hash_seconds,
            checkpoint_rehash_seconds=checkpoint_rehash_seconds,export_seconds=export_seconds)
        record['seconds']=time.monotonic()-tick
        artifacts.append(record);del model
        print(json.dumps(dict(event='endpoint',seed=entry['seed'],arm=entry['arm'],examples=count,seconds=record['seconds'])),flush=True)
    manifest=dict(config=config,artifacts=artifacts,process_seconds=time.monotonic()-start,preflight_seconds=preflight_seconds,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='supplied public renderer/case/canonical vocabulary prior; no learned rename-invariance claim; profile outcomes hashed only')
    stage_tick=time.monotonic();write_gzip(out/'manifest.json.gz',manifest)
    (out/'completion.json').write_text(json.dumps(dict(success=True,process_seconds=time.monotonic()-start,
        manifest_export_seconds=time.monotonic()-stage_tick))+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args()
    run(json.loads(Path(a.config).read_text()))
