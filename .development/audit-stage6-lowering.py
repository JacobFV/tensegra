#!/usr/bin/env python3
"""Artifact-only semantic boundary audit; no Torch import or model execution.

Pass complete shard directories (manifest.json + metrics.jsonl[.gz]). Source is
loaded from the declared git commit, not today's checkout. Refuses missing cells.
"""
import argparse, ast, collections, dataclasses, gzip, hashlib, itertools, json
from pathlib import Path
import subprocess, sys, types


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def frozen_api(commit):
    package = types.ModuleType('topoformer'); package.__path__ = []
    sys.modules['topoformer'] = package
    hashes = {}
    def source(name):
        data = subprocess.check_output(['git','show',f'{commit}:src/topoformer/{name}.py'])
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data.decode()
    for name in ('semantic_graph','thinking_runtime','thinking_tasks'):
        module = types.ModuleType('topoformer.'+name); sys.modules[module.__name__] = module
        exec(compile(source(name),name+'.py','exec'),module.__dict__)
    tree = ast.parse(source('thinking_study'))
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('evaluation_episode','evaluation_conditions')]
    namespace = {'replace':dataclasses.replace,'__name__':'topoformer.thinking_study','__package__':'topoformer'}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'frozen-evaluation-api','exec'),namespace)
    return namespace,hashes


def read_rows(directory):
    path = directory/'metrics.jsonl'
    if not path.exists(): path = directory/'metrics.jsonl.gz'
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as stream:
        for line in stream:
            yield json.loads(line)


def exact(primitive, values):
    if primitive=='add': return values[0]+values[1]
    if primitive=='sub': return values[0]-values[1]
    if primitive=='mul': return values[0]*values[1]
    if primitive=='neg': return -values[0]
    if primitive=='compare': return values[0]<values[1]
    raise ValueError(primitive)


def fraction(n,d):
    return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--source',default='c258785'); parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); api,source_hashes=frozen_api(args.source)
    counters=collections.defaultdict(collections.Counter); observed=set(); expected=set(); train=set(); expected_train=set(); input_checks=0; failures=[]; manifests=[]
    cache={}; training_hash_cache={}
    for directory in args.directories:
        manifest=json.loads((directory/'manifest.json').read_text()); manifests.append(manifest)
        cfg=types.SimpleNamespace(**manifest['config'])
        for name,h in source_hashes.items():
            assert manifest['sources']['src/topoformer/'+name+'.py']==h, ('source mismatch',name)
        for seed,variant,step in itertools.product(cfg.seeds,cfg.variants,cfg.checkpoints):
            for condition in api['evaluation_conditions'](cfg,step):
                expected.add((seed,variant,step,condition['condition'],condition['depth'],None))
                if variant=='local' and step==cfg.steps and condition['condition']=='depth' and condition['depth'] in (min(cfg.eval_depths),max(cfg.eval_depths)):
                    for intervention in ('oracle_trace','oracle_minimal','event_drop','event_shuffle','event_wrong_value','runtime_off','graph_permuted','graph_drop50','readiness_0.5','readiness_0.95'):
                        kind=intervention if intervention.startswith('oracle_') else 'frozen_intervention'
                        expected.add((seed,variant,step,kind,condition['depth'],intervention))
        expected_train.update(itertools.product(cfg.seeds,cfg.variants,range(1,cfg.steps+1)))
        for row in read_rows(directory):
            if row['kind']=='training':
                key=(row['seed'],row['variant'],row['step']); assert key not in train,('duplicate training',key); train.add(key)
                tk=(row['seed'],row['step']); step=row['step']-1
                if tk not in training_hash_cache:
                    generate=sys.modules['topoformer.thinking_tasks'].generate_episode
                    batch=[generate(seed=row['seed']*100000+step*cfg.batch_size+i,depth=1+step%cfg.train_depth,distractors=cfg.distractors,context=i%2,operator_composition='train') for i in range(cfg.batch_size)]
                    training_hash_cache[tk]=digest([dataclasses.asdict(e) for e in batch])
                assert training_hash_cache[tk]==row['data_hash'],('training data hash',key)
                continue
            assert row['kind']=='evaluation'
            key=(row['seed'],row['variant'],row['step'],row['condition'],row['depth'],row.get('intervention'))
            assert key not in observed,('duplicate cell',key); observed.add(key)
            conditions=api['evaluation_conditions'](cfg,row['step'])
            condition=next((c for c in conditions if c['condition']==row['condition'] and c['depth']==row['depth']),dict(depth=row['depth'],condition='depth',kwargs={}))
            assert len(row['examples'])==cfg.eval_examples
            for index,example in enumerate(row['examples']):
                ck=(row['seed'],row['depth'],condition['condition'],index)
                if ck not in cache: cache[ck]=api['evaluation_episode'](cfg,row['seed'],index,condition)
                episode=cache[ck]
                assert digest(dataclasses.asdict(episode.public))==example['input_hash'],('input hash',key,index)
                assert digest(dataclasses.asdict(episode))==example['data_hash'],('data hash',key,index)
                input_checks+=1
                if row['step']!=cfg.steps: continue
                group='|'.join(map(str,(row['variant'],row['condition'],row['depth'],row.get('intervention') or 'none')))
                counts=counters[group]; counts['examples']+=1
                gold={c.id:c for batch in episode.gold.trace for c in batch}
                dead=set(episode.public.operation_ids)-gold.keys()
                memory={v.id:(v.value,v.type) for v in episode.public.initial_values}
                executed=[]; bad_execution=[]; gold_values=dict(episode.gold.expected_values); all_values_correct=True
                for entry in example['trace']:
                    snapshot=dict(memory); updates={}
                    for proposal in entry['proposals']:
                        counts['proposals']+=1
                        identity=proposal['id']; target=gold.get(identity)
                        counts['proposal_null_argument']+=any(a=='__null__' for a in proposal['arguments'])
                        counts['proposal_unavailable_argument']+=any(a not in snapshot for a in proposal['arguments'])
                        if identity in dead: counts['proposal_dead_identity']+=1
                        elif target is None: counts['proposal_unrecognized_identity']+=1
                        else:
                            counts['proposal_recognized_identity']+=1
                            op=proposal['primitive']==target.primitive; binding=tuple(proposal['arguments'])==target.arguments
                            counts['proposal_primitive_correct']+=op; counts['proposal_binding_correct']+=binding; counts['proposal_full_correct']+=op and binding
                    for event in entry['events']:
                        status=event['status']; counts['status_'+status]+=1
                        if status!='executed': continue
                        counts['executions']+=1
                        expected_value=gold_values.get(event['register_id'])
                        all_values_correct = all_values_correct and event['register_id'] in gold_values and type(event['value'])==type(expected_value) and event['value']==expected_value
                        target=gold.get(event['candidate_id']); recognized=target is not None
                        counts['execution_recognized_identity']+=recognized
                        if recognized:
                            counts['execution_binding_correct']+=tuple(event['arguments'])==target.arguments
                            counts['execution_full_correct']+=event['primitive']==target.primitive and tuple(event['arguments'])==target.arguments
                        arity=1 if event['primitive']=='neg' else 2
                        valid=event['primitive'] in ('add','sub','mul','neg','compare') and len(event['arguments'])==arity and all(a in snapshot and snapshot[a][1] in ('integer','float') for a in event['arguments'])
                        counts['execution_valid_actual_arguments']+=valid
                        if valid:
                            actual=exact(event['primitive'],[snapshot[a][0] for a in event['arguments']])
                            correct=type(actual)==type(event['value']) and actual==event['value']
                            counts['execution_actual_math_correct']+=correct
                            if row.get('execution_backend')=='exact': assert correct,('incorrect exact backend',key,index,event)
                        else: bad_execution.append(event)
                        assert event['register_id'] not in memory and event['register_id'] not in updates
                        updates[event['register_id']]=(event['value'],event['type'])
                        executed.append((event['candidate_id'],event['primitive'],tuple(event['arguments'])))
                    memory.update(updates)
                assert not bad_execution,('invalid committed arguments',key,index)
                goldset={(c.id,c.primitive,c.arguments) for c in gold.values()}
                complete=set(executed)==goldset
                assert complete==example['transition_set_exact']
                counts['complete_correct_transition_set']+=complete
                counts['task_given_complete_numerator']+=complete and example['task_correct']
                counts['complete_correct_trajectory_values']+=complete and all_values_correct
                counts['task_given_complete_values_numerator']+=complete and all_values_correct and example['task_correct']
                if not example['task_correct'] and len(failures)<6 and row.get('intervention') is None and row['variant'] not in {f['variant'] for f in failures}:
                    failures.append(dict(seed=row['seed'],variant=row['variant'],condition=row['condition'],depth=row['depth'],index=index,public=dataclasses.asdict(episode.public),gold_trace=dataclasses.asdict(episode.gold)['trace'],observed=example))
    assert observed==expected,('incomplete evaluation grid',len(expected-observed),len(observed-expected))
    assert train==expected_train,('incomplete training',len(expected_train-train),len(train-expected_train))
    reference={k:v for k,v in manifests[0]['config'].items() if k!='seeds'}
    assert all({k:v for k,v in m['config'].items() if k!='seeds'}==reference for m in manifests)
    results={}
    for key,c in sorted(counters.items()):
        results[key]={'counts':dict(c),'metrics':{
            'recognized_task_identity':fraction(c['proposal_recognized_identity'],c['proposals']),
            'primitive_given_recognized':fraction(c['proposal_primitive_correct'],c['proposal_recognized_identity']),
            'ordered_binding_given_recognized':fraction(c['proposal_binding_correct'],c['proposal_recognized_identity']),
            'full_proposal_correct':fraction(c['proposal_full_correct'],c['proposals']),
            'execution_binding_given_recognized':fraction(c['execution_binding_correct'],c['execution_recognized_identity']),
            'actual_math_given_valid_executed_arguments':fraction(c['execution_actual_math_correct'],c['execution_valid_actual_arguments']),
            'task_given_complete_correct_transition_set':fraction(c['task_given_complete_numerator'],c['complete_correct_transition_set']),
            'task_given_complete_correct_trajectory_values':fraction(c['task_given_complete_values_numerator'],c['complete_correct_trajectory_values'])}}
    output=dict(source=args.source,source_hashes=source_hashes,evaluation_cells=len(observed),training_rows=len(train),example_hash_checks=input_checks,definitions={
        'recognized':'Identity belongs to a gold task transition; supplied dead destinations are separately counted.',
        'proposals':'All logged proposals, including those on a final step where no execution is attempted; thresholds do not filter this denominator.',
        'binding':'Exact ordered argument IDs against the recognized destination; independent of primitive correctness.',
        'actual_math':'Replayed from public literals plus actual executed event values, using snapshot isolation. Refusals and duplicates are status counts, never silently treated as binding errors.',
        'complete':'Executed transition SET equals gold task set, independent of order and learned numeric value correctness. Task readout correctness conditional on this set is reported separately. The trajectory-values conditional additionally requires every executed value and type to equal the gold expected value.',
        'failure_examples':'First final failing example from up to six distinct nonintervention arms; illustrative, not random sampling.'},results=results,failure_examples=failures)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status='passed',evaluation_cells=len(observed),training_rows=len(train),example_hash_checks=input_checks,groups=len(results))))

if __name__=='__main__': main()
