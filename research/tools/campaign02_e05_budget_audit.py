"""Frozen public-teacher counterfactuals on archived actor states, no inference."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types


def run(root, source):
    start=time.process_time()
    package=types.ModuleType('_budget_audit');package.__path__=[];sys.modules[package.__name__]=package
    hashes={}
    for name in ('campaign02_world','campaign02_references'):
        code=subprocess.check_output(['git','show',source+':src/topoformer/'+name+'.py'])
        hashes[name]=hashlib.sha256(code).hexdigest()
        module=types.ModuleType(package.__name__+'.'+name);sys.modules[module.__name__]=module
        exec(compile(code,source+':'+name,'exec'),module.__dict__)
    world=sys.modules['_budget_audit.campaign02_world'];refs=sys.modules['_budget_audit.campaign02_references']
    result={}
    for family in ('lightweight','recurrent'):
        path=Path(root)/('e05-acquire-'+family)/'development/round-0-slot-5-attempt-5.jsonl.gz'
        counters={name:Counter() for name in ('actor_kind','teacher_kind','actor_budget','teacher_budget',
                    'call_confusion','escalation_confusion','feature_differences','last_budget_feature')}
        totals=Counter();examples=[];bad_seeds=set()
        for line in gzip.open(path,'rt'):
            row=json.loads(line);totals['episodes']+=1;totals['success']+=row['outcome']['verified_success']
            seen={}
            for step in row['trace']:
                if time.process_time()-start>28:raise RuntimeError('CPU diagnostic bound exceeded')
                o=world.Observation(**step['observation']);catalog=world.action_catalog(o)
                teacher=refs.make_reference('cheap_first_fallback_v2').choose(o,catalog)
                action=world.Action(**step['action']);assert action in catalog
                totals['frames']+=1;counters['actor_kind'][action.kind]+=1;counters['teacher_kind'][teacher.kind]+=1
                if teacher.kind=='call':counters['teacher_budget'][teacher.arguments['budget']]+=1
                if action.kind!='call':continue
                budget=action.arguments['budget'];counters['actor_budget'][budget]+=1
                draft=o.problems[action.arguments['problem']]
                identity=(action.arguments['problem'],json.dumps(draft,sort_keys=True))
                previous=seen.get(identity)
                if previous is not None:
                    totals['repeat_exact_instance_calls']+=1
                    if previous==budget:totals['repeat_exact_instance_same_budget']+=1
                seen[identity]=budget
                records=[r for r in o.records if r.get('problem')==action.arguments['problem'] and r.get('problem_snapshot')==draft]
                last=records[-1] if records else None
                expected=teacher.kind+(':'+str(teacher.arguments['budget']) if teacher.kind=='call' else '')
                counters['call_confusion'][str(budget)+' -> '+expected]+=1
                af=world.encode_action(o,action)
                counters['last_budget_feature'][af[52]]+=1
                if last and last.get('status')=='timeout':
                    totals['calls_after_exact_instance_timeout']+=1
                    if budget<=last['work_units']:totals['calls_not_exceeding_prior_timeout_work']+=1;bad_seeds.add(row['seed'])
                    if teacher.kind=='call' and teacher.arguments['problem']==action.arguments['problem'] and teacher.arguments['budget']>last['work_units']:
                        totals['teacher_escalation_available']+=1
                        counters['escalation_confusion'][str(budget)+' -> '+str(teacher.arguments['budget'])]+=1
                        tf=world.encode_action(o,teacher)
                        diff=tuple(i for i,(a,b) in enumerate(zip(af,tf)) if a!=b)
                        counters['feature_differences'][str(diff)]+=1
                        if not diff:totals['identical_full_candidate_features_at_escalation']+=1
                        if len(examples)<4:
                            examples.append(dict(seed=row['seed'],step=step['step'],primitive=draft['primitive'],
                                selected_budget=budget,teacher_budget=teacher.arguments['budget'],remaining_work=o.remaining_work,
                                last_work=last['work_units'],last_budget=last['budget'],candidate_budget_feature=af[20],
                                last_budget_feature=af[52],teacher_budget_feature=tf[20],different_indices=diff))
        result[family]=dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),totals=dict(totals),
                           distinct_episodes_repeating_insufficient_budget=len(bad_seeds),
                           frequencies={k:dict(v) for k,v in counters.items()},examples=examples)
    return dict(source=source,source_hashes=hashes,arms=result,cpu_seconds=time.process_time()-start,
                scope='teacher recomputation on actor-visited public states; not teacher training-frequency audit or new policy rollout',
                limitations=['no neural logits/gradients/probe evaluation','normalization/optimization mechanisms remain hypotheses',
                             'repeated actor states do not constitute independent statistical support'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.root,a.source)
    Path(a.output).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:{'totals':v['totals'],'episodes_repeating':v['distinct_episodes_repeating_insufficient_budget'],'escalation':v['frequencies']['escalation_confusion'],'features':v['frequencies']['feature_differences']} for k,v in result['arms'].items()},indent=2));print('CPU',result['cpu_seconds'])
