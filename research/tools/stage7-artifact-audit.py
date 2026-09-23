"""Independent, standard-library-only Stage 7 raw-count audit (no Torch)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def close(actual, expected, label):
    assert abs(actual-expected) < 1e-7, (label, actual, expected)


def score_rows(scores, truth, threshold):
    selected = [i for i,score in enumerate(scores) if score >= threshold]
    tp = sum(bool(truth[i]) for i in selected)
    return dict(count=len(scores),selected=len(selected),executable=sum(truth),true_positive=tp,
        precision=tp/len(selected) if selected else 0.,recall=tp/sum(truth) if sum(truth) else 0.,
        coverage=len(selected)/len(scores),brier=sum((s-float(y))**2 for s,y in zip(scores,truth))/len(scores))


def audit_ab(directory):
    directory=Path(directory); manifest=json.loads((directory/'manifest.json').read_text())
    config=manifest['config']; summary=json.loads((directory/'summary.json').read_text())
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==manifest['config_hash']
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest
    assert [r['seed'] for r in summary['seeds']]==config['seeds']
    results=[]; metric_checks=0
    for seed in config['seeds']:
        raw_path=directory/f'seed{seed}-raw.json'
        raw=json.loads(raw_path.read_text() if raw_path.exists() else gzip.decompress(raw_path.with_suffix('.json.gz').read_bytes())); seed_result=dict(seed=seed,A={},B={})
        assert raw['optimizer_examples']==config['steps']*config['batch_size']
        assert raw['train_cardinality']==config['train_size']
        expected={f'{s}_{p}' for s in ('iid','ood') for p in (('validation','test') if config.get('main_budget_frozen') else ('validation',))}
        assert set(raw['evaluations'])==expected
        for split,controls in raw['evaluations'].items():
            assert set(controls)=={'complete','missing','permuted','reverse'}
            for control,entry in controls.items():
                p=entry['raw']; n=len(p['primitive']); assert n==config['eval_size']
                op=[a==b for a,b in zip(p['primitive'],p['target_primitive'])]
                args=[[a==b for a,b in zip(pred,gold)] for pred,gold in zip(p['pointers'],p['target_pointers'])]
                binary=[i for i,o in enumerate(p['target_primitive']) if o!=3]
                noncomm=[i for i,o in enumerate(p['target_primitive']) if o in (1,4)]
                calc=dict(count=n,primitive=sum(op)/n,destination=sum(a[0] for a in args)/n,
                    operand1=sum(a[1] for a in args)/n,operand2=sum(args[i][2] for i in binary)/len(binary),
                    operand2_count=len(binary),noncommutative_count=len(noncomm),
                    noncommutative_order=sum(args[i][1] and args[i][2] for i in noncomm)/len(noncomm),
                    full=sum(o and a[0] and a[1] and (a[2] or gold==3) for o,a,gold in zip(op,args,p['target_primitive']))/n)
                for key,value in calc.items():close(value,entry['metrics'][key],(seed,split,control,key));metric_checks+=1
                if control=='complete':seed_result['A'][split]=calc
        strict=all(seed_result['A'][split]['full']>.98 and seed_result['A'][split]['noncommutative_order']>.99 and all(seed_result['A'][split][k]>.99 for k in ('primitive','destination','operand1','operand2')) for split in ('iid_validation','ood_validation'))
        assert strict==raw.get('restricted_keyed_gate_a_validation',raw.get('typed_instruction_gate_a_validation'))
        b=raw['readiness']
        for split in ('validation','test') if config.get('main_budget_frozen') else ('validation',):
            rows=b[split]['records']; truth=[r['correct'] for r in rows];scores=[r['calibrated'] for r in rows]
            assert len(rows)==4*config['readiness_groups']
            groups={}
            for r,score in zip(rows,scores):groups.setdefault(r['group'],[]).append(score)
            global_scores=[sum(groups[r['group']])/len(groups[r['group']]) for r in rows]
            for kind,values,threshold in [('local',scores,b['threshold']),('global',global_scores,b['global_threshold']),('product',[r['product_score'] for r in rows],b['product_threshold'])]:
                calc=score_rows(values,truth,threshold)
                for key,value in calc.items():close(value,b[split][kind][key],(seed,split,kind,key));metric_checks+=1
                for bucket in b[split][kind]['reliability']:
                    lo=bucket['lower']; indices=[i for i,v in enumerate(values) if lo<=v and (v<lo+.1 or lo==.9 and v<=1)]
                    assert len(indices)==bucket['count']
                    if indices:
                        close(sum(values[i] for i in indices)/len(indices),bucket['confidence'],'reliability confidence')
                        close(sum(truth[i] for i in indices)/len(indices),bucket['accuracy'],'reliability accuracy')
            local=b[split]['local'];correct_gate=local['precision']>.99 and local['recall']>=.5 and local['local_advantage']>0
            assert correct_gate==b[split]['gate_b']
            seed_result['B'][split]={k:local[k] for k in ('selected','true_positive','executable','precision','recall')}
        results.append(seed_result)
    assert summary['gate_a_all_validation_seeds']==all(all(r['A'][s]['full']>.98 and all(r['A'][s][k]>.99 for k in ('primitive','destination','operand1','operand2','noncommutative_order')) for s in ('iid_validation','ood_validation')) for r in results)
    assert not summary['composition_authorized'] and not summary['progressive_authorized']
    return dict(source='independent standard-library raw recomputation',metric_checks=metric_checks,seeds=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    print(json.dumps(audit_ab(args.directory),indent=2))
