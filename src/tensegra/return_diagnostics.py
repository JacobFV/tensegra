"""Stage 9 archived return analysis. Standard library only; no model inference."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

FIELDS = ('value', 'type', 'operation', 'argument0', 'argument1', 'provenance')


def matches(row):
    fields = {k: [a == b for a, b in zip(row['predictions'][k], row['targets'][k])] for k in FIELDS}
    fields['nonvalue_joint'] = [all(v) for v in zip(*(fields[k] for k in FIELDS[1:]))]
    fields['joint'] = [all(v) for v in zip(*(fields[k] for k in FIELDS))]
    return fields


def paired(first, last):
    """Pair only same deterministic batch recipe; never pair marginal counts."""
    for key in ('split', 'seed', 'distractors', 'intervention', 'targets'):
        if first[key] != last[key]:
            raise ValueError(f'Cannot pair different {key}')
    if first['intervention'] != 'none':
        raise ValueError('Lifecycle/corruption pairing needs explicit supplied-fact semantics')
    left, right = matches(first), matches(last)
    return {field: dict(Counter(('correct' if a else 'wrong') + '_to_' + ('correct' if b else 'wrong')
                               for a, b in zip(left[field], right[field])))
            for field in ('value', 'nonvalue_joint', 'joint')}


def analyze(row, value_limit=8):
    correct = matches(row)
    y, p = row['targets']['value'], row['predictions']['value']
    errors = [(a-b)/2 for a,b in zip(p,y)]
    n = len(y)
    result = dict(counts={k: {'correct': sum(v), 'total': n} for k,v in correct.items()},
                  absolute_error=sum(abs(e) for e in errors)/n, signed_error=sum(errors)/n,
                  within_half_unit=sum(abs(e)<=.5 for e in errors),
                  value_given_nonvalue={'correct':sum(a and b for a,b in zip(correct['value'],correct['nonvalue_joint'])),
                                       'total':sum(correct['nonvalue_joint'])},
                  confusion=[[a,b,c] for (a,b),c in sorted(Counter(zip(y,p)).items())])
    result['groups'] = {}
    for name, labels in {'value':y, 'sign':[int(v>2*value_limit)-int(v<2*value_limit) for v in y],
                         'magnitude':[abs(v-2*value_limit)/2 for v in y],
                         'type':row['targets']['type'], 'operation':row['targets']['operation']}.items():
        result['groups'][name] = {str(label): {'total':sum(x==label for x in labels),
             'correct':sum(x==label and ok for x,ok in zip(labels,correct['value'])),
             'signed_error_sum':sum(e for x,e in zip(labels,errors) if x==label),
             'absolute_error_sum':sum(abs(e) for x,e in zip(labels,errors) if x==label)} for label in sorted(set(labels))}
    required = [v!=6 for v in row['targets']['argument1']]
    result['counts']['argument1_required'] = {'correct':sum(a and b for a,b in zip(required,correct['argument1'])), 'total':sum(required)}
    return result


def run(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    archived_sources = manifest['source']
    source_checks = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest for name,digest in archived_sources.items()}
    rows, pairs, hashes = [], [], {}
    for path in sorted(source.glob('*.jsonl.gz')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path,'rt') as f:
            data=[json.loads(line) for line in f if '"phase": "eval"' in line]
        indexed = {}
        for row in data:
            meta={k:row[k] for k in ('split','seed','distractors','steps','intervention')}
            meta['run']=path.name.removesuffix('.jsonl.gz')
            stats=analyze(row)
            for field, count in row['counts'].items():
                if field in stats['counts'] and count != stats['counts'][field]: raise AssertionError((path,meta,field))
            rows.append(dict(**meta,**stats))
            if row['intervention']=='none': indexed[(row['split'],row['seed'],row['distractors'],row['steps'])]=row
        for key, start in indexed.items():
            if key[-1] not in (0,1): continue
            for delay in (1,4,16,32):
                if delay <= key[-1]: continue
                end=indexed[(*key[:-1],delay)]
                pairs.append(dict(run=path.name.removesuffix('.jsonl.gz'),split=key[0],seed=key[1],distractors=key[2],start=key[-1],end=delay,outcomes=paired(start,end)))
    (output/'analysis.json').write_text(json.dumps(dict(rows=rows,paired=pairs),separators=(',',':')))
    (output/'manifest.json').write_text(json.dumps(dict(kind='numerical reconstruction; no inference',source_files=hashes,
        source_match=source_checks,rows=len(rows),pairs=len(pairs),pairing='same deterministic data seed/batch size/distractor condition; targets checked equal; nonce hashes not in historical raw'),indent=2))
    return rows,pairs

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(args.source,args.output)
