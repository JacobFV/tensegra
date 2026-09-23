"""Rebuild return study aggregate tables and failure examples from raw JSONL."""
import argparse
import gzip
import json
from pathlib import Path


def summarize(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    rows=[]; failures=[]
    for run in manifest['runs']:
        name=f"{run['seed']}-{run['encoding']}-{run['availability']}"
        path=directory/(name+'.jsonl')
        opener=open
        if not path.exists(): path=path.with_suffix('.jsonl.gz'); opener=gzip.open
        with opener(path,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['phase']!='eval': continue
                compact={k:row[k] for k in ('split','seed','distractors','steps','intervention','counts')}
                compact.update(model_seed=run['seed'],encoding=run['encoding'],availability=run['availability'])
                rows.append(compact)
                if row['split']=='test' and row['steps'] in (1,16) and row['intervention']=='none':
                    wrong=[i for i in range(len(row['targets']['value'])) if any(row['predictions'][field][i]!=row['targets'][field][i] for field in row['targets'])]
                    for i in wrong[:2]:
                        failures.append(dict(model_seed=run['seed'],encoding=run['encoding'],availability=run['availability'],data_seed=row['seed'],distractors=row['distractors'],steps=row['steps'],example_index=i,target={k:v[i] for k,v in row['targets'].items()},prediction={k:v[i] for k,v in row['predictions'].items()}))
    (directory/'summary.json').write_text(json.dumps(dict(runs=manifest['runs'],cells=rows),indent=2))
    (directory/'failures.json').write_text(json.dumps(failures,indent=2))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    summarize(args.directory)
