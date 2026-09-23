"""Regenerate public inputs for logged failures; never load/evaluate a model.

Run with frozen 4c36557 source on PYTHONPATH, accepting archive and output prefix:
PYTHONPATH=src python reproduce-stage6-language-failures.py ARCHIVE OUTPUT_PREFIX
"""
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
from collections import Counter
from topoformer.tcn_data import build_tcn_example, verify_vendor_manifest

archive, prefix = Path(sys.argv[1]), Path(sys.argv[2])
audit=json.loads((archive/'data-audit.json').read_text())
with gzip.open(archive/'curves.jsonl.gz','rt') as stream:
    rows=[json.loads(line) for line in stream]
final_step=max(row['step'] for row in rows)
arms=('semantic_supervision','multisurface_consistency')
final=[row for row in rows if row['step']==final_step and row['arm'] in arms]
by_digest={e['semantic_digest']:e for e in audit['examples']+audit['renamed_examples']}
original_by_seed={e['seed']:e for e in audit['examples']}
assert verify_vendor_manifest()
# Deterministic editorial coverage: both arms, all lessons, four evaluation views.
specifications=[(arms[0],'variable_binding','english'),(arms[0],'unification','spanish'),
                (arms[0],'set_operations','symbols'),(arms[1],'variable_binding','spanish'),
                (arms[1],'unification','symbols'),(arms[1],'set_operations','unseen_lexical')]
examples=[]
for arm,lesson,renderer in specifications:
    candidates=[(row,failure) for row in sorted(final,key=lambda r:r['seed']) if row['arm']==arm
                for failure in row['evaluation'][renderer]['failures']
                if by_digest[failure['semantic_digest']]['lesson']==lesson]
    row,failure=candidates[0]
    metadata=by_digest[failure['semantic_digest']]
    ex=build_tcn_example(lesson,metadata['seed'],difficulty=metadata['difficulty'],
                         identifier_renaming=metadata['identifier_renaming'])
    assert ex.audit['semantic_digest']==failure['semantic_digest']
    assert ex.audit['source_manifest_sha256']==metadata['source_manifest_sha256']
    assert ex.audit['compiler_sha256']==metadata['compiler_sha256']
    assert ex.audit['adapter_sha256']==metadata['adapter_sha256']
    language='english' if renderer=='unseen_lexical' else renderer
    surface=next(s for s in ex.public if s.language==language)
    assert hashlib.sha256(surface.text.encode()).hexdigest()==metadata['surface_sha256'][language]
    original_digest=original_by_seed[metadata['seed']]['semantic_digest']
    index=audit['semantic_eval'].index(original_digest)
    shuffle_seed=row['seed']+100000+index
    options=list(surface.options); random.Random(shuffle_seed).shuffle(options)
    expected=surface.options[ex.privileged.answer_index]
    assert expected==failure['expected'] and expected!=failure['predicted']
    assert failure['predicted'] in options
    examples.append(dict(arm=arm,model_seed=row['seed'],step=row['step'],lesson=lesson,
        renderer=renderer,actual_language=language,generator_seed=metadata['seed'],
        evaluation_index=index,choice_shuffle_seed=shuffle_seed,semantic_digest=failure['semantic_digest'],
        surface_sha256=metadata['surface_sha256'][language],identifier_renaming=metadata['identifier_renaming'],
        text=surface.text,options=options,expected=expected,recorded_prediction=failure['predicted']))
counts=[]
for row in final:
    for renderer,metrics in row['evaluation'].items():
        failures=metrics['failures']
        counts.append(dict(arm=row['arm'],seed=row['seed'],renderer=renderer,
                           recorded_failures=len(failures),
                           predicted_counts=dict(Counter(f['predicted'] for f in failures))))
result=dict(source_commit='4c36557',archive='results/stage6/language',
    archive_hashes={name:hashlib.sha256((archive/name).read_bytes()).hexdigest()
                    for name in ('data-audit.json','curves.jsonl.gz','manifest.json')},
    procedure='Regenerate constructions and public option shuffle; verify graph/surface/source hashes; join predictions from archived curves; no inference.',
    selection='First logged final-step failure for each specified arm/lesson/renderer, ordered by model seed.',
    limitations=['Failure buffers contain at most the first five errors per evaluation cell; selected examples are not a random sample.',
                 'Repeated surfaces share constructions, so repeated failures are correlated.',
                 'Recorded failure counts are not full prediction histograms or estimates of label preference prevalence.',
                 'No training-frequency comparison or causal shortcut test was performed.'],
    examples=examples,logged_failure_prediction_counts=counts)
prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
lines=['# Stage 6 TCN: reproduced failure examples','',
       'Six final-step errors below reproduce the exact public text and shuffled answer options from the frozen `4c36557` run. Predictions are copied from archived failure buffers; no model inference or training was performed. Semantic, source, compiler, adapter and surface hashes were checked during regeneration.','',
       'These examples span both graph-supervised arms, all three lessons, English, Spanish, symbols and controlled renamed entities. Spanish was exposed during multisurface training; symbols remained withheld. The renamed labels are a bounded pool reused across construction scopes.','',
       'The archived buffers retain at most the first five errors per cell. Repeated choices such as `erin`, `dave`, `o3` and `novelentity3` across different prompts are consistent with a weak, potentially input-insensitive answer preference. They do not establish a frequency shortcut: correct predictions and later errors are absent, surfaces reuse constructions, and training-label frequencies were not compared. Full logged-buffer counts, explicitly not full prediction histograms, are in the accompanying JSON.','']
for i,e in enumerate(examples,1):
    lines += [f"## {i}. {e['arm']} · {e['lesson']} · {e['renderer']}",'',
              f"Model seed {e['model_seed']}, update {e['step']}, generator seed {e['generator_seed']}; evaluation index {e['evaluation_index']}, option shuffle seed {e['choice_shuffle_seed']}.",'',
              '```text',e['text'],'```','',
              'Visible options in actor order: '+', '.join(f'`{x}`' for x in e['options'])+'.','',
              f"Recorded prediction: **{e['recorded_prediction']}**. Gold answer: **{e['expected']}**.",'',
              f"Semantic digest: `{e['semantic_digest']}`.",'']
lines += ['## Reproduction','',
          'Run `research/tools/reproduce-stage6-language-failures.py` with the frozen source on `PYTHONPATH`, passing the archived language directory and an output filename prefix. It regenerates only the six selected constructions, joins recorded predictions, verifies hashes and reproduces these JSON/Markdown files. It does not load checkpoints.','']
prefix.with_suffix('.md').write_text('\n'.join(lines))
print(json.dumps({'examples':len(examples),'final_step':final_step,'logged_cells':len(counts),'verified':True}))
