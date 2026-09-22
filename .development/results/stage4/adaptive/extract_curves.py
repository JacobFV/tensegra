"""Stream Stage4 raw artifacts to compact learning-curve summaries."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

metrics, output = map(Path, sys.argv[1:])
rows = []
with gzip.open(metrics, 'rt') as stream:
    for line in stream:
        run = json.loads(line)
        curve = []
        for checkpoint in run['training']['curve']:
            evaluation = checkpoint['diagnostics']
            curve.append(dict(step=checkpoint['step'], task_accuracy=evaluation['task_accuracy'],
                              metrics=evaluation['metrics'], losses=checkpoint['losses']))
        tail = run['training']['losses'][-10:]
        loss_means = {key: statistics.fmean(row[key] for row in tail)
                      for key in tail[0] if key != 'step'} if tail else {}
        rows.append(dict(variant=run['variant'], seed=run['seed'],
                         source=run['source'], config_hash=run['config_hash'],
                         loss_weights=run['loss_weights'], strengths=run['model']['strengths'],
                         curve=curve, final_ten_step_mean_losses=loss_means))
result = dict(metrics_file=metrics.name, metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest(),
              extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), runs=rows)
(output/'curves.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
            if p.is_file() and p.name != 'provenance.json'}
(output/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
