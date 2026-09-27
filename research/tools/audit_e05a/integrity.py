"""Controller/data integrity: controller.json hash (repo vs remote), controller training-file hashes vs the
A-HR2 branch files, controller feature names (public-only screen), remote vs committed result files.
usage: python integrity.py <data_root> <repo_root> <out.json>"""
import hashlib
import json
import os
import sys

ROOT, REPO, OUT = sys.argv[1:4]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


ctl_repo = f'{REPO}/research/results/campaign-05/a-pi-t/controller.json'
ctl_remote = f'{ROOT}/e05-apit/controller.json'
c = json.load(open(ctl_repo))
files = c['training_data']['files']
match = {f: (sha(f'{ROOT}/e05-hr2/branch/{f}') == h) for f, h in files.items()}
names = c['controller']['feature_names']
suspicious = [n for n in names if any(t in n.lower() for t in ('success', 'hidden', 'true', 'oracle', 'q_d', 'du', 'dsteps', 'agree', 'outcome', 'solution', 'belief'))]
pairs = {'a-hr/analysis.json': 'e05-hr/analysis.json', 'a-hr2/analysis.json': 'e05-hr2/analysis.json',
         'a-pi-t/score.json': 'e05-apit/score.json', 'a-cf-t/score.json': 'e05-acf/score.json',
         'a-pi-t/controller.json': 'e05-apit/controller.json'}
same = {k: sha(f'{REPO}/research/results/campaign-05/{k}') == sha(f'{ROOT}/{v}') for k, v in pairs.items()}
res = dict(controller_sha_repo=sha(ctl_repo), controller_sha_remote=sha(ctl_remote),
           training_files=len(files), training_files_hash_match=sum(match.values()),
           training_bases=sorted({f.split('-')[2] for f in files}),
           triggers=c['controller']['triggers'], margin=c['controller']['margin'], family=c['controller']['family'],
           n_features=len(names), feature_groups=sorted({n.split(':')[1] if n.startswith('telemetry:') else n.split('_')[0] for n in names}),
           suspicious_feature_names=suspicious, selection_rule=c['selection']['rule'], rule_history=c['selection']['rule_history'],
           registered_before_evaluation=c['selection']['registered_before_evaluation'],
           committed_vs_remote_outputs_identical=same)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res, indent=1))
