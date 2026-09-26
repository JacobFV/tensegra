"""Verify sha256 of every checkpoint referenced by the A2 screening configs and Track C eval summaries.
Run remotely (paths are remote). Usage: python hash_check.py <repo_or_snapshot_root> <results_root>"""
import json, sys, os, glob, hashlib

src, res = sys.argv[1], sys.argv[2]
seen, bad, n = {}, [], 0
for p in sorted(glob.glob(os.path.join(src, 'configs/campaign04/a2s-*.json'))):
    d = json.load(open(p))
    for c in (d.get('checkpoints') or []) if isinstance(d, dict) else []:
        seen[c['path']] = c['sha256']
for p in glob.glob(os.path.join(res, 'c-eval-*/summary.json')):
    d = json.load(open(p))
    base = d['base']
    cfg = os.path.join(src, 'configs/campaign04', f'a1-{base}.json')
    if os.path.exists(cfg):
        for c in json.load(open(cfg)).get('checkpoints', []):
            seen[c['path']] = c['sha256']
            if c['sha256'] != d['base_sha256']:
                bad.append(('c-eval base sha != a1 config', base))
for path, sha in sorted(seen.items()):
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest(); n += 1
    if h != sha:
        bad.append((path, sha, h))
print(json.dumps(dict(checked=n, mismatches=bad, paths=sorted(seen))))
