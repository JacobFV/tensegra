"""Shared loaders/statistics for the independent extended-05 Track A audit.

Written by the auditor from the raw row formats only (no reuse of campaign05_hr.py
analyze code or campaign05_apit.py score code).  Data root is a local read-only
copy of ~/structured-latent-dynamics-campaign05/results (e05-hr, e05-hr2,
e05-apit, e05-acf; the eval directories were renamed 'ev' in the copy).
"""
import gzip
import json
import os
import re

import numpy as np

FNAME = re.compile(r'(?:branch|apit)-(x1-r\d)-(.+?)-(hr-multi|hr-full|hr|apit|acf)-c(\d+)\.json\.gz$')


def load_dir(d):
    out = []
    for f in sorted(os.listdir(d)):
        m = FNAME.match(f)
        if not m:
            continue
        base, cond, kind, chunk = m.groups()
        data = json.load(gzip.open(os.path.join(d, f)))
        meta = json.load(open(os.path.join(d, f.replace('.json.gz', '.meta.json'))))
        out.append(dict(file=f, base=base, cond=cond, kind=kind, chunk=int(chunk), data=data, meta=meta))
    return out


def cluster_boot(values, clusters, n=2000, seed=0):
    """Mean with percentile CI over cluster resamples (cluster = world)."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    idx = rng.integers(0, k, size=(n, k))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return dict(n=int(len(values)), clusters=int(k), mean=float(values.mean()),
                ci90=[float(np.quantile(boots, .05)), float(np.quantile(boots, .95))],
                ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))])
