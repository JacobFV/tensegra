"""Explicit LR-only optimizer intervention and archived-score loss diagnostics."""
import math
import numpy as np


def override_learning_rate(optimizer,rate):
    if not math.isfinite(rate) or rate<=0:raise ValueError('positive finite learning rate required')
    previous=[group['lr'] for group in optimizer.param_groups]
    for group in optimizer.param_groups:group['lr']=rate
    return dict(before=previous,after=[group['lr'] for group in optimizer.param_groups],changed_fields=['param_groups[*].lr'])


def calibration_losses(path):
    data=np.load(path);scores=data['scores'].astype(np.float64);targets=data['targets'].astype(bool)
    losses=np.logaddexp(0.,scores)-targets*scores
    def summarize(loss,y):
        terms=[float(loss[m].mean()) for m in (y,~y) if m.any()]
        return dict(positive=int(y.sum()),negative=int((~y).sum()),natural_BCE=float(loss.mean()) if loss.size else None,balanced_BCE=sum(terms)/len(terms) if terms else None)
    return dict(scope='TRAIN calibration, conditional on predicted-present pairs; different support from sampled training loss',overall=summarize(losses,targets),per_relation=[summarize(losses[:,r],targets[:,r]) for r in range(scores.shape[1])])
