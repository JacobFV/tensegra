"""Exact no-return Bayes ceiling from the unchanged retention generator."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path

mass=defaultdict(Fraction)
for label in range(33):
    value=Fraction(label-16,2)
    mass[value]+=Fraction(2,5*33)
    mass[Fraction(round(value))]+=Fraction(2,5*33)
mass[Fraction(0)]+=Fraction(1,10);mass[Fraction(1)]+=Fraction(1,10)
assert sum(mass.values())==1
cells=[];accuracy=Fraction(0)
for label in range(33):
    threshold=Fraction(label-16,2)
    positive=sum(probability for value,probability in mass.items() if value>threshold)
    best=max(positive,1-positive);accuracy+=best/33
    cells.append(dict(threshold=float(threshold),positive_before_polarity=float(positive),bayes_accuracy=float(best)))
result=dict(source='Exact public generator:20%Boolean,40%uniform half-unit,40%ties-to-even rounded half-unit;uniform33thresholds and independent polarity',
            value_mass={str(float(k)):float(v) for k,v in sorted(mass.items())},thresholds=cells,
            no_return_bayes_fraction=str(accuracy),no_return_bayes_accuracy=float(accuracy),perfect_to_bayes_drop=float(1-accuracy))
Path('research/campaigns/extended-01/returns/R05-query-prior.json').write_text(json.dumps(result,indent=2)+'\n')
