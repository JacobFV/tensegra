"""Prospective S20 aggregate core. No model, confirmation loader, or fitted policy.

Inputs are already-audited event flags. Immutable raw loader awaits runner freeze.
"""
import collections
import numpy as np
SEEDS=(701,702,703)
CELLS=('3x3','3x4','4x3','4x4')
KNOWN=('3x3','4x3','4x4')
COMPARATORS=('original','context')
POLICIES=('raw','matched')

def require(ok,message):
 if not ok:raise ValueError(message)

def validate(flags):
 """flags[seed][arm][policy][cell] maps immutable semantic IDs to bool."""
 require(set(flags)==set(SEEDS),'exact three independent seeds required')
 identities={}
 for seed in SEEDS:
  require(set(flags[seed])=={'record',*COMPARATORS},'all three arms required')
  for arm,policies in flags[seed].items():
   require(set(policies)==({'categorical'} if arm=='record' else set(POLICIES)),'complete fixed policies required')
   for policy,cells in policies.items():
    require(set(cells)==set(CELLS),'all four cells required')
    for cell,events in cells.items():
     require(len(events)==512 and all(isinstance(k,str) and len(k)==64 and all(c in '0123456789abcdef' for c in k) for k in events),'512 unique semantic IDs required')
     require(all(type(v)is bool for v in events.values()),'Boolean graph-exact flags required')
     ids=tuple(sorted(events))
     if cell in identities:require(identities[cell]==ids,'event populations differ')
     else:identities[cell]=ids
 require(len(set().union(*(set(x) for x in identities.values())))==2048,'cells overlap')
 return identities

def count_gate(counts):
 require(set(counts)==set(CELLS) and all(type(n)is int and 0<=n<=512 for n in counts.values()),'four integer512 counts')
 return dict(known_macro_competence=sum(counts[c] for c in KNOWN)>=1383,each_known_cell_competence=all(counts[c]>=410 for c in KNOWN),heldout_competence=counts['3x4']>=52)

def transitions(old,new):
 require(len(old)==len(new),'paired length differs')
 return {f'{a}->{b}':sum(int(x)==a and int(y)==b for x,y in zip(old,new)) for a in (0,1) for b in (0,1)}

def paired_interval(difference,draws):
 """Difference shape (3,512); same draws (3,10000,512) across all models."""
 require(difference.shape==(3,512) and draws.shape==(3,10000,512),'fixed stratified bootstrap support')
 distribution=sum(difference[i][draws[i]].mean(axis=1) for i in range(3))/3
 return [float(x*100) for x in np.percentile(distribution,[2.5,97.5])],distribution

def decisions(counts,matched_deltas,matched_intervals):
 require(set(counts)==set(matched_deltas)==set(matched_intervals)==set(SEEDS),'decision seed set')
 per_seed={}
 for seed in SEEDS:
  require(set(matched_deltas[seed])==set(matched_intervals[seed])==set(COMPARATORS),'both comparators required')
  g=count_gate(counts[seed]);g['superiority_vs_both']=all(matched_deltas[seed][arm]>0 and matched_intervals[seed][arm][0]>0 for arm in COMPARATORS)
  g['known_claim']=g['known_macro_competence'] and g['each_known_cell_competence'] and g['superiority_vs_both'];per_seed[str(seed)]=g
 return dict(per_seed=per_seed,replicated_known_claim=all(g['known_claim'] for g in per_seed.values()),heldout_claim=all(g['heldout_competence'] for g in per_seed.values()),automatic_extension=False)

def aggregate(flags):
 identities=validate(flags);rng=np.random.default_rng(20020)
 draws=np.stack([rng.integers(0,512,size=(10000,512)) for cell in KNOWN])
 counts={};paired={};deltas={};intervals={};seed_distributions=collections.defaultdict(list)
 for seed in SEEDS:
  record=flags[seed]['record']['categorical'];counts[seed]={c:sum(record[c].values()) for c in CELLS};deltas[seed]={};intervals[seed]={}
  paired[str(seed)]={}
  for arm in COMPARATORS:
   for policy in POLICIES:
    baseline=flags[seed][arm][policy];cell_out={};diff=[]
    for cell in CELLS:
     ids=identities[cell];new=np.array([record[cell][i] for i in ids],dtype=np.int8);old=np.array([baseline[cell][i] for i in ids],dtype=np.int8)
     cell_out[cell]=dict(examples=512,record_complete=int(new.sum()),baseline_complete=int(old.sum()),delta_complete=int(new.sum()-old.sum()),transitions_baseline_to_record=transitions(old,new))
     if cell in KNOWN:diff.append(new-old)
    # CELLS order restricted to KNOWN is 3x3,4x3,4x4, exactly draw order.
    diff=np.stack(diff);ci,dist=paired_interval(diff,draws);delta=int(diff.sum());seed_distributions[(arm,policy)].append(dist)
    paired[str(seed)][f'{arm}/{policy}']=dict(cells=cell_out,known_delta_complete=delta,known_macro_delta_percentage_points=100*delta/1536,conditional_event_ci95_percentage_points=ci)
    if policy=='matched':deltas[seed][arm]=delta;intervals[seed][arm]=ci
 average={f'{a}/{p}':[float(x*100) for x in np.percentile(np.mean(v,axis=0),[2.5,97.5])] for (a,p),v in seed_distributions.items()}
 return dict(scope='Fresh event confirmation conditional on three fixed trained lineages. Stratified paired event bootstrap is not population-of-seeds uncertainty. Shared event draws across all seeds/comparators/policies. All continuous outcomes retained irrespective of gates.',counts={str(k):v for k,v in counts.items()},paired=paired,three_fixed_lineage_mean_conditional_ci95_percentage_points=average,decisions=decisions(counts,deltas,intervals),bootstrap=dict(seed=20020,draws=10000,strata=list(KNOWN),events_per_stratum=512))
