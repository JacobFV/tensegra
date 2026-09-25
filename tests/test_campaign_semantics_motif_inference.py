import json
from pathlib import Path
import pytest
import torch
from tensegra.campaign_semantics_motif_inference import compact,expand,expand_policies,validate
from tensegra import semantic_scaling as base


def test_compact_preserves_metrics_and_shared_calibration_archive():
    rng=torch.Generator().manual_seed(14014);n=8
    gold=dict(presence=torch.arange(n)<4,kind=torch.arange(n),value=torch.ones(n,dtype=torch.long),copy=torch.arange(n),
              edges=torch.zeros(n,n,len(base.ROLES),dtype=torch.bool),slots=torch.full((n,n),-1,dtype=torch.long))
    gold['edges'][0,1,3]=True;gold['slots'][0,1]=2
    predictions=[gold]
    for _ in range(20):
        predictions.append(dict(presence=torch.rand(n,generator=rng)>.4,kind=torch.randint(8,(n,),generator=rng),
            value=torch.randint(3,(n,),generator=rng),copy=torch.randint(4,(n,),generator=rng),
            edges=torch.rand(n,n,len(base.ROLES),generator=rng)>.8,slots=torch.randint(-1,4,(n,n),generator=rng)))
    for raw in predictions:
        cal={**raw,'edges':torch.rand(n,n,len(base.ROLES),generator=rng)>.8}
        rp,cp=compact(raw),compact(cal);pairs={(i,j) for i,j,s in rp['slots']}
        row=dict(raw=rp,calibrated_edges=cp['edges'],calibrated_extra_slots=[v for v in cp['slots'] if tuple(v[:2]) not in pairs])
        decoded=expand_policies(row)
        assert base.metrics(expand(rp),gold)==base.metrics(raw,gold)
        assert base.metrics(decoded[0],gold)==base.metrics(raw,gold)
        assert base.metrics(decoded[1],gold)==base.metrics(cal,gold)


def test_prepared_all_six_population_is_not_launch_authority():
    for job in ('profile','main'):
        c=json.loads(Path(f'configs/campaign-s14-motif-{job}-prepared.json').read_text())
        with pytest.raises(ValueError,match='freeze'):validate(c)
        c['budget_status']='frozen';validate(c)
        c['runs']=c['runs'][:-1]
        with pytest.raises(ValueError,match='six'):validate(c)
