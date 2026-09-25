"""S16 mechanical leakage/freeze guards; no semantic model forward."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from tensegra.campaign_semantics_identity_contract import canonicalize_public_identifiers
from tensegra.campaign_semantics_s16 import predict_public,tensor_hash
from tensegra.campaign_semantics_s16_freeze import SOURCE_NAMES,freeze,verify,validate
from tensegra.thinking_language import ActorInput,KINDS,ROLES
from tensegra import campaign_semantics_s16_launch as launch


def test_predict_boundary_gets_only_normalized_text_and_preserves_wrong_copy():
    original=ActorInput('You know these facts: frank parent bob. The pattern is D parent bob. What is the unify D?',())
    normalization=canonicalize_public_identifiers(original)
    seen=[]
    def spy(public):
        assert isinstance(public,ActorInput)
        assert public==normalization.public and not public.options
        assert 'frank' not in public.text and 'alice' in public.text
        seen.append(public)
        scores=torch.zeros(2,len(normalization.original_tokens));scores[0,5]=10;scores[1,0]=10
        return dict(presence=torch.ones(2),kind=torch.zeros(2,len(KINDS)),value=torch.zeros(2,4),copy=scores,
            edges=torch.zeros(2,2,len(ROLES)),slots=torch.zeros(2,2,3))
    raw,cal,copied=predict_public(spy,normalization,torch.zeros(len(ROLES)))
    assert len(seen)==1 and copied==('frank','You')
    assert raw['copy'].tolist()==[5,0] and torch.equal(raw['copy'],cal['copy'])
    assert set(inspect.signature(predict_public).parameters)=={'model','normalization','thresholds'}


def test_tensor_hash_covers_scalar_dtype_shape_and_mutation():
    state={'scalar':torch.tensor(1.),'value':torch.tensor([1.,2.])}
    before=tensor_hash(state)
    assert before==tensor_hash({k:v.clone() for k,v in state.items()})
    state['value'][1]=3
    assert tensor_hash(state)!=before
    assert tensor_hash({'x':torch.tensor([1.])})!=tensor_hash({'x':torch.tensor(1.)})


def test_profile_freeze_keeps_all_six_and_exact_cap_source():
    original=Path('research/campaigns/extended-01/semantics/S16-profile-prepared.json')
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);source=root/'src';source.mkdir()
        for name in SOURCE_NAMES:(source/name).write_text(name)
        config=freeze(original,root/'frozen.json',source,60)
        verify(config,source,60)
        changed=copy.deepcopy(config);changed['runs']=changed['runs'][:-1]
        with pytest.raises(ValueError):validate(changed)
        changed=copy.deepcopy(config);changed['examples']=9
        with pytest.raises(ValueError):validate(changed)
        changed=copy.deepcopy(config);changed['threshold_policy']='refit'
        with pytest.raises(ValueError):validate(changed)
        with pytest.raises(ValueError):verify(config,source,61)
        (source/'campaign_semantics_identity_contract.py').write_text('changed')
        with pytest.raises(ValueError):verify(config,source,60)
        with pytest.raises(FileExistsError):freeze(original,root/'frozen.json',source,60)


def test_main_is_unlaunchable_without_postprofile_cap_freeze():
    config=json.loads(Path('research/campaigns/extended-01/semantics/S16-main-prepared.json').read_text())
    validate(config)
    assert config['proposed_cap_seconds'] is None
    with pytest.raises(ValueError):verify(config,Path('src/tensegra'))
    config['budget_status']='frozen'
    with pytest.raises(ValueError,match='positive frozen process cap'):verify(config,Path('src/tensegra'))


@pytest.mark.parametrize('timeout',[False,True])
def test_lifecycle_receipt_records_failure_or_timeout_without_launch(monkeypatch,timeout):
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);prepared=root/'prepared.json'
        config=json.loads(Path('research/campaigns/extended-01/semantics/S16-profile-prepared.json').read_text())
        config['output_dir']=str(root/'results');prepared.write_text(json.dumps(config))
        freeze(prepared,root/'frozen.json',Path('src/tensegra'),60)
        monkeypatch.setattr(sys,'argv',['launch',str(root/'frozen.json'),'--cap','60','--python','fake-python','--prefix',str(root/'attempt')])
        monkeypatch.setattr(launch.subprocess,'check_output',lambda command,**kw:'' if command[0]=='nvidia-smi' else 'mock full process snapshot')
        calls=[]
        def fake_run(command,**kw):
            calls.append(command)
            if timeout:raise subprocess.TimeoutExpired(command,kw['timeout'])
            return SimpleNamespace(returncode=17)
        monkeypatch.setattr(launch.subprocess,'run',fake_run)
        with pytest.raises(SystemExit) as exit:launch.main()
        assert exit.value.code==(124 if timeout else 17)
        receipt=json.loads((root/'attempt.occupancy.json').read_text())
        assert receipt['timed_out']==timeout and receipt['exit_code']==exit.value.code
        assert receipt['full_process_state']=='mock full process snapshot'
        assert len(calls)==1 and calls[0][2]=='topoformer.campaign_semantics_s16'  # frozen launcher keeps its pre-rename module path
        assert (root/'attempt.started.json').exists() and not list(root.glob('*.tmp'))
