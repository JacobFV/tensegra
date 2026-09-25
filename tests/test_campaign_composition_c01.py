"""C01 CPU mechanics; reduced hidden widths here are tests only."""
import torch
from torch import nn
from tensegra.campaign_composition import make_lowering_batch, model_inputs
from tensegra.campaign_composition_acquire import private_labels, controlled_rows
from tensegra.campaign_composition_runtime import build_returns, consume, manipulate
from tensegra.campaign_composition_models import NeuralOperandBaseline, ContextualBaseline
from tensegra.campaign_composition_study import paired_metrics, counts


def test_actual_return_bundle_matches_every_event_field_and_refusal_stays_absent():
    data = make_lowering_batch(512000001, 128); labels = private_labels(data['labels'])
    bundle = build_returns(data['public'], labels['primitive'], labels['targets'])
    for key, value in data['reference']['public']['event'].items():
        assert torch.equal(bundle['public']['event'][key], value)
    pointers = labels['targets'].clone(); pointers[0, 0] = pointers[0, 1]  # operand cannot be destination
    refused = build_returns(data['public'], labels['primitive'], pointers)
    assert 0 not in refused['indices'].tolist()
    assert refused['total'] == 128 and len(refused['reasons']) == 128
    assert len(refused['public']['event']['values']) == 127


def test_drop_has_no_supplied_label_and_refusal_is_never_filled(monkeypatch):
    import tensegra.campaign_composition_runtime as runtime
    data = make_lowering_batch(512000002, 8); labels = private_labels(data['labels'])
    pointers = labels['targets'].clone(); pointers[0, 0] = pointers[0, 1]
    bundle = build_returns(data['public'], labels['primitive'], pointers)
    def fake_workspace(model, public, delays, drop=False):
        assert drop is True
        return {d: torch.zeros(len(public['query']), 16) for d in delays}
    monkeypatch.setattr(runtime, 'workspace', fake_workspace)
    interfaces = dict(backbone=None, accessor=nn.Linear(16,33), mean=torch.zeros(16), scale=torch.ones(16), consumer=nn.Linear(35,2))
    out = consume(interfaces, manipulate(bundle,'drop'), [0,1], 'cpu')
    assert out['supplied'].tolist() == [-1]*8
    for cell in out['cells'].values(): assert cell['predictions'][0] == -1


def test_baselines_public_only_gradients_and_order_equivariance():
    torch.manual_seed(42)
    data = make_lowering_batch(512000003, 4); public = model_inputs(data['public'])
    for model in (NeuralOperandBaseline(hidden=32), ContextualBaseline(hidden=32, heads=4)):
        model.eval(); out = model(public)
        assert out['answer'].shape == (4,2) and out['pointers'].shape == (4,3,11)
        out['answer'].sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        for kind in ('record_order','inventory_order','fresh_names'):
            altered = model_inputs(controlled_rows(data['public'], kind, 512000004))
            assert torch.allclose(out['answer'], model(altered)['answer'], atol=2e-6)


def test_causal_metrics_keep_original_actual_changed_and_absence_separate():
    m = paired_metrics(torch.tensor([1,0,-1,1]), torch.tensor([0,0,1,1]), torch.tensor([1,0,-1,-1]))
    assert m['original'] == dict(correct=2,total=4)
    assert m['supplied'] == dict(correct=2,total=2)
    assert m['changed_supplied'] == dict(correct=1,total=1)
    assert m['refused'] == 1 and m['supplied_absent'] == 2


def test_tiny_n2_training_export_is_mechanical_only(tmp_path, monkeypatch):
    import tensegra.campaign_composition_study as study
    monkeypatch.setattr(study, 'ContextualBaseline', lambda: ContextualBaseline(hidden=16, heads=4))
    config = dict(seed=502, sample_seed=512000010, control_seed=512000011, interfaces={},
                  data={split:dict(seed=512000012+i,count=8) for i,split in enumerate(('train','calibration','validation'))},
                  learning_rates=[.001], updates=1, checkpoints=[0,1], batch_size=4, public_controls=['reverse_roles'])
    output = tmp_path/'n2'; output.mkdir()
    result = study.train_baseline(config, 'n2', output, 'cpu')
    assert result['validation']['answer']['total'] == 8
    assert result['selected_step'] in (0,1)
    saved = torch.load(output/'validation.pt', weights_only=True)
    assert not {'event','reference','task','value','targets'} & saved['public'].keys()


def test_hybrid_runner_refusals_and_oracles_are_separate(tmp_path, monkeypatch):
    import tensegra.campaign_composition_study as study
    import tensegra.campaign_composition_runtime as runtime
    from tensegra.interface_proposals import ProposalModel
    fake = dict(lowerer=ProposalModel(key_dim=32,hidden=16), backbone=None,
                accessor=nn.Linear(16,33),mean=torch.zeros(16),scale=torch.ones(16),
                consumer=nn.Linear(35,2),oracle=nn.Linear(35,2),query_only=nn.Linear(35,2))
    monkeypatch.setattr(study,'frozen_interfaces',lambda spec,device:fake)
    monkeypatch.setattr(runtime,'workspace',lambda model,public,delays,drop=False:{d:torch.zeros(len(public['query']),16) for d in delays})
    config = dict(interfaces={}, data={'validation':dict(seed=512000020,count=8)},eval_distractors=[2,8],
                  delays=[0,1],intervention_delays=[0,1],capture_batch_size=4,causal_size=8,swap_seed=512000021,
                  public_controls=['reverse_roles','unrelated_instructions'],control_seed=512000022)
    output=tmp_path/'hybrid';output.mkdir()
    result=study.hybrid(config,output,'cpu')
    assert len(result['proposal_audits'])==2
    assert result['frozen_state']['unchanged']
    assert result['frozen_state']['before'] == result['frozen_state']['after']
    assert (output/'oracle_return-8.pt').exists()
    drop=torch.load(output/'intervention-drop.pt',weights_only=True)
    assert (drop['result']['supplied']==-1).all()
    assert drop['supplied_public'] is None
