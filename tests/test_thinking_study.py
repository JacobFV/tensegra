"""Stage 6 runner contracts; execute Torch tests on the remote CPU host."""
import pytest
from topoformer.thinking_study import ThinkingStudyConfig, auxiliary_weights, VARIANTS, rate


def test_resource_guard_and_step_zero():
    cfg = ThinkingStudyConfig(steps=24, checkpoints=[12])
    assert cfg.checkpoints == [0, 12, 24]
    assert cfg.batch_size == 4 and cfg.width == 32
    with pytest.raises(ValueError):
        ThinkingStudyConfig(threads=4)
    with pytest.raises(ValueError):
        ThinkingStudyConfig(batch_size=33)


def test_independent_auxiliary_anneals_and_task_only():
    cfg = ThinkingStudyConfig(steps=24, warmup_steps=12)
    for name in ('grounding', 'topology', 'transition', 'readiness'):
        w = auxiliary_weights('anneal_' + name, 12, cfg)
        assert w[name] == .5
        assert all(w[k] == 1 for k in w if k != name)
    assert set(auxiliary_weights('anneal_all', 24, cfg).values()) == {0.}
    assert set(auxiliary_weights('task_only_cold', 0, cfg).values()) == {0.}
    assert set(auxiliary_weights('task_only_warm', 0, cfg).values()) == {1.}
    assert set(auxiliary_weights('task_only_warm', 12, cfg).values()) == {0.}


def test_controls_and_undefined_denominators():
    assert {'local', 'global', 'fixed', 'runtime_off', 'event_drop', 'event_shuffle',
            'event_wrong_value', 'public_graph', 'task_only_cold', 'task_only_warm'} <= VARIANTS.keys()
    assert rate(0, 0) is None
    assert rate(1, 2) == .5


def test_free_rollout_public_boundary_and_event_value_isolation():
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_runtime import ProtectedSession, ValueRegister
    from topoformer.thinking_study import actor_inputs, build_model, rollout
    cfg = ThinkingStudyConfig(steps=0,max_microsteps=4)
    episode = generate_episode(seed=1,depth=1)
    session = ProtectedSession(episode.public.initial_values)
    session.execute(episode.gold.trace[0])
    _,a,_,_ = actor_inputs(episode.public,session,3,cfg)
    changed = ProtectedSession(tuple(ValueRegister(r.id,999,r.type) if r.id.startswith('result:') else r for r in session.registers.values()))
    _,b,_,_ = actor_inputs(episode.public,changed,3,cfg)
    assert torch.equal(a,b)
    model = build_model(cfg,0)
    output = rollout(model,episode.public,cfg,'local')
    assert output['post_event_pass']
    assert output['microsteps'] <= 4


def test_teacher_training_has_finite_separate_losses():
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_study import build_model, train_update
    cfg = ThinkingStudyConfig(steps=1,max_microsteps=6)
    model = build_model(cfg,0)
    losses,_ = train_update(model,torch.optim.Adam(model.parameters()),[generate_episode(depth=1)],cfg,'local',0,0.)
    assert {'graph','grounding','topology','transition','readiness','task','emit','policy'} <= losses.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())


def test_all_hidden_target_mutations_leave_free_actor_identical():
    from dataclasses import replace
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_study import build_model, rollout
    cfg = ThinkingStudyConfig(steps=0,max_microsteps=5)
    episode = generate_episode(seed=3,depth=1)
    mutated = replace(episode,gold=replace(episode.gold,result=-37,answer=1-episode.gold.answer,trace=(),readiness=(),hypotheses=(),expected_values=(),semantic_digest='changed',graph=None))
    model = build_model(cfg,0)
    a,b = [rollout(model,e.public,cfg) for e in (episode,mutated)]
    assert a['trace']==b['trace']
    assert torch.equal(a['workspace'],b['workspace'])


def test_future_frames_cannot_change_current_features_and_context_order_matters():
    from dataclasses import replace
    import torch
    from topoformer.thinking_tasks import generate_episode, PublicFrame
    from topoformer.thinking_runtime import ProtectedSession
    from topoformer.thinking_study import actor_inputs
    cfg = ThinkingStudyConfig(steps=0)
    episode = generate_episode(seed=3,depth=1)
    public = episode.public
    session = ProtectedSession(public.initial_values)
    modified = replace(public,frames=(public.frames[0],PublicFrame(('secret',)),PublicFrame(('future',))))
    assert torch.equal(actor_inputs(public,session,1,cfg)[0],actor_inputs(modified,session,1,cfg)[0])
    session.execute(episode.gold.trace[0])
    changed = replace(public,context_tokens=tuple(reversed(public.context_tokens)))
    assert not torch.equal(actor_inputs(public,session,3,cfg)[0],actor_inputs(changed,session,3,cfg)[0])


def test_anneal_finishes_before_training_ends():
    cfg = ThinkingStudyConfig(steps=24)
    assert set(auxiliary_weights('anneal_all',18,cfg).values()) == {0.}
    assert set(auxiliary_weights('anneal_all',23,cfg).values()) == {0.}


def test_event_reintegration_changes_future_actions_and_has_gradients():
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_runtime import ProtectedSession
    from topoformer.thinking_study import build_model, actor_inputs, inject_runtime_events, rollout
    cfg = ThinkingStudyConfig(steps=0,max_microsteps=6)
    episode = generate_episode(seed=7,depth=1)
    model = build_model(cfg,0)
    session = ProtectedSession(episode.public.initial_values)
    context,memory,ids,_ = actor_inputs(episode.public,session,3,cfg)
    workspace = model.initialize({'context':context})
    prediction = model.step(workspace,context,memory,microstep=3)
    events = session.execute(episode.gold.trace[0])
    injected,_,count,_ = inject_runtime_events(model,prediction['workspace'],prediction,events,ids,cfg,'local')
    dropped,_,_,_ = inject_runtime_events(model,prediction['workspace'],prediction,events,ids,cfg,'event_drop')
    assert count == 2
    a = model.step(injected,context,memory,microstep=4)
    b = model.step(dropped,context,memory,microstep=4)
    assert not torch.allclose(a['op_logits'],b['op_logits'])
    assert not torch.allclose(a['output_logits'],b['output_logits'])
    result = rollout(model,episode.public,cfg,gold=episode.gold,teacher_forcing=True)
    for name in ('grounding','graph','topology','transition','readiness','task','emit','event'):
        grads = torch.autograd.grad(result['losses'][name],tuple(model.parameters()),retain_graph=True,allow_unused=True)
        assert any(g is not None and bool(g.abs().sum()>0) for g in grads), name


def test_privileged_trace_audit_and_post_return_workspace_roundtrip():
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_study import build_model, evaluate
    cfg = ThinkingStudyConfig(steps=0,max_microsteps=7)
    result = evaluate(build_model(cfg,0),[generate_episode(seed=7,depth=1)],cfg,'local',oracle_trace=True)
    example = result['examples'][0]
    assert example['exact_semantic_correct'] and example['transition_set_exact']
    assert example['post_event_pass']
    stages = {r['stage'] for entry in example['trace'] for r in entry['event_roundtrips']}
    assert stages == {'encoder','post_recurrent_workspace'}


def test_hazard_weights_conserve_survival_and_mask_post_event():
    import torch
    from topoformer.thinking_study import halting_weights
    probabilities = torch.tensor([.8,.4,.7,.3],requires_grad=True)
    weights = halting_weights(probabilities,torch.tensor([False,True,False,True]))
    assert torch.allclose(weights,torch.tensor([0.,.4,0.,.6]))
    assert torch.allclose(weights.sum(),torch.tensor(1.))


def test_neural_recurrent_task_loss_trains_emit_without_oracle_execution():
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_study import build_model, rollout
    cfg = ThinkingStudyConfig(steps=0,max_microsteps=6)
    model = build_model(cfg,0)
    episode = generate_episode(seed=7,depth=1)
    result = rollout(model,episode.public,cfg,'neural_recurrent',gold=episode.gold,training_unroll=True)
    assert result['microsteps']==cfg.max_microsteps
    assert not any(entry['events'] for entry in result['trace'])
    assert result['losses']['emit'].item()==0
    assert abs(sum(result['halting_weights'])-1)<1e-6
    gradient, = torch.autograd.grad(result['losses']['task'],model.emit_probe.weight)
    assert bool(gradient.abs().sum()>0)
    assert set(auxiliary_weights('neural_recurrent',0,cfg).values())=={0.}
    fixed = rollout(model,episode.public,cfg,'neural_fixed',gold=episode.gold,training_unroll=True)
    assert fixed['microsteps']==1 and fixed['halting_weights']==[1.]


def test_context_arrival_is_exogenous_and_neural_pair_observables_match():
    import torch
    from topoformer.thinking_tasks import generate_episode
    from topoformer.thinking_runtime import ProtectedSession
    from topoformer.thinking_study import actor_inputs
    cfg = ThinkingStudyConfig(steps=0)
    episode = generate_episode(seed=7,depth=1)
    session = ProtectedSession(episode.public.initial_values)
    early = actor_inputs(episode.public,session,1,cfg)[0]
    final = actor_inputs(episode.public,session,3,cfg)[0]
    assert final.shape[1] == len(episode.public.frames[-1].tokens)+len(episode.public.context_tokens)
    assert early.shape[1] == len(episode.public.frames[0].tokens)
    fixed = actor_inputs(episode.public,session,1,cfg,complete_evidence=True)
    recurrent = actor_inputs(episode.public,session,6,cfg,complete_evidence=True)
    assert torch.equal(fixed[0],recurrent[0]) and torch.equal(fixed[1],recurrent[1])
