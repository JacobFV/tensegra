import torch
from tensegra.retention_data import make_batch
from tensegra.retention import ReturnRetentionModel, ReturnRegister


def test_nonce_ordered_targets_and_float_integral():
    batch = make_batch(19, 128, feature_dim=16)
    p,t = batch['public'],batch['targets']
    rows = torch.arange(128)
    for a in range(2):
        assert torch.equal(p['event']['arguments'][:,0,a],p['argument_keys'][rows,t[f'argument{a}']])
    assert torch.equal(p['event']['provenance'][:,0],p['provenance_keys'][rows,t['provenance']])
    assert ((t['type']==1)&(p['event']['values'][:,0]%1==0)).any()
    assert len(t['argument0'].unique()) > 1


def test_register_copies_and_authoritative_release_overwrite():
    event = make_batch(1,4)['public']['event']
    register = ReturnRegister(event)
    saved = register.read()['values'].clone()
    event['values'].zero_()
    assert torch.equal(register.read()['values'], saved)
    register.release()
    assert register.read() is None
    replacement = make_batch(2,4)['public']['event']
    register.overwrite(replacement)
    assert torch.equal(register.read()['values'], replacement['values'])


def test_late_query_no_effect_before_extra_pass_and_public_immutable():
    torch.manual_seed(0)
    model = ReturnRetentionModel(feature_dim=16,width=16)
    public = make_batch(3,4,feature_dim=16)['public']
    saved = public['event']['values'].clone()
    a = model(public,2,'gated')
    b = model(public,2,'gated',intervention='query_counterfactual')
    assert torch.equal(a['before']['value'],b['before']['value'])
    assert not torch.equal(a['after']['task'],b['after']['task'])
    assert torch.equal(saved,public['event']['values'])
    assert a['cell_passes'] == 3
    a['after']['task'].sum().backward()
    assert model.query.weight.grad.abs().sum() > 0


def test_release_invalidates_storage_without_erasing_workspace():
    torch.manual_seed(2)
    model = ReturnRetentionModel(feature_dim=16,width=16)
    p = make_batch(4,3,feature_dim=16)['public']
    q = dict(p,event=make_batch(5,3,feature_dim=16)['public']['event'])
    for mode in ('protected','gated','persistent'):
        a=model(p,2,mode,intervention='release')
        b=model(q,2,mode,intervention='release')
        assert not torch.equal(a['before']['value'],b['before']['value'])
        assert not a['register_active']
    a=model(p,2,'protected',intervention='delay_memory_drop')
    b=model(p,2,'persistent',intervention='delay_memory_drop')
    assert not torch.equal(a['before']['value'],b['before']['value'])


def test_counts_gate_requires_all_fields_seeds_and_conditions():
    from tensegra.retention_study import competence_gate
    counts={k:dict(correct=1000,total=1000) for k in ('type','operation','value','argument0','argument1','provenance')}
    rows=[dict(seed=s,distractors=d,delay=16,counts={k:dict(v) for k,v in counts.items()}) for s in (1,2) for d in (2,8)]
    assert competence_gate(rows,[1,2],[2,8])
    rows[-1]['counts']['argument1']['correct']=980
    assert not competence_gate(rows,[1,2],[2,8])
    assert not competence_gate(rows[:-1],[1,2],[2,8])


def test_empty_gate_support_never_passes():
    from tensegra.retention_study import competence_gate, usage_gate
    assert not competence_gate([],[],[])
    assert not usage_gate([],[],[])


def test_overwrite_preserves_free_workspace_and_validity_is_learned():
    torch.manual_seed(1)
    model=ReturnRetentionModel(feature_dim=16,width=16)
    p=make_batch(21,4,feature_dim=16)['public']
    q=dict(p,event=make_batch(22,4,feature_dim=16)['public']['event'])
    replacement=make_batch(23,4,feature_dim=16)['public']['event']
    a=model(p,2,'gated','overwrite',replacement)
    b=model(q,2,'gated','overwrite',replacement)
    assert not torch.equal(a['before']['value'],b['before']['value'])
    assert a['after']['validity'].shape==(4,2)
    assert a['register_active']


def test_c1_loss_excludes_late_query_task_head():
    from tensegra.retention_study import loss
    model=ReturnRetentionModel(feature_dim=16,width=16)
    batch=make_batch(32,4,feature_dim=16)
    output=model(batch['public'],1,'protected')
    total,terms=loss(output,batch['targets'])
    total.backward()
    assert 'task' not in terms
    assert model.heads['task_head'].weight.grad is None
    assert model.query.weight.grad is None


def test_runner_zero_step_artifact_and_acquisition_cannot_compose(tmp_path):
    from tensegra.retention_study import run
    config=dict(seeds=[0],modes=['protected'],feature_dim=16,width=16,batch_size=2,steps=0,
                fixed_set=True,validation_seeds=[900001],eval_size=2,eval_delays=[16],
                eval_distractors=[0],interventions=['none','event_drop','wrong_value'],train_delays=[1])
    result=run(config,tmp_path)
    assert result['runs'][0]['optimizer_steps']==0
    assert result['runs'][0]['dependent_experiments']=='blocked_acquisition_only'
    assert result['runs'][0]['late_use_status']=='untrained_diagnostic_C1'


def test_every_perfect_event_matches_existing_exact_runtime():
    from tensegra.thinking_runtime import ProtectedSession, ValueRegister, Candidate, PRIMITIVE_NAMES, VALUE_TYPES
    batch=make_batch(73,256)
    events=batch['public']['event']
    seen=set()
    for i in range(256):
        op=PRIMITIVE_NAMES[int(events['operations'][i,0])]
        target_type=VALUE_TYPES[int(events['types'][i,0])]
        operand_type='integer' if target_type=='integer' else 'float'
        cast=int if operand_type=='integer' else float
        values=events['operand_values'][i,0].tolist()
        operands=[ValueRegister(str(j),cast(values[j]),operand_type) for j in range(1 if op=='neg' else 2)]
        result=ProtectedSession(operands).execute([Candidate('c',op,tuple(x.id for x in operands))])[0]
        assert result.status=='executed'
        assert result.value==float(events['values'][i,0])
        assert result.type==target_type
        seen.add(op)
    assert seen==set(PRIMITIVE_NAMES)


def test_zero_delay_still_requires_one_late_query_pass():
    model=ReturnRetentionModel(feature_dim=16,width=16)
    public=make_batch(41,3,feature_dim=16)['public']
    output=model(public,0,'protected')
    assert output['cell_passes']==1


def test_intervention_schedule_keeps_clean_curves_and_prespecified_control_delay():
    from tensegra.retention_study import evaluation_interventions
    config={'interventions':['none','event_drop','wrong_value'],'intervention_delays':[16]}
    assert evaluation_interventions(config,0)==['none']
    assert evaluation_interventions(config,32)==['none']
    assert evaluation_interventions(config,16)==config['interventions']
