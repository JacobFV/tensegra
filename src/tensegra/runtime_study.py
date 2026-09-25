"""Reproducible Stage 5 lowering, execution and lifting experiments.

Discrete runtime execution has no pathwise task gradient. Answer-supervised
lowering therefore uses a sampled score-function estimator, explicitly recorded
separately from supervised lowering and frozen-binding maintenance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
from torch.nn import functional as F

from .grounding_study import fingerprint, source_identity, state_hash, tensor_hash

VARIANTS = {
    'oracle': dict(mode='exact', oracle=True, beta=0., policy=False),
    'supervised': dict(mode='exact', beta=1., policy=False),
    'task_only': dict(mode='exact', beta=0., policy=True),
    'warm_weak': dict(mode='exact', beta=.1, policy=True, warm=True),
    'warm_task': dict(mode='exact', beta=0., policy=True, warm=True),
    'warm_frozen': dict(mode='exact', beta=0., policy=False, warm=True, frozen=True),
    'neural': dict(mode='neural', beta=1., policy=False),
    'graph_data': dict(mode='graph_data', beta=1., policy=False),
    'soft_structure': dict(mode='soft_structure', beta=1., policy=False),
    'protected_learned': dict(mode='protected_learned', beta=1., policy=False),
}


# Freeze the original ten defaults; subsequent retrieval-prior controls are a
# separately configured, explicitly post-main-launch supplement.
MAIN_VARIANTS = tuple(VARIANTS)
VARIANTS.update({
    'selector_graph_data': dict(mode='graph_data', beta=1., policy=False, selector_read=True),
    'selector_protected': dict(mode='protected_learned', beta=1., policy=False, selector_read=True),
    'oracle_selector_protected': dict(mode='protected_learned', beta=1., policy=False,
                                      selector_read=True, oracle_selector=True),
})


@dataclass
class RuntimeStudyConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    variants: list[str] = field(default_factory=lambda: list(MAIN_VARIANTS))
    steps: int = 400
    warmup_steps: int = 200
    checkpoints: list[int] = field(default_factory=lambda: [0, 25, 50, 100, 200, 300, 400])
    batch_size: int = 16
    eval_examples: int = 64
    max_train_depth: int = 4
    train_sizes: list[int] = field(default_factory=lambda: [8, 16])
    eval_depths: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    eval_sizes: list[int] = field(default_factory=lambda: [8, 32, 64])
    width: int = 32
    learning_rate: float = .001
    threads: int = 2
    wall_seconds: float = 1800.
    confidence_thresholds: list[float] = field(default_factory=lambda: [0., .05, .1, .2, .4, .6, .8, .95])
    reward_baseline_decay: float = .9
    policy_weight: float = 1.
    save_checkpoints: bool = True
    extra_evaluations: bool = True
    failure_examples: int = 3

    def __post_init__(self):
        if not self.variants or len(set(self.variants)) != len(self.variants) or set(self.variants) - VARIANTS.keys():
            raise ValueError('unknown, duplicate or empty variants')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be nonempty and unique')
        if self.steps < 0 or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError('invalid training/warmup steps')
        if any(x <= 0 for x in [self.batch_size, self.eval_examples, self.max_train_depth, self.width, *self.train_sizes, *self.eval_depths, *self.eval_sizes]):
            raise ValueError('sizes and depths must be positive')
        if self.batch_size > 32 or self.threads not in (1, 2):
            raise ValueError('CPU resource limits: batch <=32 and threads <=2')
        if not 0 <= self.reward_baseline_decay < 1:
            raise ValueError('invalid reward baseline decay')
        if any(not 0 <= threshold <= 1 for threshold in self.confidence_thresholds):
            raise ValueError('confidence thresholds must lie in [0,1]')
        if any(step < 0 or step > self.steps for step in self.checkpoints):
            raise ValueError('checkpoints outside training schedule')
        self.checkpoints = sorted(set([0, self.steps, *self.checkpoints]))


def curriculum(variant, step, config):
    """step indexes updates; warm experiments share the first warmup updates."""
    options = VARIANTS[variant]
    warm = options.get('warm', False) and step < config.warmup_steps
    return dict(beta=1. if warm else options['beta'],
                policy=False if warm else options['policy'],
                frozen=bool(options.get('frozen') and not warm))


def policy_loss(log_prob, rewards, baseline):
    """REINFORCE with a lagged scalar baseline, never a gold-action surrogate."""
    advantage = rewards.detach() - baseline
    return -(advantage * log_prob).mean()


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def conditions(config):
    rows = [dict(name=f'n{nodes}_d{depth}', nodes=nodes, depth=depth)
            for nodes in config.eval_sizes for depth in config.eval_depths]
    if config.extra_evaluations:
        for name, changes in [('renamed', {'names': 'heldout'}),
                              ('paraphrase', {'surface': 'heldout'}),
                              ('noise_medium', {'noise': .25}),
                              ('noise_high', {'noise': .6}),
                              ('ambiguous', {'ambiguous': True}),
                              ('invalid', {'invalid': True}),
                              ('permuted', {'intervention': 'permuted'}),
                              ('wrong_graph', {'intervention': 'wrong_graph'})]:
            rows.append(dict(name=name, nodes=config.train_sizes[0], depth=config.max_train_depth, **changes))
    return rows


def data_hash(batch):
    """Hash gold and observables, with gold never forwarded into the model."""
    return fingerprint({group: {name: tensor_hash(value) for name, value in sorted(batch[group].items())}
                        for group in ('public', 'gold')})


def build_model(config, seed, variant=None):
    from .runtime_model import RuntimeBindingModel
    from . import runtime_tasks as data
    torch.manual_seed(seed)
    return RuntimeBindingModel(vocab_size=data.VOCAB_SIZE, n_ops=len(data.OPS),
                               width=config.width, heads=2, n_relations=data.N_RELATIONS,
                               selector_read=bool(variant and VARIANTS[variant].get('selector_read')))


def forward_model(model, batch, variant):
    override = None
    if VARIANTS[variant].get('oracle_selector'):
        from .runtime_tasks import OPS
        # Oracle supplies semantic equivalence classes, not exact runtime-node
        # identities. Shadowed lexical bindings retain their ambiguity here.
        binding = batch['gold']['selector_mask'].float()
        binding = binding / binding.sum(-1, keepdim=True).clamp_min(1)
        override = (F.one_hot(batch['gold']['ops'], len(OPS)).float(), binding)
    return model(batch['public'], mode=_model_mode(variant), lowering_override=override)


def select_actions(prediction, mask, *, sample=False):
    distributions = [torch.distributions.Categorical(logits=prediction[key])
                     for key in ('op_logits', 'binding_logits')]
    actions = [dist.sample() if sample else dist.logits.argmax(-1) for dist in distributions]
    log_prob = sum(dist.log_prob(action) for dist, action in zip(distributions, actions))
    # The score of the complete sampled program, not a mean that changes with depth.
    return *actions, (log_prob * mask).sum(-1)


def lowering_loss(prediction, gold, mask):
    logits = prediction['op_logits']
    op = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gold['ops'].reshape(-1), reduction='none').reshape_as(mask)
    binding_logp = prediction['binding_logits'].log_softmax(-1)
    equivalent = gold['selector_mask'].bool()
    if equivalent.shape[-1] + 1 == binding_logp.shape[-1]:
        equivalent = F.pad(equivalent, (0, 1))
    null_target = gold['selectors'].eq(binding_logp.shape[-1] - 1)
    equivalent = equivalent.clone()
    equivalent[..., -1] |= null_target
    # Padded steps can have no admissible selector; make their unused loss finite.
    equivalent[..., -1] |= ~mask.bool()
    binding = -binding_logp.masked_fill(~equivalent, float('-inf')).logsumexp(-1)
    return ((op + binding) * mask).sum() / mask.sum().clamp_min(1)


def numeric_results(executions, device):
    return torch.tensor([e.result if e.result is not None else 0. for e in executions], device=device, dtype=torch.float)


def configure_frozen_lowerer(model, frozen):
    # Exact-mode output lifting uses these parameters only. No frozen-boundary
    # experiment is described as task-supervised acquisition.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not frozen or name.startswith(('lifter.', 'style.')))


def _model_mode(variant):
    return {'exact': 'runtime', 'neural': 'none', 'soft_structure': 'soft'}.get(
        VARIANTS[variant]['mode'], VARIANTS[variant]['mode'])


def train_update(model, optimizer, batch, variant, phase, config, baseline):
    from .runtime_execution import execute_batch
    options = VARIANTS[variant]
    configure_frozen_lowerer(model, phase['frozen'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tick = time.perf_counter()
    prediction = forward_model(model, batch, variant)
    mask, gold = batch['public']['step_mask'], batch['gold']
    binding = lowering_loss(prediction, gold, mask)
    symbolic_seconds = 0.
    policy = binding * 0.
    reward = None
    if options['mode'] == 'exact':
        ops, selectors, log_prob = select_actions(prediction, mask, sample=phase['policy'])
        if options.get('oracle'):
            ops, selectors = gold['ops'], gold['selectors']
        exact_tick = time.perf_counter()
        executions = execute_batch(batch['tasks'], ops, selectors)
        symbolic_seconds = sum(e.symbolic_seconds for e in executions)
        result = numeric_results(executions, mask.device)
        output = model.lift(result, batch['public']['style'], batch['public']['comparison'])
        task = F.cross_entropy(output, gold['output'])
        if phase['policy']:
            # Reward is the END-TASK answer after learned lifting, with invalid
            # execution rejected. No intermediate/gold-binding reward is supplied.
            reward = ((output.detach().argmax(-1) == gold['output']) &
                      torch.tensor([e.valid and not e.deferred for e in executions], device=mask.device)).float()
            policy = policy_loss(log_prob, reward, baseline)
    else:
        task = F.cross_entropy(prediction['output_logits'], gold['output'])
        # Result supervision is a declared shared execution target for neural
        # controls; never require a baseline to learn only the easier lifted label.
        from .runtime_tasks import RESULT_MIN
        result_target = gold['result'].long() - RESULT_MIN
        task = task + F.cross_entropy(prediction['result_logits'], result_target)
    loss = task + phase['beta'] * binding + config.policy_weight * policy
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite Stage 5 loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if reward is not None:
        baseline = config.reward_baseline_decay * baseline + (1 - config.reward_baseline_decay) * float(reward.mean())
    return dict(task=float(task.detach()), binding=float(binding.detach()), policy=float(policy.detach()),
                total=float(loss.detach()), reward=None if reward is None else float(reward.mean()),
                reward_baseline=baseline, beta=phase['beta'], frozen=phase['frozen'],
                symbolic_seconds=symbolic_seconds, neural_and_diagnostic_seconds=time.perf_counter()-tick-symbolic_seconds), baseline


def make_data(config, seed, *, depth, nodes, count=None, **settings):
    from .runtime_tasks import make_batch
    # Evaluation-only interventions act on execution, not on the paired sample.
    if settings.get('intervention') == 'wrong_graph':
        settings['corruption'] = .5
    settings = {k: v for k, v in settings.items() if k not in ('name', 'intervention', 'names')}
    if 'surface' in settings:
        settings['template'] = 'heldout'
        settings.pop('surface')
    return make_batch(batch_size=count or config.eval_examples, nodes=nodes, depth=depth,
                      seed=seed, family='all', **settings)


@torch.no_grad()
def evaluate(model, variant, config, condition, seed):
    from .runtime_execution import execute_batch, semantic_selector
    from .runtime_tasks import RESULT_MIN
    model.eval()
    batch = make_data(config, seed, **condition)
    public, gold = batch['public'], batch['gold']
    mask = public['step_mask'].bool()
    tick = time.perf_counter()
    prediction = forward_model(model, batch, variant)
    neural_seconds = time.perf_counter() - tick
    ops, selectors, _ = select_actions(prediction, mask)
    if VARIANTS[variant].get('oracle') or VARIANTS[variant].get('oracle_selector'):
        ops, selectors = gold['ops'], gold['selectors']
    equivalent = gold['selector_mask'].bool()
    if equivalent.shape[-1] + 1 == prediction['binding_logits'].shape[-1]:
        equivalent = F.pad(equivalent, (0, 1))
    equivalent = equivalent.clone()
    equivalent[..., -1] |= gold['selectors'].eq(equivalent.shape[-1]-1)
    selector_correct = equivalent.gather(-1, selectors[..., None]).squeeze(-1)
    op_correct = ops.eq(gold['ops'])
    all_binding = (selector_correct | ~mask).all(-1)
    all_primitive = (op_correct | ~mask).all(-1)
    all_lowering = all_binding & all_primitive
    intervention = condition.get('intervention')
    exact_tick = time.perf_counter()
    executions = execute_batch(batch['tasks'], ops, selectors,
                               permuted=intervention == 'permuted', wrong_graph=intervention == 'wrong_graph')
    symbolic_seconds = sum(e.symbolic_seconds for e in executions)
    tick = time.perf_counter()
    exact_results = numeric_results(executions, mask.device)
    exact_valid = torch.tensor([e.valid and not e.deferred for e in executions], dtype=torch.bool)
    exact_correct = exact_valid & exact_results.eq(gold['result'])
    if VARIANTS[variant]['mode'] == 'exact':
        result_correct = exact_correct
        output = model.lift(exact_results, public['style'], public['comparison'])
        valid = exact_valid
    else:
        result_correct = (prediction['result_logits'].argmax(-1) + RESULT_MIN).eq(gold['result'])
        output = prediction['output_logits']
        valid = torch.ones_like(result_correct)
    task_correct = output.argmax(-1).eq(gold['output']) & valid
    # Isolate lifting with the CORRECT result, explicitly an oracle-lifting audit.
    oracle_lift = model.lift(gold['result'].float(), public['style'], public['comparison']).argmax(-1).eq(gold['output'])
    neural_seconds += time.perf_counter() - tick
    counts = dict(examples=int(mask.shape[0]), steps=int(mask.sum()),
                  binding_correct=int((selector_correct & mask).sum()),
                  primitive_correct=int((op_correct & mask).sum()),
                  all_binding_correct=int(all_binding.sum()), all_primitive_correct=int(all_primitive.sum()),
                  all_lowering_correct=int(all_lowering.sum()), valid=int(valid.sum()),
                  execution_correct=int(result_correct.sum()), task_correct=int(task_correct.sum()),
                  task_given_binding_correct=int((task_correct & all_binding).sum()),
                  execution_given_lowering_correct=int((result_correct & all_lowering).sum()),
                  task_given_execution_correct=int((task_correct & result_correct).sum()),
                  oracle_lift_correct=int(oracle_lift.sum()),
                  complete_trajectory=int(sum(e.complete for e in executions)))
    risk = []
    if VARIANTS[variant]['mode'] == 'exact':
        for threshold in config.confidence_thresholds:
            gated = execute_batch(batch['tasks'], ops, selectors, confidences=torch.ones_like(prediction['confidences']) if VARIANTS[variant].get('oracle') else prediction['confidences'], threshold=threshold,
                                  permuted=intervention == 'permuted', wrong_graph=intervention == 'wrong_graph')
            answered = torch.tensor([e.valid and not e.deferred for e in gated], dtype=torch.bool)
            defined = gold.get('expected_valid', torch.ones_like(answered)).bool()
            right = numeric_results(gated, mask.device).eq(gold['result']) & answered & defined
            local_gate = (torch.ones_like(prediction['confidences']) if VARIANTS[variant].get('oracle') else prediction['confidences']).ge(threshold) & mask
            local_right = selector_correct & op_correct & mask
            def actual_lowering_correct(i, entry):
                task = batch['tasks'][i]
                gold_op, gold_selector = task.gold_actions[entry['step']]
                selected = semantic_selector(task, entry['selector'])
                return entry['op'] == gold_op and selected is not None and selected == semantic_selector(task, gold_selector)
            risk.append(dict(threshold=threshold, examples=len(gated), defined_examples=int(defined.sum()), answered=int(answered.sum()), correct=int(right.sum()),
                             invoked=sum(e.invoked for e in gated), deferred=sum(e.deferred for e in gated),
                             rejected=sum(e.rejected for e in gated),
                             accepted_wrong=sum(entry['status']=='executed' and not actual_lowering_correct(i, entry) for i,e in enumerate(gated) for entry in e.trace),
                             accepted=sum(e.invoked for e in gated),
                             hypothetical_gate_wrong=int((local_gate & ~local_right).sum()),
                             hypothetical_gate_count=int(local_gate.sum()),
                             deferred_correct=sum(entry['status']=='deferred' and actual_lowering_correct(i, entry) for i,e in enumerate(gated) for entry in e.trace),
                             hypothetical_deferred_correct=int((~local_gate & local_right).sum()),
                             correct_lowerings=int(local_right.sum())))
    counts.update(invoked=sum(e.invoked for e in executions), rejected=sum(e.rejected for e in executions),
                  deferred=sum(e.deferred for e in executions), recovery_steps=sum(e.recovering_steps for e in executions),
                  semantic_failure=int((~all_binding).sum()), primitive_failure=int((~all_primitive).sum()),
                  schema_failure=sum(bool(e.rejected) for e in executions),
                  execution_failure_after_correct_lowering=int((all_lowering & ~exact_correct).sum()),
                  lifting_failure_after_correct_execution=int((result_correct & ~task_correct).sum()))
    rates = dict(binding_accuracy=rate(counts['binding_correct'], counts['steps']),
                 primitive_accuracy=rate(counts['primitive_correct'], counts['steps']),
                 execution_accuracy=rate(counts['execution_correct'], counts['examples']),
                 task_accuracy=rate(counts['task_correct'], counts['examples']),
                 complete_trajectory_accuracy=rate(counts['complete_trajectory'], counts['examples']),
                 task_given_binding_correct=rate(counts['task_given_binding_correct'], counts['all_binding_correct']),
                 execution_given_lowering_correct=rate(counts['execution_given_lowering_correct'], counts['all_lowering_correct']),
                 task_given_execution_correct=rate(counts['task_given_execution_correct'], counts['execution_correct']),
                 oracle_lifting_accuracy=rate(counts['oracle_lift_correct'], counts['examples']))
    expected_valid = gold.get('expected_valid', torch.ones_like(task_correct)).bool()
    valid_examples = int(expected_valid.sum())
    counts['defined_examples'] = valid_examples
    counts['appropriate_abstention'] = int((~valid & ~expected_valid).sum())
    counts['undefined_examples'] = int((~expected_valid).sum())
    counts['task_correct'] = int((task_correct & expected_valid).sum())
    counts['execution_correct'] = int((result_correct & expected_valid).sum())
    rates['task_accuracy'] = rate(counts['task_correct'], valid_examples)
    rates['execution_accuracy'] = rate(counts['execution_correct'], valid_examples)
    rates['result_accuracy'] = rates['execution_accuracy']
    rates['lowering_audit_complete_trajectory_accuracy'] = rates['complete_trajectory_accuracy']
    counts['lowering_audit_complete_trajectory'] = counts['complete_trajectory']
    if VARIANTS[variant]['mode'] != 'exact':
        rates['execution_accuracy'] = None
        rates['complete_trajectory_accuracy'] = None
        rates['execution_given_lowering_correct'] = None
        rates['oracle_lifting_accuracy'] = None
        counts['oracle_lift_correct'] = None
        counts['complete_trajectory'] = None
    if valid_examples == 0:
        for key in ('task_given_binding_correct', 'execution_given_lowering_correct',
                    'task_given_execution_correct', 'oracle_lifting_accuracy',
                    'complete_trajectory_accuracy'):
            rates[key] = None
    failures = []
    for i in range(len(executions)):
        if not bool(task_correct[i]) and len(failures) < config.failure_examples:
            failures.append(dict(index=i, defined=bool(expected_valid[i]), gold_result=int(gold['result'][i]), actual_result=executions[i].result,
                                 gold_output=int(gold['output'][i]), actual_output=int(output[i].argmax()),
                                 ops=ops[i][mask[i]].tolist(), selectors=selectors[i][mask[i]].tolist(),
                                 gold_ops=gold['ops'][i][mask[i]].tolist(),
                                 binding_correct=bool(all_binding[i]), primitive_correct=bool(all_primitive[i]),
                                 valid=bool(valid[i]), execution_correct=bool(result_correct[i]),
                                 surface=list(batch['tasks'][i].surface), trace=executions[i].trace))
    family_breakdown = {}
    for family in sorted({task.family for task in batch['tasks']}):
        selected = torch.tensor([task.family == family for task in batch['tasks']])
        defined = selected & expected_valid
        family_breakdown[family] = dict(examples=int(selected.sum()), defined_examples=int(defined.sum()),
            task_correct=int((task_correct & defined).sum()), result_correct=int((result_correct & defined).sum()),
            lowering_correct=int((all_lowering & selected).sum()),
            complete_trajectory=None if VARIANTS[variant]['mode'] != 'exact' else sum(e.complete for e,t in zip(executions,batch['tasks']) if t.family==family))
    style_breakdown = {}
    for style, name in enumerate(('numeric_report', 'comparison', 'sign')):
        selected = public['style'].eq(style)
        defined = selected & expected_valid
        denominator = int(defined.sum())
        task_count = int((task_correct & defined).sum())
        lifted_count = int((oracle_lift & defined).sum()) if VARIANTS[variant]['mode'] == 'exact' else None
        style_breakdown[name] = dict(examples=int(selected.sum()), defined_examples=denominator,
            task_correct=task_count, task_accuracy=rate(task_count, denominator),
            oracle_lift_correct=lifted_count,
            oracle_lifting_accuracy=None if lifted_count is None else rate(lifted_count, denominator))
    runtime_sizes = public['candidate_mask'].sum(-1).tolist()
    clause_counts = mask.sum(-1).tolist()
    return dict(condition=condition['name'], settings=condition, data_hash=data_hash(batch), counts=counts,
                **rates, confidence=risk, failures=failures, family_breakdown=family_breakdown, style_breakdown=style_breakdown,
                runtime_nodes=dict(min=min(runtime_sizes), max=max(runtime_sizes), mean=sum(runtime_sizes)/len(runtime_sizes)),
                neural_seconds=neural_seconds,
                symbolic_seconds=symbolic_seconds,
                per_example=dict(task=task_correct.tolist(), execution=result_correct.tolist(),
                                 binding=all_binding.tolist(), primitive=all_primitive.tolist(),
                                 runtime_node_count=runtime_sizes, clause_count=clause_counts,
                                 lowering=all_lowering.tolist(),
                                 confidence=(prediction['confidences'] * mask).sum(-1).div(mask.sum(-1).clamp_min(1)).tolist()))


def train_run(config, variant, seed, source, checkpoint_dir=None):
    model = build_model(config, seed, variant)
    initial_hashes = {name: tensor_hash(value) for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    curve, schedule, losses = [], [], []
    start = time.perf_counter()
    validation = dict(name='validation', nodes=config.train_sizes[0], depth=config.max_train_depth)
    grid = conditions(config)
    initial = [evaluate(model, variant, config, c, 40_000_000 + seed*100_000)
               for c in [validation, dict(name='joint_ood', nodes=max(config.eval_sizes), depth=max(config.eval_depths))]]
    baseline = 0.
    for step in range(config.steps + 1):
        if step in config.checkpoints:
            score = evaluate(model, variant, config, validation, 20_000_000 + seed*100_000)
            curve.append(dict(step=step, diagnostics=score, losses=losses[-1] if losses else None))
        if step == config.steps:
            break
        if time.perf_counter()-start > config.wall_seconds:
            raise TimeoutError(f'{variant}/{seed} exceeded resource budget')
        batch_seed = 1_000_000 + seed*100_000 + step
        depth = 1 + step % config.max_train_depth
        nodes = config.train_sizes[(step // config.max_train_depth) % len(config.train_sizes)]
        batch = make_data(config, batch_seed, depth=depth, nodes=nodes, count=config.batch_size,
                          template='canonical' if (step // config.max_train_depth) % 2 == 0 else 'paraphrase')
        schedule.append(dict(seed=batch_seed, depth=depth, nodes=nodes, data_hash=data_hash(batch)))
        loss, baseline = train_update(model, optimizer, batch, variant, curriculum(variant, step, config), config, baseline)
        losses.append(dict(step=step+1, **loss))
    evaluations = [evaluate(model, variant, config, c, 40_000_000 + seed*100_000) for c in grid]
    checkpoint = None
    if config.save_checkpoints and checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = f'{variant}-seed{seed}.pt'
        torch.save(dict(state_dict=model.state_dict(), config=asdict(config), variant=variant, seed=seed, source=source),
                   Path(checkpoint_dir)/checkpoint)
    return dict(schema_version=5, variant=variant, seed=seed, config_hash=fingerprint(asdict(config)), source=source,
                variant_options=VARIANTS[variant], metric_semantics=dict(execution_counts='actual runtime' if VARIANTS[variant]['mode']=='exact' else 'numeric prediction; invocation counts are auxiliary-lowering audit'), model=dict(parameters=sum(p.numel() for p in model.parameters())),
                training=dict(steps=config.steps, examples=config.steps*config.batch_size, curve=curve, losses=losses,
                              schedule_hash=fingerprint(schedule), initial_parameter_hashes=initial_hashes,
                              initial_state_hash=fingerprint(initial_hashes), final_state_hash=state_hash(model)),
                resources=dict(seconds=time.perf_counter()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                               neural_and_diagnostic_train_seconds=sum(x['neural_and_diagnostic_seconds'] for x in losses),
                               symbolic_train_seconds=sum(x['symbolic_seconds'] for x in losses),
                               threads=config.threads, torch_version=torch.__version__, device='cpu'),
                checkpoint=checkpoint, initial_evaluations=initial, evaluations=evaluations)


def run(config, output):
    torch.set_num_threads(config.threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash, source = fingerprint(asdict(config)), source_identity()
    config_path = output/'config.json'
    if config_path.exists() and fingerprint(json.loads(config_path.read_text())) != config_hash:
        raise ValueError('output contains a different configuration')
    config_path.write_text(json.dumps(asdict(config), indent=2)+'\n')
    path = output/'metrics.jsonl'
    completed = set()
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['config_hash'] != config_hash or row['source']['files'] != source['files']:
                    raise ValueError('cannot resume with different source/config')
                key = row['variant'], row['seed']
                if key in completed or key[0] not in config.variants or key[1] not in config.seeds:
                    raise ValueError('duplicate/unexpected run')
                completed.add(key)
    for seed in config.seeds:
        for variant in config.variants:
            if (variant, seed) in completed:
                continue
            print(json.dumps(dict(event='start', variant=variant, seed=seed)), flush=True)
            row = train_run(config, variant, seed, source, output/'checkpoints')
            with path.open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush()
            print(json.dumps(dict(event='complete', variant=variant, seed=seed, seconds=row['resources']['seconds'],
                                  validation=row['training']['curve'][-1]['diagnostics']['task_accuracy'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(RuntimeStudyConfig(**json.loads(Path(args.config).read_text())), args.output)


if __name__ == '__main__':
    main()
