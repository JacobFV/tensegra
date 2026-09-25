"""Stage 6 paired experiments: privileged training versus autonomous evaluation.

The adapter hashes public strings into features. No target graph enters the actor.
Discrete proposals have no pathwise gradient; cold task-only is intentionally a
hard acquisition control, rather than disguised teacher-forced execution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
import subprocess
import platform
import torch
from torch.nn import functional as F

AUXILIARIES = ('grounding', 'topology', 'transition', 'readiness')
VARIANTS = {name: {} for name in ('local', 'global', 'fixed', 'fixed_compute', 'runtime_off',
    'event_drop', 'event_shuffle', 'event_wrong_value', 'public_graph', 'neural_fixed', 'neural_recurrent',
    'task_only_cold', 'task_only_warm', 'anneal_all', 'protected_learned', 'no_structure',
    *(f'anneal_{name}' for name in AUXILIARIES))}

@dataclass
class ThinkingStudyConfig:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    variants: list[str] = field(default_factory=lambda: ['local', 'global', 'fixed', 'runtime_off', 'anneal_all', 'task_only_cold', 'task_only_warm'])
    steps: int = 24
    warmup_steps: int = 12
    checkpoints: list[int] = field(default_factory=lambda: [0, 12, 24])
    batch_size: int = 4
    width: int = 32
    feature_dim: int = 32
    train_depth: int = 2
    eval_depths: list[int] = field(default_factory=lambda: [2, 8, 16, 32])
    eval_examples: int = 8
    distractors: int = 2
    threads: int = 2
    learning_rate: float = .001
    min_microsteps: int = 2
    max_microsteps: int = 12
    eval_max_microsteps: int = 80
    readiness_threshold: float = .5
    ponder_weight: float = .001
    emit_weight: float = .1
    wall_seconds: float = 1800.
    save_checkpoints: bool = True
    extra_evaluations: bool = True
    evaluation_schedule: str = 'economy_v1'

    def __post_init__(self):
        if self.threads not in (1, 2) or not 1 <= self.batch_size <= 32:
            raise ValueError('CPU limits: threads <=2, batch <=32')
        if self.steps < 0 or not 0 <= self.warmup_steps <= max(self.steps, 12):
            raise ValueError('invalid training schedule')
        if not self.variants or set(self.variants) - VARIANTS.keys():
            raise ValueError('unknown variants')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be unique and nonempty')
        if any(x <= 0 for x in [self.width,self.feature_dim,self.train_depth,self.eval_examples,*self.eval_depths]):
            raise ValueError('positive dimensions required')
        if self.min_microsteps < 1 or self.max_microsteps < self.min_microsteps + 1:
            raise ValueError('microstep limits must allow a post-event update')
        self.checkpoints = sorted(set([0,self.steps,*[s for s in self.checkpoints if 0 <= s <= self.steps]]))


def auxiliary_weights(variant, step, config):
    weights = dict.fromkeys(AUXILIARIES, 1.)
    if variant in ('neural_fixed','neural_recurrent','task_only_cold') or (variant == 'task_only_warm' and step >= config.warmup_steps):
        return dict.fromkeys(AUXILIARIES, 0.)
    for name in weights:
        if variant in ('anneal_all', 'anneal_' + name):
            weights[name] = min(1.,max(0.,1. - (step-.25*config.steps)/max(1.,.5*config.steps)))
    return weights


def halting_weights(probabilities, allowed):
    """Differentiable hazard mass; the final pass absorbs all survival mass."""
    survival = probabilities.new_ones(())
    weights = []
    for index, probability in enumerate(probabilities):
        hazard = probability * allowed[index].to(probability.dtype)
        if index == len(probabilities)-1:
            weights.append(survival)
        else:
            weights.append(survival*hazard)
            survival = survival*(1-hazard)
    return torch.stack(weights)


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def features(text, width):
    """Fixed public token/identity features, independent of episode labels."""
    out = torch.zeros(width)
    for token in str(text).split():
        raw = hashlib.sha256(token.encode()).digest()
        for j in range(4):
            out[int.from_bytes(raw[2*j:2*j+2], 'little') % width] += 1 if raw[16+j] & 1 else -1
    return out / out.norm().clamp_min(1)


def source_manifest():
    root = Path(__file__).resolve().parents[2]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src','configs') for p in sorted((root/folder).rglob('*')) if p.is_file() and '__pycache__' not in str(p)}


def build_model(config, seed):
    from .thinking import ThinkingConfig, ThinkingModel
    torch.manual_seed(seed)
    return ThinkingModel(ThinkingConfig(feature_dim=config.feature_dim,width=config.width,
        output_classes=131,min_microsteps=config.min_microsteps,max_microsteps=config.max_microsteps))


def actor_inputs(public, session, microstep, config, *, complete_evidence=False):
    """Only physically public observations cross this boundary."""
    frame_index = len(public.frames)-1 if complete_evidence else min(microstep - 1, len(public.frames)-1)
    tokens = list(public.frames[frame_index].tokens)
    if frame_index == len(public.frames)-1:
        tokens += list(public.context_tokens)
    context = torch.stack([features(str(i) + ':' + t,config.feature_dim) + features(t,config.feature_dim) for i,t in enumerate(tokens or ['empty'])])[None]
    ids = list(public.register_ids)
    rows = []
    for key in ids:
        row = features(key,config.feature_dim)
        if key in session.registers:
            register = session.registers[key]
            row = row.clone()
            if key in {value.id for value in public.initial_values}:
                row[0] += float(register.value) / 64
            row[1] += 1
            row[2] += {'integer':0.,'float':.5,'boolean':1.}.get(register.type,0.)
        rows.append(row)
    return context, torch.stack(rows)[None], ids, frame_index


def _pick(logits, sample, log_probs):
    distribution = torch.distributions.Categorical(logits=logits)
    choice = distribution.sample() if sample else logits.argmax(-1)
    if sample:
        log_probs.append(distribution.log_prob(choice).sum())
    return choice


def predicted_proposals(prediction, ids, operation_ids, config, variant, *, sample=False):
    from .thinking_runtime import Candidate, PRIMITIVE_NAMES
    log_probs = []
    ops = _pick(prediction['op_logits'][0],sample,log_probs)
    args = _pick(prediction['binding_logits'][0],sample,log_probs)
    identity_logits = prediction['candidate_id_logits'][0]
    allowed = torch.tensor([key.removeprefix('result:') in operation_ids for key in ids])
    identity_logits = identity_logits.masked_fill(~allowed, -1e9)
    identities = _pick(identity_logits,sample,log_probs)
    readiness_logits = prediction['readiness_logits'][0]
    if variant == 'global':
        readiness_logits = readiness_logits.mean().expand_as(readiness_logits)
    if sample:
        gate_dist = torch.distributions.Bernoulli(logits=readiness_logits[:1] if variant=='global' else readiness_logits)
        draw = gate_dist.sample()
        log_probs.append(gate_dist.log_prob(draw).sum())
        gates = draw.expand_as(readiness_logits)
    else:
        gates = readiness_logits.sigmoid()
    proposals = []
    for k in range(len(ops)):
        primitive = PRIMITIVE_NAMES[int(ops[k])]
        arguments = tuple(ids[int(a)] if int(a) < len(ids) else '__null__' for a in args[k][:1 if primitive == 'neg' else 2])
        proposals.append(Candidate(ids[int(identities[k])].removeprefix('result:'),primitive,arguments,float(gates[k].detach())))
    return tuple(proposals), log_probs


def inject_runtime_events(model, workspace, prediction, events, ids, config, variant):
    """Exact values enter through learned events and candidate routes only."""
    from .thinking_runtime import PRIMITIVE_NAMES
    k = prediction['routes'].shape[1]
    successful = [(i,e) for i,e in enumerate(events[:k]) if e.status == 'executed']
    if variant == 'event_drop' or not successful:
        return workspace, workspace.sum()*0, 0, []
    values = torch.zeros(1,k); types = torch.zeros(1,k,dtype=torch.long)
    operations = torch.zeros(1,k,dtype=torch.long)
    arguments = torch.zeros(1,k,2,config.feature_dim)
    provenance = torch.zeros(1,k,config.feature_dim)
    mask = torch.zeros(1,k,dtype=torch.bool)
    for i,event in successful:
        values[0,i] = float(event.value) + (3. if variant == 'event_wrong_value' else 0.)
        types[0,i] = {'integer':0,'float':1,'boolean':2}[event.type]
        operations[0,i] = PRIMITIVE_NAMES.index(event.primitive)
        for j,arg in enumerate(event.arguments[:2]):
            arguments[0,i,j] = features(arg,config.feature_dim)
        provenance[0,i] = features(' '.join(map(str,event.provenance)),config.feature_dim)
        mask[0,i] = True
    encoded = model.encode_events(dict(values=values,types=types,operations=operations,arguments=arguments,provenance=provenance))
    routes = prediction['routes'].flip(1) if variant == 'event_shuffle' else prediction['routes']
    decoded = model.event_roundtrip(encoded)
    roundtrip_loss = F.mse_loss(decoded['value'][mask]/64,values[mask]/64) + F.cross_entropy(decoded['semantic_logits'][mask],types[mask]) + F.cross_entropy(decoded['op_logits'][mask],operations[mask])
    audit = [dict(stage='encoder',value_error=float((decoded['value'][0,i]-values[0,i]).detach().abs()),type_correct=bool(decoded['semantic_logits'][0,i].argmax()==types[0,i]),operation_correct=bool(decoded['op_logits'][0,i].argmax()==operations[0,i])) for i,_ in successful]
    return model.inject_events(workspace,encoded,routes,mask), roundtrip_loss, len(successful), audit


def auxiliary_losses(prediction, targets, ids, config):
    """Targets are consumed only after the public actor forward."""
    from .thinking_runtime import PRIMITIVE_NAMES
    zero = prediction['workspace'].sum()*0
    losses = dict.fromkeys(AUXILIARIES,zero)
    count = min(len(targets),prediction['op_logits'].shape[1])
    if not count:
        return losses
    transition = []; readiness = []; grounding = []
    graph = torch.zeros_like(prediction['predicted_adjacency'])
    for k,target in enumerate(targets[:count]):
        candidate = target.candidate
        transition.append(F.cross_entropy(prediction['op_logits'][:,k],torch.tensor([PRIMITIVE_NAMES.index(candidate.primitive)])))
        identity = 'result:' + candidate.id
        if identity in ids:
            transition.append(F.cross_entropy(prediction['candidate_id_logits'][:,k],torch.tensor([ids.index(identity)])))
        for j,arg in enumerate(candidate.arguments):
            index = ids.index(arg) if arg in ids else len(ids)
            transition.append(F.cross_entropy(prediction['binding_logits'][:,k,j],torch.tensor([index])))
            if arg in ids:
                if identity in ids:
                    graph[(0,j,ids.index(identity),index) if graph.ndim==4 else (0,ids.index(identity),index)] = 1
        valid = sorted({ids.index(arg) for arg in candidate.arguments if arg in ids})
        if valid:
            for role in ('grounding_q','grounding_k'):
                pooled = (prediction[role] * prediction['routes'][:,k,None,:,None]).sum(-2)
                grounding.append(-pooled[...,valid].sum(-1).clamp_min(1e-8).log().mean())
        if any(arg not in ids for arg in candidate.arguments):
            for role in ('grounding_q','grounding_k'):
                pooled = (prediction[role] * prediction['routes'][:,k,None,:,None]).sum(-2)
                grounding.append(-pooled[...,-1].clamp_min(1e-8).log().mean())
        readiness.append(F.binary_cross_entropy_with_logits(prediction['readiness_logits'][:,k],torch.tensor([target.readiness])))
    losses['transition'] = torch.stack(transition).mean()
    losses['readiness'] = torch.stack(readiness).mean()
    losses['grounding'] = torch.stack(grounding).mean() if grounding else zero
    losses['topology'] = F.binary_cross_entropy(prediction['predicted_adjacency'].clamp(1e-6,1-1e-6),graph)
    return losses


def rollout(model, public, config, variant='local', *, gold=None, teacher_forcing=False, sample=False, training_unroll=False):
    """Free evaluation accepts public records only; labels are optional train targets."""
    from .thinking_runtime import ProtectedSession
    if variant == 'public_graph':
        raise ValueError('public_graph requires an explicitly supplied public graph; controlled episodes do not supply one')
    session = ProtectedSession(public.initial_values)
    complete_evidence = variant in ('neural_fixed','neural_recurrent','fixed')
    context, memory, ids, _ = actor_inputs(public,session,1,config,complete_evidence=complete_evidence)
    workspace = model.initialize({'context':context})
    sums = {key:workspace.sum()*0 for key in (*AUXILIARIES,'graph','task','emit','ponder','event','policy','learned_transition')}
    neural_only = variant in ('neural_fixed','neural_recurrent')
    runtime_enabled = variant not in ('runtime_off','fixed','neural_fixed','neural_recurrent')
    trace = []; task_losses = []; hazards = []; emit_allowed = []; log_probs = []; injected = 0; last_event_step = -1; pending_events = None
    limit = 1 if variant in ('fixed','neural_fixed') else config.max_microsteps
    halted = False
    for step in range(1,limit+1):
        context,memory,ids,frame = actor_inputs(public,session,step,config,complete_evidence=complete_evidence)
        graph_controls = {}
        if variant == 'graph_permuted':
            graph_controls['graph_permutation'] = torch.tensor([sorted(range(len(ids)),key=lambda i:digest(ids[i]))])
        elif variant == 'graph_drop50':
            graph_controls['graph_keep'] = torch.tensor([[[int(digest(a+'|'+b)[0],16)%2 for b in ids] for a in ids]],dtype=torch.float)
        prediction = model.step(workspace,context,memory,microstep=step,structural_strength=0. if neural_only or variant=='no_structure' else None,**graph_controls)
        workspace = prediction['workspace']
        melted_audit = []
        if pending_events is not None:
            from .thinking_runtime import PRIMITIVE_NAMES
            prior_routes,prior_events = pending_events
            decoded = model.event_roundtrip(prior_routes @ workspace)
            for i,event in enumerate(prior_events):
                if event.status != 'executed': continue
                target_value = torch.tensor(float(event.value))
                target_type = {'integer':0,'float':1,'boolean':2}[event.type]
                target_op = PRIMITIVE_NAMES.index(event.primitive)
                sums['event'] = sums['event'] + F.mse_loss(decoded['value'][0,i]/64,target_value/64) + F.cross_entropy(decoded['semantic_logits'][:,i],torch.tensor([target_type])) + F.cross_entropy(decoded['op_logits'][:,i],torch.tensor([target_op]))
                melted_audit.append(dict(stage='post_recurrent_workspace',value_error=float((decoded['value'][0,i]-target_value).detach().abs()),type_correct=bool(decoded['semantic_logits'][0,i].argmax()==target_type),operation_correct=bool(decoded['op_logits'][0,i].argmax()==target_op)))
            pending_events = None
        if gold is not None:
            # Posterior supervision is training-only and is never an action mask.
            from .thinking_tasks import Episode, readiness_targets
            all_targets = readiness_targets(Episode(public,gold),frame,tuple(session.registers),context_visible=frame==len(public.frames)-1)
            target_step = min(max(0,step-len(public.frames)),len(gold.trace)-1)
            from .thinking_tasks import CandidateTarget
            targets = tuple(next((target for target in all_targets if target.candidate.id == candidate.id and target.candidate.primitive == candidate.primitive and target.candidate.arguments == candidate.arguments), CandidateTarget(candidate,0.)) for candidate in gold.trace[target_step])
            if len(targets)<prediction['op_logits'].shape[1]:
                null_target = next((t for t in all_targets if any(arg not in ids for arg in t.candidate.arguments)),None)
                if null_target is not None: targets = (*targets,null_target)
            losses = auxiliary_losses(prediction,targets,ids,config)
            for key,value in losses.items(): sums[key] = sums[key] + value
            full_graph = torch.zeros_like(prediction['predicted_adjacency'])
            for edge in gold.graph.edges:
                source = 'result:' + edge.source if 'result:' + edge.source in ids else edge.source
                if edge.role == 'argument' and source in ids and edge.target in ids:
                    full_graph[(0,edge.slot,ids.index(source),ids.index(edge.target)) if full_graph.ndim==4 else (0,ids.index(source),ids.index(edge.target))] = 1
            sums['graph'] = sums['graph'] + F.binary_cross_entropy(prediction['predicted_adjacency'].clamp(1e-6,1-1e-6),full_graph)
        proposals, scores = predicted_proposals(prediction,ids,public.operation_ids,config,variant,sample=sample)
        log_probs.extend(scores)
        if teacher_forcing and gold is not None and runtime_enabled:
            trace_index = step - len(public.frames)
            proposals = gold.trace[trace_index] if 0 <= trace_index < len(gold.trace) else ()
        events = (); event_audit = []
        available_before = tuple(session.registers)
        eligible_before_execution = step>=config.min_microsteps and step>last_event_step
        stop_before_execution = not training_unroll and not teacher_forcing and variant not in ('fixed','neural_fixed','fixed_compute') and eligible_before_execution and bool(prediction['emit'].reshape(-1)[0])
        # Reserve the last pass for mandatory recurrent processing of the latest event.
        if runtime_enabled and step < limit and not stop_before_execution:
            if variant=='protected_learned':
                predicted_values = model.event_roundtrip(prediction['routes'] @ workspace)['value'][0]
                overrides = {}
                expected = dict(gold.expected_values) if gold is not None else {}
                for k,proposal in enumerate(proposals):
                    value = float(predicted_values[k].detach().clamp(-64,64))
                    operands = [session.registers.get(arg) for arg in proposal.arguments]
                    overrides.setdefault(proposal.id,bool(value>.5) if proposal.primitive=='compare' else (value if any(r is not None and r.type=='float' for r in operands) else int(round(value))))
                    if 'result:'+proposal.id in expected:
                        sums['learned_transition'] = sums['learned_transition'] + F.mse_loss(predicted_values[k]/64,torch.tensor(float(expected['result:'+proposal.id]))/64)
                events = session.execute_learned(proposals,overrides,threshold=config.readiness_threshold)
            else:
                events = session.execute(proposals,threshold=config.readiness_threshold)
            workspace,event_loss,n,event_audit = inject_runtime_events(model,workspace,prediction,events,ids,config,variant)
            sums['event'] = sums['event'] + event_loss
            if any(event.status=='executed' for event in events):
                last_event_step = step
                pending_events = (prediction['routes'],events)
            injected += n
        ready_to_emit = step >= config.min_microsteps and step > last_event_step
        if gold is not None and runtime_enabled:
            emit_target = float('result:' + gold.trace[-1][0].id in session.registers and ready_to_emit)
            sums['emit'] = sums['emit'] + F.binary_cross_entropy_with_logits(prediction['emit_logits'].reshape(-1),torch.tensor([emit_target]))
        hazards.append(prediction['emit_probability'].mean())
        emit_allowed.append(eligible_before_execution and variant != 'fixed_compute')
        if gold is not None:
            task_losses.append(F.cross_entropy(prediction['output_logits'][:,:129],torch.tensor([int(gold.result)+64])) + F.cross_entropy(prediction['output_logits'][:,129:],torch.tensor([int(gold.answer)])))
        trace.append(dict(microstep=step,frame=frame,available_before=available_before,proposals=[asdict(p) for p in proposals],event_roundtrips=event_audit+melted_audit,events=[asdict(e) for e in events],
                          readiness=prediction['readiness_logits'].sigmoid().detach().tolist(),
                          emit_probability=float(prediction['emit_probability'].detach().mean()),
                          projection_overlap=float(prediction['overlap'].detach().mean()),grounding_entropy=float(-(prediction['grounding_q'].clamp_min(1e-9)*prediction['grounding_q'].clamp_min(1e-9).log()).sum(-1).detach().mean()),structural_null_mass=float(prediction['grounding_q'][...,-1].detach().mean()) if prediction['grounding_q'].shape[-1]>len(ids) else None,null_binding_mass=float(prediction['binding_logits'].softmax(-1)[...,-1].detach().mean())))
        if not training_unroll and not teacher_forcing and variant not in ('fixed','neural_fixed','fixed_compute') and ready_to_emit and bool(prediction['emit'].reshape(-1)[0]):
            halted = step < limit
            break
    logits = prediction['output_logits']
    numeric_prediction = int(logits[:,:129].argmax(-1))-64
    answer_prediction = int(logits[:,129:].argmax(-1))
    halt_weights = halting_weights(torch.stack(hazards),torch.tensor(emit_allowed))
    sums['ponder'] = (halt_weights * torch.arange(1,step+1,dtype=halt_weights.dtype)).sum()
    if gold is not None:
        sums['task'] = (halt_weights * torch.stack(task_losses)).sum()
    for key in (*AUXILIARIES,'graph','emit','event','learned_transition'): sums[key] = sums[key] / step
    output_register = session.registers.get(public.output_ids[0])
    semantic_result = None if output_register is None else output_register.value
    return dict(losses=sums,log_probs=log_probs,numeric_prediction=numeric_prediction,
        answer_prediction=answer_prediction,semantic_result=semantic_result,trace=trace,
        microsteps=step,injected_events=injected,halted=halted,
        halting_weights=halt_weights.detach().tolist(),expected_microsteps=float(sums['ponder'].detach()),max_forced=not halted and variant not in ('fixed','neural_fixed'),post_event_pass=step>last_event_step,
        workspace=workspace)


def train_update(model,optimizer,episodes,config,variant,step,baseline):
    model.train(); optimizer.zero_grad(set_to_none=True)
    weights = auxiliary_weights(variant,step,config)
    task_only = not any(weights.values())
    totals = {}; rewards = []
    for episode in episodes:
        result = rollout(model,episode.public,config,variant,gold=episode.gold,
                         teacher_forcing=not task_only,sample=task_only and not variant.startswith('neural_'),training_unroll=True)
        losses = result['losses']
        reward = float(result['semantic_result'] == episode.gold.result and result['answer_prediction'] == episode.gold.answer)
        rewards.append(reward)
        if task_only and result['log_probs']:
            losses['policy'] = -(reward-baseline)*torch.stack(result['log_probs']).sum()
        loss = losses['task'] + weights['transition']*losses['learned_transition'] + .1*weights['transition']*losses['event'] + weights['topology']*losses['graph'] + sum(weights[key]*losses[key] for key in AUXILIARIES)
        loss = loss + (0. if task_only else config.emit_weight)*losses['emit'] + config.ponder_weight*losses['ponder'] + losses['policy']
        (loss / len(episodes)).backward()
        for key,value in losses.items(): totals[key] = totals.get(key,0.) + float(value.detach())/len(episodes)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return totals, .9*baseline+.1*sum(rewards)/len(rewards)


@torch.no_grad()
def evaluate(model,episodes,config,variant,*,oracle_trace=False):
    started = time.perf_counter()
    model.eval(); examples = []; failures = {}; calibration = []
    for episode in episodes:
        result = rollout(model,episode.public,config,variant,gold=episode.gold if oracle_trace else None,teacher_forcing=oracle_trace)
        task = result['numeric_prediction'] == episode.gold.result and result['answer_prediction'] == episode.gold.answer
        exact = result['semantic_result'] == episode.gold.result
        from .thinking_tasks import readiness_targets
        trajectory_gold = [(c.id,c.primitive,c.arguments) for group in episode.gold.trace for c in group]
        executed = [(e['candidate_id'],e['primitive'],tuple(e['arguments'])) for entry in result['trace'] for e in entry['events'] if e['status']=='executed']
        for entry in result['trace']:
            posterior = readiness_targets(episode,entry['frame'],entry['available_before'],context_visible=entry['frame']==len(episode.public.frames)-1)
            for proposal in entry['proposals']:
                match = next((t for t in posterior if t.candidate.id==proposal['id'] and t.candidate.primitive==proposal['primitive'] and t.candidate.arguments==tuple(proposal['arguments'])),None)
                calibration.append(dict(probability=proposal['readiness'],target=0. if match is None else match.readiness))
        statuses = [event['status'] for entry in result['trace'] for event in entry['events']]
        row = {key:result[key] for key in ('numeric_prediction','answer_prediction','semantic_result','microsteps','injected_events','halted','max_forced','post_event_pass','trace')}
        comparison = next((event['value'] for entry in reversed(result['trace']) for event in entry['events'] if event['candidate_id']==episode.gold.trace[-1][0].id and event['status'] in ('executed','duplicate')),None)
        row.update(semantic_decision=comparison,semantic_decision_correct=comparison is not None and comparison==bool(episode.gold.answer),transition_set_exact=set(executed)==set(trajectory_gold),trajectory_exact=executed==trajectory_gold,task_correct=task,exact_semantic_correct=exact,gold_result=episode.gold.result,gold_answer=episode.gold.answer,
                   numeric_correct=result['numeric_prediction']==episode.gold.result,
                   decision_correct=result['answer_prediction']==episode.gold.answer,
                   input_hash=digest(asdict(episode.public)),data_hash=digest(asdict(episode)))
        examples.append(row)
        for status in statuses: failures[status] = failures.get(status,0)+1
    n = len(examples)
    return dict(execution_backend='learned' if variant=='protected_learned' else ('none' if variant in ('neural_fixed','neural_recurrent','runtime_off','fixed') else 'exact'),observation_protocol='complete_from_start' if variant in ('neural_fixed','neural_recurrent','fixed') else 'progressive_exogenous_context',evaluation_seconds=time.perf_counter()-started,examples=examples,counts=failures,n=n,readiness_calibration=calibration,readiness_brier=rate(sum((r['probability']-r['target'])**2 for r in calibration),len(calibration)),trajectory_exact_accuracy=rate(sum(e['trajectory_exact'] for e in examples),n),
        task_accuracy=rate(sum(e['task_correct'] for e in examples),n),
        exact_semantic_accuracy=rate(sum(e['exact_semantic_correct'] for e in examples),n),
        numeric_accuracy=rate(sum(e['numeric_correct'] for e in examples),n),
        decision_accuracy=rate(sum(e['decision_correct'] for e in examples),n),
        task_given_exact=rate(sum(e['task_correct'] and e['exact_semantic_correct'] for e in examples),sum(e['exact_semantic_correct'] for e in examples)),
        mean_microsteps=rate(sum(e['microsteps'] for e in examples),n))


def evaluation_conditions(config, step):
    depths = sorted(set(config.eval_depths))
    if step == config.steps:
        selected = depths
    elif step == 0:
        selected = sorted({depths[0],depths[-1]})
    else:
        selected = depths[:1]
    rows = [dict(depth=d,condition='depth',kwargs={}) for d in selected]
    if step == config.steps and config.extra_evaluations:
        rows += [dict(depth=max(2,config.train_depth),condition=name,kwargs=kwargs) for name,kwargs in [
            ('heldout_surface',dict(template='lexical')),('cross_motif',dict(motif='cross')),
            ('heldout_composition',dict(operator_composition='heldout')),('heldout_wordorder',dict(template='reordered')),
            ('distractors16',dict(distractors=16)),('distractors64',dict(distractors=64)),('ambiguity_delay',dict(delay=4))]]
    return rows


def evaluation_episode(config,seed,index,condition):
    from .thinking_tasks import generate_episode
    options = dict(condition['kwargs'])
    delay = options.pop('delay',0)
    distractors = options.pop('distractors',config.distractors)
    episode = generate_episode(seed=1_000_000+seed*10000+index,depth=condition['depth'],distractors=distractors,context=index%2,**options)
    if delay:
        episode = replace(episode,public=replace(episode.public,frames=(episode.public.frames[0],)*delay+episode.public.frames),gold=replace(episode.gold,hypotheses=(episode.gold.hypotheses[0],)*delay+episode.gold.hypotheses,readiness=(episode.gold.readiness[0],)*delay+episode.gold.readiness))
    return episode


def run(config,output):
    from .thinking_tasks import generate_episode
    torch.set_num_threads(config.threads)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest = source_manifest()
    try:
        git_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        git_commit = None
    run_manifest = dict(config=asdict(config),config_hash=digest(asdict(config)),sources=manifest,source_hash=digest(manifest),git_commit=git_commit,python=platform.python_version(),torch=torch.__version__,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),initializations={},checkpoints={})
    (output/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
    started = time.monotonic()
    def progress(kind,**fields):
        print(json.dumps(dict(kind=kind,elapsed_seconds=time.monotonic()-started,**fields)),flush=True)
    with (output/'metrics.jsonl').open('w') as stream:
        for seed in config.seeds:
            for variant in config.variants:
                progress('variant_start',seed=seed,variant=variant)
                model = build_model(config,seed)
                initial_hash = digest({k:v.detach().tolist() for k,v in model.state_dict().items()})
                run_manifest['initializations'][f'{variant}:seed{seed}'] = initial_hash
                optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
                baseline = 0.
                for step in range(config.steps+1):
                    if time.monotonic()-started > config.wall_seconds: raise TimeoutError('study wall budget exhausted')
                    if step in config.checkpoints:
                        progress('checkpoint_start',seed=seed,variant=variant,step=step)
                        checkpoint_hash = digest({k:v.detach().tolist() for k,v in model.state_dict().items()})
                        eval_conditions = evaluation_conditions(config,step)
                        for condition in eval_conditions:
                            depth = condition['depth']
                            eval_config = replace(config,max_microsteps=config.eval_max_microsteps)
                            previous_model_config = model.config
                            model.config = replace(model.config,max_microsteps=config.eval_max_microsteps)
                            episodes = [evaluation_episode(config,seed,i,condition) for i in range(config.eval_examples)]
                            row = dict(kind='evaluation',seed=seed,variant=variant,step=step,depth=depth,condition=condition['condition'],compute_cap=eval_config.max_microsteps,checkpoint_hash=checkpoint_hash,initial_hash=initial_hash,source_hash=digest(manifest),**evaluate(model,episodes,eval_config,variant))
                            stream.write(json.dumps(row)+'\n'); stream.flush()
                            if variant == 'local' and step==config.steps and condition['condition']=='depth' and depth in (min(config.eval_depths),max(config.eval_depths)):
                                for oracle_name in ('oracle_trace','oracle_minimal'):
                                    oracle_config = eval_config if oracle_name=='oracle_trace' else replace(eval_config,max_microsteps=len(episodes[0].public.frames)+len(episodes[0].gold.trace))
                                    oracle = dict(kind='evaluation',seed=seed,variant=variant,intervention=oracle_name,privilege='gold actions and supplied unroll; not free policy',step=step,depth=depth,condition=oracle_name,compute_cap=oracle_config.max_microsteps,checkpoint_hash=checkpoint_hash,initial_hash=initial_hash,source_hash=digest(manifest),**evaluate(model,episodes,oracle_config,variant,oracle_trace=True))
                                    stream.write(json.dumps(oracle)+'\n'); stream.flush()
                                for intervention in ('event_drop','event_shuffle','event_wrong_value','runtime_off','graph_permuted','graph_drop50','readiness_0.5','readiness_0.95'):
                                    intervention_config = replace(eval_config,readiness_threshold=float(intervention.split('_')[1])) if intervention.startswith('readiness_') else eval_config
                                    actor_variant = 'local' if intervention.startswith('readiness_') else intervention
                                    intervened = dict(kind='evaluation',seed=seed,variant=variant,intervention=intervention,step=step,depth=depth,condition='frozen_intervention',readiness_threshold=intervention_config.readiness_threshold,compute_cap=eval_config.max_microsteps,checkpoint_hash=checkpoint_hash,initial_hash=initial_hash,source_hash=digest(manifest),**evaluate(model,episodes,intervention_config,actor_variant))
                                    stream.write(json.dumps(intervened)+'\n'); stream.flush()
                            model.config = previous_model_config
                        if config.save_checkpoints:
                            checkpoint_path = output/f'{variant}-seed{seed}-step{step}.pt'
                            torch.save(model.state_dict(),checkpoint_path)
                            run_manifest['checkpoints'][checkpoint_path.name] = dict(state_hash=checkpoint_hash,file_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started)
                        run_manifest['elapsed_seconds'] = time.monotonic()-started
                        (output/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
                        progress('checkpoint_complete',seed=seed,variant=variant,step=step)
                    if step == config.steps:
                        progress('variant_complete',seed=seed,variant=variant,step=step)
                        break
                    episodes = [generate_episode(seed=seed*100000+step*config.batch_size+i,depth=1+step%config.train_depth,distractors=config.distractors,context=i%2,operator_composition='train') for i in range(config.batch_size)]
                    train_started = time.perf_counter()
                    losses,baseline = train_update(model,optimizer,episodes,config,variant,step,baseline)
                    row = dict(kind='training',seed=seed,variant=variant,step=step+1,losses=losses,objective='survival_weighted_task_ce',observation_protocol='complete_from_start' if variant in ('neural_fixed','neural_recurrent','fixed') else 'progressive_exogenous_context',auxiliary_teacher_forcing=any(auxiliary_weights(variant,step,config).values()),train_seconds=time.perf_counter()-train_started,auxiliary_weights=auxiliary_weights(variant,step,config),data_hash=digest([asdict(e) for e in episodes]),policy_baseline=baseline)
                    stream.write(json.dumps(row)+'\n'); stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(ThinkingStudyConfig(**json.loads(Path(args.config).read_text())),args.output)

if __name__ == '__main__': main()
