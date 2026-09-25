"""Outcome-free dense-MAC recurrence planner; not measured FLOPs or runtime."""
import math


def block_macs(state_rows, public_tokens, width):
    # Self qkv/output + cross q/output and k/v + 2-layer FF.
    # Cross memory includes the historical zero sentinel.
    s,t,w=state_rows,public_tokens+1,width
    return (10*s+2*t)*w*w+2*s*s*w+2*s*t*w


def costs(public_tokens, *, width=1024, workspace_rows=8, capacity=128):
    if type(public_tokens) is not int or public_tokens <= 0:
        raise ValueError('positive padded public token length required')
    one_step=4*block_macs(workspace_rows,public_tokens,width)
    read=(public_tokens+capacity)*width*width+capacity*public_tokens*width
    contextual=2*one_step+8*block_macs(public_tokens,public_tokens,width)+read
    return dict(original=2*one_step,context=contextual,workspace_per_step=one_step)


def plan(batch_max_lengths, *, batch_size=8, width=1024, workspace_rows=8, capacity=128):
    if not batch_max_lengths or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('nonempty fixed complete-batch schedule required')
    rows=[costs(n,width=width,workspace_rows=workspace_rows,capacity=capacity) for n in batch_max_lengths]
    per_step=sum(r['workspace_per_step']*batch_size for r in rows)
    target=sum(r['context']*batch_size for r in rows)
    lo=max(2,target//per_step);hi=max(2,math.ceil(target/per_step))
    # One global integer for training AND inference. Lower integer wins ties.
    steps=min((lo,hi),key=lambda n:(abs(n*per_step-target),n))
    return dict(control_microsteps=steps,context_macs=target,
                original_macs=2*per_step,control_macs=steps*per_step,
                control_vs_context_ratio=steps*per_step/target,
                per_batch_ratio_min=min(steps*r['workspace_per_step']/r['context'] for r in rows),
                per_batch_ratio_max=max(steps*r['workspace_per_step']/r['context'] for r in rows),
                rule='one nearest-total-MAC integer over fixed full training batch schedule; lower wins ties',
                exclusions='unchanged feature encoder/node decoder/graph heads; elementwise work, masking, normalization, softmax, backward and optimizer; not measured FLOPs or wall time')
