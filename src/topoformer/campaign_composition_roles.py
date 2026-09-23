"""C03 public binary-role swaps and separately evaluated supervision."""
import torch
from .campaign_composition import model_inputs
from .campaign_composition_acquire import controlled_rows, labels_from_public
from .campaign_composition_runtime import build_returns
from .campaign_returns_use import answer


def reverse_public(public):
    """Swap all binary records using public cues only; no requested-record lookup."""
    arguments = public['instruction_arguments']
    binary = public['instruction_cues'].argmax(-1) != 3
    changed = torch.where(binary[..., None, None], arguments.flip(-2), arguments)
    return dict(public, instruction_arguments=changed)


def reversed_supervision(rows):
    """Private data preparation: evaluate actual supplied reversed records.

    Fail on any refusal; no rejection/resampling, clipping or gold-target repair.
    Returned labels never enter reverse_public or model.forward.
    """
    changed = controlled_rows(rows, 'reverse_roles', 0)
    oracle = labels_from_public(changed)
    bundle = build_returns(changed, oracle['primitive'], oracle['targets'])
    if len(bundle['indices']) != len(rows):
        raise RuntimeError('role-swapped supervision refused; no hidden resampling')
    event = bundle['public']['event']; query = bundle['public']['query']
    labels = dict(oracle, task=answer(event['values'][:, 0], query), value=(event['values'][:, 0]*2+16).long(), type=event['types'][:, 0])
    return model_inputs(changed), labels


def choose_views(clean, swapped, mask):
    """Rowwise select tensors with an explicit independent presentation coin."""
    if clean.keys() != swapped.keys(): raise ValueError('view schemas differ')
    return {k:torch.where(mask.reshape((-1,)+(1,)*(v.ndim-1)), swapped[k], v) for k,v in clean.items()}
