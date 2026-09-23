"""Stage 9 versioned ordered-edge contracts; no runtime integration.

A slot label describes an existing directed edge. Existence and relation remain
separate decisions. Conditional training may use gold edges only to mask loss.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SlotScorer(nn.Module):
    """Additive reference or explicit bilinear pair interaction.

    Width is node width, not a hidden reduction of the experimental workspace.
    ``interaction_width`` controls only the additional output scorer rank.
    """
    def __init__(self, width=1024, classes=33, *, interaction=False, interaction_width=32):
        super().__init__()
        self.source=nn.Linear(width,classes); self.target=nn.Linear(width,classes)
        self.interaction=interaction
        if interaction:
            self.pair_source=nn.Linear(width,classes*interaction_width,bias=False)
            self.pair_target=nn.Linear(width,interaction_width,bias=False)
        self.classes=classes; self.interaction_width=interaction_width

    def forward(self,nodes,pairs):
        i,j=pairs.unbind(-1)
        output=self.source(nodes[i])+self.target(nodes[j])
        if self.interaction:
            source=self.pair_source(nodes[i]).reshape(-1,self.classes,self.interaction_width)
            target=self.pair_target(nodes[j])
            output=output+(source*target[:,None,:]).sum(-1)/self.interaction_width**.5
        return output


def slot_objective(logits,labels,edge_exists,*,edge_conditional):
    """Labels 0=unordered/no-slot, k+1=ordered slot k.

    Preserve Stage 8 positive/no-slot balanced objective within the selected
    population. Gold edge existence selects training entries only.
    """
    mask=edge_exists.bool() if edge_conditional else torch.ones_like(edge_exists,dtype=torch.bool)
    if not mask.any():return logits.sum()*0
    loss=F.cross_entropy(logits[mask],labels[mask],reduction='none')
    ordered=labels[mask].gt(0)
    terms=[loss[m].mean() for m in (ordered,~ordered) if m.any()]
    return sum(terms)/len(terms)


def ordered_pair_labels(graph):
    """Lossless relation/slot targets; never silently overwrite multiedges."""
    labels={}
    for edge in graph.edges:
        labels.setdefault((edge.source,edge.target),set()).add((edge.role,edge.slot))
    return labels


def single_slot_labels(graph):
    """Legacy scalar slot target is admissible only when pair slots agree."""
    labels=ordered_pair_labels(graph); result={}
    for pair,values in labels.items():
        slots={slot for _,slot in values}
        if len(slots)>1:raise ValueError(f'multiple slot labels at {pair}; use relation-conditioned multi-label targets')
        result[pair]=next(iter(slots))
    return result
