import torch
from tensegra.semantic_edge_diagnostics import negative_gradient_direction


def test_negative_gradient_direction_is_local_derivative():
    # Scalar mathematical fixture; no experimental workspace is narrowed.
    parameter=torch.tensor(1.,requires_grad=True)
    score=2*parameter;loss=3*parameter
    direction=negative_gradient_direction(score,[parameter],torch.autograd.grad(loss,[parameter],retain_graph=True))
    assert direction==-6.
