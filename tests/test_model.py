import torch

from topoformer.model import GraphPredictor


def chain(n):
    graph = torch.eye(n, dtype=torch.bool)
    for i in range(n - 1):
        graph[i, i + 1] = graph[i + 1, i] = True
    return graph


def test_variable_count_and_finite_backward():
    model = GraphPredictor(3, 12, 3, 2)
    for nodes in (4, 7):
        x = torch.randn(2, nodes, 3, requires_grad=True)
        y = model(x, torch.eye(nodes, dtype=torch.bool), mode="hard")
        assert y.shape == (2, nodes)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all()


def test_joint_permutation_equivariance():
    torch.manual_seed(1)
    model = GraphPredictor(3, 12, 3, 2).eval()
    x, graph = torch.randn(2, 5, 3), chain(5)
    p = torch.tensor([3, 0, 4, 1, 2])
    expected = model(x, graph, mode="soft", strength=1.0)[:, p]
    actual = model(x[:, p], graph[p][:, p], mode="soft", strength=1.0)
    torch.testing.assert_close(actual, expected)


def test_zero_strength_is_exactly_unbiased():
    model = GraphPredictor(2, 8, 2, 1).eval()
    x, graph = torch.randn(3, 4, 2), chain(4)
    torch.testing.assert_close(
        model(x, graph, mode="none"),
        model(x, graph, mode="soft", strength=0.0),
        rtol=0,
        atol=0,
    )


def test_hard_mask_receptive_field_and_positive_control():
    torch.manual_seed(4)
    graph = chain(4)
    for layers in (1, 2):
        model = GraphPredictor(1, 4, 1, layers).eval()
        # Controlled nonzero positive path through uniform attention and residuals.
        for block in model.blocks:
            torch.nn.init.zeros_(block.qkv.weight)
            block.qkv.weight.data[8:12] = torch.eye(4)
            block.out.weight.data.copy_(torch.eye(4))
            for layer in block.mlp:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
        model.input.weight.data[:, 0] = torch.tensor([1.0, 2.0, 4.0, 8.0]); model.input.bias.data.copy_(torch.tensor([0.0, 1.0, -1.0, 0.5]))
        model.readout.weight.data.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]])); model.readout.bias.data.zero_()
        base = torch.tensor([[[0.2], [0.5], [0.8], [1.1]]])
        far = base.clone(); far[:, 3] += 3
        near = base.clone(); near[:, 1] += 3
        assert torch.equal(model(base, graph, mode="hard")[:, 0], model(far, graph, mode="hard")[:, 0])
        assert not torch.equal(model(base, graph, mode="hard")[:, 0], model(near, graph, mode="hard")[:, 0])
    model = GraphPredictor(1, 4, 1, 3).eval()
    for block in model.blocks:
        torch.nn.init.zeros_(block.qkv.weight); block.qkv.weight.data[8:12] = torch.eye(4)
        block.out.weight.data.copy_(torch.eye(4))
        for layer in block.mlp:
            if isinstance(layer, torch.nn.Linear): torch.nn.init.zeros_(layer.weight); torch.nn.init.zeros_(layer.bias)
    model.input.weight.data[:, 0] = torch.tensor([1.0, 2.0, 4.0, 8.0]); model.input.bias.data.copy_(torch.tensor([0.0, 1.0, -1.0, 0.5]))
    model.readout.weight.data.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]])); model.readout.bias.data.zero_()
    base = torch.tensor([[[0.2], [0.5], [0.8], [1.1]]]); far = base.clone(); far[:, 3] += 3
    assert not torch.equal(model(base, graph, mode="hard")[:, 0], model(far, graph, mode="hard")[:, 0])
