import math

import pytest
import torch

from topoformer.attention import graph_structure, structural_attention


def test_zero_bias_matches_scaled_dot_product_attention():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    v = torch.randn(2, 3, 4, 6, dtype=torch.float64)

    output, weights = structural_attention(q, k, v)
    expected_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(5), dim=-1)

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_weights @ v)


def test_positive_directed_edge_biases_only_its_query_key_pair():
    q = torch.zeros(1, 1, 3, 2)
    bias = torch.zeros(1, 1, 3, 3)
    bias[0, 0, 0, 2] = 1

    _, weights = structural_attention(q, q, q, bias=bias, strength=2.0)

    assert weights[0, 0, 0, 2] > weights[0, 0, 0, 1]
    torch.testing.assert_close(weights[0, 0, 2], torch.full((3,), 1 / 3))


def test_rectangular_query_and_key_lengths_are_supported():
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    v = torch.tensor([[[[1.0], [2.0], [4.0]]]])

    output, weights = structural_attention(q, k, v)

    assert output.shape == (1, 1, 2, 1)
    assert weights.shape == (1, 1, 2, 3)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 2))


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("bias", torch.zeros(2, 3)),
        ("allowed", torch.ones(2, 3, dtype=torch.bool)),
    ],
)
def test_rejects_relations_that_do_not_broadcast_to_scores(argument, value):
    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 2, 5, 4)
    v = torch.zeros(1, 2, 5, 6)

    with pytest.raises(ValueError, match="broadcast"):
        structural_attention(q, k, v, **{argument: value})


@pytest.mark.parametrize("tensor_name", ["q", "k", "v"])
def test_rejects_nonfinite_inputs(tensor_name):
    tensors = {
        "q": torch.zeros(1, 1, 2, 3),
        "k": torch.zeros(1, 1, 2, 3),
        "v": torch.zeros(1, 1, 2, 4),
    }
    tensors[tensor_name][0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="finite"):
        structural_attention(**tensors)


def test_combined_padding_and_causal_mask_excludes_disallowed_keys():
    q = torch.zeros(1, 1, 3, 2)
    v = torch.tensor([[[[1.0], [2.0], [100.0]]]])
    allowed = torch.tensor([[[[True, False, False], [True, True, False], [True, True, False]]]])

    output, weights = structural_attention(q, q, v, allowed=allowed)

    torch.testing.assert_close(weights[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(weights[0, 0, 1], torch.tensor([0.5, 0.5, 0.0]))
    torch.testing.assert_close(output[0, 0, :, 0], torch.tensor([1.0, 1.5, 1.5]))


def test_rejects_rows_with_no_allowed_key():
    q = torch.zeros(1, 1, 2, 3)
    allowed = torch.tensor([[[[True, False], [False, False]]]])

    with pytest.raises(ValueError, match="no legal key"):
        structural_attention(q, q, q, allowed=allowed)


def test_graph_modes_handle_isolated_nodes_and_preserve_direction():
    graph = torch.tensor([[[False, True, False], [False, False, False], [False, False, False]]])

    none_bias, none_allowed = graph_structure(graph, "none")
    soft_bias, soft_allowed = graph_structure(graph, "soft")
    hard_bias, hard_allowed = graph_structure(graph, "hard")

    torch.testing.assert_close(none_bias, torch.zeros(1, 1, 3, 3))
    assert none_allowed is None
    torch.testing.assert_close(soft_bias, graph[:, None].to(torch.float32))
    assert soft_allowed is None
    assert hard_bias.shape == (1, 1, 3, 3)
    torch.testing.assert_close(hard_bias, torch.zeros_like(hard_bias))
    expected_allowed = graph | torch.eye(3, dtype=torch.bool)[None]
    assert torch.equal(hard_allowed, expected_allowed[:, None])


def test_attention_is_differentiable_and_returns_value_dtype():
    q = torch.randn(1, 2, 3, 4, requires_grad=True)
    k = torch.randn(1, 2, 5, 4, requires_grad=True)
    v = torch.randn(1, 2, 5, 6, dtype=torch.float64, requires_grad=True)

    output, weights = structural_attention(q, k, v)
    output.sum().backward()

    assert output.dtype == v.dtype
    assert weights.dtype == torch.float64
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.parametrize("strength", [float("nan"), float("inf"), torch.tensor(float("-inf"))])
def test_rejects_nonfinite_strength(strength):
    q = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="strength"):
        structural_attention(q, q, q, strength=strength)


def test_rejects_non_boolean_allowed_mask():
    q = torch.zeros(1, 1, 1, 2)
    with pytest.raises(TypeError, match="boolean"):
        structural_attention(q, q, q, allowed=torch.ones(1, 1, 1, 1))


def test_low_precision_scores_are_computed_in_float32():
    q = torch.zeros(1, 1, 2, 2, dtype=torch.float16)
    output, weights = structural_attention(q, q, q)
    assert weights.dtype == torch.float32
    assert output.dtype == torch.float16


def test_low_precision_output_accumulates_before_casting_to_value_dtype():
    torch.manual_seed(3)
    q = torch.ones(1, 1, 1, 1, dtype=torch.float16)
    k = torch.randn(1, 1, 64, 1).to(torch.float16)
    v = (torch.randn(1, 1, 64, 1) * 100).to(torch.float16)

    output, _ = structural_attention(q, k, v)

    assert output.item() == -6.984375


def test_float64_values_are_not_downcast_during_output_contraction():
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[1e12], [-1e12 + 2.0]]]], dtype=torch.float64)

    output, weights = structural_attention(q, k, v)

    assert weights.dtype == torch.float64
    torch.testing.assert_close(output, torch.ones_like(output))


def test_single_element_tensor_strength_remains_scalar_and_differentiable():
    q = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[1.0], [3.0]]]])
    bias = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    strength = torch.ones(1, 1, 1, 1, 1, requires_grad=True)

    output, weights = structural_attention(q, q, v, bias=bias, strength=strength)
    output.sum().backward()

    assert output.shape == (1, 1, 2, 1)
    assert weights.shape == (1, 1, 2, 2)
    assert strength.grad is not None
    assert strength.grad.shape == strength.shape
    assert strength.grad.abs().item() > 0
