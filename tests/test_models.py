"""Unit tests for model components."""

import pytest
import torch
from torch_geometric.data import HeteroData

from hybrid_fraud.models import RBGNN, TGNN, AttentionFusion, HybridFraudModel
from hybrid_fraud.models.rb_gnn import DEFAULT_EDGE_TYPES

from .conftest import make_edges

# ---------------------------------------------------------------- fusion


@pytest.mark.parametrize("dynamic", [True, False])
def test_attention_fusion_shapes(dynamic):
    fusion = AttentionFusion(embed_dim=64, use_dynamic_attention=dynamic)
    G, T, S = (torch.randn(4, 64) for _ in range(3))
    H, weights = fusion(G, T, S)
    assert H.shape == (4, 64)
    assert set(weights) == {"alpha", "beta", "gamma"}
    assert abs(sum(weights.values()) - 1.0) < 1e-5


@pytest.mark.parametrize("dynamic", [True, False])
def test_fusion_initial_weights_match_prior(dynamic):
    """Logits are initialised with log(prior) so softmax gives exactly 0.40/0.35/0.25."""
    fusion = AttentionFusion(embed_dim=16, use_dynamic_attention=dynamic)
    _, w = fusion(torch.randn(3, 16), torch.randn(3, 16), torch.randn(3, 16))
    assert w["alpha"] == pytest.approx(0.40, abs=1e-5)
    assert w["beta"] == pytest.approx(0.35, abs=1e-5)
    assert w["gamma"] == pytest.approx(0.25, abs=1e-5)


def test_fusion_custom_prior_is_normalised():
    fusion = AttentionFusion(embed_dim=8, use_dynamic_attention=False, init_weights=(2, 1, 1))
    _, w = fusion(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8))
    assert w["alpha"] == pytest.approx(0.5, abs=1e-5)


def test_fusion_rejects_bad_shapes_and_prior():
    fusion = AttentionFusion(embed_dim=8)
    with pytest.raises(ValueError):
        fusion(torch.randn(2, 8), torch.randn(3, 8), torch.randn(2, 8))
    with pytest.raises(ValueError):
        AttentionFusion(embed_dim=8, init_weights=(0.5, 0.5, 0.0))


def test_fusion_base_logits_receive_gradients():
    fusion = AttentionFusion(embed_dim=8, use_dynamic_attention=True)
    H, _ = fusion(torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8))
    H.sum().backward()
    assert fusion.base_logits.grad is not None


# ---------------------------------------------------------------- TGNN


def test_tgnn_forward():
    tgnn = TGNN(in_channels=32, hidden_channels=64)
    out = tgnn(torch.randn(8, 10, 32))  # batch, seq, feat
    assert out.shape == (8, 64)


def test_tgnn_rejects_wrong_feature_dim():
    with pytest.raises(ValueError):
        TGNN(in_channels=32, hidden_channels=64)(torch.randn(8, 10, 31))


# ---------------------------------------------------------------- RB-GNN


def test_rb_gnn_instantiation():
    model = RBGNN(hidden_channels=32, num_layers=2)
    assert model.hidden_channels == 32
    assert len(model.convs) == 2


def test_rb_gnn_has_reverse_shares_edges():
    assert ("phone", "rev_shares", "user") in DEFAULT_EDGE_TYPES
    assert ("email", "rev_shares", "user") in DEFAULT_EDGE_TYPES


def test_rb_gnn_per_node_type_layernorm():
    model = RBGNN(hidden_channels=16, num_layers=2)
    for norms in model.norms:
        assert set(norms.keys()) == set(model.node_types)
        assert len({id(m) for m in norms.values()}) == len(model.node_types)  # not shared


def test_rb_gnn_forward_with_subset_of_edge_types(graph):
    model = RBGNN(hidden_channels=16, num_layers=2).eval()
    out = model(graph)
    assert set(out) == {"user", "merchant", "phone"}
    assert out["user"].shape == (5, 16)
    assert out["merchant"].shape == (3, 16)
    assert out["phone"].shape == (4, 16)


def test_rb_gnn_phone_signal_reaches_users(graph):
    """With rev_shares edges, changing a phone's features changes its users' embeddings."""
    model = RBGNN(hidden_channels=16, num_layers=1, dropout=0.0).eval()
    base = model(graph)["user"]
    graph["phone"].x = graph["phone"].x + 10.0
    changed = model(graph)["user"]
    linked = graph["user", "shares", "phone"].edge_index[0].unique()
    assert not torch.allclose(base[linked], changed[linked])


def test_rb_gnn_keeps_nodes_without_incoming_messages():
    """Node types with no incoming edges (no reverse edge) must still get embeddings."""
    data = HeteroData()
    data["user"].x = torch.randn(4, 8)
    data["device"].x = torch.randn(2, 8)
    data["user", "uses", "device"].edge_index = make_edges(4, 2, 5)
    out = RBGNN(hidden_channels=8, num_layers=2).eval()(data)
    assert out["user"].shape == (4, 8) and out["device"].shape == (2, 8)


def test_rb_gnn_requires_known_node_types():
    data = HeteroData()
    data["unknown"].x = torch.randn(2, 4)
    with pytest.raises(ValueError):
        RBGNN(hidden_channels=8)(data)


def test_rb_gnn_explicit_in_channels():
    data = HeteroData()
    data["user"].x = torch.randn(3, 5)
    data["merchant"].x = torch.randn(2, 7)
    data["user", "transacts", "merchant"].edge_index = make_edges(3, 2, 4)
    model = RBGNN(hidden_channels=8, in_channels={"user": 5, "merchant": 7})
    assert model(data)["user"].shape == (3, 8)


# ---------------------------------------------------------------- Hybrid


def _inputs(batch=5, hidden=32, static=16, seq=5):
    return torch.randn(batch, seq, hidden), torch.randn(batch, static)


def test_hybrid_model_forward(graph):
    """Regression: edge src/dst indices are drawn from their own node-type ranges."""
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1)
    temporal, static = _inputs()
    logits, weights = model(graph, temporal, static)
    assert logits.shape == (5, 2)
    assert "alpha" in weights


def test_hybrid_model_user_index_selects_rows(graph):
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1).eval()
    idx = torch.tensor([4, 0, 2])
    temporal, static = _inputs(batch=3)
    logits, _ = model(graph, temporal, static, user_index=idx)
    assert logits.shape == (3, 2)

    # Selecting rows via user_index == manually slicing G
    G_all = model.rb_gnn(graph)["user"]
    S = model.static_encoder(static)
    T = model.tgnn(temporal)
    H, _ = model.fusion(G_all[idx], T, S)
    assert torch.allclose(model.risk_head(H), logits, atol=1e-6)


def test_hybrid_model_batch_mismatch_requires_user_index(graph):
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1)
    temporal, static = _inputs(batch=3)  # graph has 5 users
    with pytest.raises(ValueError, match="user_index"):
        model(graph, temporal, static)


def test_hybrid_model_user_index_validation(graph):
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1)
    temporal, static = _inputs(batch=3)
    with pytest.raises(ValueError, match="entries"):
        model(graph, temporal, static, user_index=torch.tensor([0, 1]))
    with pytest.raises(IndexError):
        model(graph, temporal, static, user_index=torch.tensor([0, 1, 5]))
    with pytest.raises(ValueError, match="integer"):
        model(graph, temporal, static, user_index=torch.tensor([0.0, 1.0, 2.0]))


def test_hybrid_model_shape_checks(graph):
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1)
    temporal, static = _inputs()
    with pytest.raises(ValueError, match="static_features"):
        model(graph, temporal, torch.randn(5, 15))
    with pytest.raises(ValueError, match="temporal_sequence"):
        model(graph, temporal[:3], static)


def test_hybrid_model_requires_user_nodes():
    data = HeteroData()
    data["merchant"].x = torch.randn(3, 16)
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=1)
    temporal, static = _inputs(batch=3)
    with pytest.raises(ValueError):
        model(data, temporal, static)


def test_hybrid_model_from_config(graph):
    cfg = {
        "model": {
            "hidden_channels": 32,
            "static_feature_dim": 16,
            "num_gnn_layers": 1,
            "use_dynamic_attention": False,
        },
        "fusion": {"initial_alpha": 0.5, "initial_beta": 0.3, "initial_gamma": 0.2},
    }
    model = HybridFraudModel.from_config(cfg).eval()
    temporal, static = _inputs()
    _, w = model(graph, temporal, static)
    assert w["alpha"] == pytest.approx(0.5, abs=1e-5)


def test_hybrid_model_backward(graph):
    model = HybridFraudModel(static_dim=16, hidden_channels=32, num_gnn_layers=2)
    temporal, static = _inputs()
    logits, _ = model(graph, temporal, static)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.risk_head.parameters())
