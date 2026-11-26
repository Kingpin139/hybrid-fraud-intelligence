import importlib.util
from pathlib import Path

import pytest
import torch

from hybrid_fraud.data import generate_synthetic_fraud_graph, stratified_node_split
from hybrid_fraud.models import AttentionFusion, RBGNNPlus
from hybrid_fraud.models.rbgnn_plus import (
    EdgeAttention,
    MemoryModule,
    RBGNNPlusLayer,
    TemporalDecay,
    TimeEncoder,
    add_reverse_edges,
)

N, F_IN, HID = 12, 6, 16


@pytest.fixture
def event_graph():
    x = torch.randn(N, F_IN)
    edge_index = torch.randint(0, N, (40, 2)).T.contiguous()
    edge_time = torch.sort(torch.rand(40) * 1000.0).values
    return x, edge_index, edge_time


# ------------------------------------------------------------ building blocks


def test_temporal_decay_properties():
    decay = TemporalDecay(init_lambda=0.5)
    assert decay.decay_rate.item() == pytest.approx(0.5, abs=1e-5)
    out = decay(torch.tensor([0.0, 1.0, 2.0]))
    assert out[0].item() == pytest.approx(1.0)
    assert out[0] > out[1] > out[2] > 0
    # negative dt is clamped (no amplification of "future" edges)
    assert decay(torch.tensor([-5.0])).item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TemporalDecay(init_lambda=0.0)


def test_time_encoder_shape_and_range():
    enc = TimeEncoder(8)
    out = enc(torch.rand(5))
    assert out.shape == (5, 8)
    assert out.abs().max() <= 1.0 + 1e-6


def test_edge_attention_sums_to_one_per_destination():
    torch.manual_seed(0)
    att = EdgeAttention(HID)
    dst = torch.tensor([0, 0, 0, 1, 2, 2, 5])
    w = att(torch.randn(7, HID), torch.randn(7, HID), dst, num_nodes=8)
    assert w.shape == (7,)
    assert torch.all(w > 0)
    sums = torch.zeros(8).index_add(0, dst, w)
    for node in dst.unique():
        assert sums[node].item() == pytest.approx(1.0, abs=1e-5)
    assert sums[3].item() == 0.0  # node without incoming edges


def test_layer_returns_per_destination_normalised_attention(event_graph):
    x, edge_index, edge_time = event_graph
    layer = RBGNNPlusLayer(HID)
    h = torch.randn(N, HID)
    dt = (edge_time.max() - edge_time) / edge_time.max()
    out, att = layer(h, edge_index, dt, return_attention=True)
    assert out.shape == (N, HID)
    sums = torch.zeros(N).index_add(0, edge_index[1], att)
    for node in edge_index[1].unique():
        assert sums[node].item() == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------ causality / time


def test_relative_time_is_non_negative_and_normalised():
    t = torch.tensor([10.0, 20.0, 30.0, 50.0])
    dt, keep = RBGNNPlus.relative_time(t)
    assert keep.all()
    assert dt.min().item() == 0.0 and dt.max().item() == pytest.approx(1.0)
    assert torch.all(dt >= 0)


def test_future_edges_are_dropped_and_dt_non_negative():
    t = torch.tensor([10.0, 20.0, 30.0, 50.0])
    dt, keep = RBGNNPlus.relative_time(t, query_time=30.0)
    assert keep.tolist() == [True, True, True, False]
    assert torch.all(dt >= 0)
    assert dt[2].item() == 0.0


def test_future_edges_do_not_influence_output(event_graph):
    """Changing edges that lie after query_time must not change the embeddings."""
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID).eval()
    q = float(edge_time[20])
    # Pin time_scale so the normalisation does not depend on the (modified) future edges.
    base = model.encode(x, edge_index, edge_time, query_time=q, time_scale=500.0)
    tampered = edge_index.clone()
    future = edge_time > q
    tampered[:, future] = torch.randint(0, N, (2, int(future.sum())))
    out = model.encode(x, tampered, edge_time, query_time=q, time_scale=500.0)
    assert torch.allclose(base, out, atol=1e-6)


def test_time_unit_invariance(event_graph):
    """Default scale normalisation: seconds vs. days give identical embeddings."""
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID).eval()
    a = model.encode(x, edge_index, edge_time)
    b = model.encode(x, edge_index, edge_time * 86400.0)
    assert torch.allclose(a, b, atol=1e-4)


def test_empty_edges_and_all_future(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID).eval()
    out = model(x, edge_index[:, :0], edge_time[:0])
    assert out.shape == (N, 2) and torch.isfinite(out).all()
    out = model(x, edge_index, edge_time, query_time=float(edge_time.min()) - 1.0)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------ reverse edges


def test_add_reverse_edges_presence():
    ei = torch.tensor([[0, 1], [2, 3]])
    et = torch.tensor([1.0, 2.0])
    ei2, et2 = add_reverse_edges(ei, et)
    pairs = set(map(tuple, ei2.T.tolist()))
    assert {(0, 2), (1, 3), (2, 0), (3, 1)} == pairs
    assert et2.tolist() == [1.0, 2.0, 1.0, 2.0]


def test_model_adds_reverse_edges_and_source_only_nodes_receive_messages():
    # node 0 is only ever a *source*; with bidirectional edges it must still get messages.
    x = torch.randn(3, F_IN)
    ei = torch.tensor([[0, 0], [1, 2]])
    et = torch.tensor([1.0, 2.0])
    bi = RBGNNPlus(F_IN, HID, bidirectional=True).eval()
    uni = RBGNNPlus(F_IN, HID, bidirectional=False).eval()
    uni.load_state_dict(bi.state_dict())
    edges, dt = bi.prepare_edges(ei, et)
    assert edges.size(1) == 4 and dt.numel() == 4
    assert 0 in edges[1].tolist()
    assert not torch.allclose(bi.encode(x, ei, et)[0], uni.encode(x, ei, et)[0])


# ------------------------------------------------------------ memory


def test_memory_is_buffer_reset_and_detached():
    mem = MemoryModule(5, 4)
    assert "memory" in dict(mem.named_buffers())
    assert "memory" not in dict(mem.named_parameters())

    nodes = torch.tensor([1, 3, 3])
    msgs = torch.randn(3, 4, requires_grad=True)
    new_state = mem.update(nodes, msgs)
    assert new_state.shape == (3, 4) and new_state.requires_grad  # differentiable return value
    assert not mem.memory.requires_grad and mem.memory.grad_fn is None  # stored detached
    assert mem.memory[1].abs().sum() > 0 and mem.memory[0].abs().sum() == 0
    assert torch.equal(new_state[1].detach(), mem.memory[3])  # duplicate ids share one state
    new_state.sum().backward()
    assert msgs.grad is not None and mem.gru.weight_ih.grad is not None

    mem.reset()
    assert mem.memory.abs().sum() == 0


def test_model_memory_is_read_only_by_default(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID, num_nodes=N, use_memory=True).eval()
    model(x, edge_index, edge_time)
    assert model.memory.memory.abs().sum() == 0
    model(x, edge_index, edge_time, update_memory=True)
    assert model.memory.memory.abs().sum() > 0
    model.memory.reset()
    assert model.memory.memory.abs().sum() == 0


def test_memory_requires_num_nodes():
    with pytest.raises(ValueError):
        RBGNNPlus(F_IN, HID, use_memory=True)


def test_memory_training_step_does_not_keep_graph(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID, num_nodes=N, use_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):  # a second backward would fail if memory kept the previous graph
        opt.zero_grad()
        model(x, edge_index, edge_time, update_memory=True).sum().backward()
        opt.step()


# ------------------------------------------------------------ model shapes


def test_forward_and_encode_shapes(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID, out_dim=2, num_layers=3)
    assert model(x, edge_index, edge_time).shape == (N, 2)
    assert model.encode(x, edge_index, edge_time).shape == (N, HID)


def test_input_validation(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID)
    with pytest.raises(ValueError):
        model(x[:, :3], edge_index, edge_time)
    with pytest.raises(ValueError):
        model(x, edge_index.T, edge_time)
    with pytest.raises(ValueError):
        model(x, edge_index, edge_time[:-1])
    with pytest.raises(ValueError):
        model(x, edge_index, edge_time, time_scale=0.0)


def test_backward_reaches_decay_attention_and_time_encoder(event_graph):
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID)
    model(x, edge_index, edge_time).sum().backward()
    layer = model.layers[0]
    assert layer.decay.raw_lambda.grad is not None
    assert layer.attention.score.weight.grad is not None
    assert layer.time_enc.lin.weight.grad is not None


def test_encoder_output_plugs_into_attention_fusion(event_graph):
    """Documented integration path: RBGNN+ embeddings can serve as T in H = aG + bT + cS."""
    x, edge_index, edge_time = event_graph
    model = RBGNNPlus(F_IN, HID).eval()
    target = torch.tensor([3, 7, 1])
    T = model.encode(x, edge_index, edge_time)[target]
    G, S = torch.randn(3, HID), torch.randn(3, HID)
    H, weights = AttentionFusion(embed_dim=HID)(G, T, S)
    assert H.shape == (3, HID) and abs(sum(weights.values()) - 1.0) < 1e-5


# ------------------------------------------------------------ synthetic data


def test_synthetic_defaults_inject_no_label_signal():
    x, ei, et, y = generate_synthetic_fraud_graph(
        num_nodes=4000, num_edges=2000, feature_dim=8, fraud_ratio=0.1, seed=1
    )
    gap = (x[y == 1].mean(0) - x[y == 0].mean(0)).abs().max()
    assert gap < 0.35  # noise level only (fraud n=400): no planted mean shift
    assert (y[ei[0]] * y[ei[1]]).float().mean() < 0.03  # no fraud-ring edges


def test_synthetic_signals_are_explicit():
    x, ei, et, y = generate_synthetic_fraud_graph(
        num_nodes=500, num_edges=2000, fraud_ratio=0.1, seed=1,
        feature_signal=1.0, ring_edge_fraction=0.3, burst_window=0.1,
    )  # fmt: skip
    assert (x[y == 1].mean() - x[y == 0].mean()) > 0.7
    ring = (y[ei[0]] == 1) & (y[ei[1]] == 1)
    assert (ring & (et >= 90.0)).sum() >= 600  # 30% of 2000 edges, all in the last 10%


def test_synthetic_graph_structure():
    x, ei, et, y = generate_synthetic_fraud_graph(
        num_nodes=100, num_edges=300, ring_edge_fraction=0.2
    )
    assert x.shape == (100, 16) and ei.shape == (2, 300) and et.shape == (300,)
    assert torch.all(et[1:] >= et[:-1])  # chronological
    assert (ei[0] != ei[1]).all()  # no self loops
    assert ei.max() < 100 and y.dtype == torch.long
    again = generate_synthetic_fraud_graph(num_nodes=100, num_edges=300, ring_edge_fraction=0.2)
    assert torch.equal(ei, again.edge_index)  # deterministic given seed


def test_synthetic_validation():
    with pytest.raises(ValueError):
        generate_synthetic_fraud_graph(fraud_ratio=1.0)
    with pytest.raises(ValueError):
        generate_synthetic_fraud_graph(ring_edge_fraction=1.5)


def test_stratified_split_is_disjoint_and_stratified():
    y = torch.zeros(200, dtype=torch.long)
    y[:20] = 1
    tr, va, te = stratified_node_split(y, seed=0)
    assert len(set(tr.tolist()) & set(va.tolist())) == 0
    assert len(set(tr.tolist()) & set(te.tolist())) == 0
    assert len(set(va.tolist()) & set(te.tolist())) == 0
    assert len(tr) + len(va) + len(te) == 200
    for part in (tr, va, te):
        assert 0.08 <= y[part].float().mean() <= 0.12


# ------------------------------------------------------------ demo


def test_rbgnn_plus_demo_runs_fast_and_labels_results_synthetic(capsys):
    path = Path(__file__).resolve().parents[1] / "examples" / "rbgnn_plus_demo.py"
    spec = importlib.util.spec_from_file_location("rbgnn_plus_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results = mod.main(num_nodes=300, num_edges=1200, hidden_dim=8, epochs=3)
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "not a benchmark" in out
    assert {"logistic (features only)", "RBGNN+ (with graph)"} <= set(results)
    for name, m in results.items():
        assert 0.0 <= m["pr_auc"] <= 1.0, name
