"""
RBGNN+ demo on SYNTHETIC data.

Trains :class:`hybrid_fraud.models.RBGNNPlus` for node-level fraud classification on a toy event
graph, with a stratified train / val / test split, a class-weighted loss, best-validation
checkpointing and two reference points:

* a feature-only logistic regression (no graph), and
* the same model with all edges removed (graph ablation).

IMPORTANT: the data is synthetic and the signal is injected on purpose (see
``hybrid_fraud.data.synthetic``): fraud nodes get a small feature shift and extra bursty
edges among themselves, which also makes them high-degree and therefore easy for a graph model to
find. The printed numbers are a smoke test of the code path, **not**
benchmark results and not evidence of real-world performance.

    python examples/rbgnn_plus_demo.py
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_fraud.data import generate_synthetic_fraud_graph, stratified_node_split
from hybrid_fraud.models import RBGNNPlus


def class_weights(y: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights computed on the *training* labels only."""
    counts = torch.bincount(y[train_idx], minlength=2).float().clamp(min=1.0)
    return counts.sum() / (2.0 * counts)


def _scores(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:  # degenerate tiny split
        return {"auc": float("nan"), "pr_auc": float("nan")}
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


@torch.no_grad()
def evaluate(model, x, edge_index, edge_time, y, idx) -> Dict[str, float]:
    model.eval()
    prob = torch.softmax(model(x, edge_index, edge_time)[idx], dim=-1)[:, 1]
    return _scores(y[idx].cpu().numpy(), prob.cpu().numpy())


def fit(model, x, edge_index, edge_time, y, splits, epochs: int, lr: float) -> Dict[str, float]:
    """Train on ``train_idx`` only; return test scores of the best-validation epoch."""
    train_idx, val_idx, test_idx = splits
    criterion = nn.CrossEntropyLoss(weight=class_weights(y, train_idx).to(x.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x, edge_index, edge_time)[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        val = evaluate(model, x, edge_index, edge_time, y, val_idx)["pr_auc"]
        if not np.isnan(val) and val > best_val:
            best_val, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return evaluate(model, x, edge_index, edge_time, y, test_idx)


def main(
    num_nodes: int = 1500,
    num_edges: int = 9000,
    feature_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 60,
    seed: int = 42,
    feature_signal: float = 0.25,
    ring_edge_fraction: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("RBGNN+ demo — SYNTHETIC data (smoke test, not a benchmark)")
    print("=" * 62)
    x, edge_index, edge_time, y = generate_synthetic_fraud_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feature_dim,
        seed=seed,
        feature_signal=feature_signal,
        ring_edge_fraction=ring_edge_fraction,
    )
    splits = stratified_node_split(y, seed=seed)
    train_idx, val_idx, test_idx = splits
    print(
        f"nodes={num_nodes} edges={edge_index.size(1)} fraud={int(y.sum())} "
        f"({y.float().mean():.1%}) | train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
    )
    if feature_signal or ring_edge_fraction:
        print(
            f"injected toy signal: feature_signal={feature_signal}, "
            f"ring_edge_fraction={ring_edge_fraction} (label-dependent by construction)"
        )
    else:
        print("no signal injected: every model is expected to stay near chance level")

    results: Dict[str, Dict[str, float]] = {}
    y_test = y[test_idx].numpy()
    results["chance (fraud rate)"] = {"auc": 0.5, "pr_auc": float(y_test.mean())}

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x[train_idx].numpy(), y[train_idx].numpy())
    results["logistic (features only)"] = _scores(
        y_test, clf.predict_proba(x[test_idx].numpy())[:, 1]
    )

    x, edge_index, edge_time, y = (t.to(device) for t in (x, edge_index, edge_time, y))
    splits = tuple(s.to(device) for s in splits)

    for name, ei, et in (
        ("RBGNN+ (no edges, ablation)", edge_index[:, :0], edge_time[:0]),
        ("RBGNN+ (with graph)", edge_index, edge_time),
    ):
        torch.manual_seed(seed)
        model = RBGNNPlus(feature_dim, hidden_dim, out_dim=2, num_layers=2).to(device)
        results[name] = fit(model, x, ei, et, y, splits, epochs, lr=0.01)

    print("\nTest-set results (SYNTHETIC; model chosen by best validation PR-AUC):")
    print(f"  {'model':<30}{'AUC':>8}{'PR-AUC':>9}")
    for name, m in results.items():
        print(f"  {name:<30}{m['auc']:>8.4f}{m['pr_auc']:>9.4f}")
    print("\nSynthetic toy data: do not cite these numbers as results.")
    return results


if __name__ == "__main__":
    main()
