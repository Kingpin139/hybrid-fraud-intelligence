"""
Synthetic toy data for smoke tests and demos.

.. warning::
   Everything here is **synthetic**. Metrics measured on it are *not* benchmarks and say nothing
   about performance on real data (IEEE-CIS, PaySim, …). By default the generator injects **no**
   label signal at all (a model can only reach chance level); any signal must be switched on
   explicitly through ``feature_signal`` / ``ring_edge_fraction``.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import torch


class SyntheticFraudGraph(NamedTuple):
    """Homogeneous timestamped event graph (tuple-unpackable as ``x, edge_index, edge_time, y``)."""

    x: torch.Tensor  # [N, F] float32
    edge_index: torch.Tensor  # [2, E] long, (src, dst), sorted by time
    edge_time: torch.Tensor  # [E] float32 in [0, time_horizon], non-decreasing
    y: torch.Tensor  # [N] long, 1 = fraud


def generate_synthetic_fraud_graph(
    num_nodes: int = 2000,
    num_edges: int = 10000,
    feature_dim: int = 16,
    fraud_ratio: float = 0.05,
    seed: int = 42,
    feature_signal: float = 0.0,
    ring_edge_fraction: float = 0.0,
    burst_window: float = 0.2,
    time_horizon: float = 100.0,
) -> SyntheticFraudGraph:
    """
    Generate a random event graph with a (optionally) planted fraud signal.

    With the default arguments features, edges and timestamps are independent of the labels, so
    no model can beat chance. Toy signals, off by default:

    * ``feature_signal``: added to **every** feature dimension of fraud nodes (a mean shift that
      leaks the label into the features — a linear baseline benefits directly).
    * ``ring_edge_fraction``: fraction of ``num_edges`` that are *fraud-ring* edges between two
      fraud nodes, with timestamps concentrated in the last ``burst_window`` of the timeline
      (coordinated, bursty activity). This is the relational/temporal signal a GNN can use.

    Edges are returned sorted by time and contain no self loops.
    """
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2.")
    if not 0.0 <= fraud_ratio < 1.0:
        raise ValueError("fraud_ratio must be in [0, 1).")
    if not 0.0 <= ring_edge_fraction <= 1.0:
        raise ValueError("ring_edge_fraction must be in [0, 1].")
    if not 0.0 < burst_window <= 1.0:
        raise ValueError("burst_window must be in (0, 1].")

    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(num_nodes, feature_dim)).astype(np.float32)

    n_fraud = int(round(num_nodes * fraud_ratio))
    fraud_nodes = rng.choice(num_nodes, size=n_fraud, replace=False)
    y = np.zeros(num_nodes, dtype=np.int64)
    y[fraud_nodes] = 1
    if feature_signal and n_fraud:
        x[fraud_nodes] += np.float32(feature_signal)

    n_ring = int(num_edges * ring_edge_fraction) if n_fraud >= 2 else 0
    n_bg = num_edges - n_ring

    src = rng.integers(0, num_nodes, size=n_bg)
    dst = rng.integers(0, num_nodes, size=n_bg)
    dst = np.where(src == dst, (dst + 1) % num_nodes, dst)  # no self loops
    t = rng.uniform(0.0, time_horizon, size=n_bg)

    if n_ring:
        a = rng.integers(0, n_fraud, size=n_ring)
        b = (a + 1 + rng.integers(0, n_fraud - 1, size=n_ring)) % n_fraud  # b != a
        src = np.concatenate([src, fraud_nodes[a]])
        dst = np.concatenate([dst, fraud_nodes[b]])
        t = np.concatenate(
            [t, rng.uniform(time_horizon * (1.0 - burst_window), time_horizon, size=n_ring)]
        )

    order = np.argsort(t, kind="stable")
    return SyntheticFraudGraph(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.stack([src[order], dst[order]])).long(),
        edge_time=torch.from_numpy(t[order].astype(np.float32)),
        y=torch.from_numpy(y),
    )


def stratified_node_split(
    y: torch.Tensor, seed: int = 42, train: float = 0.6, val: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified ``(train_idx, val_idx, test_idx)`` over node labels (test = remainder)."""
    from sklearn.model_selection import train_test_split

    if not (0.0 < train < 1.0 and 0.0 < val < 1.0 and train + val < 1.0):
        raise ValueError("train and val must be in (0, 1) with train + val < 1.")
    labels = y.cpu().numpy()
    idx = np.arange(labels.shape[0])
    train_idx, rest = train_test_split(idx, train_size=train, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest, train_size=val / (1.0 - train), stratify=labels[rest], random_state=seed
    )
    as_t = lambda a: torch.as_tensor(a, dtype=torch.long)  # noqa: E731
    return as_t(train_idx), as_t(val_idx), as_t(test_idx)
