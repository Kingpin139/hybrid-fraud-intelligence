"""
RBGNN+ — a temporal relational GNN for transaction-event graphs.

Unlike :class:`hybrid_fraud.models.RBGNN` (heterogeneous GraphSAGE over a static graph), RBGNN+
consumes a *homogeneous event graph* whose edges carry timestamps. Every message is shaped by:

* **Temporal decay** ``exp(-λ·Δt)`` — older interactions contribute less (learnable λ > 0).
* **Time encoding** — a learned-frequency cosine encoding of Δt (TGAT-style) added to the message.
* **Edge attention** — scores are normalised with a softmax over the *incoming edges of each
  destination node* (so they sum to 1 per destination).
* **Reverse edges** — messages flow in both directions (``bidirectional=True``).
* **Optional node memory** — a GRU-based per-node state stored in a non-trainable buffer.

Causality: ``Δt = query_time - edge_time`` is always ``>= 0``. Edges *after* ``query_time`` are
dropped before message passing, so nothing from the future is used. Time is rescaled to
(roughly) ``[0, 1]`` so the decay is independent of the raw time unit (seconds vs. days).

Typical use::

    model = RBGNNPlus(in_dim=16, hidden_dim=32, out_dim=2)
    logits = model(x, edge_index, edge_time)                 # [N, 2]
    emb = model.encode(x, edge_index, edge_time)             # [N, 32], e.g. the "T" input of fusion
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as segment_softmax


def add_reverse_edges(
    edge_index: torch.Tensor, edge_time: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append a ``dst -> src`` copy of every edge (same timestamp) so messages flow both ways."""
    reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, reverse], dim=1), torch.cat([edge_time, edge_time], dim=0)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class TemporalDecay(nn.Module):
    """Learnable exponential decay ``exp(-λ·Δt)`` with ``λ = softplus(raw) > 0``."""

    def __init__(self, init_lambda: float = 1.0):
        super().__init__()
        if init_lambda <= 0:
            raise ValueError("init_lambda must be positive.")
        self.raw_lambda = nn.Parameter(torch.tensor(_inverse_softplus(float(init_lambda))))

    @property
    def decay_rate(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.decay_rate * dt.clamp(min=0.0))


class EdgeAttention(nn.Module):
    """Attention logits ``w·[h_src ‖ h_dst]``, softmax-normalised per destination node."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(2 * dim, 1)

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return attention weights ``[E]`` that sum to 1 over the edges of each destination."""
        logits = self.score(torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)
        return segment_softmax(logits, dst, num_nodes=num_nodes)


class TimeEncoder(nn.Module):
    """TGAT-style functional time encoding ``cos(ω·Δt + b)`` with learnable ω, b."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(1, dim)
        with torch.no_grad():
            # Log-spaced frequencies in [1, 100] cover coarse and fine structure of Δt ∈ [0, 1].
            self.lin.weight.copy_(torch.logspace(0, 2, dim).unsqueeze(-1))
            self.lin.bias.zero_()

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        return torch.cos(self.lin(dt))


class MemoryModule(nn.Module):
    """
    Per-node memory updated by a ``GRUCell`` (TGN-style).

    The memory bank is a **buffer** (not a ``Parameter``) and is always written *detached*, so it
    never keeps an autograd graph alive. Call :meth:`reset` between independent passes (e.g. at the
    start of every epoch / evaluation) — a persistent memory carries state from earlier calls.
    """

    memory: torch.Tensor

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.register_buffer("memory", torch.zeros(num_nodes, dim))
        self.gru = nn.GRUCell(dim, dim)

    def get(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.memory[nodes]

    def update(self, nodes: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Update the memory of ``nodes`` with ``messages`` and return the new (differentiable) state.

        Duplicate node ids are collapsed by averaging their messages. The value written to the
        buffer is detached.
        """
        if nodes.numel() == 0:
            return messages.new_zeros(0, self.dim)
        unique, inverse = torch.unique(nodes, return_inverse=True)
        summed = messages.new_zeros(unique.numel(), self.dim).index_add(0, inverse, messages)
        counts = messages.new_zeros(unique.numel(), 1).index_add(
            0, inverse, messages.new_ones(messages.size(0), 1)
        )
        new_state = self.gru(summed / counts.clamp_min(1.0), self.memory[unique])
        self.memory[unique] = new_state.detach()
        return new_state[inverse]

    def reset(self) -> None:
        self.memory.zero_()


class RBGNNPlusLayer(nn.Module):
    """One temporal message-passing layer (decay × attention × (message + time encoding))."""

    def __init__(self, dim: int, dropout: float = 0.0, init_lambda: float = 1.0):
        super().__init__()
        self.lin_msg = nn.Linear(dim, dim)
        self.lin_self = nn.Linear(dim, dim)
        self.decay = TemporalDecay(init_lambda)
        self.attention = EdgeAttention(dim)
        self.time_enc = TimeEncoder(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        dt: torch.Tensor,
        return_attention: bool = False,
    ):
        src, dst = edge_index[0], edge_index[1]
        h_src = self.lin_msg(x[src])
        h_dst = self.lin_msg(x[dst])

        att = self.attention(h_src, h_dst, dst, x.size(0))  # [E], sums to 1 per destination
        decay = self.decay(dt)  # [E] in (0, 1]
        msg = (h_src + self.time_enc(dt)) * (att * decay).unsqueeze(-1)

        agg = torch.zeros_like(x).index_add(0, dst, msg)
        out = F.relu(self.norm(agg + self.lin_self(x)))
        out = self.dropout(out)
        return (out, att) if return_attention else out


class RBGNNPlus(nn.Module):
    """
    Temporal relational GNN for node-level fraud scoring on a timestamped event graph.

    Args:
        in_dim: Raw node feature size.
        hidden_dim: Embedding size.
        out_dim: Number of output classes (2 for fraud / legit).
        num_nodes: Size of the memory bank; only needed when ``use_memory=True``.
        num_layers: Number of temporal message-passing layers.
        dropout: Dropout after every layer.
        use_memory: Add a per-node GRU memory state to the input embedding.
        bidirectional: Add reverse edges internally (recommended; skip if already symmetric).
        init_lambda: Initial decay rate λ (in units of the *normalised* time span).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_nodes: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_memory: bool = False,
        bidirectional: bool = True,
        init_lambda: float = 1.0,
    ):
        super().__init__()
        if use_memory and not num_nodes:
            raise ValueError("num_nodes is required when use_memory=True.")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_memory = use_memory
        self.bidirectional = bidirectional

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            RBGNNPlusLayer(hidden_dim, dropout, init_lambda) for _ in range(num_layers)
        )
        self.memory: Optional[MemoryModule] = (
            MemoryModule(num_nodes, hidden_dim) if use_memory and num_nodes else None
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def relative_time(
        edge_time: torch.Tensor,
        query_time: Optional[float] = None,
        time_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal, scale-normalised ``Δt`` for every edge.

        Returns ``(dt, keep_mask)``. Edges with ``edge_time > query_time`` are masked out
        (``keep_mask`` is False) and their ``dt`` is clamped to 0. For the kept edges
        ``dt = (query_time - edge_time) / time_scale >= 0``; ``time_scale`` defaults to the span
        (max - min) of the kept edge times.
        """
        if edge_time.numel() == 0:
            return edge_time.clone(), torch.zeros_like(edge_time, dtype=torch.bool)
        q = edge_time.max() if query_time is None else edge_time.new_tensor(float(query_time))
        keep = edge_time <= q
        if not bool(keep.any()):
            return torch.zeros_like(edge_time), keep
        kept = edge_time[keep]
        if time_scale is None:
            scale = (kept.max() - kept.min()).clamp_min(1e-6)
        else:
            if time_scale <= 0:
                raise ValueError("time_scale must be positive.")
            scale = edge_time.new_tensor(float(time_scale))
        dt = ((q - edge_time) / scale).clamp(min=0.0)
        return dt, keep

    def prepare_edges(
        self,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        query_time: Optional[float] = None,
        time_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter future edges, add reverse edges (if enabled) and return ``(edge_index, dt)``."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        if edge_time.dim() != 1 or edge_time.numel() != edge_index.size(1):
            raise ValueError("edge_time must have shape [E] matching edge_index.")
        dt, keep = self.relative_time(edge_time, query_time, time_scale)
        edge_index, dt = edge_index[:, keep], dt[keep]
        if self.bidirectional:
            edge_index, dt = add_reverse_edges(edge_index, dt)
        return edge_index, dt

    # ------------------------------------------------------------------ forward
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        query_time: Optional[float] = None,
        time_scale: Optional[float] = None,
        node_ids: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        """
        Node embeddings ``[N, hidden_dim]``.

        Args:
            x: ``[N, in_dim]`` node features.
            edge_index: ``[2, E]`` (src, dst) event edges.
            edge_time: ``[E]`` event timestamps (any unit).
            query_time: Only edges with ``edge_time <= query_time`` are used (default: newest edge).
            time_scale: Divisor for ``Δt`` (default: span of the used edges).
            node_ids: Global memory ids of the rows of ``x`` (default: ``arange(N)``).
            update_memory: If True (and ``use_memory``), run the memory GRU on the input
                embeddings, use the new state, and write it (detached) back to the bank.
                By default the memory is only *read*.
        """
        if x.dim() != 2 or x.size(1) != self.in_dim:
            raise ValueError(f"x must have shape [N, {self.in_dim}], got {tuple(x.shape)}.")
        h = self.input_proj(x)

        if self.memory is not None:
            ids = torch.arange(x.size(0), device=x.device) if node_ids is None else node_ids
            if ids.numel() != x.size(0):
                raise ValueError("node_ids must have one entry per row of x.")
            h = h + (self.memory.update(ids, h) if update_memory else self.memory.get(ids))

        edge_index, dt = self.prepare_edges(edge_index, edge_time, query_time, time_scale)
        for layer in self.layers:
            h = layer(h, edge_index, dt)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_time: torch.Tensor,
        query_time: Optional[float] = None,
        time_scale: Optional[float] = None,
        node_ids: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        """Class logits ``[N, out_dim]`` (see :meth:`encode` for the arguments)."""
        return self.classifier(
            self.encode(x, edge_index, edge_time, query_time, time_scale, node_ids, update_memory)
        )
