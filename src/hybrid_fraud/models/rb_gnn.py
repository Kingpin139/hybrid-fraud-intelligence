"""
Relational Behavior Graph Neural Network (RB-GNN).

Models heterogeneous relationships among users, cards, devices, IPs, merchants,
phones and e-mail addresses to capture coordinated fraud patterns and multi-hop
relational signals.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

EdgeType = Tuple[str, str, str]

#: Default fraud-graph schema. Every forward relation has an explicit reverse
#: relation (``rev_<relation>``) so that messages flow *back* to users, which is
#: what makes shared devices / IPs / phones / e-mails informative for a user.
DEFAULT_EDGE_TYPES: List[EdgeType] = [
    ("user", "transacts", "merchant"),
    ("user", "uses", "device"),
    ("user", "from", "ip"),
    ("user", "owns", "card"),
    ("user", "shares", "phone"),
    ("user", "shares", "email"),
    # Reverse edges for bidirectional message passing
    ("merchant", "rev_transacts", "user"),
    ("device", "rev_uses", "user"),
    ("ip", "rev_from", "user"),
    ("card", "rev_owns", "user"),
    ("phone", "rev_shares", "user"),
    ("email", "rev_shares", "user"),
]


class RBGNN(nn.Module):
    """
    Relational Behavior Graph Neural Network.

    Uses heterogeneous message passing (one ``SAGEConv`` per edge type) over the
    multi-type nodes and edges commonly found in fraud graphs:

    - Nodes: ``user``, ``card``, ``device``, ``ip``, ``merchant``, ``phone``, ``email``
    - Edges: ``transacts``, ``uses``, ``from``, ``owns``, ``shares`` (+ ``rev_*`` reverses)

    Design notes:

    * Each node type first goes through its own input projection to
      ``hidden_channels`` (a ``LazyLinear`` unless ``in_channels`` is given), so
      node types may have different raw feature sizes.
    * The schema (``edge_types``) is fixed at construction time. At run time only
      the edge types actually present in the input ``HeteroData`` are used; absent
      relations are skipped (``HeteroConv`` ignores them), so sparse batches work.
    * Node types that receive no messages in a layer keep their previous
      embedding (residual connection) instead of being dropped.
    * Normalisation is a separate ``LayerNorm`` **per node type and layer**.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_types: Optional[Iterable[EdgeType]] = None,
        in_channels: Union[int, Dict[str, int], None] = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_types: List[EdgeType] = [tuple(e) for e in (edge_types or DEFAULT_EDGE_TYPES)]

        node_types: List[str] = []
        for src, _, dst in self.edge_types:
            for nt in (src, dst):
                if nt not in node_types:
                    node_types.append(nt)
        self.node_types = node_types

        def _in(nt: str) -> Optional[int]:
            if isinstance(in_channels, dict):
                return in_channels.get(nt)
            return in_channels

        self.input_proj = nn.ModuleDict(
            {
                nt: (
                    nn.Linear(_in(nt), hidden_channels)
                    if _in(nt) is not None
                    else nn.LazyLinear(hidden_channels)
                )
                for nt in node_types
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        et: SAGEConv((hidden_channels, hidden_channels), hidden_channels)
                        for et in self.edge_types
                    },
                    aggr="sum",
                )
            )
            # Per-node-type LayerNorm (statistics differ a lot between node types)
            self.norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            )

        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing node embeddings for all known node types in ``data``.

        Args:
            data: Heterogeneous graph with node features (``x``) and edge indices.

        Returns:
            Dictionary mapping node type → embedding tensor ``[num_nodes, hidden_channels]``.
        """
        x_raw = data.collect("x", allow_empty=True)
        x_dict = {nt: self.input_proj[nt](x) for nt, x in x_raw.items() if nt in self.input_proj}
        if not x_dict:
            raise ValueError(
                f"HeteroData has no node features for any known node type {self.node_types}."
            )

        known = set(self.edge_types)
        edge_index_dict = {
            et: ei for et, ei in data.collect("edge_index", allow_empty=True).items() if et in known
        }

        for conv, norms in zip(self.convs, self.norms, strict=True):
            out = conv(x_dict, edge_index_dict)
            new_x = {}
            for nt, h in x_dict.items():
                if nt in out:
                    upd = norms[nt](F.relu(out[nt]))
                    upd = F.dropout(upd, p=self.dropout, training=self.training)
                    h = h + upd  # residual: also keeps nodes with no incoming messages
                new_x[nt] = h
            x_dict = new_x

        return {nt: self.lin(h) for nt, h in x_dict.items()}

    def get_user_embeddings(self, data: HeteroData) -> torch.Tensor:
        """Convenience method returning only user node embeddings (primary G signal)."""
        embeddings = self.forward(data)
        if "user" not in embeddings:
            raise ValueError("HeteroData must contain 'user' node features.")
        return embeddings["user"]
