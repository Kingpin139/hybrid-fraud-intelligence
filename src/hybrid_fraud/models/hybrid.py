"""
Complete Hybrid Model combining RB-GNN, TGNN, and Attention Fusion.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .fusion import AttentionFusion
from .rb_gnn import RBGNN
from .tgnn import TGNN


class HybridFraudModel(nn.Module):
    """
    End-to-end hybrid fraud detection model.

    Implements: H = αG + βT + γS

    * **G** — RB-GNN embeddings of the ``user`` nodes of the graph.
    * **T** — TGNN embedding of a per-target behavioural sequence.
    * **S** — encoded static features, one row per target.

    ``T`` and ``S`` are *per target* (one row per scored entity), whereas the graph
    contains *all* user nodes. Use ``user_index`` to say which user nodes in the graph
    correspond to the rows of ``temporal_sequence`` / ``static_features``.
    """

    def __init__(
        self,
        static_dim: int = 32,
        hidden_channels: int = 64,
        num_gnn_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        temporal_dim: Optional[int] = None,
        use_dynamic_attention: bool = True,
        fusion_init_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
    ):
        super().__init__()
        self.static_dim = static_dim
        self.hidden_channels = hidden_channels
        self.temporal_dim = temporal_dim if temporal_dim is not None else hidden_channels

        # Static feature encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Relational GNN
        self.rb_gnn = RBGNN(
            hidden_channels=hidden_channels, num_layers=num_gnn_layers, dropout=dropout
        )

        # Temporal encoder
        self.tgnn = TGNN(
            in_channels=self.temporal_dim, hidden_channels=hidden_channels, dropout=dropout
        )

        # Attention fusion
        self.fusion = AttentionFusion(
            embed_dim=hidden_channels,
            use_dynamic_attention=use_dynamic_attention,
            init_weights=fusion_init_weights,
        )

        # Risk head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HybridFraudModel:
        """Build a model from the parsed ``config.yaml`` (``model`` and ``fusion`` sections)."""
        m = config.get("model", {})
        f = config.get("fusion", {})
        return cls(
            static_dim=m.get("static_feature_dim", 32),
            hidden_channels=m.get("hidden_channels", 64),
            num_gnn_layers=m.get("num_gnn_layers", 2),
            dropout=m.get("dropout", 0.1),
            use_dynamic_attention=m.get("use_dynamic_attention", True),
            fusion_init_weights=(
                f.get("initial_alpha", 0.40),
                f.get("initial_beta", 0.35),
                f.get("initial_gamma", 0.25),
            ),
        )

    def forward(
        self,
        graph: HeteroData,
        temporal_sequence: torch.Tensor,
        static_features: torch.Tensor,
        user_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full forward pass.

        Args:
            graph: Heterogeneous graph; must contain ``user`` node features.
            temporal_sequence: ``[batch, seq_len, temporal_dim]`` behavioural sequences.
            static_features: ``[batch, static_dim]`` static features.
            user_index: Optional 1-D integer tensor of length ``batch`` with the indices of
                the target users in ``graph['user']``; row ``i`` of the batch corresponds to
                user node ``user_index[i]``. If ``None``, the graph must contain exactly
                ``batch`` user nodes, in batch order.

        Returns:
            logits: Classification logits ``[batch, num_classes]``
            fusion_weights: Batch-mean attention coefficients (alpha, beta, gamma)
        """
        if static_features.dim() != 2 or static_features.shape[-1] != self.static_dim:
            raise ValueError(
                f"static_features must have shape [batch, {self.static_dim}], "
                f"got {tuple(static_features.shape)}."
            )
        batch = static_features.shape[0]
        if temporal_sequence.dim() != 3 or temporal_sequence.shape[0] != batch:
            raise ValueError(
                f"temporal_sequence must have shape [batch={batch}, seq_len, {self.temporal_dim}], "
                f"got {tuple(temporal_sequence.shape)}."
            )

        # Static embeddings (S)
        S = self.static_encoder(static_features)

        # Graph embeddings (G) for ALL user nodes, then select the batch rows
        G_all = self.rb_gnn(graph).get("user")
        if G_all is None:
            raise ValueError("graph must contain 'user' node features.")

        if user_index is None:
            if G_all.shape[0] != batch:
                raise ValueError(
                    f"The graph has {G_all.shape[0]} user nodes but the batch has {batch} rows. "
                    "Pass `user_index` (indices of the target users in graph['user']) to select "
                    "the rows of G that correspond to the batch."
                )
            G = G_all
        else:
            user_index = torch.as_tensor(user_index, device=G_all.device)
            if user_index.dtype not in (torch.int32, torch.int64) or user_index.dim() != 1:
                raise ValueError("user_index must be a 1-D integer tensor.")
            if user_index.numel() != batch:
                raise ValueError(
                    f"user_index has {user_index.numel()} entries but the batch has {batch} rows."
                )
            if user_index.numel() and (
                int(user_index.min()) < 0 or int(user_index.max()) >= G_all.shape[0]
            ):
                raise IndexError(
                    f"user_index values must be in [0, {G_all.shape[0]}), "
                    f"got range [{int(user_index.min())}, {int(user_index.max())}]."
                )
            G = G_all[user_index.long()]

        # Temporal embeddings (T)
        T = self.tgnn(temporal_sequence)

        # Fuse
        H, weights = self.fusion(G, T, S)

        # Risk score
        logits = self.risk_head(H)
        return logits, weights
