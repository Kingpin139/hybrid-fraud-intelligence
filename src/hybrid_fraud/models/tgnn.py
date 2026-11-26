"""
Temporal encoder (TGNN layer).

Captures evolving behavioral patterns, sudden changes, velocity spikes,
and coordinated temporal activity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TGNN(nn.Module):
    """
    Temporal encoder for a sequence of node or subgraph embeddings (GRU-based).

    This is a sequence encoder rather than a full temporal GNN; it can be replaced
    by TGAT, TGN or DySAT style models when richer temporal edge information is available.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a temporal sequence into a single embedding.

        Args:
            temporal_sequence: Tensor of shape ``[batch, seq_len, features]``

        Returns:
            Temporal embedding of shape ``[batch, hidden_channels]``
        """
        if temporal_sequence.dim() != 3 or temporal_sequence.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected temporal_sequence of shape [batch, seq_len, {self.in_channels}], "
                f"got {tuple(temporal_sequence.shape)}."
            )
        _, hidden = self.gru(temporal_sequence)
        # Use last layer hidden state
        last_hidden = hidden[-1]
        return self.dropout(self.out_proj(last_hidden))
