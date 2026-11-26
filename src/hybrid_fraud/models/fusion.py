"""
Attention Fusion Layer.

Implements the hybrid intelligence equation:

    H = αG + βT + γS

where (α, β, γ) lie on the probability simplex (they are non-negative and sum to 1).
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Fuses Graph (G), Temporal (T), and Static (S) embeddings.

    The coefficients are always produced by a softmax, so they sum to 1:

    * ``use_dynamic_attention=False`` — three learnable *logits* are initialised to
      ``log(init_weights)``, so that ``softmax(logits)`` equals ``init_weights``
      exactly at initialisation (default ``0.40 / 0.35 / 0.25``). The coefficients are
      then global (the same for every sample).
    * ``use_dynamic_attention=True`` — a small MLP predicts per-sample logits that are
      *added* to the same learnable base logits. The last MLP layer is zero-initialised,
      so training starts from the prior ``init_weights`` and moves away from it as needed.

    Note that the learnable parameters are logits, not the coefficients themselves.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        use_dynamic_attention: bool = True,
        init_weights: Sequence[float] = (0.40, 0.35, 0.25),
    ):
        super().__init__()
        if len(init_weights) != 3 or any(w <= 0 for w in init_weights):
            raise ValueError("init_weights must be three positive numbers (alpha, beta, gamma).")
        total = float(sum(init_weights))
        prior = [w / total for w in init_weights]  # normalise so they sum to 1

        self.embed_dim = embed_dim
        self.use_dynamic_attention = use_dynamic_attention

        # Learnable base logits: softmax(base_logits) == prior at init.
        self.base_logits = nn.Parameter(torch.tensor([math.log(w) for w in prior]))

        if use_dynamic_attention:
            self.attention = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 3),
            )
            nn.init.zeros_(self.attention[-1].weight)
            nn.init.zeros_(self.attention[-1].bias)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        G: torch.Tensor,
        T: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Fuse the three embedding sources.

        Args:
            G: Graph embeddings [batch, embed_dim]
            T: Temporal embeddings [batch, embed_dim]
            S: Static feature embeddings [batch, embed_dim]

        Returns:
            H: Fused representation [batch, embed_dim]
            weights: Batch-mean coefficients ``{"alpha", "beta", "gamma"}`` (floats, for
                logging / inspection; they sum to 1).
        """
        if not (G.shape == T.shape == S.shape) or G.dim() != 2 or G.shape[-1] != self.embed_dim:
            raise ValueError(
                "G, T and S must all have shape [batch, embed_dim="
                f"{self.embed_dim}], got {tuple(G.shape)}, {tuple(T.shape)}, {tuple(S.shape)}."
            )

        if self.use_dynamic_attention:
            logits = self.base_logits + self.attention(torch.cat([G, T, S], dim=-1))  # [B, 3]
            w = F.softmax(logits, dim=-1)
        else:
            w = F.softmax(self.base_logits, dim=-1).unsqueeze(0).expand(G.shape[0], -1)  # [B, 3]

        alpha, beta, gamma = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        H = self.out(alpha * G + beta * T + gamma * S)

        mean_w = w.detach().mean(dim=0)
        weights = {
            "alpha": mean_w[0].item(),
            "beta": mean_w[1].item(),
            "gamma": mean_w[2].item(),
        }
        return H, weights
