# Architecture Deep Dive

## Hybrid intelligence equation

```
H = αG + βT + γS
```

- **G**: embeddings produced by the Relational Behavior GNN (RB-GNN) from the heterogeneous transaction graph.
- **T**: temporal embeddings produced by the TGNN layer from recent behavioural sequences.
- **S**: static feature embeddings (MLP over tabular features).
- **α, β, γ**: coefficients on the probability simplex, produced by `AttentionFusion`.

### Fusion coefficients

The learnable parameters of `AttentionFusion` are *logits* `ℓ = (ℓ_α, ℓ_β, ℓ_γ)`, not the coefficients.

- Static mode: `(α, β, γ) = softmax(ℓ)`, identical for every sample.
- Dynamic mode: `(α, β, γ) = softmax(ℓ + MLP([G; T; S]))`, per sample. The last MLP layer starts at zero.
- Initialisation: `ℓ = log(prior)`, hence `softmax(ℓ)` equals the prior exactly (default
  `0.40 / 0.35 / 0.25`, normalised if they do not sum to one). Passing the raw prior through a softmax would
  *not* reproduce it (it would give ≈ 0.36 / 0.34 / 0.30).

## Shapes and batching

| Tensor | Shape | Notes |
|--------|-------|-------|
| `graph['user'].x` | `[N_users, F_u]` | All users in the (sub)graph |
| `temporal_sequence` | `[B, L, temporal_dim]` | `temporal_dim` defaults to `hidden_channels` |
| `static_features` | `[B, static_dim]` | |
| `user_index` | `[B]` (int) | Row `i` of the batch is user node `user_index[i]` |
| G (selected) / T / S | `[B, hidden_channels]` | Fed to the fusion layer |

The RB-GNN embeds *every* user node of the graph (`[N_users, hidden]`), whereas T and S exist only for the
`B` entities being scored. `user_index` selects `G_all[user_index]`. When `user_index` is omitted the graph must
contain exactly `B` users, in batch order; otherwise a `ValueError` explains the mismatch.

## RB-GNN schema

| Relation | Reverse |
|----------|---------|
| `user –transacts→ merchant` | `merchant –rev_transacts→ user` |
| `user –uses→ device` | `device –rev_uses→ user` |
| `user –from→ ip` | `ip –rev_from→ user` |
| `user –owns→ card` | `card –rev_owns→ user` |
| `user –shares→ phone` | `phone –rev_shares→ user` |
| `user –shares→ email` | `email –rev_shares→ user` |

Without reverse edges a user would never receive messages, so shared devices/IPs/phones would not influence user
embeddings. `HeteroConv` skips edge types absent from the input data, so graphs with only some relations work.
Each layer applies a residual connection, ReLU, dropout and a `LayerNorm` that is specific to the node type.

## RBGNN+ (temporal relational GNN)

`hybrid_fraud.models.RBGNNPlus` is a node-level encoder for a **homogeneous, timestamped event graph**
(`x: [N, F]`, `edge_index: [2, E]`, `edge_time: [E]`). For an edge `(u → v, t)` the message to `v` is

```
m = (W·h_u + φ(Δt)) · a_uv · exp(-λ·Δt),      Δt = (query_time - t) / time_scale ≥ 0
```

- `a_uv` — edge attention, a softmax of `w·[W·h_u ‖ W·h_v]` over the incoming edges of each destination `v`
  (sums to 1 per destination). The decay multiplies *after* the softmax, so a node whose evidence is old
  receives a weaker total message.
- `φ(Δt)` — learned-frequency cosine time encoding (TGAT-style); `λ = softplus(raw) > 0` is learnable.
- **Causality**: edges with `t > query_time` are dropped before message passing (default `query_time` = newest
  edge). `Δt` is rescaled by the span of the used edges (or an explicit `time_scale`), so the decay does not depend
  on whether timestamps are in seconds or days.
- **Reverse edges** are added internally (`bidirectional=True`) with the same timestamps.
- **Memory** (`use_memory=True`): per-node GRU state in a non-trainable buffer, written detached. By default it is
  only read; `update_memory=True` runs the GRU on the input embeddings and stores the result. The memory persists
  across calls, so call `model.memory.reset()` between independent passes (epochs, evaluation) and never feed it
  state derived from validation / test periods when training.

### Plugging RBGNN+ into fusion

`HybridFraudModel` keeps the GRU `TGNN` for `T` because RBGNN+ needs a timestamped event graph per batch rather
than per-target sequences. To use RBGNN+ as the temporal branch, compute `T` yourself and call the fusion layer:

```python
from hybrid_fraud.models import AttentionFusion, RBGNNPlus

temporal = RBGNNPlus(in_dim=F, hidden_dim=H)
T = temporal.encode(x_events, edge_index, edge_time, query_time=now)[target_rows]   # [B, H]
H_fused, weights = AttentionFusion(embed_dim=H)(G, T, S)                             # G, S: [B, H]
```

Making this a first-class option of `HybridFraudModel` is on the roadmap.

### Synthetic data

`hybrid_fraud.data.generate_synthetic_fraud_graph` is for smoke tests. By default it injects **no** label signal.
`feature_signal` (mean shift of fraud nodes — a label leak into features) and `ring_edge_fraction` (bursty
fraud-to-fraud edges, which also make fraud nodes high-degree) are explicit toy options. Metrics measured on
synthetic data are not benchmarks.

## Why this design?

1. **Relational blindness** of tabular models is addressed by explicit multi-hop message passing.
2. **Concept drift** is addressed by a temporal layer that can adapt to changing behaviour.
3. **Inspectability**: graph structure and fusion weights can both be examined.
4. **Production shape**: hot path (subgraph + GNN) separated from cold path (full GDS embeddings) — planned.

## Neo4j role (planned integration)

Neo4j serves as the system of record for the relational graph:

- Real-time neighbourhood extraction for RB-GNN inference
- Fraud-ring pattern matching via Cypher
- Storage of confirmed fraud labels for continuous learning
- Source of graph features (PageRank, community id, embeddings) for hybrid training

The schema is in [`neo4j/schema.cypher`](../neo4j/schema.cypher); the Python client is on the roadmap.

## Recommended deployment pattern

```
Kafka → Feature Service → Neo4j Subgraph Query → Hybrid Model → Decision
                              ↓
                         Cold Path (GDS)
```
