# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-27

Initial public release.

### Added
- `RBGNN`: heterogeneous GraphSAGE network (`HeteroConv` + `SAGEConv`) over users, cards,
  devices, IPs, merchants, phones and e-mails, with reverse edges for every relation
  (including `rev_shares` for phone / e-mail).
- `TGNN`: GRU-based temporal encoder.
- `AttentionFusion`: `H = αG + βT + γS` with static (global) or dynamic (per-sample) coefficients.
- `HybridFraudModel`: end-to-end model with an optional `user_index` argument that selects the
  target users' rows of `G`, plus `HybridFraudModel.from_config(...)`.
- `hybrid_fraud.config`: YAML config loader with `HFI_CONFIG` and `NEO4J_*` environment overrides.
- `risk_engine`: risk-score → `approve` / `review` / `decline` mapping, stable sigmoid calibration.
- FastAPI service (`/health`, `/score`, `/batch_score`) — scoring is a clearly-labelled
  **placeholder heuristic** (`mode: "placeholder"`) until the trained model and Neo4j subgraph
  extraction are wired in.
- Neo4j schema, Dockerfile, docker-compose, example script, docs.
- GitHub Actions CI (ruff + pytest on Python 3.10 / 3.11, CPU PyTorch), issue / PR templates.
- 47 tests covering models, config, risk engine, API and the example.

### Fixed (relative to the pre-release sources)
- Test graph built merchant edges with `randint(0, 5)` although only 3 merchants exist; source and
  destination indices are now drawn from their own node-type ranges.
- `HybridFraudModel` used every user-node embedding as `G` while `T` / `S` are per batch; target rows
  are now selected with `user_index`, and mismatches raise a clear `ValueError`.
- `('user', 'shares', 'phone' | 'email')` had no reverse edges, so those signals never reached users.
- A single `LayerNorm` was shared by all node types; it is now one per node type and layer.
- Node types without incoming messages were silently dropped after a layer; they are now kept (residual).
- `AttentionFusion`: the learnable α/β/γ were passed through a softmax (so they did not start at
  0.40/0.35/0.25) and were unused when dynamic attention was on. They are now logits initialised with
  `log(prior)`, and are added to the attention logits in dynamic mode.
- `config.yaml`: `weight_decay: 1e-5` was parsed by YAML as a *string*; now `1.0e-5`. Removed the
  unused/ambiguous `decline` threshold and aligned API thresholds with the config.
- `calibrate_score` raised `OverflowError` for large negative logits.
- API no longer returns fabricated fusion weights, component scores or confidence.
- Dockerfile did not copy `README.md` (required by `pyproject.toml`), used an editable install and
  the CUDA PyTorch wheel; it now installs CPU PyTorch, runs as a non-root user and has a healthcheck.
- docker-compose: removed obsolete `version`, password configurable through `.env`.
- `.gitignore` pattern `data/` would also have ignored the `hybrid_fraud/data` package.
- Placeholder `your-org` URLs replaced with `Kingpin139`.

[0.1.0]: https://github.com/Kingpin139/hybrid-fraud-intelligence/releases/tag/v0.1.0
