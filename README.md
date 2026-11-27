# Hybrid Fraud Intelligence

**A relational-temporal graph neural network (RB-GNN + TGNN + attention fusion) for real-time fraud detection.**

[![CI](https://github.com/Kingpin139/hybrid-fraud-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Kingpin139/hybrid-fraud-intelligence/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.4+-brightgreen.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Tabular models and pure GNNs miss coordinated, evolving fraud. This project combines three signals
in a single differentiable model:

```
H = αG + βT + γS
```

| Symbol | Meaning | Produced by |
|--------|---------|-------------|
| **G** | Graph embedding of the user (shared devices, IPs, cards, merchants, phones, e-mails) | `RBGNN` – heterogeneous GraphSAGE |
| **T** | Temporal embedding of recent behaviour (velocity, bursts, drift) | `TGNN` – GRU sequence encoder (or `RBGNNPlus` embeddings, see below) |
| **S** | Static feature embedding | MLP static encoder |
| **α, β, γ** | Fusion coefficients (non-negative, sum to 1) | `AttentionFusion` – softmax over learnable logits (+ optional per-sample attention) |

> **Project status: alpha (v0.1.0).** The model code, config, Neo4j schema, Docker setup and tests are in
> place, plus an IEEE-CIS loader and a time-split training script. **No benchmark has been run in this
> repository yet; Neo4j subgraph extraction, mini-batch training and LLM explanations are not implemented**
> (see the [roadmap](#roadmap)). The `/score` API endpoint currently returns a
> clearly-labelled *placeholder* heuristic (`"mode": "placeholder"`), not model output.

## Architecture

```mermaid
flowchart LR
    TX[Transaction] --> FE[Feature engineering]
    FE --> SG["Subgraph extraction<br/>(Neo4j / PyG HeteroData)"]
    FE --> SEQ[Behaviour sequence]
    FE --> STAT[Static features]

    SG --> RB["RB-GNN<br/>(HeteroConv + SAGEConv)"]
    RB -->|"all user nodes"| SEL["select target rows<br/>(user_index)"]
    SEQ --> TG["TGNN<br/>(GRU)"]
    STAT --> SE[Static encoder]

    SEL -->|G| FU
    TG -->|T| FU
    SE -->|S| FU
    FU["Attention fusion<br/>H = αG + βT + γS"] --> RH[Risk head]
    RH --> RE["Risk engine<br/>thresholds"]
    RE --> D{{"approve / review / decline"}}

    classDef planned stroke-dasharray: 5 5;
    class SG,FE planned;
```

Dashed boxes are planned components (not yet implemented); the models, fusion, risk engine and API are implemented.

### Components

- **RB-GNN** (`hybrid_fraud.models.RBGNN`) – one `SAGEConv` per edge type inside a `HeteroConv`.
  - Nodes: `user`, `card`, `device`, `ip`, `merchant`, `phone`, `email`.
  - Edges: `transacts`, `uses`, `from`, `owns`, `shares` plus a reverse (`rev_*`) relation for each,
    including `rev_shares` for phones and e-mails, so shared identifiers flow back to users.
  - Per-node-type input projections (feature sizes may differ) and per-node-type `LayerNorm`s.
  - Only edge types present in the input `HeteroData` are used; node types that receive no messages keep
    their previous embedding (residual connection).
- **TGNN** (`TGNN`) – GRU over a `[batch, seq_len, features]` sequence; can be replaced with TGAT/TGN/DySAT.
- **RBGNN+** (`hybrid_fraud.models.RBGNNPlus`) – temporal relational GNN for a timestamped event graph:
  learnable exponential temporal decay, TGAT-style time encoding, edge attention softmax-normalised per
  destination node, reverse edges, causal scale-normalised `Δt` (edges after `query_time` are ignored) and an
  optional per-node GRU memory (detached buffer with explicit `reset()`). It is a **standalone module**:
  `HybridFraudModel` still uses the GRU `TGNN`, and `RBGNNPlus.encode(...)[target_rows]` can be passed as `T`
  to `AttentionFusion` (see [docs/architecture.md](docs/architecture.md#rbgnn-temporal-relational-gnn)).
  A runnable demo on **synthetic** data is in [`examples/rbgnn_plus_demo.py`](examples/rbgnn_plus_demo.py)
  — it is a smoke test, not a benchmark.
- **Attention fusion** (`AttentionFusion`) – the learnable parameters are **logits**; coefficients are
  `softmax(logits)`. They are initialised with `log(prior)`, so training starts exactly at the configured
  `0.40 / 0.35 / 0.25`. With `use_dynamic_attention=True`, an MLP adds per-sample logits (zero-initialised,
  so it starts at the prior). The returned `weights` dict is the batch mean.
- **Hybrid model** (`HybridFraudModel`) – G comes from *all* user nodes in the graph, while T and S are
  per scored entity. Pass `user_index` (indices of the target users in `graph['user']`) to select the
  matching rows of G; without it the graph must contain exactly `batch` users, otherwise a clear
  `ValueError` is raised.
- **Risk engine** – maps a risk score to `approve` (< 0.30), `review` (< 0.65) or `decline`, configurable in
  `config/config.yaml`.
- **Neo4j schema** (`neo4j/schema.cypher`) – constraints, indexes and a sample fraud-ring query.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Kingpin139/hybrid-fraud-intelligence.git
cd hybrid-fraud-intelligence

python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch (skip --index-url if you want the default build / a GPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"
```

## Usage

### Python

```python
import torch
from torch_geometric.data import HeteroData
from hybrid_fraud.models import HybridFraudModel

graph = HeteroData()
graph["user"].x = torch.randn(6, 32)
graph["merchant"].x = torch.randn(3, 32)
graph["user", "transacts", "merchant"].edge_index = torch.tensor([[0, 1, 2, 5], [0, 1, 2, 0]])
graph["merchant", "rev_transacts", "user"].edge_index = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 5]])

model = HybridFraudModel(static_dim=32, hidden_channels=64).eval()

user_index = torch.tensor([5, 0, 2])        # score users 5, 0 and 2 of the graph
temporal = torch.randn(3, 8, 64)            # [batch, seq_len, hidden_channels]
static = torch.randn(3, 32)                 # [batch, static_dim]

with torch.no_grad():
    logits, weights = model(graph, temporal, static, user_index=user_index)
risk = logits.softmax(-1)[:, 1]             # untrained model: meaningless numbers
```

A runnable version is in [`examples/inference_demo.py`](examples/inference_demo.py):

```bash
python examples/inference_demo.py
```

#### RBGNN+ on a timestamped event graph

```python
from hybrid_fraud.data import generate_synthetic_fraud_graph
from hybrid_fraud.models import RBGNNPlus

x, edge_index, edge_time, y = generate_synthetic_fraud_graph(num_nodes=200, num_edges=800)  # synthetic
model = RBGNNPlus(in_dim=x.size(1), hidden_dim=32, out_dim=2)
logits = model(x, edge_index, edge_time)                      # [200, 2]
emb = model.encode(x, edge_index, edge_time, query_time=60.0) # only edges with t <= 60 are used
```

```bash
python examples/rbgnn_plus_demo.py   # synthetic data, stratified split, baseline + ablation
```

The generator injects **no** label signal by default; `feature_signal` and `ring_edge_fraction` are explicit
toy options. Numbers from the demo are smoke-test output on synthetic data, not benchmarks.

Build a model from the YAML config with `HybridFraudModel.from_config(load_config())`.

### Data: IEEE-CIS Fraud Detection

The loader in `hybrid_fraud.data.ieee_cis` reads the Kaggle
[`ieee-fraud-detection`](https://www.kaggle.com/c/ieee-fraud-detection) files. The dataset is **not**
included and must not be committed (`data/` is git-ignored). Accept the competition rules on Kaggle, then:

```bash
pip install kaggle                       # API token in ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY
kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection
```

```python
from hybrid_fraud.data import build_ieee_cis_graph, load_ieee_cis, snapshot, time_split, to_event_graph

df = load_ieee_cis("data/ieee-fraud-detection", nrows=100_000)  # chronological prefix; V1-V339 opt-in
split = time_split(df, val_frac=0.15, test_frac=0.15)           # by TransactionDT, no shuffling
graph = build_ieee_cis_graph(df, split)                          # HeteroData + fitted featurizer
train_view = snapshot(graph.data, split.train_end)               # edges up to the end of train only
events = to_event_graph(graph.data)                              # homogeneous view for RBGNNPlus
```

- **Nodes:** `transaction` (features, `y = isFraud`, `time`, `train/val/test_mask`) and the entities `card`
  (pseudo-user, see below), `device` (DeviceInfo), `email` (P_ and R_ domains share one node set), `address`
  (addr1 + addr2) and `product` (ProductCD). Every relation has a `rev_*` reverse edge, and every edge has
  `edge_time = TransactionDT`.
- **Card / pseudo-user key** (`card_key`): `card1`, `card1_addr1`, or (default) `card1_addr1_d1`. The default
  adds `floor(TransactionDT / 86400) - D1`, which is constant per account if D1 means "days since the
  account was first seen" — a common heuristic from public solutions, not ground truth.
- **Leakage controls:** the featurizer (imputation, scaling, category vocabularies) is fit on the training
  period only. `isFraud`, `TransactionID` and raw `TransactionDT` are never features, and there is no target
  encoding. Entity features are `[1, log1p(train-period degree)]`. Train, validate and test on time snapshots
  (or with `RBGNNPlus(query_time=...)`) so no prediction uses later transactions.
- **Memory:** columns are downcast (`float32` / `category`). V1–V339 are skipped unless you ask for them
  (`v_columns=["V258", ...]` or `"all"`), and `nrows` / `sample_frac` limit the rows.

Train the logistic baseline, RBGNN+ and the heterogeneous RB-GNN on the same time split:

```bash
python examples/train_ieee_cis.py --data-dir data/ieee-fraud-detection --nrows 100000 --epochs 50
```

Add `--model logistic,lightgbm` (after `pip install -e ".[lightgbm]"`) for a gradient-boosted tree
baseline, `--lgbm-raw-v` to give it the raw V1–V339 columns, and `--epochs 200 --patience 25` for early
stopping of the graph models.

The script prints val / test ROC-AUC and PR-AUC exactly as measured, with `--output results.json` to save
them. Without the data it exits with the download instructions above. Training is full-batch: start with
`--nrows`; mini-batch neighbour sampling is on the roadmap.

### API

```bash
uvicorn hybrid_fraud.api.main:app --reload     # run from the repository root
```

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_001",
    "user_id": "user_123",
    "amount": 149.99,
    "merchant_id": "merch_456",
    "device_id": "dev_789",
    "timestamp": "2026-11-27T00:00:00Z"
  }'
```

Endpoints: `GET /health`, `POST /score`, `POST /batch_score` (max 1000 per call). Interactive docs at
`/docs`. Responses carry `"mode": "placeholder"` until the trained model is wired in.

### Docker (API + Neo4j)

```bash
cp .env.example .env        # set NEO4J_PASSWORD (>= 8 characters)
docker compose up --build
```

This starts Neo4j (browser on `:7474`, Bolt on `:7687`) and the API on `:8000`. Load the schema with:

```bash
docker compose exec neo4j sh -c 'cypher-shell -u neo4j -p "${NEO4J_AUTH#neo4j/}" -f /schema/schema.cypher'
```

### Configuration

`config/config.yaml` holds model, fusion prior, risk thresholds, Neo4j, API and (reserved) training
settings. Set `HFI_CONFIG` to use another file; `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` and
`NEO4J_DATABASE` override the `neo4j` section. See [`.env.example`](.env.example).

## Project structure

```
hybrid-fraud-intelligence/
├── src/hybrid_fraud/
│   ├── models/          # RBGNN, RBGNNPlus, TGNN, AttentionFusion, HybridFraudModel
│   ├── data/            # IEEE-CIS loader + graph builder, synthetic toy data
│   ├── scoring/         # Risk engine (score -> decision)
│   ├── api/             # FastAPI service
│   └── config.py        # YAML + environment configuration
├── config/              # config.yaml
├── neo4j/               # Cypher schema and sample query
├── examples/            # Inference demo, RBGNN+ synthetic demo, IEEE-CIS training script
├── tests/               # pytest suite
├── docs/                # Architecture notes
├── .github/             # CI workflow, issue and PR templates
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Testing

```bash
pytest                        # unit + API tests
ruff check . && ruff format --check .
```

CI runs ruff and pytest on Python 3.10 and 3.11 with CPU-only PyTorch.

## Expected performance (hypothetical targets – not measured)

> ⚠️ **These numbers are design targets / hypotheses, not results.** No benchmark has been run in this
> repository yet: there is no trained model, dataset loader or evaluation script. Do not cite them as
> measured performance. They will be replaced by reproducible results (with dataset, split and seeds)
> once the evaluation pipeline exists.

| Model                    | Target AUC (hypothetical) |
|--------------------------|---------------------------|
| Logistic Regression      | 0.81                      |
| XGBoost                  | 0.89                      |
| Pure GNN                 | 0.93                      |
| **Hybrid RB-GNN + TGNN** | **0.97+**                 |

## Roadmap

- [x] IEEE-CIS loader + `HeteroData` / event-graph builder with a time-based split, and a training script
- [ ] Run and publish IEEE-CIS results (dataset, split, seeds, mean ± std)
- [ ] More dataset loaders (PaySim, Elliptic, Amazon review fraud)
- [ ] Mini-batch training with time-respecting neighbour sampling, for the full 590k-transaction graph
- [ ] Training / evaluation scripts (focal loss, class-imbalance handling) and measured benchmarks
- [ ] Neo4j client: k-hop subgraph extraction within the latency budget (`neo4j.subgraph_timeout_ms`)
- [ ] Wire trained weights into the API (replace the placeholder heuristic)
- [x] RBGNN+ temporal relational GNN (decay, time encoding, per-destination attention, memory) — standalone, synthetic-data demo only
- [ ] Wire `RBGNNPlus` into `HybridFraudModel` as a selectable temporal encoder (needs a timestamped event graph per batch)
- [ ] Full temporal GNN variants: chronological batching with time-respecting neighbour sampling (TGAT), TGN message store
  and memory updates, DySAT
- [ ] Experiment runner: seeds, time-based splits, baselines (logistic / MLP / static SAGE), mean ± std over runs
- [ ] Explainability: path tracing, fusion-weight reporting, optional LLM narratives
- [ ] Streaming ingestion (e.g. Kafka) and online updates for concept drift

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and ideas are welcome via issues.

## Citation

```bibtex
@software{hybrid_fraud_intelligence,
  title  = {Hybrid Fraud Intelligence: RB-GNN + TGNN Architecture},
  author = {Mahtab, Tanvir},
  year   = {2026},
  url    = {https://github.com/Kingpin139/hybrid-fraud-intelligence}
}
```

## License

[MIT](LICENSE)
