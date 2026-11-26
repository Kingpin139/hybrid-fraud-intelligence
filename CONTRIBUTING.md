# Contributing to Hybrid Fraud Intelligence

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/Kingpin139/hybrid-fraud-intelligence.git
cd hybrid-fraud-intelligence
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build keeps it small
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length 100):

```bash
ruff format .
ruff check . --fix
```

## Pull request process

1. Fork the repository and create a feature branch.
2. Add tests for new functionality (`tests/`).
3. Make sure `pytest`, `ruff check .` and `ruff format --check .` pass.
4. Update the README / docs / `CHANGELOG.md` if behaviour changes.
5. Open a pull request with a clear description.

Please do not add performance numbers unless they come from a reproducible script that is part of the PR.

## Areas for contribution

- Additional temporal GNN architectures (TGAT, TGN)
- Dataset loaders (IEEE-CIS, PaySim, Elliptic) and graph builders
- Training and evaluation scripts
- Neo4j subgraph sampling
- Explanation layer (path tracing, LLM prompts)
