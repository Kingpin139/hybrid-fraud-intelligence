from pathlib import Path

from hybrid_fraud.config import get_thresholds, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def test_repo_config_parses_with_numeric_types():
    cfg = load_config(REPO_CONFIG)
    assert isinstance(cfg["training"]["weight_decay"], float)  # "1e-5" would be a str in YAML 1.1
    assert isinstance(cfg["training"]["learning_rate"], float)
    assert cfg["model"]["use_dynamic_attention"] is True
    f = cfg["fusion"]
    assert abs(f["initial_alpha"] + f["initial_beta"] + f["initial_gamma"] - 1.0) < 1e-9


def test_thresholds_from_config_and_defaults():
    assert get_thresholds(load_config(REPO_CONFIG)) == {"approve": 0.30, "review": 0.65}
    assert get_thresholds({}) == {"approve": 0.30, "review": 0.65}


def test_missing_file_yields_empty_config(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == {}


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", "s3cret-pass")
    cfg = load_config(REPO_CONFIG)
    assert cfg["neo4j"]["uri"] == "bolt://neo4j:7687"
    assert cfg["neo4j"]["password"] == "s3cret-pass"
    assert cfg["neo4j"]["user"] == "neo4j"  # untouched


def test_hfi_config_env_var(monkeypatch, tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("risk_thresholds: {approve: 0.1, review: 0.2}\n")
    monkeypatch.setenv("HFI_CONFIG", str(p))
    assert get_thresholds(load_config()) == {"approve": 0.1, "review": 0.2}
