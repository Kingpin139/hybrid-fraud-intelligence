"""Configuration loading (YAML file + environment overrides)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

#: Environment variable pointing at the YAML config file.
CONFIG_ENV_VAR = "HFI_CONFIG"
DEFAULT_CONFIG_PATH = Path("config/config.yaml")

#: ``NEO4J_*`` environment variables override the ``neo4j`` section of the YAML file.
_NEO4J_ENV = {
    "NEO4J_URI": "uri",
    "NEO4J_USER": "user",
    "NEO4J_PASSWORD": "password",
    "NEO4J_DATABASE": "database",
}

DEFAULT_THRESHOLDS = {"approve": 0.30, "review": 0.65}


def load_config(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Load the YAML configuration.

    Resolution order for the file: explicit ``path`` → ``$HFI_CONFIG`` →
    ``config/config.yaml`` (relative to the working directory). A missing file yields
    an empty config (callers fall back to built-in defaults). ``NEO4J_*`` environment
    variables override the corresponding ``neo4j`` keys.
    """
    resolved: Optional[Path] = Path(path) if path else None
    if resolved is None:
        env = os.environ.get(CONFIG_ENV_VAR)
        resolved = Path(env) if env else DEFAULT_CONFIG_PATH

    config: Dict[str, Any] = {}
    if resolved.is_file():
        with resolved.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

    neo4j_cfg = dict(config.get("neo4j") or {})
    for env_name, key in _NEO4J_ENV.items():
        if env_name in os.environ:
            neo4j_cfg[key] = os.environ[env_name]
    if neo4j_cfg:
        config["neo4j"] = neo4j_cfg
    return config


def get_thresholds(config: Dict[str, Any]) -> Dict[str, float]:
    """Return the ``approve`` / ``review`` decision thresholds from a config dict."""
    cfg = config.get("risk_thresholds") or {}
    return {k: float(cfg.get(k, v)) for k, v in DEFAULT_THRESHOLDS.items()}
