"""
Leakage-safe entity ("graph neighbourhood") features for tabular models such as LightGBM.

Every transaction is linked to entities (pseudo-user, card1, address, e-mail domains, device) — the
same keys as the IEEE-CIS graph. For each entity, this module summarises the transaction's
*one-hop neighbourhood in the past*:

* ``cnt_past`` — earlier transactions of the entity (strictly earlier timestamp).
* ``amt_mean_past`` / ``amt_ratio`` — mean amount of those transactions, and amount / that mean.
* ``secs_since_last`` — seconds since the entity's previous (strictly earlier) transaction.
* ``fraud_te`` / ``labeled_past`` — smoothed fraud rate over the entity's earlier transactions
  **that are in the training period**, and how many such labels exist.

Leakage rules:

* Only strictly earlier rows contribute (ties at the same second are excluded), so a row never sees
  itself or anything later.
* Labels come from the **training period only**. Because the time split is strict
  (``max(train) < min(val)``), a validation / test row sees the labels of *all* training rows of its
  entity and nothing else; val / test labels are never read.
* Caveat: for training rows the target encoding uses labels of earlier training rows as if they
  were known at that moment. In IEEE-CIS, labels arrive with a chargeback delay, so this is
  optimistic for training (not for evaluation).
* The smoothing prior is the overall training fraud rate (one scalar for all rows).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .ieee_cis import AMOUNT_COL, LABEL_COL, TIME_COL, TimeSplit, _key_series, entity_keys

#: (feature prefix, relation key in :func:`entity_keys` or ``None`` for a raw column)
DEFAULT_ENTITIES: List[Tuple[str, str]] = [
    ("uid", "uses_card"),
    ("card1", "card1"),
    ("address", "at_address"),
    ("p_email", "p_email"),
    ("r_email", "r_email"),
    ("device", "uses_device"),
]
FEATURES_PER_ENTITY = (
    "cnt_past",
    "amt_mean_past",
    "amt_ratio",
    "secs_since_last",
    "fraud_te",
    "labeled_past",
)


def _strict_past(frame: pd.DataFrame, col: str) -> np.ndarray:
    """Per-row sum of ``col`` over earlier rows of the same key with a strictly smaller time."""
    by_key = frame.groupby("k", sort=False)[col].cumsum()
    by_key_time = frame.groupby(["k", "t"], sort=False)[col].cumsum()
    return (by_key - by_key_time).to_numpy(dtype=np.float64)


def entity_aggregate_features(
    df: pd.DataFrame,
    split: TimeSplit,
    card_key: str = "card1_addr1_d1",
    smoothing: float = 20.0,
    entities: Sequence[Tuple[str, str]] = DEFAULT_ENTITIES,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return ``(features [N, 6 * len(entities)] float32, names)`` aligned with the rows of ``df``.

    ``df`` must be sorted by ``TransactionDT`` (as returned by ``load_ieee_cis``) and ``split`` must
    be the :func:`~hybrid_fraud.data.ieee_cis.time_split` of that frame. Missing keys give NaN.
    """
    t = df[TIME_COL].to_numpy(dtype=np.float64)
    if np.any(np.diff(t) < 0):
        raise ValueError("df must be sorted by TransactionDT.")
    if len(split.train) != len(df):
        raise ValueError("split masks do not match the DataFrame length.")
    amt = df[AMOUNT_COL].to_numpy(dtype=np.float64)
    train = np.asarray(split.train, dtype=bool)
    y_train = np.where(train, df[LABEL_COL].to_numpy(dtype=np.float64), 0.0)
    prior = float(y_train[train].mean()) if train.any() else 0.0

    keys: Dict[str, pd.Series] = entity_keys(df, card_key)
    keys["card1"] = _key_series(df, ["card1"])

    blocks, names = [], []
    for prefix, rel in entities:
        k = keys[rel]
        codes, _ = pd.factorize(k)  # -1 for missing
        missing = codes < 0
        frame = pd.DataFrame(
            {
                "k": codes,
                "t": t,
                "one": 1.0,
                "a": amt,
                "y": y_train,
                "is_tr": train.astype(np.float64),
            }
        )
        cnt = _strict_past(frame, "one")
        amt_sum = _strict_past(frame, "a")
        y_sum = _strict_past(frame, "y")
        n_lab = _strict_past(frame, "is_tr")

        with np.errstate(divide="ignore", invalid="ignore"):
            amt_mean = np.where(cnt > 0, amt_sum / np.maximum(cnt, 1), np.nan)
            amt_ratio = np.where(cnt > 0, amt / amt_mean, np.nan)
            te = (y_sum + prior * smoothing) / (n_lab + smoothing)
            prev = frame.groupby("k", sort=False)["t"].shift(1)
        # For tied timestamps use the previous time of the first row in the tie group.
        prev = prev.groupby([frame["k"], frame["t"]], sort=False).transform("first")
        since = t - prev.to_numpy(dtype=np.float64)

        feats = np.stack([cnt, amt_mean, amt_ratio, since, te, n_lab], axis=1)
        feats[missing] = np.nan
        blocks.append(feats)
        names += [f"{prefix}__{f}" for f in FEATURES_PER_ENTITY]
    return np.concatenate(blocks, axis=1).astype(np.float32), names
