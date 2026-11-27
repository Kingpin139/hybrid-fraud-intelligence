"""Leakage tests for the LightGBM entity-aggregate ("graph") features."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_fraud.data import load_ieee_cis, time_split
from hybrid_fraud.data.graph_features import FEATURES_PER_ENTITY, entity_aggregate_features
from hybrid_fraud.data.ieee_cis import TimeSplit

CARD1 = [("card1", "card1")]
COL = {f: i for i, f in enumerate(FEATURES_PER_ENTITY)}


def _toy():
    # card1 1: t=0, 10, 10 (tie), 20 ; card1 2: t=5 ; card1 missing: t=15
    df = pd.DataFrame(
        {
            "TransactionDT": [0, 5, 10, 10, 15, 20],
            "TransactionAmt": [10.0, 7.0, 20.0, 30.0, 1.0, 40.0],
            "isFraud": [1, 0, 0, 1, 0, 0],
            "card1": [1.0, 2.0, 1.0, 1.0, np.nan, 1.0],
        }
    )
    train = np.array([True, True, True, True, False, False])
    val = np.array([False, False, False, False, True, False])
    split = TimeSplit(train=train, val=val, test=~(train | val), train_end=10, val_end=15)
    return df, split


def test_strictly_past_counts_amounts_recency_and_ties():
    df, split = _toy()
    x, names = entity_aggregate_features(df, split, smoothing=0.0, entities=CARD1)
    assert names == [f"card1__{f}" for f in FEATURES_PER_ENTITY]
    np.testing.assert_array_equal(x[[0, 1, 2, 3, 5], COL["cnt_past"]], [0, 0, 1, 1, 3])
    # Tied rows (t=10) see only t=0, not each other.
    np.testing.assert_allclose(x[[2, 3], COL["amt_mean_past"]], [10.0, 10.0])
    np.testing.assert_allclose(x[5, COL["amt_mean_past"]], 20.0)  # (10 + 20 + 30) / 3
    np.testing.assert_allclose(x[5, COL["amt_ratio"]], 2.0)
    np.testing.assert_allclose(x[[2, 3, 5], COL["secs_since_last"]], [10.0, 10.0, 10.0])
    assert np.isnan(x[0, COL["secs_since_last"]]) and np.isnan(x[0, COL["amt_mean_past"]])
    # Train-label fraud rate (smoothing 0): rows 2/3 see row 0 (fraud); row 5 sees rows 0, 2, 3.
    np.testing.assert_allclose(x[[2, 3], COL["fraud_te"]], [1.0, 1.0])
    np.testing.assert_allclose(x[5, COL["fraud_te"]], 2 / 3, rtol=1e-6)
    assert np.isnan(x[4]).all()  # missing key -> NaN


def test_smoothing_falls_back_to_train_prior():
    df, split = _toy()
    x, _ = entity_aggregate_features(df, split, smoothing=20.0, entities=CARD1)
    prior = 2 / 4  # train rows 0..3
    np.testing.assert_allclose(x[0, COL["fraud_te"]], prior)
    np.testing.assert_allclose(x[5, COL["fraud_te"]], (2 + 20 * prior) / (3 + 20), rtol=1e-6)


@pytest.fixture(scope="module")
def fake(ieee_cis_dir):
    df = load_ieee_cis(ieee_cis_dir)
    return df, time_split(df)


def test_val_test_labels_never_used(fake):
    df, split = fake
    base, _ = entity_aggregate_features(df, split)
    flipped = df.copy()
    held_out = ~split.train
    flipped.loc[held_out, "isFraud"] = 1 - flipped.loc[held_out, "isFraud"]
    np.testing.assert_array_equal(entity_aggregate_features(flipped, split)[0], base)


def test_no_future_information(fake):
    df, split = fake
    base, _ = entity_aggregate_features(df, split)
    t = df["TransactionDT"].to_numpy()
    cut = np.quantile(t[split.train], 0.5)
    future = t > cut
    changed = df.copy()
    changed.loc[future, "TransactionAmt"] = changed.loc[future, "TransactionAmt"] * 3 + 1
    # Permute (not flip) future train labels: the smoothing prior is the global train fraud rate,
    # a single constant by design, so keep it fixed and test per-row leakage only.
    fut_tr = future & split.train
    labels = changed.loc[fut_tr, "isFraud"].to_numpy()
    changed.loc[fut_tr, "isFraud"] = np.random.default_rng(0).permutation(labels)
    new, _ = entity_aggregate_features(changed, split)
    np.testing.assert_array_equal(new[~future], base[~future])
    assert not np.array_equal(new[future], base[future], equal_nan=True)


def test_rejects_unsorted_input(fake):
    df, split = fake
    with pytest.raises(ValueError, match="sorted"):
        entity_aggregate_features(df.iloc[::-1].reset_index(drop=True), split)


def test_training_script_graph_features_flag(ieee_cis_dir):
    pytest.importorskip("lightgbm")
    path = Path(__file__).resolve().parents[1] / "examples" / "train_ieee_cis.py"
    spec = importlib.util.spec_from_file_location("train_ieee_cis_gf", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    base = ["--data-dir", str(ieee_cis_dir), "--model", "lightgbm", "--lgbm-rounds", "20"]
    plain = script.main(base)
    graph = script.main([*base, "--lgbm-graph-features"])
    assert plain["data"]["lgbm_graph_features"] is False
    assert graph["data"]["lgbm_graph_features"] is True
    assert graph["data"]["features"] == plain["data"]["features"]
    assert 0.0 <= graph["lightgbm"]["test"]["pr_auc"] <= 1.0
