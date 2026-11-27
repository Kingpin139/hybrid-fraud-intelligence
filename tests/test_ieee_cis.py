"""IEEE-CIS loader tests on a tiny FAKE fixture in the real Kaggle schema (see conftest.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from hybrid_fraud.data import (
    TransactionFeaturizer,
    build_ieee_cis_graph,
    load_ieee_cis,
    snapshot,
    time_split,
    to_event_graph,
)
from hybrid_fraud.data.ieee_cis import RELATIONS, entity_keys
from hybrid_fraud.models import RBGNN, RBGNNPlus

from .conftest import write_fake_ieee_cis


@pytest.fixture(scope="module")
def df(ieee_cis_dir):
    return load_ieee_cis(ieee_cis_dir)


@pytest.fixture(scope="module")
def built(df):
    return build_ieee_cis_graph(df, time_split(df))


# ------------------------------------------------------------------ loading / schema


def test_load_parses_schema_and_merges_identity(df):
    assert len(df) == 600
    assert df["isFraud"].dtype == np.int8
    assert df["TransactionAmt"].dtype == np.float32
    assert isinstance(df["P_emaildomain"].dtype, pd.CategoricalDtype)
    assert isinstance(df["DeviceInfo"].dtype, pd.CategoricalDtype)
    assert df["id_01"].dtype == np.float32
    assert {"id_38", "DeviceType", "M4", "D15", "C14"} <= set(df.columns)
    assert df["DeviceInfo"].notna().mean() < 0.5  # left join: most rows have no identity
    assert (df["TransactionDT"].diff().dropna() >= 0).all()


def test_v_columns_are_opt_in(ieee_cis_dir):
    assert not any(c.startswith("V") for c in load_ieee_cis(ieee_cis_dir, nrows=20).columns)
    some = load_ieee_cis(ieee_cis_dir, nrows=20, v_columns=["V1", "V258"])
    assert {"V1", "V258"} <= set(some.columns) and "V2" not in some.columns
    assert (
        sum(c.startswith("V") for c in load_ieee_cis(ieee_cis_dir, nrows=5, v_columns="all")) == 339
    )


def test_nrows_usecols_sample_and_identity_toggle(ieee_cis_dir):
    head = load_ieee_cis(ieee_cis_dir, nrows=50)
    assert len(head) == 50 and head["TransactionID"].min() == 2_987_000
    few = load_ieee_cis(ieee_cis_dir, usecols=["card1"], include_identity=False)
    assert set(few.columns) == {
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "card1",
    }
    assert len(load_ieee_cis(ieee_cis_dir, sample_frac=0.5, seed=1)) == 300
    with pytest.raises(ValueError):
        load_ieee_cis(ieee_cis_dir, usecols=["not_a_column"])


def test_missing_files_give_kaggle_instructions(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="kaggle competitions download -c ieee-fraud-detection"
    ):
        load_ieee_cis(tmp_path)


def test_missing_required_column(tmp_path):
    pd.DataFrame({"TransactionID": [1], "TransactionDT": [1]}).to_csv(
        tmp_path / "train_transaction.csv", index=False
    )
    with pytest.raises(ValueError, match="isFraud"):
        load_ieee_cis(tmp_path)


def test_test_identity_hyphen_columns_are_normalised(tmp_path):
    d = write_fake_ieee_cis(tmp_path / "x", n=40)
    ident = pd.read_csv(d / "train_identity.csv")
    ident.columns = [c.replace("id_", "id-") for c in ident.columns]
    ident.to_csv(d / "train_identity.csv", index=False)
    assert "id_01" in load_ieee_cis(d).columns


# ------------------------------------------------------------------ split


def test_time_split_is_chronological_disjoint_and_complete(df):
    s = time_split(df, val_frac=0.2, test_frac=0.2)
    t = df["TransactionDT"].to_numpy()
    assert (
        not (s.train & s.val).any() and not (s.val & s.test).any() and not (s.train & s.test).any()
    )
    assert (s.train | s.val | s.test).all()
    assert t[s.train].max() < t[s.val].min() <= t[s.val].max() < t[s.test].min()
    assert t[s.train].max() == s.train_end and t[s.val].max() == s.val_end
    assert abs(s.train.mean() - 0.6) < 0.02


def test_time_split_handles_ties_and_rejects_unsorted():
    tied = pd.DataFrame({"TransactionDT": [1, 2, 2, 2, 3, 4, 5, 6, 7, 8]})
    s = time_split(tied, val_frac=0.2, test_frac=0.2)
    t = tied["TransactionDT"].to_numpy()
    assert t[s.train].max() < t[s.val].min()
    with pytest.raises(ValueError, match="sorted"):
        time_split(pd.DataFrame({"TransactionDT": [3, 1, 2, 4, 5]}))


# ------------------------------------------------------------------ leakage


def test_featurizer_is_fit_on_train_rows_only(df, built):
    s = built.split
    fz = built.featurizer
    train_amt = np.log1p(df.loc[s.train, "TransactionAmt"].to_numpy(dtype=np.float64))
    all_amt = np.log1p(df["TransactionAmt"].to_numpy(dtype=np.float64))
    assert fz.means_["TransactionAmt"] == pytest.approx(train_amt.mean(), rel=1e-6)
    assert fz.means_["TransactionAmt"] != pytest.approx(all_amt.mean(), rel=1e-6)
    # Changing val/test rows must not change any fitted statistic or any training feature.
    poisoned = df.copy()
    poisoned.loc[~s.train, "TransactionAmt"] = 1e9
    poisoned.loc[~s.train, "C1"] = -1e6
    other = build_ieee_cis_graph(poisoned, s)
    assert other.featurizer.means_ == fz.means_ and other.featurizer.vocab_ == fz.vocab_
    tx_mask = torch.from_numpy(s.train)
    assert torch.equal(other.data["transaction"].x[tx_mask], built.data["transaction"].x[tx_mask])


def test_label_ids_and_raw_time_are_not_features(built):
    names = built.featurizer.feature_names_
    for banned in ("isFraud", "TransactionID", "TransactionDT"):
        assert not any(n == banned or n.startswith(banned + "=") for n in names)
    assert len(names) == built.data["transaction"].x.size(1)


def test_label_is_not_recoverable_from_entity_features(df, built):
    """Entity features are [1, log1p(train degree)] — identical if the labels are flipped."""
    flipped = df.copy()
    flipped["isFraud"] = (1 - flipped["isFraud"]).astype(np.int8)
    other = build_ieee_cis_graph(flipped, built.split)
    for nt in ("card", "device", "email", "address", "product"):
        assert torch.equal(other.data[nt].x, built.data[nt].x)
    assert torch.equal(other.data["transaction"].x, built.data["transaction"].x)


def test_entity_degree_uses_training_period_only(df, built):
    data, s = built.data, built.split
    ei = data["transaction", "is_product", "product"].edge_index
    train_edges = torch.from_numpy(s.train)[ei[0]]
    deg = torch.bincount(ei[1][train_edges], minlength=data["product"].num_nodes).float()
    assert torch.allclose(data["product"].x[:, 1], torch.log1p(deg))


def test_featurizer_handles_unseen_categories(df):
    s = time_split(df)
    fz = TransactionFeaturizer(max_categories=2).fit(df.loc[s.train])
    new = df.iloc[:5].copy()
    new["card4"] = pd.Categorical(["never-seen"] * 5)
    x = fz.transform(new)
    j = fz.feature_names_.index("card4=<other>")
    assert (x[:, j] == 1).all() and np.isfinite(x).all()
    with pytest.raises(RuntimeError):
        TransactionFeaturizer().transform(df)


# ------------------------------------------------------------------ graph


def test_graph_shapes_and_masks(df, built):
    data = built.data
    tx = data["transaction"]
    assert tx.x.shape[0] == len(df) == tx.y.numel() == tx.time.numel()
    assert torch.isfinite(tx.x).all()
    assert int(tx.train_mask.sum() + tx.val_mask.sum() + tx.test_mask.sum()) == len(df)
    assert set(data.node_types) == {"transaction", "card", "device", "email", "address", "product"}
    for nt in data.node_types:
        if nt != "transaction":
            assert data[nt].x.shape == (data[nt].num_nodes, 2)
    data.validate(raise_on_error=True)


def test_every_relation_has_reverse_edges_with_times(df, built):
    data = built.data
    for rel, nt in RELATIONS:
        fwd = data["transaction", rel, nt]
        rev = data[nt, f"rev_{rel}", "transaction"]
        assert torch.equal(rev.edge_index, fwd.edge_index.flip(0))
        assert torch.equal(rev.edge_time, fwd.edge_time)
        # edge_time is the TransactionDT of the transaction endpoint
        assert torch.equal(fwd.edge_time, data["transaction"].time[fwd.edge_index[0]].float())


def test_missing_keys_create_no_edges(df, built):
    n_with_device = int(df["DeviceInfo"].notna().sum())
    assert built.data["transaction", "uses_device", "device"].edge_index.size(1) == n_with_device
    n_r = int(df["R_emaildomain"].notna().sum())
    assert built.data["transaction", "r_email", "email"].edge_index.size(1) == n_r
    # P_ and R_ e-mail domains share one 'email' node set
    assert built.data["email"].num_nodes == len(
        set(df["P_emaildomain"].dropna().astype(str))
        | set(df["R_emaildomain"].dropna().astype(str))
    )


def test_card_key_heuristics(df):
    n = {
        k: entity_keys(df, k)["uses_card"].nunique()
        for k in ("card1", "card1_addr1", "card1_addr1_d1")
    }
    assert n["card1"] <= n["card1_addr1"] <= n["card1_addr1_d1"]
    row = df.iloc[0]
    key = entity_keys(df.iloc[:1], "card1_addr1_d1")["uses_card"].iloc[0]
    assert key.startswith(f"{int(row['card1'])}_")
    with pytest.raises(ValueError):
        entity_keys(df, "bogus")


def test_snapshot_drops_future_edges(built):
    data, s = built.data, built.split
    snap = snapshot(data, s.train_end)
    for et in data.edge_types:
        assert snap[et].edge_time.numel() == 0 or snap[et].edge_time.max() <= s.train_end
        assert snap[et].edge_index.size(1) == int((data[et].edge_time <= s.train_end).sum())
    assert snap["transaction"].x is data["transaction"].x  # node tensors shared, not copied


def test_event_graph_view(built):
    data = built.data
    ev = to_event_graph(data)
    n_tx = data["transaction"].num_nodes
    total = sum(data[nt].num_nodes for nt in data.node_types)
    assert ev.x.shape[0] == total and ev.y.numel() == total
    n_fwd = sum(
        data[et].edge_index.size(1) for et in data.edge_types if not et[1].startswith("rev_")
    )
    assert ev.edge_index.shape == (2, n_fwd) and ev.edge_time.numel() == n_fwd
    assert (ev.edge_index[0] < n_tx).all() and (ev.edge_index[1] >= n_tx).all()  # txn -> entity
    assert not ev.train_mask[n_tx:].any() and not ev.test_mask[n_tx:].any()
    assert torch.equal(ev.y[:n_tx], data["transaction"].y)
    assert ev.x[:, -len(ev.node_types) :].sum(1).eq(1).all()  # one-hot node type


def test_models_run_on_loaded_graph(built):
    data = built.data
    ev = to_event_graph(data)
    plus = RBGNNPlus(ev.x.size(1), 8)
    out = plus(ev.x, ev.edge_index, ev.edge_time, query_time=built.split.train_end)
    assert out.shape == (ev.x.size(0), 2)
    gnn = RBGNN(
        hidden_channels=8,
        edge_types=data.edge_types,
        in_channels={nt: data[nt].x.size(1) for nt in data.node_types},
    )
    assert gnn(data)["transaction"].shape == (data["transaction"].num_nodes, 8)


# ------------------------------------------------------------------ script


def _load_script():
    path = Path(__file__).resolve().parents[1] / "examples" / "train_ieee_cis.py"
    spec = importlib.util.spec_from_file_location("train_ieee_cis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_training_script_runs_end_to_end(ieee_cis_dir, tmp_path, capsys):
    out = tmp_path / "res.json"
    res = _load_script().main(
        ["--data-dir", str(ieee_cis_dir), "--epochs", "2", "--hidden", "8", "--output", str(out)]
    )
    assert {"logistic", "rbgnn_plus", "hetero", "data"} <= set(res)
    for name in ("logistic", "rbgnn_plus", "hetero"):
        for phase in ("val", "test"):
            assert 0.0 <= res[name][phase]["pr_auc"] <= 1.0
    assert out.is_file()
    assert "time-based split" in capsys.readouterr().out


def test_training_script_missing_data_is_helpful(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _load_script().main(["--data-dir", str(tmp_path / "nope")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "kaggle competitions download -c ieee-fraud-detection" in err and "rules" in err


def test_training_script_lightgbm_baseline(ieee_cis_dir):
    pytest.importorskip("lightgbm")
    script = _load_script()
    base = ["--data-dir", str(ieee_cis_dir), "--lgbm-rounds", "30"]
    res = script.main([*base, "--model", "logistic,lightgbm"])
    assert {"logistic", "lightgbm"} <= set(res) and "rbgnn_plus" not in res
    lgbm = res["lightgbm"]
    assert 1 <= lgbm["best_iteration"] <= 30
    for phase in ("val", "test"):
        assert 0.0 <= lgbm[phase]["pr_auc"] <= 1.0
    raw = script.main([*base, "--model", "lightgbm", "--lgbm-raw-v"])
    assert raw["data"]["lgbm_raw_v"] is True
    # Raw V columns go to the tree model only; the shared (scaled) features are unchanged.
    assert raw["data"]["features"] == res["data"]["features"]


def test_training_script_v_columns_all_and_model_validation(ieee_cis_dir):
    script = _load_script()
    base = ["--data-dir", str(ieee_cis_dir), "--model", "logistic"]
    no_v = script.main(base)["data"]["features"]
    with_v = script.main([*base, "--v-columns", "all"])["data"]["features"]
    assert with_v > no_v + 300  # 339 V columns (+ missing flags) reach the featurizer
    with pytest.raises(SystemExit, match="Unknown model"):
        script.main(["--data-dir", str(ieee_cis_dir), "--model", "xgboost"])


def test_training_script_early_stopping_patience(ieee_cis_dir):
    res = _load_script().main(
        [
            "--data-dir", str(ieee_cis_dir), "--model", "hetero",
            "--epochs", "40", "--patience", "2", "--hidden", "8",
        ]
    )  # fmt: skip
    r = res["hetero"]
    assert r["epochs_run"] <= 40 and r["best_epoch"] < r["epochs_run"]
    assert r["epochs_run"] - 1 - r["best_epoch"] <= 2 or r["epochs_run"] == 40
    assert r["seconds"] >= 0
