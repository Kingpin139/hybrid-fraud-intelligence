import pytest
import torch
from torch_geometric.data import HeteroData


def make_edges(n_src: int, n_dst: int, n_edges: int) -> torch.Tensor:
    """Random edges with src/dst indices drawn from their own node-type ranges."""
    return torch.stack([torch.randint(0, n_src, (n_edges,)), torch.randint(0, n_dst, (n_edges,))])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def graph() -> HeteroData:
    """5 users, 3 merchants, 4 phones; forward + reverse edges."""
    data = HeteroData()
    data["user"].x = torch.randn(5, 16)
    data["merchant"].x = torch.randn(3, 16)
    data["phone"].x = torch.randn(4, 8)  # different feature size on purpose
    data["user", "transacts", "merchant"].edge_index = make_edges(5, 3, 10)
    data["merchant", "rev_transacts", "user"].edge_index = data[
        "user", "transacts", "merchant"
    ].edge_index.flip(0)
    data["user", "shares", "phone"].edge_index = make_edges(5, 4, 6)
    data["phone", "rev_shares", "user"].edge_index = data[
        "user", "shares", "phone"
    ].edge_index.flip(0)
    return data


# ----------------------------------------------------------------------------- IEEE-CIS fixture
# Tiny FAKE data in the Kaggle ``ieee-fraud-detection`` schema (generated, not real data).


def write_fake_ieee_cis(path, n: int = 600, seed: int = 0, identity_frac: float = 0.4):
    """Write fake ``train_transaction.csv`` / ``train_identity.csv`` with the real column names."""
    import numpy as np
    import pandas as pd

    from hybrid_fraud.data.ieee_cis import IDENTITY_CATEGORICAL, IDENTITY_COLUMNS

    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)

    def with_nan(values, frac):
        values = np.asarray(values, dtype=object)
        values[rng.random(len(values)) < frac] = np.nan
        return values

    n_cards = max(8, n // 15)
    card_pool = rng.integers(1000, 18000, size=n_cards)
    card_id = rng.integers(0, n_cards, size=n)
    risky_cards = rng.choice(n_cards, size=max(1, n_cards // 6), replace=False)
    p_fraud = np.where(np.isin(card_id, risky_cards), 0.45, 0.04)
    is_fraud = (rng.random(n) < p_fraud).astype(int)
    is_fraud[:3] = 1  # guarantee positives early (train) ...
    is_fraud[-3:] = 1  # ... and late (test)

    dt = 86_400 + np.cumsum(rng.integers(1, 3_000, size=n))
    tx = {
        "TransactionID": 2_987_000 + np.arange(n),
        "isFraud": is_fraud,
        "TransactionDT": dt,
        "TransactionAmt": np.round(rng.lognormal(3.5, 1.0, size=n) * (1 + is_fraud), 3),
        "ProductCD": rng.choice(list("WHCSR"), size=n),
        "card1": card_pool[card_id],
        "card2": with_nan(rng.integers(100, 600, size=n).astype(float), 0.02),
        "card3": rng.choice([150.0, 185.0], size=n),
        "card4": with_nan(
            rng.choice(["visa", "mastercard", "discover", "american express"], n), 0.01
        ),
        "card5": with_nan(rng.choice([226.0, 224.0, 166.0], size=n), 0.01),
        "card6": rng.choice(["debit", "credit"], size=n),
        "addr1": with_nan(rng.choice([299.0, 325.0, 204.0, 264.0], size=n), 0.1),
        "addr2": with_nan(np.full(n, 87.0), 0.1),
        "dist1": with_nan(rng.integers(0, 500, size=n).astype(float), 0.6),
        "dist2": with_nan(rng.integers(0, 500, size=n).astype(float), 0.9),
        "P_emaildomain": with_nan(rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n), 0.15),
        "R_emaildomain": with_nan(rng.choice(["gmail.com", "anonymous.com"], n), 0.7),
    }
    for i in range(1, 15):
        tx[f"C{i}"] = rng.poisson(1.5, size=n).astype(float) + 3 * is_fraud
    for i in range(1, 16):
        tx[f"D{i}"] = with_nan(rng.integers(0, 400, size=n).astype(float), 0.3 if i > 1 else 0.05)
    for i in range(1, 10):
        choices = ["M0", "M1", "M2"] if i == 4 else ["T", "F"]
        tx[f"M{i}"] = with_nan(rng.choice(choices, size=n), 0.4)
    for i in range(1, 340):
        tx[f"V{i}"] = with_nan(rng.integers(0, 3, size=n).astype(float), 0.3)
    tx_df = pd.DataFrame(tx)
    tx_df.to_csv(path / "train_transaction.csv", index=False)

    ids = tx_df["TransactionID"][rng.random(n) < identity_frac].to_numpy()
    m = len(ids)
    ident = {"TransactionID": ids}
    for c in IDENTITY_COLUMNS[1:]:
        if c == "DeviceType":
            ident[c] = with_nan(rng.choice(["mobile", "desktop"], size=m), 0.05)
        elif c == "DeviceInfo":
            ident[c] = with_nan(rng.choice(["Windows", "iOS Device", "MacOS", "SM-G9650"], m), 0.2)
        elif c in IDENTITY_CATEGORICAL:
            ident[c] = with_nan(rng.choice(["Found", "NotFound", "New"], size=m), 0.3)
        else:
            ident[c] = with_nan(rng.normal(0, 10, size=m).round(1), 0.3)
    pd.DataFrame(ident).to_csv(path / "train_identity.csv", index=False)
    return path


@pytest.fixture(scope="session")
def ieee_cis_dir(tmp_path_factory):
    return write_fake_ieee_cis(tmp_path_factory.mktemp("ieee") / "ieee-fraud-detection")
