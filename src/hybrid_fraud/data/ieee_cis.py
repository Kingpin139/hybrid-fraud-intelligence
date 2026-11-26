"""
IEEE-CIS Fraud Detection (Kaggle ``ieee-fraud-detection``) loader and graph builder.

Pipeline::

    df = load_ieee_cis("data/ieee-fraud-detection", nrows=100_000)  # lean read + merge
    split = time_split(df, val_frac=0.15, test_frac=0.15)  # by TransactionDT, no shuffling
    graph = build_ieee_cis_graph(df, split)  # PyG HeteroData + metadata
    events = to_event_graph(graph.data)  # homogeneous view for RBGNNPlus

Leakage rules enforced here:

* ``isFraud`` is only stored as the label ``y``; it is never a feature and there is no target
  encoding.
* The :class:`TransactionFeaturizer` (medians, means, stds, category vocabularies) is fit on the
  **training split only**; val / test rows are transformed with those statistics.
* Entity node features are ``[1, log1p(degree within the training period)]`` — no labels, no future
  counts.
* Every edge carries ``edge_time = TransactionDT`` of its transaction. Use :func:`snapshot` (or
  ``RBGNNPlus(query_time=...)``) so a model scoring a period only sees edges up to the end of it.

Data: https://www.kaggle.com/c/ieee-fraud-detection (accept the competition rules first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# --------------------------------------------------------------------------- schema

DOWNLOAD_HELP = """\
IEEE-CIS data not found in {path!s}.

Download it with the Kaggle CLI (you must first accept the competition rules at
https://www.kaggle.com/c/ieee-fraud-detection/rules while logged in):

    pip install kaggle
    # put your API token in ~/.kaggle/kaggle.json (chmod 600) or set KAGGLE_USERNAME / KAGGLE_KEY
    kaggle competitions download -c ieee-fraud-detection -p data/ieee-fraud-detection
    unzip data/ieee-fraud-detection/ieee-fraud-detection.zip -d data/ieee-fraud-detection

Expected files: {files}. The data/ directory is git-ignored; never commit the dataset.
"""

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"

ID_COL, LABEL_COL, TIME_COL, AMOUNT_COL = (
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
)

TRANSACTION_CATEGORICAL: List[str] = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
] + [f"M{i}" for i in range(1, 10)]
IDENTITY_CATEGORICAL: List[str] = [
    f"id_{i:02d}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)
] + ["DeviceType", "DeviceInfo"]

V_COLUMNS: List[str] = [f"V{i}" for i in range(1, 340)]
TRANSACTION_COLUMNS: List[str] = (
    [ID_COL, LABEL_COL, TIME_COL, AMOUNT_COL, "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"M{i}" for i in range(1, 10)]
    + V_COLUMNS
)
IDENTITY_COLUMNS: List[str] = (
    [ID_COL]
    + [f"id_{i:02d}" for i in range(1, 39)]
    + [
        "DeviceType",
        "DeviceInfo",
    ]
)
REQUIRED_COLUMNS: List[str] = [ID_COL, LABEL_COL, TIME_COL, AMOUNT_COL]

#: Columns used as graph keys (entity nodes), not as transaction features.
ENTITY_KEY_COLUMNS: List[str] = [
    "card1",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "ProductCD",
]
#: Never features: identifiers, the label and raw time (time trends do not transfer across a split).
NON_FEATURE_COLUMNS: List[str] = [ID_COL, LABEL_COL, TIME_COL]

SECONDS_PER_DAY = 86_400


def _dtype_map(columns: Sequence[str]) -> Dict[str, str]:
    categorical = set(TRANSACTION_CATEGORICAL) | set(IDENTITY_CATEGORICAL)
    out: Dict[str, str] = {}
    for c in columns:
        if c == ID_COL or c == TIME_COL:
            out[c] = "int64"
        elif c == LABEL_COL:
            out[c] = "int8"
        elif c in categorical:
            out[c] = "category"
        else:
            out[c] = "float32"
    return out


def _check_files(data_dir: Path) -> Tuple[Path, Optional[Path]]:
    tx = data_dir / TRANSACTION_FILE
    if not tx.is_file():
        raise FileNotFoundError(
            DOWNLOAD_HELP.format(path=data_dir, files=f"{TRANSACTION_FILE}, {IDENTITY_FILE}")
        )
    ident = data_dir / IDENTITY_FILE
    return tx, (ident if ident.is_file() else None)


def load_ieee_cis(
    data_dir: Union[str, Path],
    nrows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    v_columns: Union[None, str, Sequence[str]] = None,
    usecols: Optional[Sequence[str]] = None,
    include_identity: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Read ``train_transaction.csv`` (+ ``train_identity.csv``) into one memory-lean DataFrame.

    Args:
        data_dir: Directory with the Kaggle CSV files.
        nrows: Read only the first ``nrows`` transactions (the file is ordered by TransactionDT, so
            this is a chronological prefix — the cheapest way to work on a subset).
        sample_frac: Optional random subsample (seeded) *after* reading. Subsampling thins out the
            entity graph; prefer ``nrows`` for graph experiments.
        v_columns: ``None`` (default, no V1–V339 — they are ~85% of the file), ``"all"``, or an
            explicit list such as ``["V258", "V294"]``.
        usecols: Explicit transaction columns to read (overrides ``v_columns``); the required
            columns are always added.
        include_identity: Left-join ``train_identity.csv`` on TransactionID if present.
        seed: Seed for ``sample_frac``.

    Returns:
        DataFrame sorted by ``(TransactionDT, TransactionID)``; numeric columns ``float32``,
        categoricals ``category``, ``isFraud`` ``int8``.
    """
    data_dir = Path(data_dir)
    tx_path, id_path = _check_files(data_dir)

    header = pd.read_csv(tx_path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{tx_path.name} is missing required columns {missing}.")

    if usecols is not None:
        wanted = list(dict.fromkeys(list(REQUIRED_COLUMNS) + list(usecols)))
    else:
        if v_columns is None:
            v_keep: List[str] = []
        elif isinstance(v_columns, str):
            if v_columns != "all":
                raise ValueError("v_columns must be None, 'all' or a list of V column names.")
            v_keep = V_COLUMNS
        else:
            v_keep = list(v_columns)
        wanted = [c for c in TRANSACTION_COLUMNS if not c.startswith("V")] + v_keep
    unknown = [c for c in wanted if c not in header]
    if usecols is not None and unknown:
        raise ValueError(f"Columns not found in {tx_path.name}: {unknown}")
    cols = [c for c in wanted if c in header]

    df = pd.read_csv(tx_path, usecols=cols, dtype=_dtype_map(cols), nrows=nrows)

    if include_identity and id_path is not None:
        id_header = pd.read_csv(id_path, nrows=0).columns.tolist()
        rename = {
            c: c.replace("id-", "id_") for c in id_header if c.startswith("id-")
        }  # test files
        id_cols = [c for c in id_header if rename.get(c, c) in IDENTITY_COLUMNS]
        ident = pd.read_csv(
            id_path,
            usecols=id_cols,
            dtype={c: _dtype_map([rename.get(c, c)])[rename.get(c, c)] for c in id_cols},
        ).rename(columns=rename)
        ident = ident[ident[ID_COL].isin(df[ID_COL])]
        df = df.merge(ident, on=ID_COL, how="left", validate="one_to_one")

    if sample_frac is not None:
        if not 0.0 < sample_frac <= 1.0:
            raise ValueError("sample_frac must be in (0, 1].")
        df = df.sample(frac=sample_frac, random_state=seed)

    return df.sort_values([TIME_COL, ID_COL], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- split


class TimeSplit(NamedTuple):
    """Boolean masks over the rows of a DataFrame sorted by time, plus the period boundaries."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: float  # last TransactionDT in train
    val_end: float  # last TransactionDT in val


def time_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15) -> TimeSplit:
    """
    Chronological train / val / test split by ``TransactionDT`` (no shuffling).

    Rows are split by position in time order, then rows that share a timestamp with the last row of
    an earlier period are moved into that period, so ``max(train) < min(val)`` and
    ``max(val) < min(test)`` hold strictly.
    """
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0 and val_frac + test_frac < 1.0):
        raise ValueError("val_frac and test_frac must be in (0, 1) with val_frac + test_frac < 1.")
    t = df[TIME_COL].to_numpy()
    if np.any(np.diff(t) < 0):
        raise ValueError("DataFrame must be sorted by TransactionDT (use load_ieee_cis).")
    n = len(t)
    n_train = int(round(n * (1.0 - val_frac - test_frac)))
    n_val = int(round(n * val_frac))
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise ValueError(f"Too few rows ({n}) for the requested split.")
    train_end = t[n_train - 1]
    val_end = max(t[min(n_train + n_val, n) - 1], train_end)
    train = t <= train_end
    val = (t > train_end) & (t <= val_end)
    test = t > val_end
    if not (train.any() and val.any() and test.any()):
        raise ValueError("Time split produced an empty period (too many tied timestamps).")
    return TimeSplit(train, val, test, float(train_end), float(val_end))


# --------------------------------------------------------------------------- features


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class TransactionFeaturizer:
    """
    Transaction feature encoder, **fit on the training rows only**.

    * Numeric columns: signed ``log1p`` → median imputation → standardisation → clip to ``±clip``,
      plus a missing-indicator for every column that has missing values in train.
    * Low-cardinality categoricals: one-hot over the ``max_categories`` most frequent training
      values, plus ``<other>`` (unseen / rare) and ``<missing>``.
    * Time of day / day of week from ``TransactionDT`` as sin/cos (raw time is never a feature).
    * Entity keys (card1, addr1/2, e-mail domains, DeviceInfo, ProductCD) become graph nodes;
      ProductCD is also one-hot encoded because it has only 5 values.
    """

    max_categories: int = 20
    clip: float = 10.0
    numeric_cols_: List[str] = field(default_factory=list)
    categorical_cols_: List[str] = field(default_factory=list)
    medians_: Dict[str, float] = field(default_factory=dict)
    means_: Dict[str, float] = field(default_factory=dict)
    stds_: Dict[str, float] = field(default_factory=dict)
    missing_cols_: List[str] = field(default_factory=list)
    vocab_: Dict[str, List[str]] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> TransactionFeaturizer:
        excluded = set(NON_FEATURE_COLUMNS) | (set(ENTITY_KEY_COLUMNS) - {"ProductCD"})
        self.categorical_cols_ = [
            c
            for c in df.columns
            if c not in excluded
            and (isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object)
        ]
        self.numeric_cols_ = [
            c
            for c in df.columns
            if c not in excluded
            and c not in self.categorical_cols_
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        for c in self.numeric_cols_:
            v = _signed_log1p(df[c].to_numpy(dtype=np.float64))
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            v = np.where(np.isnan(v), med, v)
            self.medians_[c] = float(med)
            self.means_[c] = float(v.mean())
            self.stds_[c] = float(v.std()) or 1.0
        self.missing_cols_ = [c for c in self.numeric_cols_ if df[c].isna().any()]
        for c in self.categorical_cols_:
            counts = df[c].astype("object").value_counts(dropna=True)
            self.vocab_[c] = [str(v) for v in counts.index[: self.max_categories]]

        names = list(self.numeric_cols_) + [f"{c}__missing" for c in self.missing_cols_]
        for c in self.categorical_cols_:
            names += [f"{c}={v}" for v in self.vocab_[c]] + [f"{c}=<other>", f"{c}=<missing>"]
        names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        self.feature_names_ = names
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TransactionFeaturizer must be fit before transform.")
        n = len(df)
        blocks: List[np.ndarray] = []
        for c in self.numeric_cols_:
            v = _signed_log1p(df[c].to_numpy(dtype=np.float64)) if c in df else np.full(n, np.nan)
            v = np.where(np.isnan(v), self.medians_[c], v)
            blocks.append(np.clip((v - self.means_[c]) / self.stds_[c], -self.clip, self.clip))
        for c in self.missing_cols_:
            blocks.append(df[c].isna().to_numpy(dtype=np.float64) if c in df else np.ones(n))
        for c in self.categorical_cols_:
            vocab = self.vocab_[c]
            col = df[c].astype("object") if c in df else pd.Series([None] * n)
            index = {v: i for i, v in enumerate(vocab)}
            onehot = np.zeros((n, len(vocab) + 2))
            isna = col.isna().to_numpy()
            codes = np.array([index.get(str(v), len(vocab)) for v in col.to_numpy()])
            codes[isna] = len(vocab) + 1
            onehot[np.arange(n), codes] = 1.0
            blocks.append(onehot)
        t = df[TIME_COL].to_numpy(dtype=np.float64)
        hour = (t / 3600.0) % 24.0
        dow = (t / SECONDS_PER_DAY) % 7.0
        blocks += [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ]
        cols = [b.reshape(n, -1) for b in blocks]
        return np.concatenate(cols, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- graph

#: (relation, entity node type) for every transaction -> entity edge type. Reverse edges are
#: ``(entity, "rev_<relation>", "transaction")``.
RELATIONS: List[Tuple[str, str]] = [
    ("uses_card", "card"),
    ("uses_device", "device"),
    ("p_email", "email"),
    ("r_email", "email"),
    ("at_address", "address"),
    ("is_product", "product"),
]


def _key_series(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """String key from ``cols`` joined by '_'; NaN if the *first* column is missing."""
    first = df[cols[0]] if cols[0] in df else pd.Series(np.nan, index=df.index)
    parts = []
    for c in cols:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        if pd.api.types.is_float_dtype(s):
            s = s.astype("Int64")  # 13926.0 -> "13926"
        parts.append(s.astype("string").fillna("NA"))
    key = parts[0]
    for p in parts[1:]:
        key = key + "_" + p
    return key.where(first.notna(), other=pd.NA)


def entity_keys(df: pd.DataFrame, card_key: str = "card1_addr1_d1") -> Dict[str, pd.Series]:
    """
    Entity key per transaction for every relation (``<NA>`` = no edge).

    Card / pseudo-user heuristic (``card_key``):

    * ``"card1"`` — raw card1 (a card-number bucket; many people share one value).
    * ``"card1_addr1"`` — card1 + billing region addr1.
    * ``"card1_addr1_d1"`` (default) — additionally ``floor(TransactionDT / 86400) - D1``. D1 is
      commonly read as "days since the card / account was first seen", so this difference is a
      constant per account and approximates a client id (the "uid" trick from public IEEE-CIS
      solutions). It is a heuristic: D1 is anonymised and partly missing (missing → "NA").
    """
    card_cols = {"card1": ["card1"], "card1_addr1": ["card1", "addr1"]}
    if card_key in card_cols:
        card = _key_series(df, card_cols[card_key])
    elif card_key == "card1_addr1_d1":
        day = np.floor(df[TIME_COL].to_numpy(dtype=np.float64) / SECONDS_PER_DAY)
        d1 = df["D1"] if "D1" in df else pd.Series(np.nan, index=df.index, dtype="float32")
        tmp = pd.DataFrame(
            {"card1": df["card1"], "addr1": df.get("addr1"), "d1n": day - d1.to_numpy()},
            index=df.index,
        )
        card = _key_series(tmp, ["card1", "addr1", "d1n"])
    else:
        raise ValueError("card_key must be 'card1', 'card1_addr1' or 'card1_addr1_d1'.")
    return {
        "uses_card": card,
        "uses_device": _key_series(df, ["DeviceInfo"]),
        "p_email": _key_series(df, ["P_emaildomain"]),
        "r_email": _key_series(df, ["R_emaildomain"]),
        "at_address": _key_series(df, ["addr1", "addr2"]),
        "is_product": _key_series(df, ["ProductCD"]),
    }


@dataclass
class IEEECISGraph:
    """Result of :func:`build_ieee_cis_graph`."""

    data: HeteroData
    featurizer: TransactionFeaturizer
    split: TimeSplit
    entity_index: Dict[str, Dict[str, int]]  # node type -> {key: node index}
    card_key: str


def build_ieee_cis_graph(
    df: pd.DataFrame,
    split: Optional[TimeSplit] = None,
    card_key: str = "card1_addr1_d1",
    featurizer: Optional[TransactionFeaturizer] = None,
) -> IEEECISGraph:
    """
    Build a heterogeneous transaction graph.

    Node types: ``transaction`` (x, y, time, train/val/test masks, transaction_id) and the entities
    ``card``, ``device``, ``email``, ``address``, ``product`` (x = ``[1, log1p(train degree)]``).
    Edge types: ``(transaction, <rel>, <entity>)`` and ``(<entity>, rev_<rel>, transaction)`` for
    every relation in :data:`RELATIONS`, each with ``edge_time`` = TransactionDT.

    The featurizer is fit on the training rows only (pass a pre-fit one to reuse it).
    """
    if split is None:
        split = time_split(df)
    if len(split.train) != len(df):
        raise ValueError("split masks do not match the DataFrame length.")
    if featurizer is None:
        featurizer = TransactionFeaturizer().fit(df.loc[split.train])
    elif not featurizer.fitted_:
        raise ValueError("featurizer must already be fit (on training rows).")

    data = HeteroData()
    tx = data["transaction"]
    tx.x = torch.from_numpy(featurizer.transform(df))
    tx.y = torch.tensor(df[LABEL_COL].to_numpy(dtype=np.int64))
    tx.time = torch.tensor(df[TIME_COL].to_numpy(dtype=np.float64))
    tx.transaction_id = torch.tensor(df[ID_COL].to_numpy(dtype=np.int64))
    tx.train_mask = torch.from_numpy(split.train.copy())
    tx.val_mask = torch.from_numpy(split.val.copy())
    tx.test_mask = torch.from_numpy(split.test.copy())

    keys = entity_keys(df, card_key)
    entity_index: Dict[str, Dict[str, int]] = {}
    for rel, ntype in RELATIONS:  # shared vocab per node type (P and R e-mails share 'email')
        vocab = entity_index.setdefault(ntype, {})
        for k in keys[rel].dropna().unique():
            vocab.setdefault(str(k), len(vocab))

    train_deg = {nt: np.zeros(len(v), dtype=np.float64) for nt, v in entity_index.items()}
    edge_time = tx.time.to(torch.float32)
    for rel, ntype in RELATIONS:
        k = keys[rel]
        rows = np.flatnonzero(k.notna().to_numpy())
        vocab = entity_index[ntype]
        cols = np.fromiter((vocab[str(v)] for v in k.iloc[rows]), dtype=np.int64, count=len(rows))
        np.add.at(train_deg[ntype], cols[split.train[rows]], 1.0)
        ei = torch.from_numpy(np.stack([rows, cols]))
        et = edge_time[torch.from_numpy(rows)]
        data["transaction", rel, ntype].edge_index = ei
        data["transaction", rel, ntype].edge_time = et
        data[ntype, f"rev_{rel}", "transaction"].edge_index = ei.flip(0)
        data[ntype, f"rev_{rel}", "transaction"].edge_time = et.clone()

    for ntype, vocab in entity_index.items():
        deg = train_deg[ntype]
        data[ntype].x = torch.from_numpy(
            np.stack([np.ones_like(deg), np.log1p(deg)], axis=1).astype(np.float32)
        )
        data[ntype].num_nodes = len(vocab)

    return IEEECISGraph(data, featurizer, split, entity_index, card_key)


def snapshot(data: HeteroData, t_max: float) -> HeteroData:
    """
    Copy of ``data`` keeping only edges with ``edge_time <= t_max`` (node tensors are shared).

    Train on ``snapshot(data, split.train_end)``, validate on ``snapshot(data, split.val_end)``,
    test on the full graph: then no prediction for a period uses transactions from a later period.
    """
    out = HeteroData()
    for ntype in data.node_types:
        for key, value in data[ntype].items():
            out[ntype][key] = value
    for etype in data.edge_types:
        store = data[etype]
        keep = store.edge_time <= t_max
        out[etype].edge_index = store.edge_index[:, keep]
        out[etype].edge_time = store.edge_time[keep]
    return out


# --------------------------------------------------------------------------- event view


class EventGraph(NamedTuple):
    """Homogeneous timestamped view for :class:`hybrid_fraud.models.RBGNNPlus`."""

    x: torch.Tensor  # [N_total, F] — transaction features | entity features | node-type one-hot
    edge_index: torch.Tensor  # [2, E] transaction -> entity (RBGNNPlus adds reverses itself)
    edge_time: torch.Tensor  # [E] TransactionDT
    y: torch.Tensor  # [N_total] (0 for entity nodes; use the masks)
    train_mask: torch.Tensor  # [N_total] bool, transaction rows only
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    node_type: torch.Tensor  # [N_total] index into node_types
    node_types: List[str]
    offsets: Dict[str, int]  # first global index of each node type


def to_event_graph(data: HeteroData) -> EventGraph:
    """Flatten an IEEE-CIS ``HeteroData`` into one node set with forward event edges only."""
    node_types = ["transaction"] + [nt for nt in data.node_types if nt != "transaction"]
    offsets, total = {}, 0
    for nt in node_types:
        offsets[nt] = total
        total += data[nt].num_nodes
    f_tx = data["transaction"].x.size(1)
    f_ent = max((data[nt].x.size(1) for nt in node_types[1:]), default=0)
    x = torch.zeros(total, f_tx + f_ent + len(node_types))
    node_type = torch.zeros(total, dtype=torch.long)
    for i, nt in enumerate(node_types):
        lo, n = offsets[nt], data[nt].num_nodes
        feats = data[nt].x
        start = 0 if nt == "transaction" else f_tx
        x[lo : lo + n, start : start + feats.size(1)] = feats
        x[lo : lo + n, f_tx + f_ent + i] = 1.0
        node_type[lo : lo + n] = i

    srcs, times = [], []
    for src_t, rel, dst_t in data.edge_types:
        if rel.startswith("rev_"):
            continue
        ei = data[src_t, rel, dst_t].edge_index
        srcs.append(torch.stack([ei[0] + offsets[src_t], ei[1] + offsets[dst_t]]))
        times.append(data[src_t, rel, dst_t].edge_time)
    edge_index = torch.cat(srcs, dim=1) if srcs else torch.zeros(2, 0, dtype=torch.long)
    edge_time = torch.cat(times) if times else torch.zeros(0)

    n_tx = data["transaction"].num_nodes
    y = torch.zeros(total, dtype=torch.long)
    y[:n_tx] = data["transaction"].y
    masks = []
    for name in ("train_mask", "val_mask", "test_mask"):
        m = torch.zeros(total, dtype=torch.bool)
        m[:n_tx] = data["transaction"][name]
        masks.append(m)
    return EventGraph(x, edge_index, edge_time, y, *masks, node_type, node_types, offsets)
