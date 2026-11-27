"""
Train fraud models on IEEE-CIS (Kaggle ``ieee-fraud-detection``) with a time-based split.

Models (all evaluated on the same chronological val / test periods):

* ``logistic`` — sklearn logistic regression on the transaction features (no graph).
* ``rbgnn_plus`` — :class:`hybrid_fraud.models.RBGNNPlus` on the homogeneous event graph;
  training uses ``query_time = train_end``, validation ``val_end``, test all edges.
* ``lightgbm`` — gradient-boosted trees on the same transaction features (optional dependency:
  ``pip install -e ".[lightgbm]"``), early-stopped on validation PR-AUC. ``--lgbm-raw-v`` appends
  the raw V1–V339 columns (trees need no scaling, so nothing is fit on val / test).
  ``--lgbm-graph-features`` appends leakage-safe entity aggregates (graph neighbourhood
  summaries, see ``hybrid_fraud.data.graph_features``).
* ``hetero`` — the heterogeneous :class:`hybrid_fraud.models.RBGNN` + a linear head on transaction
  nodes, trained / evaluated on time snapshots of the graph.

Full-batch training: start with ``--nrows`` (a chronological prefix, e.g. 100000) on CPU.

    python examples/train_ieee_cis.py --data-dir data/ieee-fraud-detection --nrows 100000

Results are printed exactly as measured; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_fraud.data import build_ieee_cis_graph, load_ieee_cis, snapshot, time_split
from hybrid_fraud.data.graph_features import entity_aggregate_features
from hybrid_fraud.data.ieee_cis import DOWNLOAD_HELP, IDENTITY_FILE, TRANSACTION_FILE
from hybrid_fraud.data.ieee_cis import to_event_graph as _to_event_graph
from hybrid_fraud.models import RBGNN, RBGNNPlus

MODELS = ("logistic", "lightgbm", "rbgnn_plus", "hetero")
DEFAULT_MODELS = ("logistic", "rbgnn_plus", "hetero")  # ``--model all``


def scores(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auc": float("nan"), "pr_auc": float("nan")}
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y, minlength=2).float().clamp(min=1.0)
    return counts.sum() / (2.0 * counts)


def train_full_batch(
    model: nn.Module,
    logits_fn: Callable[[str], torch.Tensor],
    y: Dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    patience: int = 0,
) -> Dict[str, object]:
    """
    ``logits_fn(phase)`` returns logits for the rows of ``phase``; best epoch by val PR-AUC.

    With ``patience > 0`` training stops after ``patience`` epochs without a val PR-AUC improvement.
    """
    t0 = time.time()
    criterion = nn.CrossEntropyLoss(weight=class_weights(y["train"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def evaluate(phase: str) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(logits_fn(phase), dim=-1)[:, 1]
        return scores(y[phase].numpy(), prob.numpy())

    best, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), -1
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        opt.zero_grad()
        loss = criterion(logits_fn("train"), y["train"])
        loss.backward()
        opt.step()
        val = evaluate("val")["pr_auc"]
        if not np.isnan(val) and val > best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        elif patience and epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return {
        "val": evaluate("val"),
        "test": evaluate("test"),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "seconds": round(time.time() - t0, 1),
    }


def train_lightgbm(
    x: Dict[str, np.ndarray], y: Dict[str, np.ndarray], seed: int, max_rounds: int
) -> Dict[str, object]:
    """LightGBM fit on train, early-stopped on validation average precision."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            'lightgbm is not installed: pip install -e ".[lightgbm]" (or pip install lightgbm)'
        ) from exc
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=max_rounds,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        x["train"],
        y["train"],
        eval_set=[(x["val"], y["val"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
    )
    best = int(clf.best_iteration_ or max_rounds)
    out: Dict[str, object] = {
        p: scores(y[p], clf.predict_proba(x[p], num_iteration=best)[:, 1]) for p in ("val", "test")
    }
    out.update({"best_iteration": best, "seconds": round(time.time() - t0, 1)})
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    models: List[str] = (
        list(DEFAULT_MODELS) if args.model == "all" else [m.strip() for m in args.model.split(",")]
    )
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}; choose from {list(MODELS)} or 'all'.")

    if args.v_columns == "all":
        v_columns: object = "all"
    else:
        v_columns = args.v_columns.split(",") if args.v_columns else None
    raw_v = args.lgbm_raw_v and "lightgbm" in models
    df = load_ieee_cis(args.data_dir, nrows=args.nrows, v_columns="all" if raw_v else v_columns)
    split = time_split(df, val_frac=args.val_frac, test_frac=args.test_frac)
    v_raw: Optional[np.ndarray] = None
    if raw_v:
        # Raw V1-V339 for the tree model only; the shared featurizer keeps --v-columns.
        v_all = [c for c in df.columns if c.startswith("V")]
        keep = set(v_columns) if isinstance(v_columns, list) else set()
        if v_columns == "all":
            keep = set(v_all)
        v_raw = df[v_all].to_numpy(dtype=np.float32)
        df = df.drop(columns=[c for c in v_all if c not in keep])
    graph = build_ieee_cis_graph(df, split, card_key=args.card_key)
    graph_feats: Optional[np.ndarray] = None
    if args.lgbm_graph_features and "lightgbm" in models:
        # Past-only entity aggregates + train-label target encodings (see graph_features.py).
        graph_feats, _ = entity_aggregate_features(df, split, card_key=args.card_key)
    data = graph.data
    tx = data["transaction"]
    days = lambda s: s / 86_400  # noqa: E731
    info = {
        "rows": len(df),
        "fraud_rate": float(df["isFraud"].mean()),
        "train/val/test": [int(split.train.sum()), int(split.val.sum()), int(split.test.sum())],
        "train_end_day": round(days(split.train_end), 2),
        "val_end_day": round(days(split.val_end), 2),
        "features": tx.x.size(1),
        "entities": {nt: data[nt].num_nodes for nt in data.node_types if nt != "transaction"},
        "card_key": args.card_key,
        "models": models,
        "seed": args.seed,
        "lgbm_raw_v": bool(raw_v),
        "lgbm_graph_features": graph_feats is not None,
    }
    del df

    print("IEEE-CIS training — time-based split (measured results, not targets)")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k}: {v}")

    masks = {"train": tx.train_mask, "val": tx.val_mask, "test": tx.test_mask}
    y = {p: tx.y[m] for p, m in masks.items()}
    results: Dict[str, object] = {"data": info}

    if "logistic" in models:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=args.logistic_c)
        clf.fit(tx.x[masks["train"]].numpy(), y["train"].numpy())
        results["logistic"] = {
            p: scores(y[p].numpy(), clf.predict_proba(tx.x[masks[p]].numpy())[:, 1])
            for p in ("val", "test")
        }

    if "lightgbm" in models:
        feats = tx.x.numpy()
        xs = {p: feats[masks[p].numpy()] for p in masks}
        if graph_feats is not None:
            xs = {p: np.hstack([xs[p], graph_feats[masks[p].numpy()]]) for p in masks}
        if v_raw is not None:
            xs = {p: np.hstack([xs[p], v_raw[masks[p].numpy()]]) for p in masks}
        results["lightgbm"] = train_lightgbm(
            xs, {p: v.numpy() for p, v in y.items()}, args.seed, args.lgbm_rounds
        )
        del xs

    if "rbgnn_plus" in models:
        ev = _to_event_graph(data)
        ev_masks = {"train": ev.train_mask, "val": ev.val_mask, "test": ev.test_mask}
        horizon = {"train": split.train_end, "val": split.val_end, "test": None}
        span = float(ev.edge_time.max() - ev.edge_time.min()) if ev.edge_time.numel() else 1.0
        model = RBGNNPlus(ev.x.size(1), args.hidden, out_dim=2, num_layers=args.layers)

        def plus_logits(phase: str) -> torch.Tensor:
            out = model(
                ev.x, ev.edge_index, ev.edge_time, query_time=horizon[phase], time_scale=span or 1.0
            )
            return out[ev_masks[phase]]

        results["rbgnn_plus"] = train_full_batch(
            model, plus_logits, y, args.epochs, args.lr, args.patience
        )

    if "hetero" in models:
        snaps = {
            "train": snapshot(data, split.train_end),
            "val": snapshot(data, split.val_end),
            "test": data,
        }
        gnn = RBGNN(
            hidden_channels=args.hidden,
            num_layers=args.layers,
            edge_types=data.edge_types,
            in_channels={nt: data[nt].x.size(1) for nt in data.node_types},
        )
        model = nn.ModuleDict({"gnn": gnn, "head": nn.Linear(args.hidden, 2)})

        def hetero_logits(phase: str) -> torch.Tensor:
            emb = model["gnn"](snaps[phase])["transaction"]
            return model["head"](emb[masks[phase]])

        results["hetero"] = train_full_batch(
            model, hetero_logits, y, args.epochs, args.lr, args.patience
        )

    results["seconds"] = round(time.time() - t0, 1)
    print(f"\nResults (val / test period; fraud rate test = {y['test'].float().mean():.4f}):")
    print(f"  {'model':<12}{'val AUC':>9}{'val PR':>9}{'test AUC':>10}{'test PR':>9}")
    for name in models:
        r = results[name]
        print(
            f"  {name:<12}{r['val']['auc']:>9.4f}{r['val']['pr_auc']:>9.4f}"
            f"{r['test']['auc']:>10.4f}{r['test']['pr_auc']:>9.4f}"
        )
    print(f"  elapsed: {results['seconds']}s")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"  wrote {args.output}")
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", default="data/ieee-fraud-detection")
    p.add_argument("--nrows", type=int, default=None, help="chronological prefix of transactions")
    p.add_argument(
        "--v-columns", default=None, help="'all' or comma-separated V columns (default: none)"
    )
    p.add_argument(
        "--card-key", default="card1_addr1_d1", choices=["card1", "card1_addr1", "card1_addr1_d1"]
    )
    p.add_argument(
        "--model",
        default="all",
        help=f"'all' ({', '.join(DEFAULT_MODELS)}) or a comma list of {', '.join(MODELS)}",
    )
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience (0 = off)")
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--lgbm-rounds", type=int, default=3000, help="max LightGBM boosting rounds")
    p.add_argument(
        "--lgbm-raw-v", action="store_true", help="append raw V1-V339 to LightGBM's features"
    )
    p.add_argument(
        "--lgbm-graph-features",
        action="store_true",
        help="append leakage-safe entity aggregates (past counts, amounts, recency, train-label "
        "fraud rates per pseudo-user / card1 / address / e-mail / device) to LightGBM's features",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="optional JSON file for the results")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not (data_dir / TRANSACTION_FILE).is_file():
        print(
            DOWNLOAD_HELP.format(path=data_dir, files=f"{TRANSACTION_FILE}, {IDENTITY_FILE}"),
            file=sys.stderr,
        )
        raise SystemExit(2)
    return run(args)


if __name__ == "__main__":
    main()
