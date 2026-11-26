"""
Train fraud models on IEEE-CIS (Kaggle ``ieee-fraud-detection``) with a time-based split.

Models (all evaluated on the same chronological val / test periods):

* ``logistic`` — sklearn logistic regression on the transaction features (no graph).
* ``rbgnn_plus`` — :class:`hybrid_fraud.models.RBGNNPlus` on the homogeneous event graph;
  training uses ``query_time = train_end``, validation ``val_end``, test all edges.
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
from hybrid_fraud.data.ieee_cis import DOWNLOAD_HELP, IDENTITY_FILE, TRANSACTION_FILE
from hybrid_fraud.data.ieee_cis import to_event_graph as _to_event_graph
from hybrid_fraud.models import RBGNN, RBGNNPlus

MODELS = ("logistic", "rbgnn_plus", "hetero")


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
) -> Dict[str, Dict[str, float]]:
    """``logits_fn(phase)`` returns logits for the rows of ``phase``; best epoch by val PR-AUC."""
    criterion = nn.CrossEntropyLoss(weight=class_weights(y["train"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def evaluate(phase: str) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(logits_fn(phase), dim=-1)[:, 1]
        return scores(y[phase].numpy(), prob.numpy())

    best, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), -1
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(logits_fn("train"), y["train"])
        loss.backward()
        opt.step()
        val = evaluate("val")["pr_auc"]
        if not np.isnan(val) and val > best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return {"val": evaluate("val"), "test": evaluate("test"), "best_epoch": best_epoch}


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    df = load_ieee_cis(
        args.data_dir,
        nrows=args.nrows,
        v_columns=args.v_columns.split(",") if args.v_columns else None,
    )
    split = time_split(df, val_frac=args.val_frac, test_frac=args.test_frac)
    graph = build_ieee_cis_graph(df, split, card_key=args.card_key)
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
    }
    del df

    print("IEEE-CIS training — time-based split (measured results, not targets)")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k}: {v}")

    masks = {"train": tx.train_mask, "val": tx.val_mask, "test": tx.test_mask}
    y = {p: tx.y[m] for p, m in masks.items()}
    results: Dict[str, object] = {"data": info}

    models: List[str] = list(MODELS) if args.model == "all" else [args.model]

    if "logistic" in models:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=args.logistic_c)
        clf.fit(tx.x[masks["train"]].numpy(), y["train"].numpy())
        results["logistic"] = {
            p: scores(y[p].numpy(), clf.predict_proba(tx.x[masks[p]].numpy())[:, 1])
            for p in ("val", "test")
        }

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

        results["rbgnn_plus"] = train_full_batch(model, plus_logits, y, args.epochs, args.lr)

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

        results["hetero"] = train_full_batch(model, hetero_logits, y, args.epochs, args.lr)

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
    p.add_argument("--v-columns", default=None, help="comma-separated V columns (default: none)")
    p.add_argument(
        "--card-key", default="card1_addr1_d1", choices=["card1", "card1_addr1", "card1_addr1_d1"]
    )
    p.add_argument("--model", default="all", choices=["all", *MODELS])
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--logistic-c", type=float, default=1.0)
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
