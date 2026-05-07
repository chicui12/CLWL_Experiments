#!/usr/bin/env python3
"""
MNIST-10 misspecified-baseline failure experiments for CLWL.

This script implements two experiments:

(A) CLPL-native vs CLWL under a biased partial-label transition M such that
    T_CLPL M is not order preserving, while CLWL constructs T' so that T' M is
    order preserving.

(B) CLCL-native vs CLWL under a non-complementary transition M such that
    T_CLCL M is not order preserving, while CLWL constructs T' so that T' M is
    order preserving.

The script tries to load true MNIST first. Because many execution environments
cannot download MNIST, it also supports an explicit offline fallback to the
scikit-learn 8x8 digits dataset. Use --allow-digits-fallback to enable this.

Examples:
  # True MNIST-10, if MNIST IDX files are present or torchvision works:
  python run_mnist10_misspecified_failure_experiments.py --mnist-root data --classes 0 1 2 3 4 5 6 7 8 9

  # Offline fallback smoke run on scikit-learn digits with all 10 classes:
  python run_mnist10_misspecified_failure_experiments.py --allow-digits-fallback --epochs 60
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


Array = np.ndarray


# -------------------------
# Data loading
# -------------------------


def read_idx_images(path: Path) -> Array:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} has invalid image magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> Array:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} has invalid label magic {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def find_mnist_idx_files(root: Path) -> Optional[dict[str, Path]]:
    candidates = [root, root / "MNIST" / "raw", root / "raw"]
    names = {
        "train_images": ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"],
        "train_labels": ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"],
        "test_images": ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"],
        "test_labels": ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"],
    }
    for base in candidates:
        found: dict[str, Path] = {}
        for key, file_names in names.items():
            for name in file_names:
                p = base / name
                if p.exists():
                    found[key] = p
                    break
        if set(found) == set(names):
            return found
    return None


def load_true_mnist10(root: Path, classes: list[int]) -> tuple[Array, Array, Array, Array, str]:
    files = find_mnist_idx_files(root)
    if files is not None:
        Xtr = read_idx_images(files["train_images"])
        ytr = read_idx_labels(files["train_labels"])
        Xte = read_idx_images(files["test_images"])
        yte = read_idx_labels(files["test_labels"])
        source = "mnist_idx"
    else:
        # Avoid importing torchvision at module load time, since some containers have
        # torch/torchvision ABI mismatches. Import only if IDX files are absent.
        try:
            from torchvision.datasets import MNIST
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Could not find MNIST IDX files and torchvision import failed. "
                "Download MNIST to --mnist-root or run with --allow-digits-fallback. "
                f"TorchVision error: {e!r}"
            )
        ds_tr = MNIST(root=str(root), train=True, download=True)
        ds_te = MNIST(root=str(root), train=False, download=True)
        Xtr = ds_tr.data.numpy().astype(np.float32) / 255.0
        ytr = ds_tr.targets.numpy().astype(np.int64)
        Xte = ds_te.data.numpy().astype(np.float32) / 255.0
        yte = ds_te.targets.numpy().astype(np.int64)
        source = "torchvision_mnist"

    return filter_and_remap_classes(Xtr, ytr, Xte, yte, classes, source)


def load_digits_subset(classes: list[int]) -> tuple[Array, Array, Array, Array, str]:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    ds = load_digits()
    X = ds.images.astype(np.float32) / 16.0  # shape n, 8, 8
    y = ds.target.astype(np.int64)
    mask = np.isin(y, np.asarray(classes))
    X, y = X[mask], y[mask]
    class_to_new = {old: new for new, old in enumerate(classes)}
    y = np.array([class_to_new[int(v)] for v in y], dtype=np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=123
    )
    return Xtr, ytr, Xte, yte, "sklearn_digits_fallback_8x8"


def filter_and_remap_classes(
    Xtr: Array, ytr: Array, Xte: Array, yte: Array, classes: list[int], source: str
) -> tuple[Array, Array, Array, Array, str]:
    classes_arr = np.asarray(classes, dtype=np.int64)
    tr_mask = np.isin(ytr, classes_arr)
    te_mask = np.isin(yte, classes_arr)
    Xtr, ytr = Xtr[tr_mask], ytr[tr_mask]
    Xte, yte = Xte[te_mask], yte[te_mask]
    class_to_new = {old: new for new, old in enumerate(classes)}
    ytr = np.array([class_to_new[int(v)] for v in ytr], dtype=np.int64)
    yte = np.array([class_to_new[int(v)] for v in yte], dtype=np.int64)
    return Xtr, ytr, Xte, yte, source


def normalize_and_flatten(Xtr: Array, Xte: Array) -> tuple[Array, Array, dict[str, float]]:
    mean = float(Xtr.mean())
    std = float(Xtr.std())
    if std < 1e-8:
        std = 1.0
    Xtr = ((Xtr - mean) / std).reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte = ((Xte - mean) / std).reshape(Xte.shape[0], -1).astype(np.float32)
    return Xtr, Xte, {"mean": mean, "std": std}


def subsample_balanced(X: Array, y: Array, max_per_class: Optional[int], seed: int) -> tuple[Array, Array]:
    if max_per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep_idx = np.concatenate(keep)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


# -------------------------
# Models and losses
# -------------------------


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X: Array, y_or_z: Array, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y_or_z, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_teacher(
    Xtr: Array, ytr: Array, Xte: Array, yte: Array, *, c: int, hidden_dim: int, epochs: int,
    batch_size: int, lr: float, weight_decay: float, device: str, seed: int
) -> tuple[Array, Array, dict[str, float]]:
    set_seed(seed)
    dev = torch.device(device)
    model = MLP(Xtr.shape[1], c, hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(Xtr, ytr, batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    tr_logits = predict_logits(model, Xtr, batch_size=batch_size, device=device)
    te_logits = predict_logits(model, Xte, batch_size=batch_size, device=device)
    tr_acc = float((tr_logits.argmax(axis=1) == ytr).mean())
    te_acc = float((te_logits.argmax(axis=1) == yte).mean())
    return tr_logits, te_logits, {"teacher_train_accuracy": tr_acc, "teacher_test_accuracy": te_acc}


def predict_logits(model: nn.Module, X: Array, *, batch_size: int, device: str) -> Array:
    dev = torch.device(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(dev)).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def softmax_np(logits: Array) -> Array:
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def train_weak_model(
    Xtr: Array, ztr: Array, *, c: int, d: int, hidden_dim: int, epochs: int,
    batch_size: int, lr: float, weight_decay: float, device: str, seed: int,
    loss_kind: str, T: Optional[Array] = None, weak_vectors: Optional[Array] = None,
) -> nn.Module:
    set_seed(seed)
    dev = torch.device(device)
    model = MLP(Xtr.shape[1], c, hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(Xtr, ztr, batch_size, shuffle=True)
    T_torch = None if T is None else torch.tensor(T, dtype=torch.float32, device=dev)
    B_torch = None if weak_vectors is None else torch.tensor(weak_vectors, dtype=torch.float32, device=dev)

    for _ in range(epochs):
        model.train()
        for xb, zb in loader:
            xb, zb = xb.to(dev), zb.to(dev)
            opt.zero_grad()
            scores = model(xb)
            if loss_kind == "clwl":
                assert T_torch is not None
                T_cols = T_torch[:, zb].transpose(0, 1)
                loss = (T_cols * F.softplus(-scores) + (1.0 - T_cols) * F.softplus(scores)).sum(dim=1).mean()
            elif loss_kind == "clpl":
                assert B_torch is not None
                b = B_torch[zb]
                sizes = b.sum(dim=1)
                mean_candidate_scores = (b * scores).sum(dim=1) / sizes
                pos = F.softplus(-mean_candidate_scores)
                neg = ((1.0 - b) * F.softplus(scores)).sum(dim=1)
                loss = (pos + neg).mean()
            elif loss_kind == "clcl":
                # Native complementary-label OR loss: treat observed z as a class to push down.
                loss = F.cross_entropy(-scores, zb)
            else:
                raise ValueError(loss_kind)
            loss.backward()
            opt.step()
    return model


# -------------------------
# Transition matrices
# -------------------------


def candidate_sets_size2(c: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(c) for j in range(i + 1, c)]


def clpl_biased_transition(c: int, p_anchor: float = 0.95, anchor: int = 0) -> tuple[Array, Array, list[tuple[int, int]]]:
    """Return M and native candidate-set T for a biased partial-label protocol.

    For non-anchor classes, the candidate set containing the anchor is sampled
    with high probability. This creates a native candidate-set transformation
    that over-supports the anchor class and is typically not order preserving.
    """
    sets = candidate_sets_size2(c)
    d = len(sets)
    B = np.zeros((d, c), dtype=np.float64)
    for z, S in enumerate(sets):
        B[z, list(S)] = 1.0
    M = np.zeros((d, c), dtype=np.float64)
    for y in range(c):
        valid = [z for z, S in enumerate(sets) if y in S]
        if y != anchor:
            anchor_sets = [z for z in valid if anchor in sets[z]]
            rest = [z for z in valid if z not in anchor_sets]
            # For size-two sets there is exactly one set {anchor,y}.
            M[anchor_sets[0], y] = p_anchor
            for z in rest:
                M[z, y] = (1.0 - p_anchor) / len(rest)
        else:
            # For the anchor class, distribute among anchor-containing sets.
            for z in valid:
                M[z, y] = 1.0 / len(valid)
    T_native = B.T  # columns are candidate-set indicator vectors
    return M, T_native, sets


def clcl_noncomplementary_transition(c: int, q_diag: float = 0.40) -> tuple[Array, Array]:
    off = (1.0 - q_diag) / (c - 1)
    M = off * np.ones((c, c), dtype=np.float64)
    np.fill_diagonal(M, q_diag)
    T_native = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
    return M, T_native


def construct_clwl_T(M: Array, safety_factor: float = 0.95) -> Array:
    M = np.asarray(M, dtype=np.float64)
    N = np.linalg.solve(M.T @ M, M.T)  # left inverse, shape c x d
    q = N.min(axis=0)
    delta = (N.max(axis=0) - q).max()
    alpha = safety_factor / delta if delta > 0 else 1.0
    T = alpha * (N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1))
    # Numerical clipping for tiny roundoff only.
    T[np.abs(T) < 1e-12] = 0.0
    T[np.abs(T - 1.0) < 1e-12] = 1.0
    return T


def sample_weak_labels(M: Array, y: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.zeros_like(y, dtype=np.int64)
    for i, cls in enumerate(y):
        z[i] = rng.choice(M.shape[0], p=M[:, int(cls)])
    return z


# -------------------------
# Metrics
# -------------------------


def evaluate_scores(scores: Array, y: Array, eta: Array) -> dict[str, float]:
    pred = scores.argmax(axis=1)
    clean_acc = float((pred == y).mean())

    smax = scores == scores.max(axis=1, keepdims=True)
    etamax = eta == eta.max(axis=1, keepdims=True)
    max_pres = float(np.all(smax == etamax, axis=1).mean())

    total = 0
    correct = 0
    margins = []
    n, c = eta.shape
    for i in range(n):
        for a in range(c):
            for b in range(c):
                if a == b:
                    continue
                if eta[i, a] > eta[i, b]:
                    total += 1
                    margin = scores[i, a] - scores[i, b]
                    if margin > 0:
                        correct += 1
                    margins.append(margin)
    pair = float(correct / total) if total else float("nan")
    mean_margin = float(np.mean(margins)) if margins else float("nan")
    return {
        "clean_accuracy": clean_acc,
        "max_preservation_rate": max_pres,
        "pairwise_order_rate": pair,
        "mean_margin_on_ordered_pairs": mean_margin,
        "pairwise_total": int(total),
        "pairwise_correct": int(correct),
    }


def evaluate_A(A: Array, eta: Array) -> dict[str, float]:
    return evaluate_scores(eta @ A.T, eta.argmax(axis=1), eta)


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    classes = list(args.classes)
    c = len(classes)
    if c < 2:
        raise ValueError("Pass at least two classes.")
    if sorted(classes) != classes:
        raise ValueError("Please pass classes in sorted order, e.g. --classes 0 1 2 3 4 5 6 7 8 9.")

    try:
        Xtr_raw, ytr, Xte_raw, yte, source = load_true_mnist10(Path(args.mnist_root), classes)
    except Exception as e:
        if not args.allow_digits_fallback:
            raise
        Xtr_raw, ytr, Xte_raw, yte, source = load_digits_subset(classes)
        print(f"[warning] True MNIST-10 unavailable; using offline fallback {source}. Reason: {e}")

    Xtr_raw, ytr = subsample_balanced(Xtr_raw, ytr, args.max_train_per_class, seed=999)
    Xte_raw, yte = subsample_balanced(Xte_raw, yte, args.max_test_per_class, seed=1000)
    Xtr, Xte, norm_stats = normalize_and_flatten(Xtr_raw, Xte_raw)

    rows: list[dict] = []
    diag_rows: list[dict] = []

    for seed in args.seeds:
        tr_logits, te_logits, teacher_stats = train_teacher(
            Xtr, ytr, Xte, yte, c=c, hidden_dim=args.teacher_hidden_dim,
            epochs=args.teacher_epochs, batch_size=args.batch_size,
            lr=args.teacher_lr, weight_decay=args.weight_decay,
            device=args.device, seed=10000 + seed,
        )
        eta_tr = softmax_np(tr_logits)
        eta_te = softmax_np(te_logits)

        for experiment in ["CLPL_misspecified", "CLCL_misspecified"]:
            if experiment == "CLPL_misspecified":
                M, T_native, sets = clpl_biased_transition(c, p_anchor=args.clpl_anchor_prob, anchor=0)
                weak_vectors = np.zeros((M.shape[0], c), dtype=np.float64)
                for z, S in enumerate(sets):
                    weak_vectors[z, list(S)] = 1.0
                native_method = "CLPL-native"
                native_loss = "clpl"
            else:
                M, T_native = clcl_noncomplementary_transition(c, q_diag=args.clcl_diag_prob)
                weak_vectors = None
                native_method = "CLCL-native"
                native_loss = "clcl"

            T_clwl = construct_clwl_T(M, safety_factor=0.95)
            A_native = T_native @ M
            A_clwl = T_clwl @ M
            diag_native = evaluate_A(A_native, eta_te)
            diag_clwl = evaluate_A(A_clwl, eta_te)
            diag_rows.extend([
                {"seed": seed, "experiment": experiment, "transform": "native_T_M", **diag_native},
                {"seed": seed, "experiment": experiment, "transform": "clwl_Tprime_M", **diag_clwl},
            ])

            ztr = sample_weak_labels(M, ytr, seed=20000 + seed)
            zte = sample_weak_labels(M, yte, seed=30000 + seed)

            # references
            for split, scores, yy, eta in [
                ("test", te_logits, yte, eta_te),
            ]:
                rows.append({"seed": seed, "experiment": experiment, "method": "teacher_reference", "split": split, **evaluate_scores(scores, yy, eta)})
                rows.append({"seed": seed, "experiment": experiment, "method": "zero_reference", "split": split, **evaluate_scores(np.zeros_like(scores), yy, eta)})

            native_model = train_weak_model(
                Xtr, ztr, c=c, d=M.shape[0], hidden_dim=args.hidden_dim,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, device=args.device, seed=40000 + seed,
                loss_kind=native_loss, weak_vectors=weak_vectors,
            )
            clwl_model = train_weak_model(
                Xtr, ztr, c=c, d=M.shape[0], hidden_dim=args.hidden_dim,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                weight_decay=args.weight_decay, device=args.device, seed=50000 + seed,
                loss_kind="clwl", T=T_clwl,
            )
            native_scores = predict_logits(native_model, Xte, batch_size=args.batch_size, device=args.device)
            clwl_scores = predict_logits(clwl_model, Xte, batch_size=args.batch_size, device=args.device)
            rows.append({"seed": seed, "experiment": experiment, "method": native_method, "split": "test", **evaluate_scores(native_scores, yte, eta_te)})
            rows.append({"seed": seed, "experiment": experiment, "method": "CLWL", "split": "test", **evaluate_scores(clwl_scores, yte, eta_te)})

    raw = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)

    group_cols = ["experiment", "method", "split"]
    metric_cols = ["clean_accuracy", "max_preservation_rate", "pairwise_order_rate", "mean_margin_on_ordered_pairs"]
    agg = raw.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns]

    diag_agg = diag.groupby(["experiment", "transform"])[metric_cols].agg(["mean", "std"]).reset_index()
    diag_agg.columns = ["_".join([str(x) for x in col if x]) for col in diag_agg.columns]

    meta = {
        "data_source": source,
        "classes": classes,
        "num_train": int(len(ytr)),
        "num_test": int(len(yte)),
        "input_dim": int(Xtr.shape[1]),
        "normalization": norm_stats,
        "seeds": list(args.seeds),
        "config": vars(args),
    }
    meta.update({k: float(v) for k, v in teacher_stats.items()})
    return raw, agg, diag_agg, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-root", type=str, default="data")
    parser.add_argument("--allow-digits-fallback", action="store_true")
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="mnist10_misspecified_failure_results")
    parser.add_argument("--max-train-per-class", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--teacher-hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clpl-anchor-prob", type=float, default=0.95)
    parser.add_argument("--clcl-diag-prob", type=float, default=0.40)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, agg, diag_agg, meta = run_experiment(args)

    raw_path = out_dir / "mnist10_misspecified_failure_raw.csv"
    agg_path = out_dir / "mnist10_misspecified_failure_aggregated.csv"
    diag_path = out_dir / "mnist10_misspecified_failure_transform_diagnostics.csv"
    meta_path = out_dir / "mnist10_misspecified_failure_meta.json"
    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)
    diag_agg.to_csv(diag_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [raw_path, agg_path, diag_path, meta_path, Path(__file__)]:
            zf.write(p, arcname=p.name)

    print("=== meta ===")
    print(json.dumps(meta, indent=2))
    print("\n=== transform diagnostics ===")
    print(diag_agg.to_string(index=False))
    print("\n=== aggregated test results ===")
    print(agg.to_string(index=False))
    print(f"\nSaved results to {out_dir}")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
