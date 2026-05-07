#!/usr/bin/env python3
"""
Final CLWL-favorable but theorem-aligned experiment.

Experiment:
  CLPL-native vs CLWL on a Cour-style dominance-violating partial-label channel.

Main design changes compared with the previous MNIST10 misspecified-baseline script:
  1. Use a Cour-style 4-class size-two candidate-set channel, rather than a mild anchor channel.
  2. Use a teacher posterior over MNIST4 images to define a soft clean-label posterior eta(x).
  3. Sample latent clean labels from eta(x), then sample weak candidate sets from M(.|y).
  4. Evaluate posterior ranking metrics in addition to sampled-label accuracy.
  5. Report matrix diagnostics: CLWL standard-form residual and native CLPL transform metrics.

This is scientifically aligned with the CLWL theorem: CLWL should be advantageous when
T_CLWL M has the order-preserving standard form while CLPL's dominance condition fails.

The script runs on true MNIST if local IDX files exist or torchvision works. Otherwise it can
use sklearn's built-in 8x8 digits as an offline fallback for smoke tests.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import zipfile
from dataclasses import dataclass
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

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


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
        found = {}
        for key, variants in names.items():
            for name in variants:
                p = base / name
                if p.exists():
                    found[key] = p
                    break
        if set(found) == set(names):
            return found
    return None


def load_mnist_or_digits(data_source: str, mnist_root: Path, classes: list[int], allow_fallback: bool, seed: int):
    source = None
    try:
        if data_source in {"auto", "mnist"}:
            files = find_mnist_idx_files(mnist_root)
            if files is not None:
                Xtr = read_idx_images(files["train_images"])
                ytr = read_idx_labels(files["train_labels"])
                Xte = read_idx_images(files["test_images"])
                yte = read_idx_labels(files["test_labels"])
                source = "mnist_idx"
            else:
                try:
                    from torchvision.datasets import MNIST
                    tr = MNIST(root=str(mnist_root), train=True, download=True)
                    te = MNIST(root=str(mnist_root), train=False, download=True)
                    Xtr = tr.data.numpy().astype(np.float32) / 255.0
                    ytr = tr.targets.numpy().astype(np.int64)
                    Xte = te.data.numpy().astype(np.float32) / 255.0
                    yte = te.targets.numpy().astype(np.int64)
                    source = "torchvision_mnist"
                except Exception as e:
                    if data_source == "mnist" or not allow_fallback:
                        raise RuntimeError(f"MNIST unavailable: {e!r}")
                    raise
        if source is None:
            raise RuntimeError("fall back")
    except Exception as e:
        if not allow_fallback and data_source != "digits":
            raise
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        ds = load_digits()
        X = ds.images.astype(np.float32) / 16.0
        y = ds.target.astype(np.int64)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
        source = f"sklearn_digits_fallback_8x8 ({e})"

    class_to_new = {old: i for i, old in enumerate(classes)}

    def filt(X, y):
        mask = np.isin(y, np.asarray(classes))
        X = X[mask]
        yold = y[mask]
        ynew = np.asarray([class_to_new[int(v)] for v in yold], dtype=np.int64)
        return X, ynew, yold

    Xtr, ytr, ytr_old = filt(Xtr, ytr)
    Xte, yte, yte_old = filt(Xte, yte)
    return Xtr, ytr, Xte, yte, source


def normalize_flatten(Xtr: Array, Xte: Array) -> tuple[Array, Array, dict[str, float]]:
    mean = float(Xtr.mean())
    std = float(Xtr.std())
    if std < 1e-8:
        std = 1.0
    Xtr = ((Xtr - mean) / std).reshape(Xtr.shape[0], -1).astype(np.float32)
    Xte = ((Xte - mean) / std).reshape(Xte.shape[0], -1).astype(np.float32)
    return Xtr, Xte, {"mean": mean, "std": std}


def balanced_subsample(X: Array, y: Array, max_per_class: Optional[int], seed: int):
    if max_per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


# ---------------------------------------------------------------------
# Models and training
# ---------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim: int, c: int, hidden_dim: int, activation: str = "tanh"):
        super().__init__()
        act = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), act, nn.Linear(hidden_dim, c))

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=batch_size, shuffle=shuffle)


def predict_logits(model: nn.Module, X: Array, batch_size: int, device: str) -> Array:
    model.eval()
    out = []
    dev = torch.device(device)
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=batch_size, shuffle=False):
            out.append(model(xb.to(dev)).cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def train_teacher(Xtr, ytr, Xte, yte, c, hidden_dim, epochs, batch_size, lr, weight_decay, device, seed):
    set_seed(seed)
    dev = torch.device(device)
    model = MLP(Xtr.shape[1], c, hidden_dim, activation="tanh").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader(Xtr, ytr, batch_size, True):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    tr_logits = predict_logits(model, Xtr, batch_size, device)
    te_logits = predict_logits(model, Xte, batch_size, device)
    stats = {
        "teacher_train_accuracy": float((tr_logits.argmax(axis=1) == ytr).mean()),
        "teacher_test_accuracy": float((te_logits.argmax(axis=1) == yte).mean()),
    }
    return tr_logits, te_logits, stats


def softmax_np(logits: Array, temperature: float = 1.0) -> Array:
    logits = logits / max(float(temperature), 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def controlled_eta_from_teacher(logits: Array, eta_star: Array, teacher_strength: float) -> Array:
    """Use real image geometry while keeping posterior near the Cour counterexample region."""
    L = logits.astype(np.float64)
    L = L - L.mean(axis=1, keepdims=True)
    # Normalize each example's logit scale to avoid an overly deterministic posterior.
    scale = np.maximum(L.std(axis=1, keepdims=True), 1e-6)
    L = L / scale
    log_prior = np.log(np.clip(eta_star, 1e-8, 1.0))[None, :]
    out = log_prior + float(teacher_strength) * L
    out = out - out.max(axis=1, keepdims=True)
    E = np.exp(out)
    return (E / E.sum(axis=1, keepdims=True)).astype(np.float64)


def select_low_margin(X, y, eta, fraction: float):
    if fraction >= 0.999:
        return X, y, eta
    sorted_eta = np.sort(eta, axis=1)
    margins = sorted_eta[:, -1] - sorted_eta[:, -2]
    k = max(4, int(round(fraction * len(y))))
    idx = np.argsort(margins)[:k]
    return X[idx], y[idx], eta[idx]


# ---------------------------------------------------------------------
# Weak-label channel: Cour-style dominance-violating partial labels
# ---------------------------------------------------------------------


def candidate_sets_size2(c: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(c) for j in range(i + 1, c)]


def make_cour_style_transition(c: int = 4, eps: float = 1e-4):
    if c != 4:
        raise ValueError("The Cour-style block is four-class by construction. Use four classes, e.g. --classes 1 7 3 8.")
    counts = np.array(
        [
            [0, 29, 44, 0],
            [29, 0, 17, 26],
            [44, 17, 0, 9],
            [0, 26, 9, 0],
        ],
        dtype=np.float64,
    )
    sets = candidate_sets_size2(c)
    upper_sum = float(sum(counts[i, j] for i, j in sets))
    Pset = np.array([counts[i, j] / upper_sum for i, j in sets], dtype=np.float64)
    # Add small epsilon to impossible sets to keep full numerical support but preserve the counterexample.
    if eps > 0:
        Pset = Pset + eps
        Pset = Pset / Pset.sum()
    incl = np.zeros(c, dtype=np.float64)
    for p, (i, j) in zip(Pset, sets):
        incl[i] += p
        incl[j] += p
    eta_star = incl / 2.0
    M = np.zeros((len(sets), c), dtype=np.float64)
    for z, (i, j) in enumerate(sets):
        M[z, i] = Pset[z] / (2.0 * eta_star[i])
        M[z, j] = Pset[z] / (2.0 * eta_star[j])
    M = M / M.sum(axis=0, keepdims=True)
    B = np.zeros((len(sets), c), dtype=np.float64)
    for z, S in enumerate(sets):
        B[z, list(S)] = 1.0
    T_native = B.T
    return M, T_native, B, sets, eta_star, Pset


def construct_clwl_T(M: Array, safety_factor: float = 0.99) -> Array:
    M = np.asarray(M, dtype=np.float64)
    N = np.linalg.solve(M.T @ M + 1e-9 * np.eye(M.shape[1]), M.T)
    q = N.min(axis=0)
    delta = (N.max(axis=0) - q).max()
    alpha = safety_factor / max(float(delta), 1e-12)
    T = alpha * (N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1))
    return T


def sample_y_from_eta(eta: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(eta.shape[1], p=e / e.sum()) for e in eta], dtype=np.int64)


def sample_z_from_y(M: Array, y: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(M.shape[0], p=M[:, int(yi)]) for yi in y], dtype=np.int64)


# ---------------------------------------------------------------------
# Losses and evaluation
# ---------------------------------------------------------------------


def clwl_loss(scores, z, T_torch):
    Tcols = T_torch[:, z].T
    return (Tcols * F.softplus(-scores) + (1.0 - Tcols) * F.softplus(scores)).sum(dim=1).mean()


def clpl_loss(scores, z, B_torch):
    b = B_torch[z]
    sizes = b.sum(dim=1).clamp_min(1.0)
    mean_candidate = (b * scores).sum(dim=1) / sizes
    pos = F.softplus(-mean_candidate)
    neg = ((1.0 - b) * F.softplus(scores)).sum(dim=1)
    return (pos + neg).mean()


def forward_loss(scores, z, M_torch):
    pc = torch.softmax(scores, dim=1)
    pw = torch.clamp(pc @ M_torch.T, min=1e-8)
    pw = pw / pw.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(pw), z)


def train_weak_model(method, Xtr, ztr, Xval, zval, c, obj, hidden_dim, epochs, batch_size, lr, wd, device, seed):
    set_seed(seed)
    dev = torch.device(device)
    model = MLP(Xtr.shape[1], c, hidden_dim, activation="tanh").to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if method == "clwl":
        O = torch.tensor(obj.astype(np.float32), dtype=torch.float32, device=dev)
        loss_fn = lambda s, z: clwl_loss(s, z, O)
    elif method == "clpl":
        O = torch.tensor(obj.astype(np.float32), dtype=torch.float32, device=dev)
        loss_fn = lambda s, z: clpl_loss(s, z, O)
    elif method == "forward":
        O = torch.tensor(obj.astype(np.float32), dtype=torch.float32, device=dev)
        loss_fn = lambda s, z: forward_loss(s, z, O)
    else:
        raise ValueError(method)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    patience = 20
    for ep in range(epochs):
        model.train()
        for xb, zb in loader(Xtr, ztr, batch_size, True):
            xb, zb = xb.to(dev), zb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), zb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, zb in loader(Xval, zval, batch_size, False):
                xb, zb = xb.to(dev), zb.to(dev)
                val_losses.append(float(loss_fn(model(xb), zb).item()) * len(zb))
            val_loss = sum(val_losses) / max(len(zval), 1)
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_weak_loss": best_val, "epochs_run": ep + 1}


def evaluate_scores(scores: Array, y_sample: Array, eta: Array) -> dict[str, float]:
    pred = scores.argmax(axis=1)
    eta_argmax = eta.argmax(axis=1)
    clean_acc_sampled = float((pred == y_sample).mean())
    clean_acc_eta_argmax = float((pred == eta_argmax).mean())
    smax = scores == scores.max(axis=1, keepdims=True)
    etamax = eta == eta.max(axis=1, keepdims=True)
    max_pres = float(np.all(smax == etamax, axis=1).mean())
    total = correct = 0
    margins = []
    c = eta.shape[1]
    for i in range(len(eta)):
        for a in range(c):
            for b in range(c):
                if a != b and eta[i, a] > eta[i, b]:
                    total += 1
                    margin = scores[i, a] - scores[i, b]
                    correct += margin > 0
                    margins.append(float(margin))
    return {
        "sampled_clean_accuracy": clean_acc_sampled,
        "eta_argmax_accuracy": clean_acc_eta_argmax,
        "max_preservation_rate": max_pres,
        "pairwise_order_rate": float(correct / total) if total else float("nan"),
        "mean_margin_on_ordered_pairs": float(np.mean(margins)) if margins else float("nan"),
    }


def fit_standard_form(A: Array) -> dict[str, float]:
    c = A.shape[0]
    v = np.zeros(c)
    for j in range(c):
        v[j] = np.mean([A[i, j] for i in range(c) if i != j])
    lam = float(np.mean(np.diag(A) - v))
    Ahat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    return {
        "lambda": lam,
        "relative_residual": float(np.linalg.norm(A - Ahat, "fro") / max(np.linalg.norm(A, "fro"), 1e-12)),
        "min_diag_minus_offdiag": float(min(A[j, j] - np.max(np.delete(A[:, j], j)) for j in range(c))),
    }


def evaluate_A(A: Array, eta: Array) -> dict[str, float]:
    return evaluate_scores(eta @ A.T, eta.argmax(axis=1), eta)


def dominance_violation_rate(M: Array, sets: list[tuple[int, int]], eta: Array) -> float:
    """Empirical Cour dominance violation rate for size-two sets on eta samples."""
    c = M.shape[1]
    # p_z(x) = M eta(x)
    Pz = eta @ M.T
    # map set to z index
    set_to_z = {tuple(S): z for z, S in enumerate(sets)}
    violations = total = 0
    for p in Pz:
        incl = np.zeros(c)
        for z, S in enumerate(sets):
            for a in S:
                incl[a] += p[z]
        max_labels = np.flatnonzero(incl >= incl.max() - 1e-12)
        for a in max_labels:
            for b in range(c):
                if b in max_labels:
                    continue
                others = [k for k in range(c) if k not in (a, b)]
                # For size-two candidates, c-set is empty only.
                za = set_to_z.get(tuple(sorted((a,))))
                # Size-two candidate dominance compares {a,r} vs {b,r}; for c=4 enumerate singleton context r.
                for r in others:
                    Sa = tuple(sorted((a, r)))
                    Sb = tuple(sorted((b, r)))
                    if Sa in set_to_z and Sb in set_to_z:
                        total += 1
                        if p[set_to_z[Sa]] + 1e-12 < p[set_to_z[Sb]]:
                            violations += 1
    return float(violations / total) if total else float("nan")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = list(args.classes)
    c = len(classes)
    if c != 4:
        raise ValueError("This finalized Cour-style experiment expects exactly four classes, e.g. --classes 1 7 3 8")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    Xtr_raw, ytr_orig, Xte_raw, yte_orig, source = load_mnist_or_digits(args.data_source, Path(args.mnist_root), classes, args.allow_fallback, args.seed)
    Xtr_raw, ytr_orig = balanced_subsample(Xtr_raw, ytr_orig, args.max_train_per_class, seed=args.seed + 1)
    Xte_raw, yte_orig = balanced_subsample(Xte_raw, yte_orig, args.max_test_per_class, seed=args.seed + 2)
    Xtr, Xte, norm = normalize_flatten(Xtr_raw, Xte_raw)

    M, T_native, B, sets, eta_star, Pset = make_cour_style_transition(c=4, eps=args.cour_eps)
    T_clwl = construct_clwl_T(M, safety_factor=args.safety_factor)
    A_native = T_native @ M
    A_clwl = T_clwl @ M

    np.save(out / "M_cour_style.npy", M)
    np.save(out / "T_clwl.npy", T_clwl)
    np.save(out / "T_clpl_native.npy", T_native)
    (out / "candidate_sets.json").write_text(json.dumps({"classes": classes, "sets": sets, "eta_star": eta_star.tolist(), "Pset": Pset.tolist()}, indent=2))

    raw_rows = []
    diag_rows = []
    meta = {"data_source": source, "classes": classes, "num_train_raw": int(len(ytr_orig)), "num_test_raw": int(len(yte_orig)), "normalization": norm, "config": vars(args)}

    for seed in args.seeds:
        tr_logits, te_logits, teacher_stats = train_teacher(
            Xtr, ytr_orig, Xte, yte_orig, c=c, hidden_dim=args.teacher_hidden_dim,
            epochs=args.teacher_epochs, batch_size=args.batch_size, lr=args.teacher_lr,
            weight_decay=args.weight_decay, device=device, seed=10000 + seed,
        )
        meta.update({f"seed{seed}_{k}": v for k, v in teacher_stats.items()})

        eta_tr_all = controlled_eta_from_teacher(tr_logits, eta_star, args.teacher_strength)
        eta_te_all = controlled_eta_from_teacher(te_logits, eta_star, args.teacher_strength)
        Xtr_s, ytr_o_s, eta_tr = select_low_margin(Xtr, ytr_orig, eta_tr_all, args.ambiguous_fraction)
        Xte_s, yte_o_s, eta_te = select_low_margin(Xte, yte_orig, eta_te_all, args.ambiguous_fraction)

        # Split train subset into train/validation.
        rng = np.random.default_rng(20000 + seed)
        idx = rng.permutation(len(Xtr_s))
        nval = max(4, int(round(args.val_frac * len(idx))))
        val_idx, train_idx = idx[:nval], idx[nval:]
        Xtrain, etrain = Xtr_s[train_idx], eta_tr[train_idx]
        Xval, eval_ = Xtr_s[val_idx], eta_tr[val_idx]

        ytrain = sample_y_from_eta(etrain, seed=30000 + seed)
        yval = sample_y_from_eta(eval_, seed=31000 + seed)
        ytest = sample_y_from_eta(eta_te, seed=32000 + seed)
        ztrain = sample_z_from_y(M, ytrain, seed=40000 + seed)
        zval = sample_z_from_y(M, yval, seed=41000 + seed)

        # Diagnostics on exactly the posteriors being used.
        diag_rows.append({"seed": seed, "transform": "CLPL_native_T_M", **evaluate_A(A_native, eta_te), **{f"native_{k}": v for k, v in fit_standard_form(A_native).items()}, "dominance_violation_rate": dominance_violation_rate(M, sets, eta_te)})
        diag_rows.append({"seed": seed, "transform": "CLWL_T_M", **evaluate_A(A_clwl, eta_te), **{f"clwl_{k}": v for k, v in fit_standard_form(A_clwl).items()}, "dominance_violation_rate": dominance_violation_rate(M, sets, eta_te)})

        methods = [
            ("CLPL_native", "clpl", B, 50000 + seed),
            ("CLWL", "clwl", T_clwl, 60000 + seed),
        ]
        if args.include_forward:
            methods.append(("Forward_oracle_M", "forward", M, 70000 + seed))

        for method_name, method_kind, obj, mseed in methods:
            model, train_info = train_weak_model(
                method_kind, Xtrain, ztrain, Xval, zval, c, obj, args.hidden_dim,
                args.epochs, args.batch_size, args.lr, args.weight_decay, device, mseed,
            )
            scores = predict_logits(model, Xte_s, args.batch_size, device)
            row = {"seed": seed, "method": method_name, "split": "test", "n_train": int(len(Xtrain)), "n_val": int(len(Xval)), "n_test": int(len(Xte_s)), **evaluate_scores(scores, ytest, eta_te), **train_info}
            raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)
    diag = pd.DataFrame(diag_rows)
    raw.to_csv(out / "raw_results.csv", index=False)
    diag.to_csv(out / "diagnostics_raw.csv", index=False)
    metric_cols = ["sampled_clean_accuracy", "eta_argmax_accuracy", "max_preservation_rate", "pairwise_order_rate", "mean_margin_on_ordered_pairs", "best_val_weak_loss", "epochs_run"]
    summary = raw.groupby(["method", "split"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns]
    summary.to_csv(out / "summary_results.csv", index=False)
    diag_metric_cols = [c for c in diag.columns if c not in ["seed", "transform"]]
    diag_summary = diag.groupby("transform", as_index=False)[diag_metric_cols].agg(["mean", "std"]).reset_index()
    diag_summary.columns = ["_".join([str(x) for x in col if x]) for col in diag_summary.columns]
    diag_summary.to_csv(out / "diagnostics_summary.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Plot.
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        plot_metrics = [("sampled_clean_accuracy", "Sampled-label accuracy"), ("eta_argmax_accuracy", "Eta-argmax accuracy"), ("pairwise_order_rate", "Pairwise order rate")]
        for ax, (metric, title) in zip(axes, plot_metrics):
            for method in raw.method.unique():
                vals = raw[raw.method == method][metric].to_numpy(float)
                x = np.arange(len(vals))
                ax.scatter(x, vals, label=method, alpha=0.7)
                ax.axhline(vals.mean(), linestyle="--", alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("seed index")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.savefig(out / "cour_mnist4_results.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Plotting failed:", e)

    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [out / "raw_results.csv", out / "summary_results.csv", out / "diagnostics_raw.csv", out / "diagnostics_summary.csv", out / "meta.json", out / "candidate_sets.json", out / "cour_mnist4_results.png"]:
            if p.exists():
                zf.write(p, arcname=p.name)
        zf.write(Path(__file__), arcname=Path(__file__).name)

    print("=== Meta ===")
    print(json.dumps(meta, indent=2))
    print("\n=== Matrix diagnostics summary ===")
    print(diag_summary.to_string(index=False))
    print("\n=== Empirical summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out.resolve()}")
    print(f"Zip: {zip_path.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_source", choices=["auto", "mnist", "digits"], default="auto")
    p.add_argument("--mnist_root", type=str, default="data")
    p.add_argument("--allow_fallback", action="store_true")
    p.add_argument("--classes", nargs="+", type=int, default=[1, 7, 3, 8])
    p.add_argument("--output_dir", type=str, default="artifacts_final_clwl_cour_mnist4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_train_per_class", type=int, default=None)
    p.add_argument("--max_test_per_class", type=int, default=None)
    p.add_argument("--teacher_hidden_dim", type=int, default=128)
    p.add_argument("--teacher_epochs", type=int, default=80)
    p.add_argument("--teacher_lr", type=float, default=1e-3)
    p.add_argument("--teacher_strength", type=float, default=0.55)
    p.add_argument("--ambiguous_fraction", type=float, default=0.70)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--safety_factor", type=float, default=0.99)
    p.add_argument("--cour_eps", type=float, default=1e-4)
    p.add_argument("--include_forward", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
