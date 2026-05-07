
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


Array = np.ndarray


# ============================================================
# Utilities
# ============================================================

def ensure_col_stochastic(M: Array, name: str = "M") -> Array:
    M = np.asarray(M, dtype=np.float64)
    if np.min(M) < -1e-10:
        raise ValueError(f"{name} has negative entries: min={np.min(M)}")
    col_sums = M.sum(axis=0, keepdims=True)
    if np.any(col_sums <= 0):
        raise ValueError(f"{name} has an empty column.")
    return M / col_sums


def construct_clwl_T(M: Array, safety_factor: float = 0.95) -> Array:
    """
    Standalone module1-style construction:
      N = pinv(M),
      T = alpha (N - 1 q^T),
    where q is the columnwise minimum of N.
    """
    M = ensure_col_stochastic(M, "M")
    N = np.linalg.pinv(M)
    q = N.min(axis=0)
    R = N - q[None, :]
    col_ranges = N.max(axis=0) - N.min(axis=0)
    alpha = safety_factor / max(float(np.max(col_ranges)), 1e-12)
    return (alpha * R).astype(np.float64)


def fit_standard_form(A: Array) -> dict:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    v = np.zeros(c, dtype=np.float64)
    for j in range(c):
        off = [A[i, j] for i in range(c) if i != j]
        v[j] = float(np.mean(off)) if off else 0.0
    lam = float(np.mean(np.diag(A) - v))
    A_hat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    res = float(np.linalg.norm(A - A_hat, ord="fro"))
    rel = res / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    return {"lambda": lam, "residual": res, "relative_residual": rel}


def make_pair_map(c: int) -> dict[int, int]:
    if c % 2 != 0:
        raise ValueError("This script expects an even number of classes.")
    pair = {}
    for i in range(0, c, 2):
        pair[i] = i + 1
        pair[i + 1] = i
    return pair


# ============================================================
# Matrix construction: MNIST-4 with 3 weak response modes per digit
# ============================================================

def make_digit_response_Mtrue_3mode(
    c: int,
    p_high: float = 0.3819518877,
    p_low: float = 0.0408696747,
    p_amb: float = 0.0455088471,
    q_high: float = 0.0403115924,
    q_low: float = 0.1827273197,
    q_amb: float = 0.0926864795,
) -> tuple[Array, dict[int, int]]:
    """
    d=3c weak labels:
      z_{3k}   = high-confidence response for class k,
      z_{3k+1} = low-confidence response for class k,
      z_{3k+2} = ambiguous/fallback response for class k.

    For true class y, mass is assigned to:
      y_H, y_L, y_A,
      r(y)_H, r(y)_L, r(y)_A,
      and ambiguous responses of other digits.

    For MNIST4 we recommend original classes [1,7,3,8], so pairs are:
      1 <-> 7 and 3 <-> 8, internally represented as 0<->1 and 2<->3.
    """
    pair = make_pair_map(c)
    d = 3 * c
    M = np.zeros((d, c), dtype=np.float64)
    rem = 1.0 - p_high - p_low - p_amb - q_high - q_low - q_amb
    if rem < -1e-12:
        raise ValueError("M_true parameters sum to more than 1.")

    for y in range(c):
        r = pair[y]
        M[3 * y, y] += p_high
        M[3 * y + 1, y] += p_low
        M[3 * y + 2, y] += p_amb
        M[3 * r, y] += q_high
        M[3 * r + 1, y] += q_low
        M[3 * r + 2, y] += q_amb

        others = [k for k in range(c) if k not in (y, r)]
        for k in others:
            M[3 * k + 2, y] += rem / max(len(others), 1)

    return ensure_col_stochastic(M, "M_true"), pair


def make_structured_H_3mode(
    c: int,
    pair: dict[int, int],
    boost_pair_high: float = 2.4100825344,
    boost_pair_low: float = 5.8820760550,
    boost_pair_amb: float = 5.0043179425,
    penalize_true_high: float = 3.4760276078,
    penalize_true_low: float = 0.9485361404,
    penalize_true_amb: float = 2.1795081387,
) -> Array:
    """
    Pilot-estimation bias H.

    Positive means the pilot estimate over-counts that response mode for class y.
    Negative means the pilot estimate under-counts that response mode for class y.

    This H models a realistic pilot bias:
      - the pilot/weak-label generator overestimates responses for the confusing digit r(y);
      - it underestimates the true digit responses.
    """
    H = np.zeros((3 * c, c), dtype=np.float64)
    for y in range(c):
        r = pair[y]
        H[3 * r, y] += boost_pair_high
        H[3 * r + 1, y] += boost_pair_low
        H[3 * r + 2, y] += boost_pair_amb
        H[3 * y, y] -= penalize_true_high
        H[3 * y + 1, y] -= penalize_true_low
        H[3 * y + 2, y] -= penalize_true_amb
    return H


def make_Mhat_row_tilt(M_true: Array, H: Array, s: float) -> Array:
    W = np.asarray(M_true, dtype=np.float64) * np.exp(float(s) * np.asarray(H, dtype=np.float64))
    return ensure_col_stochastic(W, "M_hat")


# ============================================================
# Data loading
# ============================================================

@dataclass
class SplitData:
    X: Array
    y: Array


def load_data(
    data_source: str,
    classes: Sequence[int],
    max_train: int | None,
    max_test: int | None,
    seed: int,
) -> tuple[SplitData, SplitData]:
    classes = list(map(int, classes))
    class_to_new = {old: i for i, old in enumerate(classes)}
    rng = np.random.default_rng(seed)

    if data_source == "openml":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
        Xtr_all, ytr_all = X[:60000], y[:60000]
        Xte_all, yte_all = X[60000:], y[60000:]

    elif data_source == "torchvision_mnist":
        from torchvision.datasets import MNIST

        root = "./data"
        tr = MNIST(root=root, train=True, download=True)
        te = MNIST(root=root, train=False, download=True)

        Xtr_all = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        ytr_all = tr.targets.numpy().astype(int)

        Xte_all = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        yte_all = te.targets.numpy().astype(int)


    elif data_source == "sklearn_digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        X_all = data.data.astype(np.float32) / 16.0
        y_all = data.target.astype(int)
        Xtr_all, Xte_all, ytr_all, yte_all = train_test_split(
            X_all, y_all, test_size=0.25, random_state=seed, stratify=y_all
        )
    else:
        raise ValueError("data_source must be 'openml', 'torchvision_mnist', or 'sklearn_digits'.")

    def subset(X, y, max_n):
        mask = np.isin(y, classes)
        Xs, ys_old = X[mask], y[mask]
        ys = np.asarray([class_to_new[int(v)] for v in ys_old], dtype=np.int64)
        if max_n is not None and len(ys) > max_n:
            idx = rng.choice(len(ys), size=max_n, replace=False)
            Xs, ys = Xs[idx], ys[idx]
        return SplitData(X=Xs.astype(np.float32), y=ys.astype(np.int64))

    train = subset(Xtr_all, ytr_all, max_train)
    test = subset(Xte_all, yte_all, max_test)

    mean = train.X.mean(axis=0, keepdims=True)
    std = np.maximum(train.X.std(axis=0, keepdims=True), 1e-6)
    train.X = (train.X - mean) / std
    test.X = (test.X - mean) / std
    return train, test


def make_val_split(train: SplitData, val_frac: float, seed: int) -> tuple[SplitData, SplitData]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train.y))
    n_val = int(round(val_frac * len(idx)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return SplitData(train.X[tr_idx], train.y[tr_idx]), SplitData(train.X[val_idx], train.y[val_idx])


def sample_weak_labels(y: Array, M_true: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(M_true.shape[0], p=M_true[:, int(yi)]))
    return z


# ============================================================
# Torch training
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, c: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c),
        )

    def forward(self, x):
        return self.net(x)


def clwl_loss(logits: torch.Tensor, z: torch.Tensor, T_torch: torch.Tensor) -> torch.Tensor:
    target = T_torch[:, z].T
    loss = target * F.softplus(-logits) + (1.0 - target) * F.softplus(logits)
    return loss.sum(dim=1).mean()


def forward_loss(logits: torch.Tensor, z: torch.Tensor, M_torch: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    weak_probs = torch.clamp(probs @ M_torch.T, min=1e-8)
    weak_probs = weak_probs / weak_probs.sum(dim=1, keepdim=True)
    return F.nll_loss(torch.log(weak_probs), z)


def train_model(
    method: str,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    z_train: Array,
    M_or_T: Array,
    c: int,
    seed: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    Xtr = torch.tensor(train.X, dtype=torch.float32, device=device)
    ztr = torch.tensor(z_train, dtype=torch.long, device=device)
    Xv = torch.tensor(val.X, dtype=torch.float32, device=device)
    yv = torch.tensor(val.y, dtype=torch.long, device=device)
    Xte = torch.tensor(test.X, dtype=torch.float32, device=device)
    yte = torch.tensor(test.y, dtype=torch.long, device=device)

    model = MLP(train.X.shape[1], c, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if method == "clwl":
        T_torch = torch.tensor(M_or_T, dtype=torch.float32, device=device)
    else:
        M_torch = torch.tensor(M_or_T, dtype=torch.float32, device=device)

    best_state = None
    best_val = -1.0
    patience = 25
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        if method == "clwl":
            loss = clwl_loss(logits, ztr, T_torch)
        else:
            loss = forward_loss(logits, ztr, M_torch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_acc = float((model(Xv).argmax(dim=1) == yv).float().mean().item())
            if val_acc > best_val + 1e-5:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_acc = float((model(Xte).argmax(dim=1) == yte).float().mean().item())
    return {"test_acc": test_acc, "best_val_acc": best_val}


# ============================================================
# Experiment runner
# ============================================================

def run_experiment(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = list(map(int, args.classes))
    c = len(classes)
    if c % 2 != 0:
        raise ValueError("Use an even number of classes with paired confusions.")

    train_full, test = load_data(
        args.data_source,
        classes,
        args.max_trainval_samples,
        args.max_test_samples,
        args.seed,
    )

    M_true, pair = make_digit_response_Mtrue_3mode(c)
    H = make_structured_H_3mode(c, pair)

    np.save(out_dir / "M_true.npy", M_true)
    np.save(out_dir / "H.npy", H)

    diag_rows = []
    for s in args.s_grid:
        M_hat = make_Mhat_row_tilt(M_true, H, float(s))
        T_hat = construct_clwl_T(M_hat)
        fit = fit_standard_form(T_hat @ M_true)
        diag_rows.append({"s": s, **fit})
        np.save(out_dir / f"M_hat_s_{s:.2f}.npy", M_hat)
        np.save(out_dir / f"T_hat_s_{s:.2f}.npy", T_hat)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)

    print("classes:", classes)
    print("data_source:", args.data_source)
    print("M_true shape:", M_true.shape, "H shape:", H.shape)
    print("diagnostics:")
    print(diag_df)

    rows = []
    for train_seed in args.train_seeds:
        train, val = make_val_split(train_full, args.val_frac, train_seed)
        z_train = sample_weak_labels(train.y, M_true, seed=10000 + train_seed)

        oracle = train_model(
            "forward", train, val, test, z_train, M_true,
            c=c, seed=train_seed, hidden_dim=args.hidden_dim,
            epochs=args.epochs, lr=args.lr, device=args.device,
        )

        for s in args.s_grid:
            M_hat = make_Mhat_row_tilt(M_true, H, float(s))
            T_hat = construct_clwl_T(M_hat)

            clwl = train_model(
                "clwl", train, val, test, z_train, T_hat,
                c=c, seed=train_seed, hidden_dim=args.hidden_dim,
                epochs=args.epochs, lr=args.lr, device=args.device,
            )
            fwd = train_model(
                "forward", train, val, test, z_train, M_hat,
                c=c, seed=train_seed, hidden_dim=args.hidden_dim,
                epochs=args.epochs, lr=args.lr, device=args.device,
            )

            rows.append({"method": "CLWL", "seed": train_seed, "s": s, **clwl})
            rows.append({"method": "Forward_Mhat", "seed": train_seed, "s": s, **fwd})
            rows.append({"method": "Forward_oracle_Mtrue", "seed": train_seed, "s": s, **oracle})

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw_results.csv", index=False)

    summary = raw.groupby(["method", "s"], as_index=False).agg({
        "test_acc": ["mean", "std"],
        "best_val_acc": ["mean", "std"],
    })
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(out_dir / "summary_results.csv", index=False)

    print("summary:")
    print(summary)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for method in ["CLWL", "Forward_Mhat", "Forward_oracle_Mtrue"]:
        dfm = summary[summary["method"] == method].sort_values("s")
        x = dfm["s"].to_numpy(float)
        y = dfm["test_acc_mean"].to_numpy(float)
        ystd = np.nan_to_num(dfm["test_acc_std"].to_numpy(float), nan=0.0)
        axes[0].plot(x, y, marker="o", label=method)
        axes[0].fill_between(x, y - ystd, y + ystd, alpha=0.15)

    axes[0].set_xlabel("estimate bias strength s")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(f"{args.data_source}: classes {classes}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(diag_df["s"], diag_df["lambda"], marker="o", label="lambda")
    axes[1].plot(diag_df["s"], diag_df["relative_residual"], marker="o", label="relative residual")
    axes[1].axhline(0, linestyle="--", alpha=0.6)
    axes[1].set_xlabel("estimate bias strength s")
    axes[1].set_title("CLWL matrix diagnostics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(out_dir / "mnist4_3mode_row_tilt_early_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return raw, summary, diag_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_source", choices=["openml", "torchvision_mnist", "sklearn_digits"], default="torchvision_mnist")
    p.add_argument("--classes", nargs="+", type=int, default=[1, 7, 3, 8])
    p.add_argument("--out_dir", type=str, default="artifacts_mnist4_3mode_row_tilt_early")
    p.add_argument("--max_trainval_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--s_grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
