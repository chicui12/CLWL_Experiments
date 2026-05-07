#!/usr/bin/env python3
"""
Incomplete asymmetric partial-label experiment for CLPL vs CLWL.

Default mode uses MNIST from OpenML:
    python clwl_mnist_incomplete_asym_experiment.py --dataset mnist --epochs 10 --max_train 12000 --max_test 2000

If OpenML is unavailable, run the same weak-label mechanism on sklearn's built-in digits proxy:
    python clwl_mnist_incomplete_asym_experiment.py --dataset digits --epochs 400

The weak-label model is:
    with probability 1-rho: B = {y, u}, u uniform over non-y labels
    with probability rho:   B = {pi(y), pi^2(y)}
where pi is a fixed digit-confusion permutation.

The script saves:
    raw.csv
    summary.csv
    epoch_history.csv
    accuracy_vs_probability.pdf
    loss_trend_over_epochs.pdf
"""

import argparse
import itertools
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pi():
    pi = np.zeros(10, dtype=int)
    for cyc in ([0, 6, 8, 3, 5], [1, 7, 9, 4, 2]):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            pi[a] = b
    return pi


def build_weak_model(rho: float, c: int = 10):
    pi = make_pi()
    pairs = [tuple(p) for p in itertools.combinations(range(c), 2)]
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    Z = np.zeros((c, len(pairs)), dtype=np.float64)
    for k, pair in enumerate(pairs):
        Z[list(pair), k] = 1.0

    M = np.zeros((len(pairs), c), dtype=np.float64)
    for y in range(c):
        # Strict partial-label branch: true digit plus one uniformly random wrong digit.
        for u in range(c):
            if u == y:
                continue
            M[pair_to_idx[tuple(sorted((y, u)))], y] += (1.0 - rho) / (c - 1)

        # Incomplete asymmetric branch: true digit is absent; two decoys are shown.
        decoy_pair = tuple(sorted((int(pi[y]), int(pi[pi[y]]))))
        M[pair_to_idx[decoy_pair], y] += rho

    return pairs, Z, M, pi


def construct_T(M: np.ndarray, alpha_scale: float = 0.99):
    # M is d x c, column stochastic, full column rank.
    N = np.linalg.inv(M.T @ M) @ M.T
    q = N.min(axis=0)
    ranges = N.max(axis=0) - N.min(axis=0)
    Delta = ranges.max()
    alpha = alpha_scale / Delta if Delta > 0 else 1.0
    T = alpha * (N - np.ones((N.shape[0], 1)) @ q.reshape(1, -1))
    A = T @ M
    return T, A, alpha


def dominance_diagnostics(M: np.ndarray, Z: np.ndarray, pairs):
    c = Z.shape[0]
    argmax_set_ok = 0
    dom_violations = 0
    dom_checks = 0

    labels = set(range(c))
    for y in range(c):
        p = M[:, y]
        q = Z @ p
        argq = set(np.where(np.abs(q - q.max()) < 1e-10)[0].tolist())
        if argq == {y}:
            argmax_set_ok += 1

        prob = {tuple(pair): p[k] for k, pair in enumerate(pairs) if p[k] > 1e-15}

        # Cour-style dominance checked relative to the clean Bayes class y:
        # P(C union {y}) >= P(C union {b}) for every b != y and every C excluding y,b.
        for b in range(c):
            if b == y:
                continue
            others = list(labels - {y, b})
            for r in range(len(others) + 1):
                for C in itertools.combinations(others, r):
                    S_y = tuple(sorted(C + (y,)))
                    S_b = tuple(sorted(C + (b,)))
                    py = prob.get(S_y, 0.0)
                    pb = prob.get(S_b, 0.0)
                    dom_checks += 1
                    if py + 1e-12 < pb:
                        dom_violations += 1

    return {
        "argmax_set_equal_rate": argmax_set_ok / c,
        "dominance_violation_rate": dom_violations / dom_checks,
        "dominance_violations": dom_violations,
        "dominance_checks": dom_checks,
    }


class SmallMLP(nn.Module):
    def __init__(self, d_in: int, c: int = 10, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, c),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_weak_indices(y: np.ndarray, M: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yy in enumerate(y.astype(int)):
        z[i] = rng.choice(M.shape[0], p=M[:, yy])
    return z


def compute_loss(s, ztr_t, method: str, Z_rows, T_rows=None):
    if method == "clpl":
        b = Z_rows[ztr_t]
        mean_s = (b * s).sum(dim=1) / b.sum(dim=1).clamp_min(1.0)
        return (F.softplus(-mean_s) + ((1.0 - b) * F.softplus(s)).sum(dim=1)).mean()
    if method == "clwl":
        if T_rows is None:
            raise ValueError("T_rows must be provided for CLWL")
        tz = T_rows[ztr_t]
        return (tz * F.softplus(-s) + (1.0 - tz) * F.softplus(s)).sum(dim=1).mean()
    raise ValueError(method)


def train_one(Xtr, ytr, Xte, yte, rho, method, seed, epochs, lr, hidden, weight_decay):
    """Train one model and return final accuracies plus a per-epoch history.

    The per-epoch history supports the requested loss-over-epochs visualization.
    It stores train_loss, train_acc, and test_acc after every epoch.
    """
    set_seed(seed)
    pairs, Z, M, _ = build_weak_model(rho)
    ztr = sample_weak_indices(ytr, M, seed + 12345)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    yte_t = torch.tensor(yte, dtype=torch.long)
    ztr_t = torch.tensor(ztr, dtype=torch.long)

    model = SmallMLP(Xtr.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Z_rows = torch.tensor(Z.T, dtype=torch.float32)  # d x c
    T_rows = None
    if method == "clwl":
        T, _, _ = construct_T(M)
        T_rows = torch.tensor(T.T, dtype=torch.float32)  # d x c

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        s = model(Xtr_t)
        loss = compute_loss(s, ztr_t, method, Z_rows, T_rows)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_tr = model(Xtr_t).argmax(dim=1)
            pred_te = model(Xte_t).argmax(dim=1)
            train_acc = (pred_tr == ytr_t).float().mean().item()
            test_acc = (pred_te == yte_t).float().mean().item()

        history.append({
            "rho": rho,
            "seed": seed,
            "method": method,
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu().item()),
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

    final = history[-1]
    return final["test_acc"], final["train_acc"], history


def load_data(dataset: str, max_train: int | None, max_test: int | None, seed: int, mnist_npz: str | None = None):
    if dataset == "digits":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        X = X.astype(np.float32) / 16.0
        Xtr, Xte, ytr, yte = train_test_split(
            X, y.astype(int), test_size=0.3, random_state=42, stratify=y
        )
    elif dataset == "mnist":
        if mnist_npz is not None:
            data = np.load(mnist_npz)
            Xtr = data["x_train"].reshape(len(data["x_train"]), -1).astype(np.float32) / 255.0
            ytr = data["y_train"].astype(int)
            Xte = data["x_test"].reshape(len(data["x_test"]), -1).astype(np.float32) / 255.0
            yte = data["y_test"].astype(int)
        else:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X = mnist.data.astype(np.float32) / 255.0
            y = mnist.target.astype(int)
            # OpenML MNIST follows the usual 60k/10k ordering.
            Xtr, ytr = X[:60000], y[:60000]
            Xte, yte = X[60000:], y[60000:]
    else:
        raise ValueError(dataset)

    rng = np.random.default_rng(seed)
    if max_train is not None and max_train < len(ytr):
        idx = np.concatenate([
            rng.choice(np.where(ytr == k)[0], size=min(max_train // 10, np.sum(ytr == k)), replace=False)
            for k in range(10)
        ])
        rng.shuffle(idx)
        Xtr, ytr = Xtr[idx], ytr[idx]

    if max_test is not None and max_test < len(yte):
        idx = np.concatenate([
            rng.choice(np.where(yte == k)[0], size=min(max_test // 10, np.sum(yte == k)), replace=False)
            for k in range(10)
        ])
        rng.shuffle(idx)
        Xte, yte = Xte[idx], yte[idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    return Xtr, ytr.astype(int), Xte, yte.astype(int)



def plot_accuracy_vs_probability(summary: pd.DataFrame, out_path: Path):
    """Plot final test accuracy as a function of rho/probability."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("rho")
        yerr = sub["test_acc_std"].fillna(0.0).to_numpy()
        ax.errorbar(
            sub["rho"],
            sub["test_acc_mean"],
            yerr=yerr,
            marker="o",
            capsize=3,
            label=method.upper(),
        )
    ax.set_xlabel(r"Incomplete asymmetric probability $\rho$")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. incomplete asymmetric probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=1.0)
    ax.legend(framealpha=1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_trend_over_epochs(epoch_df: pd.DataFrame, out_path: Path, rhos_to_plot=None):
    """Plot mean training loss over epochs, averaged across seeds.

    If many rho values are used, the plot can become crowded. By default we plot
    all rho values. Use --loss_plot_rhos to select a smaller subset, e.g.
    --loss_plot_rhos 0.0 0.3 0.6
    """
    plot_df = epoch_df.copy()
    if rhos_to_plot is not None and len(rhos_to_plot) > 0:
        wanted = np.array(rhos_to_plot, dtype=float)
        keep = np.zeros(len(plot_df), dtype=bool)
        for rho in wanted:
            keep |= np.isclose(plot_df["rho"].to_numpy(dtype=float), rho)
        plot_df = plot_df.loc[keep]

    mean_df = plot_df.groupby(["rho", "method", "epoch"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (rho, method), sub in mean_df.groupby(["rho", "method"]):
        sub = sub.sort_values("epoch")
        ax.plot(
            sub["epoch"],
            sub["train_loss_mean"],
            linewidth=1.8,
            label=f"{method.upper()}, rho={rho:g}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss trend over epochs")
    ax.grid(True, alpha=1.0)
    ax.legend(fontsize=8, ncol=2, framealpha=1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--clpl_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=12000)
    parser.add_argument("--max_test", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="clwl_mnist_results")
    parser.add_argument("--mnist_npz", type=str, default=None, help="Optional Keras-style mnist.npz with x_train/y_train/x_test/y_test.")
    parser.add_argument(
        "--loss_plot_rhos",
        type=float,
        nargs="*",
        default=None,
        help="Optional subset of rho values to include in loss_trend_over_epochs.pdf. Example: --loss_plot_rhos 0.0 0.3 0.6",
    )
    args = parser.parse_args()

    torch.set_num_threads(2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = load_data(args.dataset, args.max_train, args.max_test, seed=123, mnist_npz=args.mnist_npz)
    rows = []
    epoch_rows = []
    t0 = time.time()

    for rho in args.rhos:
        pairs, Z, M, _ = build_weak_model(rho)
        diag = dominance_diagnostics(M, Z, pairs)
        T, A, alpha = construct_T(M)
        v = (A - alpha * np.eye(10))[0, :]
        A_resid = np.linalg.norm(A - (alpha * np.eye(10) + np.ones((10, 1)) @ v.reshape(1, -1)))

        for seed in args.seeds:
            for method in ["clpl", "clwl"]:
                run_epochs = args.clpl_epochs if (method == "clpl" and np.isclose(rho, 0.0)) else args.epochs

                test_acc, train_acc, history = train_one(
                    Xtr, ytr, Xte, yte, rho, method, seed,
                    epochs=run_epochs,
                    lr=args.lr,
                    hidden=args.hidden,
                    weight_decay=args.weight_decay,
                )
                for h in history:
                    h.update({
                        "dataset": args.dataset,
                        "epochs_run": run_epochs,
                        "rankM": np.linalg.matrix_rank(M),
                        "alpha": alpha,
                        "A_resid": A_resid,
                        **diag,
                    })
                epoch_rows.extend(history)
                rows.append({
                    "dataset": args.dataset,
                    "rho": rho,
                    "seed": seed,
                    "method": method,
                    "epochs_run": run_epochs,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "rankM": np.linalg.matrix_rank(M),
                    "alpha": alpha,
                    "A_resid": A_resid,
                    **diag,
                })
                print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby(["dataset", "rho", "method"]).agg(
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        train_acc_mean=("train_acc", "mean"),
        rankM=("rankM", "first"),
        alpha=("alpha", "first"),
        A_resid=("A_resid", "first"),
        argmax_set_equal_rate=("argmax_set_equal_rate", "first"),
        dominance_violation_rate=("dominance_violation_rate", "first"),
        dominance_violations=("dominance_violations", "first"),
        dominance_checks=("dominance_checks", "first"),
    ).reset_index()

    epoch_df = pd.DataFrame(epoch_rows)

    df.to_csv(out_dir / "raw.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    epoch_df.to_csv(out_dir / "epoch_history.csv", index=False)

    plot_accuracy_vs_probability(summary, out_dir / "accuracy_vs_probability.pdf")
    plot_loss_trend_over_epochs(epoch_df, out_dir / "loss_trend_over_epochs.pdf", args.loss_plot_rhos)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_dir}. Runtime: {time.time() - t0:.1f}s")
    print(f"Saved plot: {out_dir / 'accuracy_vs_probability.pdf'}")
    print(f"Saved plot: {out_dir / 'loss_trend_over_epochs.pdf'}")
    print(f"Saved per-epoch history: {out_dir / 'epoch_history.csv'}")


if __name__ == "__main__":
    main()
