from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# Optimized multiplicative row-tilt experiment for CLWL robustness
# ============================================================
# Fixed true weak-label transition M_true.
# Estimated transition:
#   Mhat_s(z|y) = M_true(z|y) exp(s H[z,y]) / sum_z' M_true(z'|y) exp(s H[z',y])
#
# Scenario:
#   Binary clean task with six weak response modes:
#     z0: clear class-0 response
#     z1: clear class-1 response
#     z2: class-0-leaning ambiguous response
#     z3: class-1-leaning ambiguous response
#     z4: shared low-confidence response
#     z5: shared fallback response
#
# H models pilot-set response-mode bias / annotator-interface bias:
#   certain weak response modes are over-represented in the pilot estimate relative
#   to deployment. Data are always generated from fixed M_true; only the estimate
#   Mhat_s changes.


Array = np.ndarray


def construct_clwl_T(M: Array, safety: float = 0.95) -> Array:
    """Self-contained version of module1's full-column-rank construction."""
    M = np.asarray(M, dtype=np.float64)
    N = np.linalg.inv(M.T @ M) @ M.T  # c x d
    q = N.min(axis=0)                # d
    shifted = N - np.ones((N.shape[0], 1), dtype=np.float64) @ q.reshape(1, -1)
    ranges = N.max(axis=0) - N.min(axis=0)
    alpha = safety / max(float(np.max(ranges)), 1e-12)
    return alpha * shifted


def make_true_M_and_H() -> tuple[Array, Array]:
    M_true = np.array(
        [
            [0.35, 0.05],
            [0.05, 0.35],
            [0.20, 0.08],
            [0.08, 0.20],
            [0.17, 0.17],
            [0.15, 0.15],
        ],
        dtype=np.float64,
    )

    H = np.array(
        [
            [ 1.13087686,  1.08104571],
            [-1.31103908, -2.75816408],
            [-1.59079107,  1.54276357],
            [ 0.21854245, -0.02970892],
            [-0.53226267, -1.28978768],
            [ 2.08467351,  1.45385140],
        ],
        dtype=np.float64,
    )
    H = H - H.mean(axis=0, keepdims=True)
    return M_true, H


def tilted_estimate(M_true: Array, H: Array, s: float) -> Array:
    X = M_true * np.exp(s * H)
    return X / X.sum(axis=0, keepdims=True)


def standard_form_fit(A: Array) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    v = np.zeros(c, dtype=np.float64)
    for j in range(c):
        v[j] = np.mean([A[i, j] for i in range(c) if i != j])
    lam = float(np.mean(np.diag(A) - v))
    Ahat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    res = float(np.linalg.norm(A - Ahat, ord="fro"))
    rel = res / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    return {"lambda": lam, "residual": res, "relative_residual": rel}


def forward_bayes_tstar(M_hat: Array, M_true: Array, pi: float, grid_size: int = 1001) -> float:
    eta = np.array([1.0 - pi, pi], dtype=np.float64)
    p_true = M_true @ eta
    ts = np.linspace(0.0, 1.0, grid_size)
    P = (1.0 - ts[:, None]) * M_hat[:, 0][None, :] + ts[:, None] * M_hat[:, 1][None, :]
    P = np.clip(P, 1e-12, 1.0)
    ce = -(p_true[None, :] * np.log(P)).sum(axis=1)
    return float(ts[int(np.argmin(ce))])


@dataclass
class Split:
    X: Array
    y: Array
    eta: Array


def sigmoid(t: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-t))


def make_synthetic_splits(n: int, seed: int, label_seed: int, input_dim: int = 8) -> dict[str, Split]:
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(label_seed)
    X = rng.normal(size=(n, input_dim)).astype(np.float32)
    w = rng.normal(size=input_dim).astype(np.float64)
    w /= max(np.linalg.norm(w), 1e-12)
    raw = X @ w
    raw += 0.35 * np.sin(X[:, 0]) - 0.25 * np.cos(X[:, 1]) + 0.15 * X[:, 2] ** 2
    raw *= 1.8
    p1 = sigmoid(raw)
    eta = np.stack([1.0 - p1, p1], axis=1).astype(np.float32)
    y = np.asarray([label_rng.choice(2, p=e) for e in eta], dtype=np.int64)

    idx = rng.permutation(n)
    ntr = int(round(0.6 * n))
    nv = int(round(0.2 * n))
    ids = {"train": idx[:ntr], "val": idx[ntr:ntr+nv], "test": idx[ntr+nv:]}
    return {k: Split(X=X[ii], y=y[ii], eta=eta[ii]) for k, ii in ids.items()}


def sample_weak_labels(y: Array, M_true: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(M_true.shape[0], p=M_true[:, int(yi)]) for yi in y], dtype=np.int64)


class MLP(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, c: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def eval_model(model: nn.Module, split: Split) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(split.X, dtype=torch.float32))
        scores = logits.detach().cpu().numpy()
    pred = np.argmax(scores, axis=1)
    acc = float(np.mean(pred == split.y))
    order = float(np.mean((scores[:, 1] > scores[:, 0]) == (split.eta[:, 1] > split.eta[:, 0])))
    return acc, order


def train_clwl(train: Split, ztr: Array, val: Split, zv: Array, T: Array, seed: int, epochs: int) -> nn.Module:
    torch.manual_seed(seed)
    model = MLP(input_dim=train.X.shape[1], hidden_dim=64, c=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    Tt = torch.tensor(T.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(torch.tensor(train.X), torch.tensor(ztr)), batch_size=256, shuffle=True)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    for _ in range(epochs):
        model.train()
        for xb, zb in loader:
            opt.zero_grad()
            logits = model(xb)
            Tcols = Tt[:, zb].T
            loss = (Tcols * F.softplus(-logits) + (1.0 - Tcols) * F.softplus(logits)).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(val.X))
            Tcols = Tt[:, torch.tensor(zv)].T
            val_loss = (Tcols * F.softplus(-logits) + (1.0 - Tcols) * F.softplus(logits)).sum(dim=1).mean().item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_forward(train: Split, ztr: Array, val: Split, zv: Array, M_hat: Array, seed: int, epochs: int) -> nn.Module:
    torch.manual_seed(seed)
    model = MLP(input_dim=train.X.shape[1], hidden_dim=64, c=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    Mt = torch.tensor(M_hat.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(torch.tensor(train.X), torch.tensor(ztr)), batch_size=256, shuffle=True)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    for _ in range(epochs):
        model.train()
        for xb, zb in loader:
            opt.zero_grad()
            logits = model(xb)
            pc = torch.softmax(logits, dim=1)
            pw = pc @ Mt.T
            pw = torch.clamp(pw, min=1e-8)
            pw = pw / pw.sum(dim=1, keepdim=True)
            loss = F.nll_loss(torch.log(pw), zb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(val.X))
            pc = torch.softmax(logits, dim=1)
            pw = pc @ Mt.T
            pw = torch.clamp(pw, min=1e-8)
            pw = pw / pw.sum(dim=1, keepdim=True)
            val_loss = F.nll_loss(torch.log(pw), torch.tensor(zv)).item()
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="artifacts_row_tilt_optimized_response_bias")
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    M_true, H = make_true_M_and_H()
    s_grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])

    diag_rows = []
    for s in s_grid:
        M_hat = tilted_estimate(M_true, H, s)
        T = construct_clwl_T(M_hat)
        fit = standard_form_fit(T @ M_true)
        diag_rows.append({
            "s": s,
            "lambda": fit["lambda"],
            "relative_residual": fit["relative_residual"],
            "tstar_pi_02": forward_bayes_tstar(M_hat, M_true, 0.2),
            "tstar_pi_08": forward_bayes_tstar(M_hat, M_true, 0.8),
        })
        np.save(out / f"M_hat_s_{s:.2f}.npy", M_hat)
        np.save(out / f"T_hat_s_{s:.2f}.npy", T)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out / "diagnostics.csv", index=False)
    np.save(out / "M_true.npy", M_true)
    np.save(out / "H_optimized.npy", H)

    rows = []
    for seed in args.seeds:
        splits = make_synthetic_splits(args.n, seed=100 + seed, label_seed=200 + seed)
        ztr = sample_weak_labels(splits["train"].y, M_true, seed=300 + seed)
        zv = sample_weak_labels(splits["val"].y, M_true, seed=400 + seed)
        zte = sample_weak_labels(splits["test"].y, M_true, seed=500 + seed)

        oracle = train_forward(splits["train"], ztr, splits["val"], zv, M_true, seed, args.epochs)
        oracle_acc, oracle_order = eval_model(oracle, splits["test"])

        for s in s_grid:
            M_hat = tilted_estimate(M_true, H, s)
            T = construct_clwl_T(M_hat)

            clwl = train_clwl(splits["train"], ztr, splits["val"], zv, T, seed, args.epochs)
            acc, order = eval_model(clwl, splits["test"])
            rows.append({"method": "CLWL", "seed": seed, "s": s, "acc": acc, "order": order})

            fwd = train_forward(splits["train"], ztr, splits["val"], zv, M_hat, seed, args.epochs)
            acc, order = eval_model(fwd, splits["test"])
            rows.append({"method": "Forward_Mhat", "seed": seed, "s": s, "acc": acc, "order": order})

            rows.append({"method": "Oracle_Forward", "seed": seed, "s": s, "acc": oracle_acc, "order": oracle_order})

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_results.csv", index=False)
    summary = raw.groupby(["method", "s"]).agg(
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        order_mean=("order", "mean"),
        order_std=("order", "std"),
    ).reset_index()
    summary.to_csv(out / "summary_results.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for method in ["CLWL", "Forward_Mhat", "Oracle_Forward"]:
        d = summary[summary.method == method].sort_values("s")
        x = d.s.to_numpy()
        y = d.acc_mean.to_numpy()
        ys = d.acc_std.fillna(0).to_numpy()
        axes[0].plot(x, y, marker="o", label=method)
        axes[0].fill_between(x, y - ys, y + ys, alpha=0.15)
        y = d.order_mean.to_numpy()
        ys = d.order_std.fillna(0).to_numpy()
        axes[1].plot(x, y, marker="o", label=method)
        axes[1].fill_between(x, y - ys, y + ys, alpha=0.15)

    cl = summary[summary.method == "CLWL"].sort_values("s")
    fw = summary[summary.method == "Forward_Mhat"].sort_values("s")
    axes[2].plot(cl.s.to_numpy(), cl.acc_mean.to_numpy() - fw.acc_mean.to_numpy(), marker="o", label="CLWL-Forward_Mhat")
    axes[2].axhline(0, linestyle="--", alpha=0.6)

    axes[0].set_title("Clean accuracy")
    axes[1].set_title("Pairwise order rate")
    axes[2].set_title("Accuracy gap")
    for ax in axes:
        ax.set_xlabel("estimate bias strength s")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.savefig(out / "row_tilt_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    x = diag_df.s.to_numpy()
    axes[0].plot(x, diag_df["lambda"], marker="o")
    axes[0].axhline(0, linestyle="--", alpha=0.6)
    axes[0].set_title("lambda of T(Mhat_s)Mtrue")
    axes[1].plot(x, diag_df.relative_residual, marker="o")
    axes[1].set_title("standard-form rel. residual")
    axes[2].plot(x, diag_df.tstar_pi_02, marker="o")
    axes[2].axhline(0.5, linestyle="--", alpha=0.6)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Forward t*(pi=0.2)")
    axes[3].plot(x, diag_df.tstar_pi_08, marker="o")
    axes[3].axhline(0.5, linestyle="--", alpha=0.6)
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Forward t*(pi=0.8)")
    for ax in axes:
        ax.set_xlabel("estimate bias strength s")
        ax.grid(True, alpha=0.3)
    fig.savefig(out / "row_tilt_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved to", out.resolve())
    print("Diagnostics:")
    print(diag_df)
    print("Summary:")
    print(summary)


if __name__ == "__main__":
    main()
