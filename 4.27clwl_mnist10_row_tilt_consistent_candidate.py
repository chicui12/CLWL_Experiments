
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# MNIST10 row-tilt response-bias experiment
# ============================================================
# Protocol:
#   - Fix M_true.
#   - Generate weak labels from M_true.
#   - Vary the estimated matrix M_hat(s) via a multiplicative row-tilt:
#       M_hat_s[z,y] ∝ M_true[z,y] exp(s H[z,y]).
#   - CLWL uses T(M_hat_s).
#   - Forward uses M_hat_s.
#   - Oracle Forward uses fixed M_true.
#
# Weak response modes for each digit k:
#   k_H: high-confidence response for digit k
#   k_L: low-confidence response for digit k
#   k_A: ambiguous response for digit k
#   k_F: fallback/style response for digit k
#
# Default MNIST10 confusion pairs:
#   0<->6, 1<->7, 2<->5, 3<->8, 4<->9.
#
# In this candidate, d = 4c = 40 for c=10.
# The provided M_true and H were selected to keep T(M_hat_s) M_true
# reasonably close to ranking-consistent form while making Forward(M_hat_s)
# degrade under large estimate bias.


Array = np.ndarray


def confusion_partner_map() -> dict[int, int]:
    return {0: 6, 6: 0, 1: 7, 7: 1, 2: 5, 5: 2, 3: 8, 8: 3, 4: 9, 9: 4}


def make_mnist10_mtrue_and_h() -> tuple[Array, Array, dict[int, int]]:
    c = 10
    modes = 4
    d = c * modes
    pair = confusion_partner_map()

    # Search-selected interpretable parameters.
    # M_true:
    # correct digit modes: H,L,A,F
    # paired-confuser modes: H,L,A,F
    # remaining mass: mostly other ambiguous/fallback modes
    pH = 0.4360881678804948
    pL = 0.16721373447109728
    pA = 0.0802561665161117
    pF = 0.03277422008276444
    qH = 0.10011206874989617
    qL = 0.05982115090602809
    qA = 0.02383979986362264
    qF = 0.03277906591933194
    restA = 0.7150920766785394
    restF = 0.19605272145752176

    M = np.zeros((d, c), dtype=np.float64)
    for y in range(c):
        r = pair[y]
        for m, val in enumerate([pH, pL, pA, pF]):
            M[4 * y + m, y] += val
        for m, val in enumerate([qH, qL, qA, qF]):
            M[4 * r + m, y] += val

        rem = 1.0 - M[:, y].sum()
        others = [j for j in range(c) if j not in (y, r)]
        for j in others:
            M[4 * j + 2, y] += rem * restA / len(others)
            M[4 * j + 3, y] += rem * restF / len(others)
            M[4 * j + 1, y] += rem * (1.0 - restA - restF) / len(others)

    # H:
    # the pilot estimate slightly undercounts true-digit modes and overcounts
    # paired confusing-digit modes. The strongest overcount is on ambiguous
    # response for the paired confuser, which is realistic for biased pilot
    # annotations on visually ambiguous MNIST digits.
    aH = 0.1337070204983062
    aL = 0.12426401919151742
    aA = 0.1888751838795088
    aF = -0.03823428956190106
    bH = 1.0418833494207815
    bL = 1.4083050898855802
    bA = 1.9944416969841758
    bF = 0.6374323757127076
    otherA = 0.005629515986911332
    otherF = 0.14785696474608592

    H = np.zeros((d, c), dtype=np.float64)
    for y in range(c):
        r = pair[y]
        for m, val in enumerate([aH, aL, aA, aF]):
            H[4 * y + m, y] -= val
        for m, val in enumerate([bH, bL, bA, bF]):
            H[4 * r + m, y] += val
        for j in range(c):
            if j not in (y, r):
                H[4 * j + 2, y] += otherA
                H[4 * j + 3, y] += otherF

    H = H - H.mean(axis=0, keepdims=True)
    validate_transition(M)
    return M, H, pair


def validate_transition(M: Array, atol: float = 1e-8) -> None:
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got {M.shape}")
    if np.min(M) < -atol:
        raise ValueError(f"M has negative entries: min={np.min(M)}")
    if not np.allclose(M.sum(axis=0), np.ones(M.shape[1]), atol=atol, rtol=0.0):
        raise ValueError(f"M is not column-stochastic: {M.sum(axis=0)}")


def row_tilt(M_true: Array, H: Array, s: float) -> Array:
    L = np.log(np.clip(M_true, 1e-12, 1.0)) + s * H
    L = L - L.max(axis=0, keepdims=True)
    E = np.exp(L)
    M_hat = E / E.sum(axis=0, keepdims=True)
    validate_transition(M_hat)
    return M_hat


def construct_clwl_T(M: Array, safety_factor: float = 0.99) -> Array:
    """Module1-compatible theorem-style construction."""
    M = np.asarray(M, dtype=np.float64)
    d, c = M.shape
    N = np.linalg.solve(M.T @ M + 1e-9 * np.eye(c), M.T)
    q = N.min(axis=0)
    col_ranges = N.max(axis=0) - N.min(axis=0)
    alpha = safety_factor / max(float(col_ranges.max()), 1e-12)
    T = alpha * (N - np.ones((c, 1), dtype=np.float64) @ q.reshape(1, -1))
    return T


def standard_form_fit(A: Array) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    v = (A.sum(axis=0) - np.diag(A)) / max(c - 1, 1)
    lam = float(np.mean(np.diag(A) - v))
    A_hat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - A_hat, ord="fro"))
    relative = residual / max(float(np.linalg.norm(A, ord="fro")), 1e-12)
    margin = float(min(A[j, j] - np.delete(A[:, j], j).max() for j in range(c)))
    return {
        "lambda_hat": lam,
        "relative_residual": relative,
        "ranking_margin": margin,
    }


def forward_pure_class_diagnostic(M_true: Array, M_hat: Array, pair: dict[int, int]) -> dict[str, object]:
    c = M_true.shape[1]
    CE = -(M_true.T @ np.log(np.clip(M_hat, 1e-12, 1.0)))
    pred = CE.argmin(axis=1)
    scores = -CE
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    true_prob = float(np.mean(probs[np.arange(c), np.arange(c)]))
    pair_prob = float(np.mean([probs[y, pair[y]] for y in range(c)]))
    margin = float(np.mean([np.min(np.delete(CE[y], y)) - CE[y, y] for y in range(c)]))
    return {
        "forward_pure_acc": float(np.mean(pred == np.arange(c))),
        "forward_true_prob": true_prob,
        "forward_pair_prob": pair_prob,
        "forward_margin": margin,
        "forward_pure_pred": pred.tolist(),
    }


def compute_diagnostics(M_true: Array, H: Array, s_grid: list[float], pair: dict[int, int]) -> pd.DataFrame:
    rows = []
    for s in s_grid:
        M_hat = row_tilt(M_true, H, s)
        T_hat = construct_clwl_T(M_hat)
        fit = standard_form_fit(T_hat @ M_true)
        fdiag = forward_pure_class_diagnostic(M_true, M_hat, pair)
        rows.append({"s": s, **fit, **fdiag})
    return pd.DataFrame(rows)


def load_dataset(data_source, max_train, max_test, seed):
    """Load MNIST10."""
    if data_source == "openml":
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
            parser="auto",
        )
        y = y.astype(np.int64)
        X = X.astype(np.float32) / 255.0
        X_train, y_train = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]

    elif data_source == "torchvision_mnist":
        from torchvision.datasets import MNIST

        tr = MNIST(root="./data", train=True, download=True)
        te = MNIST(root="./data", train=False, download=True)

        X_train = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_train = tr.targets.numpy().astype(np.int64)

        X_test = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = te.targets.numpy().astype(np.int64)

    elif data_source == "sklearn_digits":
        from sklearn.datasets import load_digits
        data = load_digits()
        X = data.data.astype(np.float32) / 16.0
        y = data.target.astype(np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )

    else:
        raise ValueError(
            "data_source must be 'openml', 'torchvision_mnist', or 'sklearn_digits'."
        )

    rng = np.random.default_rng(seed)

    if max_train is not None and len(X_train) > max_train:
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if max_test is not None and len(X_test) > max_test:
        idx = rng.choice(len(X_test), size=max_test, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test


def sample_weak_labels(y: Array, M_true: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.empty_like(y, dtype=np.int64)
    for i, yi in enumerate(y):
        z[i] = int(rng.choice(M_true.shape[0], p=M_true[:, int(yi)]))
    return z


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def clwl_loss(logits, z, T_torch):
    targets = T_torch[:, z].T.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, targets)


def forward_loss(logits, z, M_torch):
    probs_clean = torch.softmax(logits, dim=1)
    probs_weak = probs_clean @ M_torch.T
    probs_weak = torch.clamp(probs_weak, min=1e-8)
    return F.nll_loss(torch.log(probs_weak), z)


def train_one(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    z_train: Array,
    method: str,
    M_hat: Array,
    T_hat: Array,
    seed: int,
    hidden_dim: int,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,


    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[float, int]:
    torch.manual_seed(seed)
    model = MLP(X_train.shape[1], hidden_dim, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    z_t = torch.tensor(z_train, dtype=torch.long, device=device)
    M_t = torch.tensor(M_hat, dtype=torch.float32, device=device)
    T_t = torch.tensor(T_hat, dtype=torch.float32, device=device)

    
    n = X_train.shape[0]

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0


    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for st in range(0, n, batch_size):
            idx = perm[st : st + batch_size]
            logits = model(X_t[idx])
            if method == "clwl":
                loss = clwl_loss(logits, z_t[idx], T_t)
            elif method == "forward":
                loss = forward_loss(logits, z_t[idx], M_t)
            else:
                raise ValueError(method)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            batch_n = idx.shape[0]
            epoch_loss += float(loss.item()) * batch_n
            epoch_count += batch_n

        mean_epoch_loss = epoch_loss / max(epoch_count, 1)

        if mean_epoch_loss < best_loss - early_stop_min_delta:
            best_loss = mean_epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)



    model.eval()
    with torch.no_grad():
        X_eval = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred = model(X_eval).argmax(dim=1).cpu().numpy()
    return float(np.mean(pred == y_test)), best_epoch


def run_experiment(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    M_true, H, pair = make_mnist10_mtrue_and_h()
    np.save(out_dir / "M_true.npy", M_true)
    np.save(out_dir / "H.npy", H)

    s_grid = [float(x) for x in args.s_grid]
    diag_df = compute_diagnostics(M_true, H, s_grid, pair)
    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)

    raw_rows = []
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)

    for split_seed in args.train_seeds:
        X_train, y_train, X_test, y_test = load_dataset(
            args.data_source, args.max_train_samples, args.max_test_samples, split_seed
        )
        z_train = sample_weak_labels(y_train, M_true, seed=10_000 + split_seed)

        # oracle Forward with fixed M_true
        oracle_acc, oracle_epochs = train_one(
            X_train, y_train, X_test, y_test, z_train,
            method="forward",
            M_hat=M_true,
            T_hat=construct_clwl_T(M_true),
            seed=30_000 + split_seed,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,

            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,


            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        for s in s_grid:
            M_hat = row_tilt(M_true, H, s)
            T_hat = construct_clwl_T(M_hat)

            clwl_acc, clwl_epochs = train_one(
                X_train, y_train, X_test, y_test, z_train,
                method="clwl",
                M_hat=M_hat,
                T_hat=T_hat,
                seed=10_000 + split_seed,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,

                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,

        
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            raw_rows.append({
                "method": "CLWL_T_Mhat",
                "seed": split_seed,
                "s": s,
                "clean_accuracy": clwl_acc,
                "epochs": clwl_epochs,
            })

            fwd_acc, fwd_epochs = train_one(
                X_train, y_train, X_test, y_test, z_train,
                method="forward",
                M_hat=M_hat,
                T_hat=T_hat,
                seed=20_000 + split_seed,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,

                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            raw_rows.append({
                "method": "Forward_Mhat",
                "seed": split_seed,
                "s": s,
                "clean_accuracy": fwd_acc,
                "epochs": fwd_epochs,
            })

            raw_rows.append({
                "method": "Forward_oracle_Mtrue",
                "seed": split_seed,
                "s": s,
                "clean_accuracy": oracle_acc,
                "epochs": oracle_epochs,
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(out_dir / "raw_results.csv", index=False)

    summary_df = (
        raw_df.groupby(["method", "s"], as_index=False)
        .agg(
            clean_accuracy_mean=("clean_accuracy", "mean"),
            clean_accuracy_std=("clean_accuracy", "std"),
            epochs_mean=("epochs", "mean"),
        )
    )
    summary_df.to_csv(out_dir / "summary_results.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for method in ["CLWL_T_Mhat", "Forward_Mhat", "Forward_oracle_Mtrue"]:
            df = summary_df[summary_df["method"] == method].sort_values("s")
            x = df["s"].to_numpy()
            y = df["clean_accuracy_mean"].to_numpy()
            e = df["clean_accuracy_std"].fillna(0).to_numpy()
            axes[0].plot(x, y, marker="o", label=method)
            axes[0].fill_between(x, y - e, y + e, alpha=0.15)
        axes[0].set_xlabel("estimate bias strength s")
        axes[0].set_ylabel("clean test accuracy")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        dfc = summary_df[summary_df["method"] == "CLWL_T_Mhat"].sort_values("s")
        dff = summary_df[summary_df["method"] == "Forward_Mhat"].sort_values("s")
        axes[1].plot(dfc["s"], dfc["clean_accuracy_mean"].to_numpy() - dff["clean_accuracy_mean"].to_numpy(), marker="o")
        axes[1].axhline(0.0, linestyle="--", alpha=0.7)
        axes[1].set_xlabel("estimate bias strength s")
        axes[1].set_ylabel("accuracy gap")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(diag_df["s"], diag_df["forward_true_prob"], marker="o", label="true-class prob")
        axes[2].plot(diag_df["s"], diag_df["forward_pair_prob"], marker="o", label="paired-confuser prob")
        axes[2].axhline(0.5, linestyle="--", alpha=0.7)
        axes[2].set_xlabel("estimate bias strength s")
        axes[2].set_ylim(0.3, 0.75)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        fig.savefig(out_dir / "results.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        axes[0].plot(diag_df["s"], diag_df["lambda_hat"], marker="o")
        axes[0].axhline(0.0, linestyle="--", alpha=0.7)
        axes[0].set_title("lambda")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(diag_df["s"], diag_df["relative_residual"], marker="o")
        axes[1].set_title("standard-form residual")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(diag_df["s"], diag_df["ranking_margin"], marker="o")
        axes[2].axhline(0.0, linestyle="--", alpha=0.7)
        axes[2].set_title("ranking margin")
        axes[2].grid(True, alpha=0.3)
        fig.savefig(out_dir / "diagnostics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Plotting failed:", e)

    print("Diagnostics:")
    print(diag_df)
    print("\nSummary:")
    print(summary_df)
    print("\nSaved to:", out_dir.resolve())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
    "--data_source",
    choices=["openml", "torchvision_mnist", "sklearn_digits"],
    default="torchvision_mnist",
)
    p.add_argument("--out_dir", type=str, default="artifacts_mnist10_row_tilt_consistent")
    p.add_argument("--s_grid", type=float, nargs="+", default=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--train_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
