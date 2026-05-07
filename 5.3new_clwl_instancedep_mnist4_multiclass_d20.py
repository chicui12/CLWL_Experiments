"""MNIST4-style instance-dependent weak-label experiment for CLWL vs Forward.

This script constructs a 4-class, d=20 weak-label family with two instance-dependent
transition matrices M_hard and M_easy whose class-conditional average is Mbar.
Both CLWL and Forward only receive Mbar. CLWL uses T(Mbar), while Forward uses Mbar.

Default run uses sklearn's built-in 8x8 handwritten digits proxy because it is
available offline. Use --data_source openml to run true MNIST via sklearn OpenML.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

Array = np.ndarray

# --------------------------- matrix utilities ---------------------------

def construct_clwl_T(M: Array, safety: float = 0.95) -> tuple[Array, Array, float]:
    M = np.asarray(M, dtype=np.float64)
    d, c = M.shape
    N = np.linalg.solve(M.T @ M, M.T)  # c x d
    q = np.min(N, axis=0)              # d
    R = N - np.ones((c, 1)) @ q.reshape(1, -1)
    max_R = float(np.max(R))
    alpha = safety / max(max_R, 1e-12)
    T = alpha * R
    return T, N, alpha


def standard_form_fit(A: Array) -> dict:
    A = np.asarray(A, dtype=np.float64)
    c = A.shape[0]
    v = np.zeros(c)
    for j in range(c):
        v[j] = np.mean([A[i, j] for i in range(c) if i != j])
    lam = float(np.mean(np.diag(A) - v))
    Ahat = lam * np.eye(c) + np.ones((c, 1)) @ v.reshape(1, -1)
    residual = float(np.linalg.norm(A - Ahat) / max(np.linalg.norm(A), 1e-12))
    margins = []
    for j in range(c):
        margins.append(float(A[j, j] - max(A[i, j] for i in range(c) if i != j)))
    return {"lambda": lam, "relative_residual": residual, "min_margin": float(min(margins))}


def qstar_forward(Mbar: Array, p: Array) -> Array:
    c = Mbar.shape[1]
    def obj(q):
        prob = np.clip(Mbar @ q, 1e-12, 1.0)
        return float(-np.sum(p * np.log(prob)))
    cons = ({"type": "eq", "fun": lambda q: np.sum(q) - 1.0},)
    bounds = [(0.0, 1.0)] * c
    starts = [np.ones(c) / c] + [np.eye(c)[i] for i in range(c)]
    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"ftol": 1e-11, "maxiter": 300, "disp": False})
        val = obj(res.x)
        if best is None or val < best[0]:
            best = (val, res.x)
    q = np.clip(best[1], 0.0, 1.0)
    q = q / max(q.sum(), 1e-12)
    return q


def make_mbar(c: int = 4, modes: int = 5, eps: float = 0.4) -> Array:
    """MNIST4 weak-label template with paired confusions 1<->7 and 3<->8.

    Classes are indexed [1,7,3,8] -> [0,1,2,3]. Each class has five weak
    response modes: H,L,A,F,S.
    """
    d = c * modes
    pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    template = np.zeros((d, c), dtype=np.float64)
    own = np.array([0.33, 0.20, 0.13, 0.08, 0.06], dtype=np.float64)
    own = own / own.sum() * 0.78
    conf = np.array([0.08, 0.06, 0.035, 0.025, 0.02], dtype=np.float64)
    conf = conf / conf.sum() * 0.17
    rem = 1.0 - own.sum() - conf.sum()
    for y in range(c):
        r = pairs[y]
        template[y*modes:(y+1)*modes, y] += own
        template[r*modes:(r+1)*modes, y] += conf
        others = [k for k in range(c) if k not in (y, r)]
        idx = []
        for k in others:
            for m in range(modes):
                idx.append(k*modes + m)
        template[idx, y] += rem / len(idx)
    M = (1.0 - eps) * template + eps / d
    M /= M.sum(axis=0, keepdims=True)
    return M


def build_instance_dependent_family(pi: float = 0.2, mu: float = 0.35, eps: float = 0.4) -> tuple[Array, Array, Array, Array]:
    """Construct Mbar, Mhard, Measy using an LP in the nullspace of T(Mbar).

    Mhard is selected to make Forward with Mbar confuse each class with its paired
    confuser, while T(Mbar)Mhard and T(Mbar)Measy remain standard-form.
    """
    c, modes = 4, 5
    pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    Mbar = make_mbar(c=c, modes=modes, eps=eps)
    T, N, _ = construct_clwl_T(Mbar)
    d = Mbar.shape[0]
    B = mu * np.eye(c) + (1.0 - mu) / c * np.ones((c, c))
    Mhard = np.zeros_like(Mbar)
    for y in range(c):
        r = pairs[y]
        obj = -(np.log(np.clip(Mbar[:, r], 1e-12, 1.0)) - np.log(np.clip(Mbar[:, y], 1e-12, 1.0)))
        Aeq = np.vstack([N, np.ones((1, d))])
        beq = np.concatenate([B[:, y], [1.0]])
        bounds = [(0.0, float(Mbar[i, y] / pi)) for i in range(d)]
        res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed for class {y}: {res.message}")
        Mhard[:, y] = res.x
    Measy = (Mbar - pi * Mhard) / (1.0 - pi)
    if np.min(Measy) < -1e-8 or np.min(Mhard) < -1e-8:
        raise RuntimeError("Negative entries in constructed matrices.")
    Measy = np.maximum(Measy, 0.0); Measy /= Measy.sum(axis=0, keepdims=True)
    Mhard = np.maximum(Mhard, 0.0); Mhard /= Mhard.sum(axis=0, keepdims=True)
    return Mbar, Mhard, Measy, T


def interpolate_matrix(Mbar: Array, Mend: Array, kappa: float) -> Array:
    M = (1.0 - kappa) * Mbar + kappa * Mend
    M = np.maximum(M, 0.0)
    M /= M.sum(axis=0, keepdims=True)
    return M

# --------------------------- data utilities ---------------------------

def load_data(data_source: str, classes: list[int], max_samples: int | None, seed: int):
    if data_source == "sklearn_digits":
        data = load_digits()
        X = data.data.astype(np.float64) / 16.0
        y_orig = data.target.astype(int)
    elif data_source == "openml":
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float64) / 255.0
        y_orig = mnist.target.astype(int)
    else:
        raise ValueError(data_source)
    cls = list(classes)
    mask = np.isin(y_orig, cls)
    X = X[mask]
    y_orig = y_orig[mask]
    mapper = {c: i for i, c in enumerate(cls)}
    y = np.asarray([mapper[int(v)] for v in y_orig], dtype=np.int64)
    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def hard_flags_by_prototype(X: Array, y: Array, hard_fraction: float, pairs: Dict[int, int]) -> Array:
    c = len(np.unique(y))
    protos = np.stack([X[y == k].mean(axis=0) for k in range(c)], axis=0)
    hard = np.zeros(len(y), dtype=bool)
    for k in range(c):
        idx = np.where(y == k)[0]
        own = np.linalg.norm(X[idx] - protos[k], axis=1)
        conf = np.linalg.norm(X[idx] - protos[pairs[k]], axis=1)
        score = own - conf  # large means farther from own / closer to confuser
        m = max(1, int(round(hard_fraction * len(idx))))
        hard[idx[np.argsort(score)[-m:]]] = True
    return hard


def sample_weak_labels(y: Array, hard: Array, Mhard: Array, Measy: Array, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    z = np.empty(len(y), dtype=np.int64)
    for i, yi in enumerate(y):
        M = Mhard if hard[i] else Measy
        z[i] = rng.choice(M.shape[0], p=M[:, yi])
    return z

# --------------------------- models and training ---------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, c: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, c))
    def forward(self, x):
        return self.net(x)


def clwl_loss(logits, z, T):
    tz = T.T[z]
    return (tz * F.softplus(-logits) + (1.0 - tz) * F.softplus(logits)).sum(dim=1).mean()


def forward_loss(logits, z, M):
    q = torch.softmax(logits, dim=1)
    pw = q @ M.T
    return F.nll_loss(torch.log(torch.clamp(pw, min=1e-8)), z)


def oracle_forward_loss(logits, z, hard, Mhard, Measy):
    q = torch.softmax(logits, dim=1)
    Mh = Mhard.T.unsqueeze(0)  # 1 x c x d
    Me = Measy.T.unsqueeze(0)
    Mall = torch.where(hard.view(-1, 1, 1), Mh, Me)  # n x c x d
    pw = torch.bmm(q.unsqueeze(1), Mall).squeeze(1)
    return F.nll_loss(torch.log(torch.clamp(pw, min=1e-8)), z)


def train_method(method: str, Xtr, ytr, ztr, hard_tr, Xva, yva, zva, hard_va, Mbar, Mhard, Measy, T, seed, epochs, hidden_dim, lr, weight_decay):
    """Full-batch trainer for this small MNIST4 experiment."""
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], hidden_dim, 4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ztr_t = torch.tensor(ztr, dtype=torch.long)
    hard_tr_t = torch.tensor(hard_tr, dtype=torch.bool)
    Tt = torch.tensor(T, dtype=torch.float32)
    Mbar_t = torch.tensor(Mbar, dtype=torch.float32)
    Mh_t = torch.tensor(Mhard, dtype=torch.float32)
    Me_t = torch.tensor(Measy, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xtr_t)
        if method == "CLWL-average":
            loss = clwl_loss(logits, ztr_t, Tt)
        elif method == "Forward-average":
            loss = forward_loss(logits, ztr_t, Mbar_t)
        elif method == "Oracle-forward-instance":
            loss = oracle_forward_loss(logits, ztr_t, hard_tr_t, Mh_t, Me_t)
        else:
            raise ValueError(method)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xva, dtype=torch.float32))
        pred = logits.argmax(dim=1).numpy()
    return float(np.mean(pred == yva))

# --------------------------- experiment ---------------------------

def diagnostics(Mbar, Mhard, Measy, T, kappas):
    pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    rows = []
    for kappa in kappas:
        Mh = interpolate_matrix(Mbar, Mhard, kappa)
        Me = interpolate_matrix(Mbar, Measy, kappa)
        fit_h = standard_form_fit(T @ Mh)
        fit_e = standard_form_fit(T @ Me)
        fwd_margins = []
        q_true = []
        q_conf = []
        for y in range(4):
            q = qstar_forward(Mbar, Mh[:, y])
            r = pairs[y]
            fwd_margins.append(q[y] - q[r])
            q_true.append(q[y]); q_conf.append(q[r])
        rows.append({
            "kappa": kappa,
            "hard_clwl_margin": fit_h["min_margin"],
            "easy_clwl_margin": fit_e["min_margin"],
            "hard_residual": fit_h["relative_residual"],
            "easy_residual": fit_e["relative_residual"],
            "forward_hard_pair_margin_mean": float(np.mean(fwd_margins)),
            "forward_hard_true_prob_mean": float(np.mean(q_true)),
            "forward_hard_confuser_prob_mean": float(np.mean(q_conf)),
        })
    return pd.DataFrame(rows)


def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    classes = args.classes
    X, y = load_data(args.data_source, classes, args.max_samples, args.seed)
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.30, random_state=args.seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.20, random_state=args.seed + 1, stratify=ytrva)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)
    pairs = {0: 1, 1: 0, 2: 3, 3: 2}
    hard_tr = hard_flags_by_prototype(Xtr, ytr, args.hard_fraction, pairs)
    hard_va = hard_flags_by_prototype(Xva, yva, args.hard_fraction, pairs)
    hard_te = hard_flags_by_prototype(Xte, yte, args.hard_fraction, pairs)

    Mbar, Mhard_end, Measy_end, T = build_instance_dependent_family(pi=args.hard_fraction, mu=args.mu, eps=args.eps)
    np.save(out / "Mbar.npy", Mbar); np.save(out / "Mhard_end.npy", Mhard_end); np.save(out / "Measy_end.npy", Measy_end); np.save(out / "T.npy", T)
    kappas = args.kappa_grid
    diag = diagnostics(Mbar, Mhard_end, Measy_end, T, kappas)
    diag.to_csv(out / "diagnostics.csv", index=False)
    print("=== diagnostics ===")
    print(diag)

    rows = []
    for kappa in kappas:
        Mh = interpolate_matrix(Mbar, Mhard_end, kappa)
        Me = interpolate_matrix(Mbar, Measy_end, kappa)
        for seed in args.seeds:
            ztr = sample_weak_labels(ytr, hard_tr, Mh, Me, seed=1000 + seed)
            zva = sample_weak_labels(yva, hard_va, Mh, Me, seed=2000 + seed)
            zte = sample_weak_labels(yte, hard_te, Mh, Me, seed=3000 + seed)
            for method in ["CLWL-average", "Forward-average", "Oracle-forward-instance"]:
                acc = train_method(method, Xtr, ytr, ztr, hard_tr, Xte, yte, zte, hard_te,
                                   Mbar, Mh, Me, T, seed=seed, epochs=args.epochs,
                                   hidden_dim=args.hidden_dim, lr=args.lr, weight_decay=args.weight_decay)
                rows.append({"kappa": kappa, "seed": seed, "method": method, "test_accuracy": acc})
                print(kappa, seed, method, acc, flush=True)
    raw = pd.DataFrame(rows); raw.to_csv(out / "raw_results.csv", index=False)
    summary = raw.groupby(["method", "kappa"], as_index=False).agg(test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"))
    summary.to_csv(out / "summary_results.csv", index=False)
    print("=== summary ===")
    print(summary)
    plot(summary, diag, out / "results.png")


def plot(summary, diag, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for method in ["CLWL-average", "Forward-average", "Oracle-forward-instance"]:
        df = summary[summary.method == method].sort_values("kappa")
        x = df.kappa.to_numpy(float); y = df.test_accuracy_mean.to_numpy(float); e = df.test_accuracy_std.fillna(0).to_numpy(float)
        axes[0].plot(x, y, marker="o", label=method)
        axes[0].fill_between(x, y-e, y+e, alpha=0.15)
    axes[0].set_title("Test accuracy")
    axes[0].set_xlabel("instance-dependence strength $\\kappa$")
    axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=.3); axes[0].legend()
    axes[1].plot(diag.kappa, diag.hard_clwl_margin, marker="o", label="hard CLWL margin")
    axes[1].plot(diag.kappa, diag.easy_clwl_margin, marker="o", label="easy CLWL margin")
    axes[1].axhline(0, linestyle="--", alpha=.6)
    axes[1].set_title("CLWL ranking margins")
    axes[1].set_xlabel("$\\kappa$"); axes[1].grid(True, alpha=.3); axes[1].legend()
    axes[2].plot(diag.kappa, diag.forward_hard_pair_margin_mean, marker="o", label="Forward hard pair margin")
    axes[2].axhline(0, linestyle="--", alpha=.6)
    axes[2].set_title("Forward posterior diagnostic")
    axes[2].set_xlabel("$\\kappa$"); axes[2].grid(True, alpha=.3); axes[2].legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_source", choices=["sklearn_digits", "openml"], default="sklearn_digits")
    p.add_argument("--classes", nargs="+", type=int, default=[1, 7, 3, 8])
    p.add_argument("--out_dir", default="artifacts_instancedep_mnist4_d20_multiclass")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hard_fraction", type=float, default=0.2)
    p.add_argument("--mu", type=float, default=0.35)
    p.add_argument("--eps", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--kappa_grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
