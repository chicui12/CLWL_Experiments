import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Module 1: build T from M
# ============================================================

def build_T_from_M(M: torch.Tensor, alpha_scale: float = 0.99, eps: float = 1e-12):
    """
    Build T from M using the explicit construction.

    M: shape (d, c)
    T: shape (c, d)
    """
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {tuple(M.shape)}")

    d, c = M.shape
    if d < c:
        raise ValueError(f"Need d >= c for full column rank, got M.shape={(d, c)}")

    rank = torch.linalg.matrix_rank(M)
    if int(rank.item()) < c:
        raise ValueError(f"M must have full column rank c={c}, got rank={int(rank.item())}")

    dtype = M.dtype
    device = M.device

    MtM = M.T @ M
    N = torch.linalg.inv(MtM) @ M.T                    # (c, d)

    q = N.min(dim=0).values                            # (d,)
    ones_c = torch.ones((c, 1), dtype=dtype, device=device)
    T_unnormalized = N - ones_c @ q.unsqueeze(0)      # (c, d)

    col_max = N.max(dim=0).values
    col_min = N.min(dim=0).values
    col_ranges = col_max - col_min
    max_range = col_ranges.max()

    if max_range <= eps:
        raise ValueError("alpha upper bound is ill-defined because max column range is too small")

    alpha_upper = 1.0 / max_range
    alpha = alpha_scale * alpha_upper
    T = alpha * T_unnormalized

    info = {
        "N": N,
        "q": q,
        "alpha_upper": alpha_upper,
        "alpha": alpha,
        "NM": N @ M,
        "A": T @ M,
        "rank_M": rank,
        "T_min": T.min(),
        "T_max": T.max(),
    }
    return T, info


# ============================================================
# Module 2: CLWL loss
# ============================================================

class CLWLLoss(nn.Module):
    """
    CLWL loss with logistic beta:
        beta(f) = log(1 + exp(-f)) = softplus(-f)

    Formula:
        Psi(z, s) = z^T T^T beta(s) + (1_c - Tz)^T beta(-s)

    Batch implementation:
        logits: (B, c)
        Z:      (B, d) one-hot weak labels
        T:      (c, d)
    """

    def __init__(self, T: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        if T.ndim != 2:
            raise ValueError(f"T must be 2D, got shape {tuple(T.shape)}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")
        self.register_buffer("T", T)
        self.reduction = reduction

    @staticmethod
    def beta(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(-x)

    def forward(self, logits: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B, c), got {tuple(logits.shape)}")
        if Z.ndim != 2:
            raise ValueError(f"Z must have shape (B, d), got {tuple(Z.shape)}")

        B, c = logits.shape
        c_T, d = self.T.shape
        Bz, d_Z = Z.shape

        if c != c_T:
            raise ValueError(f"logits second dimension {c} must equal T.shape[0]={c_T}")
        if B != Bz:
            raise ValueError(f"Batch size mismatch: {B} vs {Bz}")
        if d != d_Z:
            raise ValueError(f"Z second dimension {d_Z} must equal T.shape[1]={d}")

        row_sums = Z.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
            raise ValueError("Each row of Z must sum to 1")

        support = Z @ self.T.T                          # (B, c), equals Tz in batch form
        beta_pos = self.beta(logits)                    # beta(s)
        beta_neg = self.beta(-logits)                   # beta(-s)

        ones_c = torch.ones((1, c), dtype=logits.dtype, device=logits.device)
        loss_per_sample = (support * beta_pos + (ones_c - support) * beta_neg).sum(dim=1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


# ============================================================
# Module 3: synthetic data + weak labels + training loop
# ============================================================

@dataclass
class SyntheticConfig:
    c: int = 3
    d: int = 3
    input_dim: int = 2
    train_per_class: int = 200
    test_per_class: int = 100
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-2
    seed: int = 123
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, c: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def make_class_means(c: int, input_dim: int, radius: float = 3.0, dtype=torch.float64, device="cpu"):
    """
    Put class means on a circle when input_dim=2, otherwise use a simple separated construction.
    """
    if input_dim == 2:
        means = []
        for k in range(c):
            angle = 2.0 * math.pi * k / c
            means.append([radius * math.cos(angle), radius * math.sin(angle)])
        return torch.tensor(means, dtype=dtype, device=device)

    means = torch.zeros((c, input_dim), dtype=dtype, device=device)
    for k in range(c):
        means[k, k % input_dim] = 3.0
        means[k] += 0.5 * k
    return means


@torch.no_grad()
def generate_gaussian_dataset(cfg: SyntheticConfig):
    """
    Generate Gaussian classes with true labels y.

    Returns:
        X_train, y_train, X_test, y_test
    """
    device = cfg.device
    dtype = cfg.dtype
    c = cfg.c
    input_dim = cfg.input_dim

    means = make_class_means(c, input_dim, dtype=dtype, device=device)
    cov_std = 0.8

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for cls in range(c):
        train_noise = cov_std * torch.randn((cfg.train_per_class, input_dim), dtype=dtype, device=device)
        test_noise = cov_std * torch.randn((cfg.test_per_class, input_dim), dtype=dtype, device=device)

        X_train_cls = means[cls].unsqueeze(0) + train_noise
        X_test_cls = means[cls].unsqueeze(0) + test_noise

        y_train_cls = torch.full((cfg.train_per_class,), cls, dtype=torch.long, device=device)
        y_test_cls = torch.full((cfg.test_per_class,), cls, dtype=torch.long, device=device)

        X_train_list.append(X_train_cls)
        y_train_list.append(y_train_cls)
        X_test_list.append(X_test_cls)
        y_test_list.append(y_test_cls)

    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    X_test = torch.cat(X_test_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)

    # Shuffle train/test sets independently.
    train_perm = torch.randperm(X_train.shape[0], device=device)
    test_perm = torch.randperm(X_test.shape[0], device=device)

    X_train = X_train[train_perm]
    y_train = y_train[train_perm]
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    return X_train, y_train, X_test, y_test, means


@torch.no_grad()
def sample_weak_labels_from_M(y: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    Sample weak labels z from the column M[:, y].

    Args:
        y: shape (n,), true-label indices in {0, ..., c-1}
        M: shape (d, c), where column j is P(z | y=j)

    Returns:
        Z: shape (n, d), one-hot weak labels
    """
    if y.ndim != 1:
        raise ValueError("y must have shape (n,)")
    if M.ndim != 2:
        raise ValueError("M must have shape (d, c)")

    d, c = M.shape
    if y.min().item() < 0 or y.max().item() >= c:
        raise ValueError("y contains class indices outside [0, c-1]")

    n = y.shape[0]
    probs = M[:, y].T                                  # (n, d)
    z_idx = torch.multinomial(probs, num_samples=1).squeeze(1)   # (n,)
    Z = torch.nn.functional.one_hot(z_idx, num_classes=d).to(dtype=M.dtype)
    return Z


@torch.no_grad()
def empirical_order_preservation_rate(A: torch.Tensor, n_samples: int = 2000):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    c = A.shape[0]
    eta = torch.rand((n_samples, c), dtype=A.dtype, device=A.device)
    eta = eta / eta.sum(dim=1, keepdim=True)
    q = eta @ A.T

    total = 0
    good = 0
    for b in range(n_samples):
        for i in range(c):
            for j in range(c):
                if i == j:
                    continue
                if eta[b, i] > eta[b, j]:
                    total += 1
                    if q[b, i] > q[b, j]:
                        good += 1
    return good / total if total > 0 else float("nan")


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits = model(X)
    pred = logits.argmax(dim=1)
    return (pred == y).double().mean().item()


def train_one_run(cfg: SyntheticConfig):
    set_seed(cfg.seed)
    device = cfg.device
    dtype = cfg.dtype

    # Hand-crafted full-column-rank 3x3 weak-label transition matrix.
    # Columns sum to 1: M[:, j] = P(z | y=j).
    M = torch.tensor([
        [0.70, 0.10, 0.20],
        [0.20, 0.70, 0.20],
        [0.10, 0.20, 0.60],
    ], dtype=dtype, device=device)

    # Build T and diagnostics.
    T, info = build_T_from_M(M, alpha_scale=0.99)
    A = info["A"]
    order_rate = empirical_order_preservation_rate(A, n_samples=3000)

    # Synthetic Gaussian dataset.
    X_train, y_train, X_test, y_test, means = generate_gaussian_dataset(cfg)

    # Sample weak labels only for training.
    Z_train = sample_weak_labels_from_M(y_train, M)

    train_ds = TensorDataset(X_train, Z_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # Model and loss.
    model = LinearClassifier(cfg.input_dim, cfg.c).to(device=device, dtype=dtype)
    criterion = CLWLLoss(T, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for xb, zb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, zb)
            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            running_loss += loss.item() * batch_size
            n_seen += batch_size

        train_loss = running_loss / n_seen
        train_acc = evaluate_accuracy(model, X_train, y_train)
        test_acc = evaluate_accuracy(model, X_test, y_test)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"test_acc={test_acc:.4f}"
            )

    diagnostics = {
        "M": M,
        "T": T,
        "A": A,
        "means": means,
        "order_preservation_rate": order_rate,
        "T_min": float(info["T_min"].item()),
        "T_max": float(info["T_max"].item()),
        "alpha": float(info["alpha"].item()),
        "alpha_upper": float(info["alpha_upper"].item()),
        "max_abs_NM_minus_I": float((info["NM"] - torch.eye(cfg.c, dtype=dtype, device=device)).abs().max().item()),
    }

    return model, history, diagnostics


if __name__ == "__main__":
    cfg = SyntheticConfig(
        c=3,
        d=3,
        input_dim=2,
        train_per_class=200,
        test_per_class=100,
        batch_size=64,
        epochs=60,
        lr=1e-2,
        seed=123,
        device="cpu",
        dtype=torch.float64,
    )

    model, history, diagnostics = train_one_run(cfg)

    print("\n===== Diagnostics =====")
    print("M =\n", diagnostics["M"])
    print("T =\n", diagnostics["T"])
    print("A = T @ M =\n", diagnostics["A"])
    print("T_min =", diagnostics["T_min"])
    print("T_max =", diagnostics["T_max"])
    print("alpha =", diagnostics["alpha"])
    print("alpha_upper =", diagnostics["alpha_upper"])
    print("max_abs_NM_minus_I =", diagnostics["max_abs_NM_minus_I"])
    print("empirical order-preservation rate =", diagnostics["order_preservation_rate"])
    print("final train_acc =", history["train_acc"][-1])
    print("final test_acc =", history["test_acc"][-1])
