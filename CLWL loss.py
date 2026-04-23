import torch
import torch.nn as nn
import torch.nn.functional as F


class CLWLLoss(nn.Module):
    """
    CLWL loss for weak labels with one-hot weak-label vectors z.

    Paper formula:
        Psi(z, s) = z^T T^T beta(s) + (1_c - Tz)^T beta(-s)

    With z one-hot and T in [0,1]^{c x d}, we can compute batch-wise:
        support = T z                      # shape (c,) per sample
        beta_pos = beta(s)
        beta_neg = beta(-s)
        loss = <support, beta_pos> + <1-support, beta_neg>

    Here beta is the logistic loss:
        beta(f) = log(1 + exp(-f)) = softplus(-f)

    Assumptions:
        - logits s have shape (B, c)
        - weak labels Z have shape (B, d) and are one-hot row vectors
        - T has shape (c, d)
    """

    def __init__(self, T: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        if T.ndim != 2:
            raise ValueError(f"T must be 2D, got shape {tuple(T.shape)}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")

        # Register T as a buffer because it is fixed by the weak-label mechanism.
        self.register_buffer("T", T)
        self.reduction = reduction

    @staticmethod
    def beta(x: torch.Tensor) -> torch.Tensor:
        """
        Logistic beta:
            beta(x) = log(1 + exp(-x)) = softplus(-x)
        """
        return F.softplus(-x)

    def forward(self, logits: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (B, c), raw logits.
            Z: Tensor of shape (B, d), one-hot weak-label vectors.

        Returns:
            Scalar loss if reduction in {'mean', 'sum'}, else per-sample loss (B,).
        """
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape (B, c), got {tuple(logits.shape)}")
        if Z.ndim != 2:
            raise ValueError(f"Z must have shape (B, d), got {tuple(Z.shape)}")

        B, c = logits.shape
        c_T, d = self.T.shape
        Bz, d_Z = Z.shape

        if c != c_T:
            raise ValueError(
                f"logits second dimension c={c} must match T.shape[0]={c_T}"
            )
        if B != Bz:
            raise ValueError(
                f"Batch sizes must match: logits has {B}, Z has {Bz}"
            )
        if d != d_Z:
            raise ValueError(
                f"Z second dimension d={d_Z} must match T.shape[1]={d}"
            )

        # Optional sanity check: each weak label should be one-hot.
        row_sums = Z.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6):
            raise ValueError("Each row of Z must sum to 1 (one-hot weak labels expected)")
        if not torch.all((Z >= -1e-6) & (Z <= 1.0 + 1e-6)):
            raise ValueError("Z should contain one-hot entries in {0,1}")

        # support[b] = T @ z_b, but with row-wise one-hot labels stored in Z,
        # batch computation is support = Z @ T^T, shape (B, c).
        support = Z @ self.T.T

        beta_pos = self.beta(logits)     # beta(s), shape (B, c)
        beta_neg = self.beta(-logits)    # beta(-s), shape (B, c)

        ones_c = torch.ones((1, c), dtype=logits.dtype, device=logits.device)
        loss_per_sample = (support * beta_pos + (ones_c - support) * beta_neg).sum(dim=1)

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


@torch.no_grad()
def manual_single_sample_clwl(logits: torch.Tensor, z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Direct single-sample implementation of formula (39), useful for debugging.

    Args:
        logits: shape (c,)
        z: shape (d,), one-hot
        T: shape (c, d)

    Returns:
        Scalar tensor with the single-sample CLWL loss.
    """
    if logits.ndim != 1:
        raise ValueError("logits must have shape (c,)")
    if z.ndim != 1:
        raise ValueError("z must have shape (d,)")
    if T.ndim != 2:
        raise ValueError("T must have shape (c, d)")

    c, d = T.shape
    if logits.shape[0] != c:
        raise ValueError("logits length must match T.shape[0]")
    if z.shape[0] != d:
        raise ValueError("z length must match T.shape[1]")

    support = T @ z
    beta_pos = F.softplus(-logits)
    beta_neg = F.softplus(logits)  # beta(-s) = softplus(s)
    return torch.dot(support, beta_pos) + torch.dot(1.0 - support, beta_neg)


@torch.no_grad()
def compare_batch_and_manual(logits: torch.Tensor, Z: torch.Tensor, T: torch.Tensor):
    """
    Compare batch implementation against the direct single-sample formula.
    This is a very useful sanity check before training.
    """
    criterion = CLWLLoss(T, reduction="none")
    batch_loss = criterion(logits, Z)

    manual_losses = []
    for b in range(logits.shape[0]):
        manual_losses.append(manual_single_sample_clwl(logits[b], Z[b], T))
    manual_losses = torch.stack(manual_losses)

    return {
        "batch_loss": batch_loss,
        "manual_loss": manual_losses,
        "max_abs_diff": (batch_loss - manual_losses).abs().max(),
        "allclose": torch.allclose(batch_loss, manual_losses, atol=1e-8, rtol=1e-8),
    }


if __name__ == "__main__":
    # Tiny standalone sanity check for module 2.
    # Example T: shape (c, d) = (3, 3)
    T = torch.tensor([
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.2],
        [0.0, 0.1, 0.6],
    ], dtype=torch.float64)

    # Batch of 2 samples, logits shape (B, c)
    logits = torch.tensor([
        [1.2, -0.3, 0.4],
        [0.1, 0.7, -1.1],
    ], dtype=torch.float64)

    # Weak labels are one-hot row vectors, shape (B, d)
    Z = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float64)

    criterion = CLWLLoss(T, reduction="none")
    losses = criterion(logits, Z)
    cmp = compare_batch_and_manual(logits, Z, T)

    print("T =\n", T)
    print("logits =\n", logits)
    print("Z =\n", Z)
    print("losses =\n", losses)
    print("compare batch vs manual =", cmp)
