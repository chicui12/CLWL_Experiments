import torch


def build_T_from_M(M: torch.Tensor, alpha_scale: float = 0.99, eps: float = 1e-12):
    """
    Build the CLWL support matrix T from a weak-label transition matrix M.

    Theory-aligned construction:
        N = (M^T M)^{-1} M^T
        q_k = min_i N_{ik}
        T = alpha * (N - 1_c q^T)
        alpha = 0.99 / max_k (max_i N_{ik} - min_i N_{ik})

    Assumptions:
        - M has shape (d, c): weak labels x true classes
        - M has full column rank
        - z is a one-hot vector in R^d
        - T should have shape (c, d)

    Args:
        M: Tensor of shape (d, c)
        alpha_scale: Uses alpha = alpha_scale * theoretical_upper_bound.
                     Per your choice, default is 0.99.
        eps: Small tolerance only for numerical checks.

    Returns:
        T: Tensor of shape (c, d)
        info: dict with diagnostic tensors and scalars
    """
    if M.ndim != 2:
        raise ValueError(f"M must be a 2D tensor, got shape {tuple(M.shape)}")

    d, c = M.shape
    if d < c:
        raise ValueError(
            f"Full column rank requires d >= c, but got M with shape {(d, c)}"
        )

    dtype = M.dtype
    device = M.device

    # Check rank explicitly to match the theorem assumptions.
    rank = torch.linalg.matrix_rank(M)
    if int(rank.item()) < c:
        raise ValueError(
            f"M must have full column rank c={c}, but rank(M)={int(rank.item())}"
        )

    # Exact theorem-defined left inverse.
    MtM = M.T @ M                              # (c, c)
    N = torch.linalg.inv(MtM) @ M.T           # (c, d)

    # Column-wise minima: q_k = min_i N_{ik}
    q = N.min(dim=0).values                   # (d,)

    # Shift each column by its minimum.
    ones_c = torch.ones((c, 1), dtype=dtype, device=device)
    T_unnormalized = N - ones_c @ q.unsqueeze(0)   # (c, d)

    # Theoretical upper bound for alpha.
    col_max = N.max(dim=0).values
    col_min = N.min(dim=0).values
    col_ranges = col_max - col_min
    max_range = col_ranges.max()

    if max_range <= eps:
        raise ValueError(
            "Cannot construct alpha because all columns of N are nearly constant. "
            "This would make the upper bound ill-defined."
        )

    alpha_upper = 1.0 / max_range
    alpha = alpha_scale * alpha_upper

    T = alpha * T_unnormalized

    # Diagnostics.
    NM = N @ M                                # should be close to I_c
    A = T @ M                                # should be order-preserving
    I_c = torch.eye(c, dtype=dtype, device=device)

    info = {
        "N": N,
        "q": q,
        "alpha_upper": alpha_upper,
        "alpha": alpha,
        "NM": NM,
        "A": A,
        "rank_M": rank,
        "max_abs_NM_minus_I": (NM - I_c).abs().max(),
        "T_min": T.min(),
        "T_max": T.max(),
        "col_ranges_N": col_ranges,
    }

    return T, info


def check_T_construction(M: torch.Tensor, T: torch.Tensor, info: dict, tol: float = 1e-6):
    """
    Basic sanity checks for the explicit construction.
    Returns a dictionary of booleans and summary values.
    """
    d, c = M.shape
    expected_T_shape = (c, d)
    if T.shape != expected_T_shape:
        raise ValueError(f"Expected T shape {expected_T_shape}, got {tuple(T.shape)}")

    NM = info["NM"]
    A = info["A"]
    I_c = torch.eye(c, dtype=M.dtype, device=M.device)

    results = {
        "shape_ok": T.shape == expected_T_shape,
        "NM_close_to_I": torch.allclose(NM, I_c, atol=tol, rtol=tol),
        "T_in_0_1": bool((T >= -tol).all() and (T <= 1.0 + tol).all()),
        "T_min": float(T.min().item()),
        "T_max": float(T.max().item()),
        "max_abs_NM_minus_I": float((NM - I_c).abs().max().item()),
        "A": A,
    }
    return results


def sample_random_eta(batch_size: int, c: int, device=None, dtype=torch.float64):
    """
    Sample eta from the simplex using a Dirichlet(1,...,1)-style construction.
    """
    x = torch.rand((batch_size, c), device=device, dtype=dtype)
    return x / x.sum(dim=1, keepdim=True)


def empirical_order_preservation_rate(A: torch.Tensor, n_samples: int = 2000):
    """
    Empirically test whether A is order-preserving on random eta vectors.

    For each sampled eta and each pair (i, j), whenever eta_i > eta_j,
    we check whether (A eta)_i > (A eta)_j.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix of shape (c, c)")

    c = A.shape[0]
    etas = sample_random_eta(n_samples, c, device=A.device, dtype=A.dtype)   # (n, c)
    q = etas @ A.T                                                            # (n, c)

    total = 0
    good = 0
    strict_failures = []

    for b in range(n_samples):
        eta_b = etas[b]
        q_b = q[b]
        for i in range(c):
            for j in range(c):
                if i == j:
                    continue
                if eta_b[i] > eta_b[j]:
                    total += 1
                    if q_b[i] > q_b[j]:
                        good += 1
                    elif len(strict_failures) < 10:
                        strict_failures.append({
                            "sample_index": b,
                            "i": i,
                            "j": j,
                            "eta_i": float(eta_b[i].item()),
                            "eta_j": float(eta_b[j].item()),
                            "q_i": float(q_b[i].item()),
                            "q_j": float(q_b[j].item()),
                        })

    rate = good / total if total > 0 else float("nan")
    return {
        "rate": rate,
        "total_pairs_checked": total,
        "good_pairs": good,
        "example_failures": strict_failures,
    }


if __name__ == "__main__":
    # Tiny synthetic sanity-check example.
    # M has shape (d, c) = (3, 3), columns sum to 1, and rank is full.
    M = torch.tensor([
        [0.70, 0.10, 0.20],
        [0.20, 0.70, 0.20],
        [0.10, 0.20, 0.60],
    ], dtype=torch.float64)

    T, info = build_T_from_M(M)
    checks = check_T_construction(M, T, info)
    order_stats = empirical_order_preservation_rate(info["A"], n_samples=3000)

    print("M =\n", M)
    print("N =\n", info["N"])
    print("q =\n", info["q"])
    print("alpha_upper =", float(info["alpha_upper"].item()))
    print("alpha =", float(info["alpha"].item()))
    print("T =\n", T)
    print("NM =\n", info["NM"])
    print("A = T @ M =\n", info["A"])
    print("checks =", checks)
    print("empirical order preservation =", order_stats)
