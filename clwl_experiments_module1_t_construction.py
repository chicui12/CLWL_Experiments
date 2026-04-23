from __future__ import annotations

"""
Module 1: weak-label transition utilities and T construction for CLWL.

This module solves one core problem:
- represent a weak-label transition matrix M;
- validate the stochastic / rank assumptions used by CLWL;
- construct T from M using the explicit left-inverse-style recipe;
- verify the resulting A = T M has the target order-preserving form.

Intended interface to later modules:
- data-generation modules will output M and call `construct_clwl_T(M)`;
- loss modules will consume the returned `T` and `A = T @ M`;
- experiment modules can call `summarize_clwl_construction(M)` to log diagnostics.

Conventions used here:
- M has shape (d, c), where d = number of weak labels, c = number of clean classes.
- M is column-stochastic: each column sums to 1.
- T has shape (c, d), so A = T @ M has shape (c, c).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass
class CLWLConstructionResult:
    """Container for the CLWL T-construction output."""

    M: Array
    N: Array
    T: Array
    A: Array
    alpha: float
    q: Array
    delta_per_column: Array
    delta_max: float
    column_sums_M: Array
    rank_M: int
    lambda_value: float
    v: Array
    t_min: float
    t_max: float
    reconstruction_error: float


class TransitionMatrixError(ValueError):
    """Raised when the weak-label transition matrix is invalid."""



def set_seed(seed: int) -> None:
    """Set numpy seed for reproducibility."""
    np.random.seed(seed)



def as_float_array(x: Array | list[list[float]] | list[float]) -> Array:
    """Convert input to a float64 NumPy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise TransitionMatrixError(f"Expected a 2D matrix, got shape {arr.shape}.")
    return arr



def validate_transition_matrix(
    M: Array,
    *,
    atol: float = 1e-8,
    require_nonnegative: bool = True,
    require_column_stochastic: bool = True,
) -> None:
    """
    Validate the weak-label transition matrix M.

    Parameters
    ----------
    M:
        Matrix of shape (d, c), where M[z, y] = P(z | y).
    atol:
        Numerical tolerance.
    require_nonnegative:
        If True, all entries must be >= -atol.
    require_column_stochastic:
        If True, each column must sum to 1 within tolerance.
    """
    if M.ndim != 2:
        raise TransitionMatrixError(f"M must be 2D, got ndim={M.ndim}.")

    d, c = M.shape
    if d < c:
        raise TransitionMatrixError(
            f"Expected d >= c for full column rank construction, got M.shape={M.shape}."
        )

    if require_nonnegative and np.min(M) < -atol:
        raise TransitionMatrixError(
            f"M contains entries below 0 beyond tolerance: min(M)={np.min(M):.3e}."
        )

    if require_column_stochastic:
        col_sums = M.sum(axis=0)
        if not np.allclose(col_sums, np.ones(c), atol=atol, rtol=0.0):
            raise TransitionMatrixError(
                "M is not column-stochastic. "
                f"Column sums are {col_sums}, expected all ones."
            )



def matrix_rank(M: Array, *, tol: Optional[float] = None) -> int:
    """Compute numerical rank of M."""
    return int(np.linalg.matrix_rank(M, tol=tol))



def left_inverse_from_full_column_rank(M: Array) -> Array:
    """
    Return a left inverse N of M using the normal-equation formula.

    Since M has shape (d, c) with full column rank c, we use
        N = (M^T M)^{-1} M^T,
    so that N M = I_c.
    """
    gram = M.T @ M
    if matrix_rank(gram) != gram.shape[0]:
        raise TransitionMatrixError(
            "M^T M is singular; M does not appear to have full column rank."
        )
    return np.linalg.solve(gram, M.T)



def construct_clwl_T(
    M: Array,
    *,
    alpha: Optional[float] = None,
    safety_factor: float = 0.95,
    atol: float = 1e-8,
) -> CLWLConstructionResult:
    """
    Construct T for CLWL from a full-column-rank transition matrix M.

    The construction follows the intended recipe of the explicit theorem:
        1. take a left inverse N such that N M = I_c;
        2. for each weak-label column k of N, subtract its minimum q_k;
        3. scale by alpha so all entries of T lie in [0, 1].

    Parameters
    ----------
    M:
        Weak-label transition matrix of shape (d, c).
    alpha:
        Optional manual scaling. If None, choose a conservative value.
    safety_factor:
        When alpha is chosen automatically and Delta > 0, use
            alpha = safety_factor / Delta.
        Must satisfy 0 < safety_factor <= 1.
    atol:
        Numerical tolerance used in checks.
    """
    M = as_float_array(M)
    validate_transition_matrix(M, atol=atol)

    d, c = M.shape
    rank = matrix_rank(M)
    if rank != c:
        raise TransitionMatrixError(
            f"M must have full column rank c={c}, but rank(M)={rank}."
        )

    if not (0.0 < safety_factor <= 1.0):
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}.")

    N = left_inverse_from_full_column_rank(M)  # shape (c, d)

    q = N.min(axis=0)  # shape (d,)
    delta_per_column = N.max(axis=0) - q
    delta_max = float(np.max(delta_per_column))

    if alpha is None:
        if delta_max > 0.0:
            alpha = float(safety_factor / delta_max)
        else:
            alpha = 1.0
    else:
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        if delta_max > 0.0 and alpha > (1.0 / delta_max) + atol:
            raise ValueError(
                f"alpha={alpha:.6f} is too large: need alpha <= 1/Delta = {1.0/delta_max:.6f}."
            )

    T = alpha * (N - np.outer(np.ones(c, dtype=np.float64), q))
    A = T @ M

    # Since N M = I and T = alpha(N - 1 q^T), we should have
    # A = alpha I - alpha 1 (q^T M).
    # Write A = lambda I + 1 v^T with lambda = alpha, v^T = -alpha q^T M.
    lambda_value = alpha
    v = -alpha * (q @ M)  # shape (c,)
    A_target = lambda_value * np.eye(c, dtype=np.float64) + np.outer(np.ones(c), v)

    t_min = float(np.min(T))
    t_max = float(np.max(T))
    reconstruction_error = float(np.max(np.abs(A - A_target)))
    column_sums_M = M.sum(axis=0)

    if t_min < -atol or t_max > 1.0 + atol:
        raise TransitionMatrixError(
            "Constructed T is outside [0, 1] beyond tolerance: "
            f"min={t_min:.3e}, max={t_max:.3e}."
        )

    return CLWLConstructionResult(
        M=M,
        N=N,
        T=T,
        A=A,
        alpha=alpha,
        q=q,
        delta_per_column=delta_per_column,
        delta_max=delta_max,
        column_sums_M=column_sums_M,
        rank_M=rank,
        lambda_value=lambda_value,
        v=v,
        t_min=t_min,
        t_max=t_max,
        reconstruction_error=reconstruction_error,
    )



def is_order_preserving_standard_form(
    A: Array,
    *,
    atol: float = 1e-8,
) -> tuple[bool, float, Array]:
    """
    Check whether A numerically matches the form A = lambda I + 1 v^T.

    Returns
    -------
    is_ok:
        Whether the matrix matches the form up to tolerance and lambda > 0.
    lambda_value:
        Estimated lambda.
    v:
        Estimated v.
    """
    A = as_float_array(A)
    if A.shape[0] != A.shape[1]:
        raise TransitionMatrixError(f"A must be square, got shape {A.shape}.")

    
    c = A.shape[0]
    v = (A.sum(axis=0) - np.diag(A)) / (c - 1)
    lambda_estimates = np.diag(A) - v
    lambda_value = float(lambda_estimates.mean())
    
    A_rebuilt = lambda_value * np.eye(c) + np.outer(np.ones(c), v)
    ok = np.max(np.abs(A - A_rebuilt)) <= atol and lambda_value > 0.0
    return bool(ok), lambda_value, v



def summarize_clwl_construction(M: Array, *, atol: float = 1e-8) -> dict[str, float | int | bool]:
    """
    Convenience wrapper that constructs T and returns compact diagnostics.

    This is intended for logging inside later experiment modules.
    """
    result = construct_clwl_T(M, atol=atol)
    ok, lambda_value, _ = is_order_preserving_standard_form(result.A, atol=max(atol, 1e-6))
    return {
        "d": int(result.M.shape[0]),
        "c": int(result.M.shape[1]),
        "rank_M": int(result.rank_M),
        "alpha": float(result.alpha),
        "delta_max": float(result.delta_max),
        "t_min": float(result.t_min),
        "t_max": float(result.t_max),
        "lambda_value": float(lambda_value),
        "reconstruction_error": float(result.reconstruction_error),
        "A_has_standard_form": bool(ok),
    }



def make_uniform_complementary_M(c: int) -> Array:
    """
    Construct the standard single complementary-label transition matrix.

    For c classes, M[z, y] = 1/(c-1) if z != y and 0 otherwise.
    Shape is (c, c).
    """
    if c < 2:
        raise ValueError(f"Need c >= 2, got {c}.")
    M = np.ones((c, c), dtype=np.float64) - np.eye(c, dtype=np.float64)
    M /= (c - 1)
    return M



def make_random_full_rank_column_stochastic_M(
    d: int,
    c: int,
    *,
    seed: int = 0,
    min_entry: float = 1e-3,
) -> Array:
    """
    Generate a random nonnegative column-stochastic matrix with high probability of full rank.

    Parameters
    ----------
    d, c:
        Matrix shape (d, c) with d >= c.
    seed:
        Random seed.
    min_entry:
        Small positive floor before normalization to avoid exact zeros.
    """
    if d < c:
        raise ValueError(f"Need d >= c, got d={d}, c={c}.")
    if min_entry < 0:
        raise ValueError(f"min_entry must be nonnegative, got {min_entry}.")

    rng = np.random.default_rng(seed)
    raw = rng.random((d, c)) + min_entry
    M = raw / raw.sum(axis=0, keepdims=True)

    if matrix_rank(M) != c:
        # Retry with a deterministic perturbation if needed.
        M = M + 1e-4 * rng.standard_normal((d, c))
        M = np.clip(M, 1e-8, None)
        M = M / M.sum(axis=0, keepdims=True)

    if matrix_rank(M) != c:
        raise TransitionMatrixError("Failed to generate a full-column-rank random M.")
    return M


if __name__ == "__main__":
    # Minimal smoke tests.
    set_seed(0)

    print("=== Uniform complementary example ===")
    M_comp = make_uniform_complementary_M(c=4)
    summary_comp = summarize_clwl_construction(M_comp)
    for k, v in summary_comp.items():
        print(f"{k}: {v}")

    print("\n=== Random full-rank weak-label example ===")
    M_rand = make_random_full_rank_column_stochastic_M(d=6, c=4, seed=123)
    summary_rand = summarize_clwl_construction(M_rand)
    for k, v in summary_rand.items():
        print(f"{k}: {v}")
