import logging
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, eig, eigh, lstsq, qr
from scipy.sparse.linalg import eigs

from operator_learning.linalg import add_diagonal_, stable_topk
from operator_learning.structs import FitResult

logger = logging.getLogger("operator_learning")


def fit(
    kernel_X: ArrayLike,  # Kernel matrix of the input data
    kernel_Y: ArrayLike,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    # Number of data points
    npts = kernel_X.shape[0]
    eps = 1000.0 * np.finfo(kernel_X.dtype).eps
    penalty = max(eps, tikhonov_reg) * npts

    A = (kernel_Y / sqrt(npts)) @ (kernel_X / sqrt(npts))
    add_diagonal_(kernel_X, penalty)
    # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
    # Prefer svd_solver == 'randomized' in such a case.
    if svd_solver == "arnoldi":
        # Adding a small buffer to the Arnoldi-computed eigenvalues.
        num_arnoldi_eigs = min(rank + 5, npts)
        values, vectors = eigs(A, k=num_arnoldi_eigs, M=kernel_X)
    elif svd_solver == "full":  # 'full'
        values, vectors = eig(A, kernel_X, overwrite_a=True, overwrite_b=True)
    else:
        raise ValueError(f"Unknown svd_solver: {svd_solver}")
    # Remove the penalty from kernel_X (inplace)
    add_diagonal_(kernel_X, -penalty)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    kernel_X_vecs = np.dot(kernel_X / sqrt(npts), vectors)
    vecs_norm = np.sqrt(
        np.sum(
            kernel_X_vecs**2 + tikhonov_reg * kernel_X_vecs * vectors * sqrt(npts),
            axis=0,
        )
    )

    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    U = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]

    # Ordering the results
    V = kernel_X @ U
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def fit_nystroem(
    kernel_X: ArrayLike,  # Kernel matrix of the input inducing points
    kernel_Y: ArrayLike,  # Kernel matrix of the output inducing points
    kernel_Xnys: ArrayLike,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: ArrayLike,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    svd_solver: Literal["arnoldi", "full"] = "arnoldi",
) -> FitResult:
    num_points = kernel_Xnys.shape[0]
    num_centers = kernel_X.shape[0]

    eps = 1000 * np.finfo(kernel_X.dtype).eps * num_centers
    reg = max(eps, tikhonov_reg)

    # LHS of the generalized eigenvalue problem
    sqrt_Mn = sqrt(num_centers * num_points)
    kernel_YX_nys = (kernel_Ynys.T / sqrt_Mn) @ (kernel_Xnys / sqrt_Mn)

    _tmp_YX = lstsq(kernel_Y * (num_centers**-1), kernel_YX_nys)[0]
    kernel_XYX = kernel_YX_nys.T @ _tmp_YX

    # RHS of the generalized eigenvalue problem
    kernel_Xnys_sq = (kernel_Xnys.T / sqrt_Mn) @ (
        kernel_Xnys / sqrt_Mn
    ) + reg * kernel_X * (num_centers**-1)

    add_diagonal_(kernel_Xnys_sq, eps)
    A = lstsq(kernel_Xnys_sq, kernel_XYX)[0]
    if svd_solver == "full":
        values, vectors = eigh(
            kernel_XYX, kernel_Xnys_sq
        )  # normalization leads to needing to invert evals
    elif svd_solver == "arnoldi":
        _oversampling = max(10, 4 * int(np.sqrt(rank)))
        _num_arnoldi_eigs = min(rank + _oversampling, num_centers)
        values, vectors = eigs(kernel_XYX, k=_num_arnoldi_eigs, M=kernel_Xnys_sq)
    else:
        raise ValueError(f"Unknown svd_solver {svd_solver}")
    add_diagonal_(kernel_Xnys_sq, -eps)

    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )
    # Eigenvector normalization
    vecs_norm = np.sqrt(np.abs(np.sum(vectors.conj() * (kernel_XYX @ vectors), axis=0)))
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(vecs_norm, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs] / vecs_norm[stable_values_idxs]
    U = A @ vectors
    V = _tmp_YX @ vectors
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U.real, "V": V.real, "svals": svals}
    return result


def fit_randomized(
    kernel_X: ArrayLike,  # Kernel matrix of the input data
    kernel_Y: ArrayLike,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter
    rank: int,  # Rank of the estimator
    n_oversamples: int = 5,  # Number of oversamples
    optimal_sketching: bool = False,  # Whether to use optimal sketching (slower but more accurate) or not.
    iterated_power: int = 1,  # Number of iterations of the power method
    rng_seed: int | None = None,
    precomputed_cholesky=None,  # Precomputed Cholesky decomposition. Should be the output of cho_factor evaluated on the regularized kernel matrix.
) -> FitResult:
    rng = np.random.default_rng(rng_seed)
    npts = kernel_X.shape[0]

    penalty = npts * tikhonov_reg
    add_diagonal_(kernel_X, penalty)
    if precomputed_cholesky is None:
        cholesky_decomposition = cho_factor(kernel_X)
    else:
        cholesky_decomposition = precomputed_cholesky
    add_diagonal_(kernel_X, -penalty)

    sketch_dimension = rank + n_oversamples

    if optimal_sketching:
        cov = kernel_Y / npts
        sketch = rng.multivariate_normal(
            np.zeros(npts, dtype=kernel_Y.dtype), cov, size=sketch_dimension
        ).T
    else:
        sketch = rng.standard_normal(size=(npts, sketch_dimension))

    for _ in range(iterated_power):
        # Powered randomized rangefinder
        sketch = (kernel_Y / npts) @ (
            sketch - penalty * cho_solve(cholesky_decomposition, sketch)
        )
        sketch, _ = qr(sketch, mode="economic")  # QR re-orthogonalization

    kernel_X_sketch = cho_solve(cholesky_decomposition, sketch)
    _M = sketch - penalty * kernel_X_sketch

    F_0 = sketch.T @ sketch - penalty * (sketch.T @ kernel_X_sketch)  # Symmetric
    F_0 = 0.5 * (F_0 + F_0.T)
    F_1 = _M.T @ ((kernel_Y @ _M) / npts)

    values, vectors = eig(lstsq(F_0, F_1)[0])
    values, stable_values_idxs = stable_topk(values, rank, ignore_warnings=False)
    vectors = vectors[:, stable_values_idxs]

    # Remove elements in the kernel of F_0
    relative_norm_sq = np.abs(
        np.sum(vectors.conj() * (F_0 @ vectors), axis=0)
        / np.linalg.norm(vectors, axis=0) ** 2
    )
    norm_rcond = 1000.0 * np.finfo(values.dtype).eps
    values, stable_values_idxs = stable_topk(relative_norm_sq, rank, rcond=norm_rcond)
    vectors = vectors[:, stable_values_idxs]

    vecs_norms = (np.sum(vectors.conj() * (F_0 @ vectors), axis=0).real) ** 0.5
    vectors = vectors / vecs_norms

    U = sqrt(npts) * kernel_X_sketch @ vectors
    V = sqrt(npts) * _M @ vectors
    svals = np.sqrt(values)
    result: FitResult = {"U": U, "V": V, "svals": svals}
    return result
