import logging
from math import sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig, eigh, lstsq
from scipy.sparse.linalg import eigs

from operator_learning.kernel.base import FitResult
from operator_learning.linalg import add_diagonal_, rank_reveal

logger = logging.getLogger("operator_learning")


def fit(
    kernel_X: ArrayLike,  # Kernel matrix of the input data
    kernel_Y: ArrayLike,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
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

    stable_values_idxs = rank_reveal(values, rank, ignore_warnings=False)
    values = values[stable_values_idxs]
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )

    # Eigenvector normalization
    kernel_X_vecs = np.dot(kernel_X / sqrt(npts), vectors)
    vecs_norms = np.sum(
        kernel_X_vecs**2 + tikhonov_reg * kernel_X_vecs * vectors * sqrt(npts),
        axis=0,
    ) ** (0.5)
    U = vectors / vecs_norms
    # Ordering the results
    V = kernel_X @ U
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U, "V": V, "svals": svals}
    return result


def fit_nystroem(
    kernel_X: ArrayLike,  # Kernel matrix of the input inducing points
    kernel_Y: ArrayLike,  # Kernel matrix of the output inducing points
    kernel_Xnys: ArrayLike,  # Kernel matrix between the input data and the input inducing points
    kernel_Ynys: ArrayLike,  # Kernel matrix between the output data and the output inducing points
    tikhonov_reg: float = 0.0,  # Tikhonov (ridge) regularization parameter
    rank: int | None = None,  # Rank of the estimator
    svd_solver: Literal[
        "arnoldi", "full"
    ] = "arnoldi",  # Solver for the generalized eigenvalue problem. 'arnoldi' or 'full'
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

    stable_values_idxs = rank_reveal(values, rank, ignore_warnings=False)
    values = values[stable_values_idxs]
    vectors = vectors[:, stable_values_idxs]
    # Compare the filtered eigenvalues with the regularization strength, and warn if there are any eigenvalues that are smaller than the regularization strength.
    if not np.all(np.abs(values) >= tikhonov_reg):
        logger.warning(
            f"Warning: {(np.abs(values) < tikhonov_reg).sum()} out of the {len(values)} squared singular values are smaller than the regularization strength {tikhonov_reg:.2e}. Consider redudcing the regularization strength to avoid overfitting."
        )
    # Eigenvector normalization
    normalization_csts = np.sqrt(
        np.abs(np.sum(vectors.conj() * (kernel_XYX @ vectors), axis=0))
    )
    well_conditioned_indices = rank_reveal(
        normalization_csts, rank, rcond=1000.0 * np.finfo(vectors.dtype).eps
    )
    vectors = (
        vectors[:, well_conditioned_indices]
        / normalization_csts[well_conditioned_indices]
    )
    values = values[well_conditioned_indices]
    U = A @ vectors
    V = _tmp_YX @ vectors
    svals = np.sqrt(np.abs(values))
    result: FitResult = {"U": U, "V": V, "svals": svals}
    return result
