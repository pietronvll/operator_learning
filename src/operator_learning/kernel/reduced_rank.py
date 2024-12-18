import logging
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig, eigh, lstsq
from scipy.sparse.linalg import eigs

from operator_learning.kernel.base import postprocess_UV
from operator_learning.linalg import add_diagonal_, rank_reveal

logger = logging.getLogger("operator_learning")


def fit(
    kernel_X: ArrayLike,  # Kernel matrix of the input data
    kernel_Y: ArrayLike,  # Kernel matrix of the output data
    tikhonov_reg: float,  # Tikhonov (ridge) regularization parameter, can be 0
    rank: int,  # Rank of the estimator
    svd_solver: str = "arnoldi",  # SVD solver to use. 'arnoldi' is faster but might be numerically unstable.
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    # Number of data points
    npts = kernel_X.shape[0]
    eps = 1000.0 * np.finfo(kernel_X.dtype).eps
    if tikhonov_reg == 0.0:
        logger.warning(
            "The least-squares solution (tikhonov_reg == 0) of the reduced rank problem in the kernel setting is computationally very inefficient. Consider adding a small regularization parameter."
        )
        values_X, U_X = eigh(kernel_X)
        stable_UX_cols = rank_reveal(values_X, npts)
        U_X = U_X[:, stable_UX_cols]
        proj_X = U_X @ U_X.T
        L = proj_X @ kernel_Y
        if svd_solver != "full":
            values, vectors = eigs(L, rank + 3)
        else:
            values, vectors = eig(L, overwrite_a=True)
        stable_V_cols = rank_reveal(values, rank)
        values = values[stable_V_cols]
        V = vectors[stable_V_cols]
        # Normalize V
        V_norm = np.linalg.norm(V, ord=2, axis=0) / sqrt(npts)
        eps = 1000.0 * np.finfo(kernel_X.dtype).eps * npts
        inv_V_norm = np.where(V_norm < eps, 0.0, V_norm**-1)
        raise ValueError("Debug from here. Is the line below a bug?")
        V = V / inv_V_norm
        # Solve the least squares problem to determine U
        U = lstsq(kernel_X, V)[0]
        svals = np.flip(np.sort(np.abs(values))) ** 0.5
        U, V = postprocess_UV(U, V, rank)
    else:
        penalty = max(eps, tikhonov_reg) * npts

        A = (kernel_Y / sqrt(npts)) @ (kernel_X / sqrt(npts))

        add_diagonal_(kernel_X, penalty)
        # Find U via Generalized eigenvalue problem equivalent to the SVD. If K is ill-conditioned might be slow.
        # Prefer svd_solver == 'randomized' in such a case.
        if svd_solver == "arnoldi":
            # Adding a small buffer to the Arnoldi-computed eigenvalues.
            num_arnoldi_eigs = min(rank + 5, npts)
            values, vectors = eigs(A, k=num_arnoldi_eigs, M=kernel_X)
        else:  # 'full'
            values, vectors = eig(A, kernel_X, overwrite_a=True, overwrite_b=True)
        # Remove the penalty from kernel_X (inplace)
        add_diagonal_(kernel_X, -penalty)

        stable_values_idxs = rank_reveal(values, rank)
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
        svals = sqrt(np.abs(values))
    return U, V, svals
