import logging
from math import sqrt
from typing import Literal, TypedDict

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from operator_learning.linalg import weighted_norm
from operator_learning.utils import fuzzy_parse_complex

logger = logging.getLogger("operator_learning")


class FitResult(TypedDict):
    U: ArrayLike
    V: ArrayLike
    svals: ArrayLike | None


class EigResult(TypedDict):
    values: ArrayLike
    left: ArrayLike
    right: ArrayLike


def predict(
    num_steps: int,  # Number of steps to predict (return the last one)
    fit_result: FitResult,
    K_YX: ArrayLike,  # Kernel matrix between the output data and the input data (or inducing points, in the case of Nystroem).
    K_Xin_X: ArrayLike,  # Kernel matrix between the initial conditions and the input data (or the inducing points, in the case of Nystroem)
    obs_train_Y: ArrayLike,  # Observable to be predicted evaluated on the output training data (or inducing points, in the case of Nystroem)
) -> ArrayLike:
    # G = S UV.T Z
    # G^n = (SU)(V.T K_YX U)^(n-1)(V.T Z)
    U = fit_result["U"]
    V = fit_result["V"]
    npts = U.shape[0]
    K_dot_U = K_Xin_X @ U / sqrt(npts)
    V_dot_obs = V.T @ obs_train_Y / sqrt(npts)
    V_K_YX_U = np.linalg.multi_dot([V.T, K_YX, U]) / npts
    M = np.linalg.matrix_power(V_K_YX_U, num_steps - 1)
    return np.linalg.multi_dot([K_dot_U, M, V_dot_obs])


def eig(
    fit_result: FitResult,
    K_X: ArrayLike,  # Kernel matrix of the input data
    K_YX: ArrayLike,  # Kernel matrix between the output data and the input data
) -> EigResult:
    # SUV.TZ -> V.T K_YX U (right ev = SUvr, left ev = ZVvl)
    U = fit_result["U"]
    V = fit_result["V"]
    r_dim = (K_X.shape[0]) ** (-1)

    W_YX = np.linalg.multi_dot([V.T, r_dim * K_YX, U])
    W_X = np.linalg.multi_dot([U.T, r_dim * K_X, U])

    values, vl, vr = scipy.linalg.eig(
        W_YX, left=True, right=True
    )  # Left -> V, Right -> U
    values = fuzzy_parse_complex(values)
    r_perm = np.argsort(values)
    vr = vr[:, r_perm]
    l_perm = np.argsort(values.conj())
    vl = vl[:, l_perm]
    values = values[r_perm]

    rcond = 1000.0 * np.finfo(U.dtype).eps
    # Normalization in RKHS
    norm_r = weighted_norm(vr, W_X)
    norm_r = np.where(norm_r < rcond, np.inf, norm_r)
    vr = vr / norm_r

    # Bi-orthogonality of left eigenfunctions
    norm_l = np.diag(np.linalg.multi_dot([vl.T, W_YX, vr]))
    norm_l = np.where(np.abs(norm_l) < rcond, np.inf, norm_l)
    vl = vl / norm_l
    result: EigResult = {"values": values, "left": V @ vl, "right": U @ vr}
    return result


def evaluate_eigenfunction(
    eig_result: EigResult,
    which: Literal["left", "right"],
    K_Xin_X_or_Y: ArrayLike,
    # Kernel matrix between the initial conditions and the input data (right eigenfunctions) or the output data
    # (left eigenfunctions)
):
    vr_or_vl = eig_result[which]
    rsqrt_dim = (K_Xin_X_or_Y.shape[1]) ** (-0.5)
    return np.linalg.multi_dot([rsqrt_dim * K_Xin_X_or_Y, vr_or_vl])
