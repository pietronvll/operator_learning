import logging
from math import sqrt
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger("operator_learning")


class FitResult(TypedDict):
    U: ArrayLike
    V: ArrayLike
    svals: ArrayLike | None


def predict(
    num_steps: int,  # Number of steps to predict (return the last one)
    fit_result: FitResult,
    K_YX: np.ndarray,  # Kernel matrix between the output data and the input data
    K_Xin_X: np.ndarray,  # Kernel matrix between the initial conditions and the input data
    obs_train_Y: np.ndarray,  # Observable to be predicted evaluated on the output training data
) -> np.ndarray:
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
