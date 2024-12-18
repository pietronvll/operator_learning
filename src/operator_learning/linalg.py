import logging

import numpy as np
from numpy.typing import ArrayLike

from operator_learning.utils import topk

logger = logging.getLogger("operator_learning")


def add_diagonal_(M: ArrayLike, alpha: float):
    """
    Add alpha to the diagonal of a matrix M in-place.

    Parameters
    ----------
    M : ArrayLike
        The matrix to modify.
    alpha : float
        The value to add to the diagonal of M.
    """
    np.fill_diagonal(M, M.diagonal() + alpha)


def rank_reveal(
    values: np.ndarray,
    rank: int,  # Desired rank
    rcond: float | None = None,  # Threshold for the singular values
    ignore_warnings: bool = True,
):
    if rcond is None:
        rcond = 10.0 * values.shape[0] * np.finfo(values.dtype).eps

    top_values, top_idxs = topk(values, rank)

    if all(top_values > rcond):
        top_idxs
    else:
        valid = top_values > rcond
        # In the case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
        first_invalid = np.argmax(np.logical_not(valid))
        _first_discarded_val = np.max(np.abs(values[first_invalid:]))

        if not ignore_warnings:
            logger.warning(
                f"Warning: Discarted {rank - values.shape[0]} dimensions of the {rank} requested due to numerical instability. Consider decreasing the rank. The largest discarded value is: {_first_discarded_val:.3e}."
            )
        return top_idxs[valid]
