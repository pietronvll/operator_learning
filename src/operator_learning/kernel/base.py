import logging
from typing import TypedDict

from numpy.typing import ArrayLike

logger = logging.getLogger("operator_learning")


class FitResult(TypedDict):
    U: ArrayLike
    V: ArrayLike
    svals: ArrayLike | None


def postprocess_UV(
    U: ArrayLike, V: ArrayLike, rank: int
) -> tuple[ArrayLike, ArrayLike]:
    assert U.shape == V.shape
    if U.shape[1] < rank:
        logger.warning(
            f"Warning: The fitting algorithm discarded {rank - U.shape[1]} dimensions of the {rank} requested out of numerical instabilities.\nThe rank attribute has been updated to {U.shape[1]}.\nConsider decreasing the rank parameter."
        )
    else:
        # Assuming that everything is in decreasing order
        U = U[:, :rank]
        V = V[:, :rank]
    return U, V
