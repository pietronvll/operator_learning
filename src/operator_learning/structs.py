from typing import TypedDict

from numpy.typing import ArrayLike


class FitResult(TypedDict):
    U: ArrayLike
    V: ArrayLike
    svals: ArrayLike | None


class EigResult(TypedDict):
    values: ArrayLike
    left: ArrayLike
    right: ArrayLike
