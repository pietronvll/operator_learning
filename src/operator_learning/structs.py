from typing import TypedDict, Union

from numpy.typing import ArrayLike
from torch import Tensor


class FitResult(TypedDict):
    U: Union[ArrayLike, Tensor]
    V: Union[ArrayLike, Tensor]
    svals: Union[ArrayLike, Tensor] | None


class EigResult(TypedDict):
    values: Union[ArrayLike, Tensor]
    left: Union[ArrayLike, Tensor] | None
    right: Union[ArrayLike, Tensor]
