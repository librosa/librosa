from __future__ import annotations

from typing import Callable, Generator, List, TypeVar, Union, Tuple, Any, Sequence
import numpy as np
from numpy.typing import ArrayLike


_WindowSpec = Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike]
_T = TypeVar("_T")
_IterableLike = Union[List[_T], Tuple[_T, ...], Generator[_T, None, None]]
_SequenceLike = Union[Sequence[_T], np.ndarray]
_ScalarOrSequence = Union[_T, _SequenceLike[_T]]

_PadMode = Union[str, Callable[..., Any]]

# The following definitions are copied from numpy/_typing/_scalars.py
# (We don't import them directly from numpy because they're an implementation detail.)
###
### START COPIED CODE
###
_CharLike_co = Union[str, bytes]
# The 6 `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, "np.unsignedinteger[Any]"]
_IntLike_co = Union[_BoolLike_co, int, "np.integer[Any]"]
_FloatLike_co = Union[_IntLike_co, float, "np.floating[Any]"]
_ComplexLike_co = Union[_FloatLike_co, complex, "np.complexfloating[Any, Any]"]
_TD64Like_co = Union[_IntLike_co, np.timedelta64]

_NumberLike_co = Union[int, float, complex, "np.number[Any]", np.bool_]
_ScalarLike_co = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]
# `_VoidLike_co` is technically not a scalar, but it's close enough
_VoidLike_co = Union[Tuple[Any, ...], np.void]
###
### END COPIED CODE
###
