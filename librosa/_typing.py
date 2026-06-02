from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Generator, List, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from typing_extensions import Literal, Never

# RNG types
SeedLike = Union[int, np.integer, Sequence[int], np.random.SeedSequence]
RNGLike = Union[np.random.Generator, np.random.BitGenerator]


_WindowSpec = Union[str, Tuple[Any, ...], float, Callable[[int], np.ndarray], ArrayLike]
_T = TypeVar("_T")
_IterableLike = Union[List[_T], Tuple[_T, ...], Generator[_T, None, None]]
_SequenceLike = Union[Sequence[_T], np.ndarray]
_ScalarOrSequence = Union[_T, _SequenceLike[_T]]

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


# Padding modes in general
_ModeKind = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]
###
### END COPIED CODE
###

# Padding modes for head/tail padding
# These rule out padding modes that depend on the entire array
_STFTPad = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "reflect",
    "symmetric",
    "empty",
]

_PadMode = Union[_ModeKind, Callable[..., Any]]

_PadModeSTFT = Union[_STFTPad, Callable[..., Any]]


def _ensure_not_reachable(*, __arg: Never):
    """
    Ensure that a code path is not reachable, like typing_extension.assert_never.

    This doesn't raise an exception so that we are forced to manually
    raise a more user friendly exception afterwards.
    """
    ...

_SparseMatrix = Union[
    sp.bsr_matrix,
    sp.coo_matrix,
    sp.csc_matrix,
    sp.csr_matrix,
    sp.dia_matrix,
    sp.dok_matrix,
    sp.lil_matrix,
]

_SparseArray = Union[
    sp.bsr_array,
    sp.coo_array,
    sp.csc_array,
    sp.csr_array,
    sp.dia_array,
    sp.dok_array,
    sp.lil_array,
]

# matches the `interp` argument in `scipy.interpolate.interp1d` on all supported SciPy versions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
_InterpKind = Literal[
    "linear",
    "nearest",
    "nearest-up",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]


# DCT normalization modes
_DCTNorm = Literal["backward", "ortho", "forward"]
_DCTType = Literal[1, 2, 3, 4]

# More specialized number types
_Real = Union[float, "np.integer[Any]", "np.floating[Any]"]
_Complex = Union[_Real, "np.complexfloating[Any, Any]"]
_Number = Union[complex, "np.number[Any]"]
