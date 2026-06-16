# ruff: noqa: PYI047

from collections.abc import Sequence
from typing import Any, Callable, Generator, Literal, Never

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

# RNG types
type SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
type RNGLike = np.random.Generator | np.random.BitGenerator


type _WindowSpec = str | tuple[Any, ...] | float | Callable[[int], np.ndarray] | ArrayLike
type _IterableLike[T] = list[T] | tuple[T, ...] | Generator[T, None, None]
type _SequenceLike[T] = Sequence[T] | np.ndarray
type _ScalarOrSequence[T] = T | _SequenceLike[T]

# The following definitions are copied from numpy/_typing/_scalars.py
# (We don't import them directly from numpy because they're an implementation detail.)
###
### START COPIED CODE
###
type _CharLike_co = str | bytes
# The `<X>Like_co` type-aliases below represent all scalars that can be
# coerced into `<X>` (with the casting rule `same_kind`)
type _BoolLike_co = bool | np.bool
type _UIntLike_co = bool | np.unsignedinteger | np.bool
type _IntLike_co = int | np.integer | np.bool
type _FloatLike_co = float | np.floating | np.integer | np.bool
type _ComplexLike_co = complex | np.number | np.bool
type _NumberLike_co = _ComplexLike_co
type _TD64Like_co = int | np.timedelta64 | np.integer | np.bool
# `_VoidLike_co` is technically not a scalar, but it's close enough
type _VoidLike_co = tuple[Any, ...] | np.void
type _ScalarLike_co = complex | str | bytes | np.generic


# Padding modes in general
type _ModeKind = Literal[
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
type _STFTPad = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "reflect",
    "symmetric",
    "empty",
]

type _PadMode = _ModeKind | Callable[..., Any]

type _PadModeSTFT = _STFTPad | Callable[..., Any]


def _ensure_not_reachable(*, __arg: Never):
    """
    Ensure that a code path is not reachable, like typing_extension.assert_never.

    This doesn't raise an exception so that we are forced to manually
    raise a more user friendly exception afterwards.
    """
    ...


type _SparseMatrix = (
    sp.bsr_matrix | sp.coo_matrix | sp.csc_matrix | sp.csr_matrix | sp.dia_matrix | sp.dok_matrix | sp.lil_matrix
)

type _SparseArray = (
    sp.bsr_array | sp.coo_array | sp.csc_array | sp.csr_array | sp.dia_array | sp.dok_array | sp.lil_array
)

# matches the `interp` argument in `scipy.interpolate.interp1d` on all supported SciPy versions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
type _InterpKind = Literal[
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
type _DCTNorm = Literal["backward", "ortho", "forward"]
type _DCTType = Literal[1, 2, 3, 4]

# More specialized number types
type _Real = float | np.integer[Any] | np.floating[Any]
type _Complex = _Real | np.complexfloating[Any, Any]
type _Number = complex | np.number[Any]

# Shape-typing
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
