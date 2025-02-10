#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

from __future__ import annotations

import scipy.ndimage
import scipy.sparse

import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided

from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import DTypeLike
from typing import (
    Any,
    Callable,
    List,
    Dict,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

__all__ = [
    "MAX_MEM_BLOCK",
    "frame",
    "pad_center",
    "expand_to",
    "fix_length",
    "valid_audio",
    "valid_int",
    "is_positive_int",
    "valid_intervals",
    "fix_frames",
    "axis_sort",
    "localmax",
    "localmin",
    "normalize",
    "peak_pick",
    "sparsify_rows",
    "shear",
    "stack",
    "fill_off_diagonal",
    "index_to_slice",
    "sync",
    "softmask",
    "buf_to_float",
    "tiny",
    "cyclic_gradient",
    "dtype_r2c",
    "dtype_c2r",
    "count_unique",
    "is_unique",
    "abs2",
    "phasor",
]


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    """Slice a data array into (overlapping) frames.

    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.

    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::

        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]

    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.

    The second way (``axis=0``) results in the array ``x_frames``::

        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]

    where each row ``x_frames[i]`` contains a contiguous slice of the input.

    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either before the framing axis (if ``axis < 0``)
    or after the framing axis (if ``axis >= 0``).

    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : int
        The axis along which to frame.
    writeable : bool
        If ``False``, then the framed view of ``x`` is read-only.
        If ``True``, then the framed view is read-write.  Note that writing to the framed view
        will also write to the input array ``x`` in this case.
    subok : bool
        If True, sub-classes will be passed-through, otherwise the returned array will be
        forced to be a base-class array (default).

    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::

            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]

        If ``axis=0`` (framing on the first dimension), then::

            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]

    Raises
    ------
    ParameterError
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.

        If ``hop_length < 1``, frames cannot advance.

    See Also
    --------
    numpy.lib.stride_tricks.as_strided

    Examples
    --------
    Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    >>> frames
    array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
           [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
           ...,
           [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
           [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
          dtype=float32)
    >>> y.shape
    (117601,)

    >>> frames.shape
    (2048, 1806)

    Or frame along the first axis instead of the last:

    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    >>> frames.shape
    (1806, 2048)

    Frame a stereo signal:

    >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    >>> y.shape
    (2, 117601)
    >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    (2, 2048, 1806)

    Carve an STFT into fixed-length patches of 32 frames with 50% overlap

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S.shape
    (1025, 230)
    >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    >>> S_patch.shape
    (1025, 32, 13)
    >>> # The first patch contains the first 32 frames of S
    >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    True
    >>> # The second patch contains frames 16 to 16+32=48, and so on
    >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    True
    """
    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


@cache(level=20)
def valid_audio(y: np.ndarray) -> bool:
    """Determine whether a variable contains valid audio data.

    The following conditions must be satisfied:

    - ``type(y)`` is ``np.ndarray``
    - ``y.dtype`` is floating-point
    - ``y.ndim != 0`` (must have at least one dimension)
    - ``np.isfinite(y).all()`` samples must be all finite values

    Parameters
    ----------
    y : np.ndarray
        The input data to validate

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        In any of the conditions specified above fails

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    We can make an array that can be interpreted as an audio signal

    >>> y = np.random.randn(5000)
    >>> librosa.util.valid_audio(y)
    True

    If we insert a non-finite sample value somewhere, it will fail

    >>> y[5] = np.nan
    >>> librosa.util.valid_audio(y)
    ...
    ParameterError: Audio buffer is not finite everywhere

    See Also
    --------
    numpy.float32
    """
    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if y.ndim == 0:
        raise ParameterError(
            f"Audio data must be at least one-dimensional, given y.shape={y.shape}"
        )

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True


def valid_int(x: float, *, cast: Optional[Callable[[float], float]] = None) -> int:
    """Ensure that an input value is integer-typed.
    This is primarily useful for ensuring integrable-valued
    array indices.

    Parameters
    ----------
    x : number
        A scalar value to be cast to int
    cast : function [optional]
        A function to modify ``x`` before casting.
        Default: `np.floor`

    Returns
    -------
    x_int : int
        ``x_int = int(cast(x))``

    Raises
    ------
    ParameterError
        If ``cast`` is provided and is not callable.
    """
    if cast is None:
        cast = np.floor

    if not callable(cast):
        raise ParameterError("cast parameter must be callable")

    return int(cast(x))


def is_positive_int(x: float) -> bool:
    """Check that x is a positive integer, i.e. 1 or greater.

    Parameters
    ----------
    x : number

    Returns
    -------
    positive : bool
    """
    # Check type first to catch None values.
    return isinstance(x, (int, np.integer)) and (x > 0)


def valid_intervals(intervals: np.ndarray) -> bool:
    """Ensure that an array is a valid representation of time intervals:

        - intervals.ndim == 2
        - intervals.shape[1] == 2
        - intervals[i, 0] <= intervals[i, 1] for all i

    Parameters
    ----------
    intervals : np.ndarray [shape=(n, 2)]
        set of time intervals

    Returns
    -------
    valid : bool
        True if ``intervals`` passes validation.
    """
    if intervals.ndim != 2 or intervals.shape[-1] != 2:
        raise ParameterError("intervals must have shape (n, 2)")

    if np.any(intervals[:, 0] > intervals[:, 1]):
        raise ParameterError(f"intervals={intervals} must have non-negative durations")

    return True


def pad_center(
    data: np.ndarray, *, size: int, axis: int = -1, **kwargs: Any
) -> np.ndarray:
    """Pad an array to a target length along a target axis.

    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, size=10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, size=7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, size=7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad ``data``
    axis : int
        Axis along which to pad and center the data
    **kwargs : additional keyword arguments
        arguments passed to `np.pad`

    Returns
    -------
    data_padded : np.ndarray
        ``data`` centered and padded to length ``size`` along the
        specified axis

    Raises
    ------
    ParameterError
        If ``size < data.shape[axis]``

    See Also
    --------
    numpy.pad
    """
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(
            f"Target size ({size:d}) must be at least input size ({n:d})"
        )

    return np.pad(data, lengths, **kwargs)


def expand_to(
    x: np.ndarray, *, ndim: int, axes: Union[int, slice, Sequence[int], Sequence[slice]]
) -> np.ndarray:
    """Expand the dimensions of an input array with

    Parameters
    ----------
    x : np.ndarray
        The input array
    ndim : int
        The number of dimensions to expand to.  Must be at least ``x.ndim``
    axes : int or slice
        The target axis or axes to preserve from x.
        All other axes will have length 1.

    Returns
    -------
    x_exp : np.ndarray
        The expanded version of ``x``, satisfying the following:
            ``x_exp[axes] == x``
            ``x_exp.ndim == ndim``

    See Also
    --------
    np.expand_dims

    Examples
    --------
    Expand a 1d array into an (n, 1) shape

    >>> x = np.arange(3)
    >>> librosa.util.expand_to(x, ndim=2, axes=0)
    array([[0],
       [1],
       [2]])

    Expand a 1d array into a (1, n) shape

    >>> librosa.util.expand_to(x, ndim=2, axes=1)
    array([[0, 1, 2]])

    Expand a 2d array into (1, n, m, 1) shape

    >>> x = np.vander(np.arange(3))
    >>> librosa.util.expand_to(x, ndim=4, axes=[1,2]).shape
    (1, 3, 3, 1)
    """
    # Force axes into a tuple
    axes_tup: Tuple[int]
    try:
        axes_tup = tuple(axes)  # type: ignore
    except TypeError:
        axes_tup = tuple([axes])  # type: ignore

    if len(axes_tup) != x.ndim:
        raise ParameterError(
            f"Shape mismatch between axes={axes_tup} and input x.shape={x.shape}"
        )

    if ndim < x.ndim:
        raise ParameterError(
            f"Cannot expand x.shape={x.shape} to fewer dimensions ndim={ndim}"
        )

    shape: List[int] = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def fix_length(
    data: np.ndarray, *, size: int, axis: int = -1, **kwargs: Any
) -> np.ndarray:
    """Fix the length an array ``data`` to exactly ``size`` along a target axis.

    If ``data.shape[axis] < n``, pad according to the provided kwargs.
    By default, ``data`` is padded with trailing zeros.

    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, size=10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, size=5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, size=10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    Parameters
    ----------
    data : np.ndarray
        array to be length-adjusted
    size : int >= 0 [scalar]
        desired length of the array
    axis : int, <= data.ndim
        axis along which to fix length
    **kwargs : additional keyword arguments
        Parameters to ``np.pad``

    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        ``data`` either trimmed or padded to length ``size``
        along the specified axis.

    See Also
    --------
    numpy.pad
    """
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def fix_frames(
    frames: _SequenceLike[int],
    *,
    x_min: Optional[int] = 0,
    x_max: Optional[int] = None,
    pad: bool = True,
) -> np.ndarray:
    """Fix a list of frames to lie within [x_min, x_max]

    Examples
    --------
    >>> # Generate a list of frame indices
    >>> frames = np.arange(0, 1000.0, 50)
    >>> frames
    array([   0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,
            400.,  450.,  500.,  550.,  600.,  650.,  700.,  750.,
            800.,  850.,  900.,  950.])
    >>> # Clip to span at most 250
    >>> librosa.util.fix_frames(frames, x_max=250)
    array([  0,  50, 100, 150, 200, 250])
    >>> # Or pad to span up to 2500
    >>> librosa.util.fix_frames(frames, x_max=2500)
    array([   0,   50,  100,  150,  200,  250,  300,  350,  400,
            450,  500,  550,  600,  650,  700,  750,  800,  850,
            900,  950, 2500])
    >>> librosa.util.fix_frames(frames, x_max=2500, pad=False)
    array([  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
           550, 600, 650, 700, 750, 800, 850, 900, 950])

    >>> # Or starting away from zero
    >>> frames = np.arange(200, 500, 33)
    >>> frames
    array([200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames, x_max=500)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497,
           500])

    Parameters
    ----------
    frames : np.ndarray [shape=(n_frames,)]
        List of non-negative frame indices
    x_min : int >= 0 or None
        Minimum allowed frame index
    x_max : int >= 0 or None
        Maximum allowed frame index
    pad : boolean
        If ``True``, then ``frames`` is expanded to span the full range
        ``[x_min, x_max]``

    Returns
    -------
    fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
        Fixed frame indices, flattened and sorted

    Raises
    ------
    ParameterError
        If ``frames`` contains negative values
    """
    frames = np.asarray(frames)

    if np.any(frames < 0):
        raise ParameterError("Negative frame index detected")

    # TODO: this whole function could be made more efficient

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((np.asarray(pad_data), frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    unique: np.ndarray = np.unique(frames).astype(int)
    return unique


@overload
def axis_sort(
    S: np.ndarray,
    *,
    axis: int = ...,
    index: Literal[False] = ...,
    value: Optional[Callable[..., Any]] = ...,
) -> np.ndarray:
    ...


@overload
def axis_sort(
    S: np.ndarray,
    *,
    axis: int = ...,
    index: Literal[True],
    value: Optional[Callable[..., Any]] = ...,
) -> Tuple[np.ndarray, np.ndarray]:
    ...


def axis_sort(
    S: np.ndarray,
    *,
    axis: int = -1,
    index: bool = False,
    value: Optional[Callable[..., Any]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Sort an array along its rows or columns.

    Examples
    --------
    Visualize NMF output for a spectrogram S

    >>> # Sort the columns of W by peak frequency bin
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> W, H = librosa.decompose.decompose(S, n_components=64)
    >>> W_sort = librosa.util.axis_sort(W)

    Or sort by the lowest frequency bin

    >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)

    Or sort the rows instead of the columns

    >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)

    Get the sorting index also, and use it to permute the rows of H

    >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
    >>> H_sort = H[idx, :]

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2)
    >>> img_w = librosa.display.specshow(librosa.amplitude_to_db(W, ref=np.max),
    ...                                  y_axis='log', ax=ax[0, 0])
    >>> ax[0, 0].set(title='W')
    >>> ax[0, 0].label_outer()
    >>> img_act = librosa.display.specshow(H, x_axis='time', ax=ax[0, 1])
    >>> ax[0, 1].set(title='H')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(W_sort,
    ...                                                  ref=np.max),
    ...                          y_axis='log', ax=ax[1, 0])
    >>> ax[1, 0].set(title='W sorted')
    >>> librosa.display.specshow(H_sort, x_axis='time', ax=ax[1, 1])
    >>> ax[1, 1].set(title='H sorted')
    >>> ax[1, 1].label_outer()
    >>> fig.colorbar(img_w, ax=ax[:, 0], orientation='horizontal')
    >>> fig.colorbar(img_act, ax=ax[:, 1], orientation='horizontal')

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        Array to be sorted

    axis : int [scalar]
        The axis along which to compute the sorting values

        - ``axis=0`` to sort rows by peak column index
        - ``axis=1`` to sort columns by peak row index

    index : boolean [scalar]
        If true, returns the index array as well as the permuted data.

    value : function
        function to return the index corresponding to the sort order.
        Default: `np.argmax`.

    Returns
    -------
    S_sort : np.ndarray [shape=(d, n)]
        ``S`` with the columns or rows permuted in sorting order
    idx : np.ndarray (optional) [shape=(d,) or (n,)]
        If ``index == True``, the sorting index used to permute ``S``.
        Length of ``idx`` corresponds to the selected ``axis``.

    Raises
    ------
    ParameterError
        If ``S`` does not have exactly 2 dimensions (``S.ndim != 2``)
    """
    if value is None:
        value = np.argmax

    if S.ndim != 2:
        raise ParameterError("axis_sort is only defined for 2D arrays")

    bin_idx = value(S, axis=np.mod(1 - axis, S.ndim))
    idx = np.argsort(bin_idx)

    sort_slice = [slice(None)] * S.ndim
    sort_slice[axis] = idx  # type: ignore

    if index:
        return S[tuple(sort_slice)], idx
    else:
        return S[tuple(sort_slice)]


@cache(level=40)
def normalize(
    S: np.ndarray,
    *,
    norm: Optional[float] = np.inf,
    axis: Optional[int] = 0,
    threshold: Optional[_FloatLike_co] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    """Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that::

        norm(S, axis=axis) == 1

    For example, ``axis=0`` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, ``axis=1`` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    ``threshold`` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.

    Parameters
    ----------
    S : np.ndarray
        The array to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : minimum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least ``threshold`` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of ``S.dtype``.

    fill : None or bool
        If None, then columns (or rows) with norm below ``threshold``
        are left as is.

        If False, then columns (rows) with norm below ``threshold``
        are set to 0.

        If True, then columns (rows) with norm below ``threshold``
        are filled uniformly such that the corresponding norm is 1.

        .. note:: ``fill=True`` is incompatible with ``norm=0`` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ParameterError
        If ``norm`` is not among the valid types defined above

        If ``S`` is not finite

        If ``fill=True`` and ``norm=0``

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    """
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise ParameterError(f"threshold={threshold} must be strictly positive")

    if fill not in [None, False, True]:
        raise ParameterError(f"fill={fill} must be None or boolean")

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S

    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    else:
        raise ParameterError(f"Unsupported norm: {repr(norm)}")

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


@numba.stencil
def _localmax_sten(x):  # pragma: no cover
    """Numba stencil for local maxima computation"""
    return (x[0] > x[-1]) & (x[0] >= x[1])


@numba.stencil
def _localmin_sten(x):  # pragma: no cover
    """Numba stencil for local minima computation"""
    return (x[0] < x[-1]) & (x[0] <= x[1])


@numba.guvectorize(
    [
        "void(int16[:], bool_[:])",
        "void(int32[:], bool_[:])",
        "void(int64[:], bool_[:])",
        "void(float32[:], bool_[:])",
        "void(float64[:], bool_[:])",
    ],
    "(n)->(n)",
    cache=True,
    nopython=True,
)
def _localmax(x, y):  # pragma: no cover
    """Vectorized wrapper for the localmax stencil"""
    y[:] = _localmax_sten(x)


@numba.guvectorize(
    [
        "void(int16[:], bool_[:])",
        "void(int32[:], bool_[:])",
        "void(int64[:], bool_[:])",
        "void(float32[:], bool_[:])",
        "void(float64[:], bool_[:])",
    ],
    "(n)->(n)",
    cache=True,
    nopython=True,
)
def _localmin(x, y):  # pragma: no cover
    """Vectorized wrapper for the localmin stencil"""
    y[:] = _localmin_sten(x)


def localmax(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Find local maxima in an array

    An element ``x[i]`` is considered a local maximum if the following
    conditions are met:

    - ``x[i] > x[i-1]``
    - ``x[i] >= x[i+1]``

    Note that the first condition is strict, and that the first element
    ``x[0]`` will never be considered as a local maximum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmax(x, axis=0)
    array([[False, False, False],
           [ True, False, False],
           [False,  True,  True]], dtype=bool)
    >>> librosa.util.localmax(x, axis=1)
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True]], dtype=bool)

    Parameters
    ----------
    x : np.ndarray [shape=(d1,d2,...)]
        input vector or array
    axis : int
        axis along which to compute local maximality

    Returns
    -------
    m : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along ``axis``

    See Also
    --------
    localmin
    """
    # Rotate the target axis to the end
    xi = x.swapaxes(-1, axis)

    # Allocate the output array and rotate target axis
    lmax = np.empty_like(x, dtype=bool)
    lmaxi = lmax.swapaxes(-1, axis)

    # Call the vectorized stencil
    _localmax(xi, lmaxi)

    # Handle the edge condition not covered by the stencil
    lmaxi[..., -1] = xi[..., -1] > xi[..., -2]

    return lmax


def localmin(x: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Find local minima in an array

    An element ``x[i]`` is considered a local minimum if the following
    conditions are met:

    - ``x[i] < x[i-1]``
    - ``x[i] <= x[i+1]``

    Note that the first condition is strict, and that the first element
    ``x[0]`` will never be considered as a local minimum.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmin(x)
    array([False,  True, False, False,  True, False,  True, False])

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmin(x, axis=0)
    array([[False, False, False],
           [False,  True,  True],
           [False, False, False]])

    >>> librosa.util.localmin(x, axis=1)
    array([[False,  True, False],
           [False,  True, False],
           [False,  True, False]])

    Parameters
    ----------
    x : np.ndarray [shape=(d1,d2,...)]
        input vector or array
    axis : int
        axis along which to compute local minimality

    Returns
    -------
    m : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local minimality along ``axis``

    See Also
    --------
    localmax
    """
    # Rotate the target axis to the end
    xi = x.swapaxes(-1, axis)

    # Allocate the output array and rotate target axis
    lmin = np.empty_like(x, dtype=bool)
    lmini = lmin.swapaxes(-1, axis)

    # Call the vectorized stencil
    _localmin(xi, lmini)

    # Handle the edge condition not covered by the stencil
    lmini[..., -1] = xi[..., -1] < xi[..., -2]

    return lmin



@numba.guvectorize(
    [
        "void(float32[:], uint32, uint32, uint32, uint32, float32, uint32, bool_[:])",
        "void(float64[:], uint32, uint32, uint32, uint32, float32, uint32, bool_[:])",
        "void(int32[:], uint32, uint32, uint32, uint32, float32, uint32, bool_[:])",
        "void(int64[:], uint32, uint32, uint32, uint32, float32, uint32, bool_[:])",
    ],
    "(n),(),(),(),(),(),()->(n)",
    nopython=True, cache=True)
def __peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait, peaks):
    """Vectorized wrapper for the peak-picker"""
    # Special case the first frame
    peaks[0] = (x[0] >= np.max(x[:min(post_max, x.shape[0])]))
    peaks[0] &= (x[0] >= np.mean(x[:min(post_avg, x.shape[0])]) + delta)

    if peaks[0]:
        n = wait + 1
    else:
        n = 1

    while n < x.shape[0]:
        maxn = np.max( x[max(0, n-pre_max):min(n+post_max, x.shape[0])])

        # Are we the local max and sufficiently above average?
        peaks[n] = (x[n] == maxn) 
        
        if not peaks[n]:
            n += 1
            continue

        avgn = np.mean(x[max(0, n-pre_avg):min(n+post_avg, x.shape[0])])
        peaks[n] &= (x[n] >= avgn + delta)

        if not peaks[n]:
            n += 1
            continue

        # Skip the next `wait` frames
        n += wait + 1


def peak_pick(
    x: np.ndarray,
    *,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    delta: float,
    wait: int,
    sparse: bool = True,
    axis: int = -1
) -> np.ndarray:
    """Use a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding ``x[n]``
    fulfills the following three conditions:

    1. ``x[n] == max(x[n - pre_max:n + post_max])``
    2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
    3. ``n - previous_n > wait``

    where ``previous_n`` is the last sample picked as a peak (greedily).

    This implementation is based on [#]_ and [#]_.

    .. [#] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [#] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

    Parameters
    ----------
    x : np.ndarray
        input signal to peak picks from
    pre_max : int >= 0 [scalar]
        number of samples before ``n`` over which max is computed
    post_max : int >= 1 [scalar]
        number of samples after ``n`` over which max is computed
    pre_avg : int >= 0 [scalar]
        number of samples before ``n`` over which mean is computed
    post_avg : int >= 1 [scalar]
        number of samples after ``n`` over which mean is computed
    delta : float >= 0 [scalar]
        threshold offset for mean
    wait : int >= 0 [scalar]
        number of samples to wait after picking a peak
    sparse : bool [scalar]
        If `True`, the output are indices of detected peaks.
        If `False`, the output is a dense boolean array of the same
        shape as ``x``.
    axis : int [scalar]
        the axis over which to detect peaks.

    Returns
    -------
    peaks : np.ndarray [shape=(n_peaks,) or shape=x.shape, dtype=int or bool]
        indices of peaks in ``x`` (sparse=True)
        or a boolean array where `peaks[..., n]` indicates a peak at frame index `n` (sparse=False)

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    >>> peaks
    array([  3,  27,  40,  61,  72,  88, 103])

    Using dense output to make a boolean array of peak indicators
    >>> librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
    ...                        delta=0.5, wait=10, sparse=False)
    array([False, False, ..., False, False])

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> D = np.abs(librosa.stft(y))
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
    >>> ax[0].vlines(times[peaks], 0,
    ...              onset_env.max(), color='r', alpha=0.8,
    ...              label='Selected peaks')
    >>> ax[0].legend(frameon=True, framealpha=0.8)
    >>> ax[0].label_outer()
    """
    if pre_max < 0:
        raise ParameterError("pre_max must be non-negative")
    if pre_avg < 0:
        raise ParameterError("pre_avg must be non-negative")
    if delta < 0:
        raise ParameterError("delta must be non-negative")
    if wait < 0:
        raise ParameterError("wait must be non-negative")
    if post_max <= 0:
        raise ParameterError("post_max must be positive")
    if post_avg <= 0:
        raise ParameterError("post_avg must be positive")
    if sparse and x.ndim != 1:
        raise ParameterError(f"sparse=True (default) does not support "
                f"{x.ndim}-dimensional inputs. "
                f"Either set sparse=False or process each dimension independently.")

    # Ensure valid index types
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)

    peaks = np.zeros_like(x, dtype=bool)
    __peak_pick(x.swapaxes(axis, -1), pre_max, post_max, pre_avg, post_avg, delta, wait, peaks.swapaxes(axis, -1))

    if sparse:
        return np.flatnonzero(peaks)

    return peaks


@cache(level=40)
def sparsify_rows(
    x: np.ndarray, *, quantile: float = 0.01, dtype: Optional[DTypeLike] = None
) -> scipy.sparse.csr_matrix:
    """Return a row-sparse matrix approximating the input

    Parameters
    ----------
    x : np.ndarray [ndim <= 2]
        The input matrix to sparsify.
    quantile : float in [0, 1.0)
        Percentage of magnitude to discard in each row of ``x``
    dtype : np.dtype, optional
        The dtype of the output array.
        If not provided, then ``x.dtype`` will be used.

    Returns
    -------
    x_sparse : ``scipy.sparse.csr_matrix`` [shape=x.shape]
        Row-sparsified approximation of ``x``

        If ``x.ndim == 1``, then ``x`` is interpreted as a row vector,
        and ``x_sparse.shape == (1, len(x))``.

    Raises
    ------
    ParameterError
        If ``x.ndim > 2``

        If ``quantile`` lies outside ``[0, 1.0)``

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct a Hann window to sparsify
    >>> x = scipy.signal.hann(32)
    >>> x
    array([ 0.   ,  0.01 ,  0.041,  0.09 ,  0.156,  0.236,  0.326,
            0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
            0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
            0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
            0.09 ,  0.041,  0.01 ,  0.   ])
    >>> # Discard the bottom percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.01)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 26 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.09 ,  0.156,  0.236,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
              0.09 ,  0.   ,  0.   ,  0.   ]])
    >>> # Discard up to the bottom 10th percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.1)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 20 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.   ,  0.   ,
              0.   ,  0.   ,  0.   ,  0.   ]])
    """
    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError(
            f"Input must have 2 or fewer dimensions. Provided x.shape={x.shape}."
        )

    if not 0.0 <= quantile < 1:
        raise ParameterError(f"Invalid quantile {quantile:.2f}")

    if dtype is None:
        dtype = x.dtype

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def buf_to_float(
    x: np.ndarray, *, n_bytes: int = 2, dtype: DTypeLike = np.float32
) -> np.ndarray:
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``
    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """
    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = f"<i{n_bytes:d}"

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def index_to_slice(
    idx: _SequenceLike[int],
    *,
    idx_min: Optional[int] = None,
    idx_max: Optional[int] = None,
    step: Optional[int] = None,
    pad: bool = True,
) -> List[slice]:
    """Generate a slice array from an index array.

    Parameters
    ----------
    idx : list-like
        Array of index boundaries
    idx_min, idx_max : None or int
        Minimum and maximum allowed indices
    step : None or int
        Step size for each slice.  If `None`, then the default
        step of 1 is used.
    pad : boolean
        If `True`, pad ``idx`` to span the range ``idx_min:idx_max``.

    Returns
    -------
    slices : list of slice
        ``slices[i] = slice(idx[i], idx[i+1], step)``
        Additional slice objects may be added at the beginning or end,
        depending on whether ``pad==True`` and the supplied values for
        ``idx_min`` and ``idx_max``.

    See Also
    --------
    fix_frames

    Examples
    --------
    >>> # Generate slices from spaced indices
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15))
    [slice(20, 35, None), slice(35, 50, None), slice(50, 65, None), slice(65, 80, None),
     slice(80, 95, None)]
    >>> # Pad to span the range (0, 100)
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100)
    [slice(0, 20, None), slice(20, 35, None), slice(35, 50, None), slice(50, 65, None),
     slice(65, 80, None), slice(80, 95, None), slice(95, 100, None)]
    >>> # Use a step of 5 for each slice
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100, step=5)
    [slice(0, 20, 5), slice(20, 35, 5), slice(35, 50, 5), slice(50, 65, 5), slice(65, 80, 5),
     slice(80, 95, 5), slice(95, 100, 5)]
    """
    # First, normalize the index set
    idx_fixed = fix_frames(idx, x_min=idx_min, x_max=idx_max, pad=pad)

    # Now convert the indices to slices
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]


@cache(level=40)
def sync(
    data: np.ndarray,
    idx: Union[Sequence[int], Sequence[slice]],
    *,
    aggregate: Optional[Callable[..., Any]] = None,
    pad: bool = True,
    axis: int = -1,
) -> np.ndarray:
    """Aggregate a multi-dimensional array between specified boundaries.

    .. note::
        In order to ensure total coverage, boundary points may be added
        to ``idx``.

        If synchronizing a feature matrix against beat tracker output, ensure
        that frame index numbers are properly aligned and use the same hop length.

    Parameters
    ----------
    data : np.ndarray
        multi-dimensional array of features
    idx : sequence of ints or slices
        Either an ordered array of boundary indices, or
        an iterable collection of slice objects.
    aggregate : function
        aggregation function (default: `np.mean`)
    pad : boolean
        If `True`, ``idx`` is padded to span the full range ``[0, data.shape[axis]]``
    axis : int
        The axis along which to aggregate data

    Returns
    -------
    data_sync : ndarray
        ``data_sync`` will have the same dimension as ``data``, except that the ``axis``
        coordinate will be reduced according to ``idx``.

        For example, a 2-dimensional ``data`` with ``axis=-1`` should satisfy::

            data_sync[:, i] = aggregate(data[:, idx[i-1]:idx[i]], axis=-1)

    Raises
    ------
    ParameterError
        If the index set is not of consistent type (all slices or all integers)

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Beat-synchronous CQT spectra

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> beats = librosa.util.fix_frames(beats)

    By default, use mean aggregation

    >>> C_avg = librosa.util.sync(C, beats)

    Use median-aggregation instead of mean

    >>> C_med = librosa.util.sync(C, beats,
    ...                              aggregate=np.median)

    Or sub-beat synchronization

    >>> sub_beats = librosa.segment.subsegment(C, beats)
    >>> sub_beats = librosa.util.fix_frames(sub_beats)
    >>> C_med_sub = librosa.util.sync(C, sub_beats, aggregate=np.median)

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> beat_t = librosa.frames_to_time(beats, sr=sr)
    >>> subbeat_t = librosa.frames_to_time(sub_beats, sr=sr)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                  ref=np.max),
    ...                          x_axis='time', ax=ax[0])
    >>> ax[0].set(title='CQT power, shape={}'.format(C.shape))
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med,
    ...                                                  ref=np.max),
    ...                          x_coords=beat_t, x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Beat synchronous CQT power, '
    ...                 'shape={}'.format(C_med.shape))
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med_sub,
    ...                                                  ref=np.max),
    ...                          x_coords=subbeat_t, x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Sub-beat synchronous CQT power, '
    ...                 'shape={}'.format(C_med_sub.shape))
    """
    if aggregate is None:
        aggregate = np.mean

    shape = list(data.shape)

    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.integer) for _ in idx]):
        slices = index_to_slice(
            np.asarray(idx), idx_min=0, idx_max=shape[axis], pad=pad
        )
    else:
        raise ParameterError(f"Invalid index set: {idx}")

    agg_shape = list(shape)
    agg_shape[axis] = len(slices)

    data_agg = np.empty(
        agg_shape, order="F" if np.isfortran(data) else "C", dtype=data.dtype
    )

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for i, segment in enumerate(slices):
        idx_in[axis] = segment  # type: ignore
        idx_agg[axis] = i  # type: ignore
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)

    return data_agg


def softmask(
    X: np.ndarray, X_ref: np.ndarray, *, power: float = 1, split_zeros: bool = False
) -> np.ndarray:
    """Robustly compute a soft-mask operation.

        ``M = X**power / (X**power + X_ref**power)``

    Parameters
    ----------
    X : np.ndarray
        The (non-negative) input array corresponding to the positive mask elements

    X_ref : np.ndarray
        The (non-negative) array of reference or background elements.
        Must have the same shape as ``X``.

    power : number > 0 or np.inf
        If finite, returns the soft mask computed in a numerically stable way

        If infinite, returns a hard (binary) mask equivalent to ``X > X_ref``.
        Note: for hard masks, ties are always broken in favor of ``X_ref`` (``mask=0``).

    split_zeros : bool
        If `True`, entries where ``X`` and ``X_ref`` are both small (close to 0)
        will receive mask values of 0.5.

        Otherwise, the mask is set to 0 for these entries.

    Returns
    -------
    mask : np.ndarray, shape=X.shape
        The output mask array

    Raises
    ------
    ParameterError
        If ``X`` and ``X_ref`` have different shapes.

        If ``X`` or ``X_ref`` are negative anywhere

        If ``power <= 0``

    Examples
    --------
    >>> X = 2 * np.ones((3, 3))
    >>> X_ref = np.vander(np.arange(3.0))
    >>> X
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]])
    >>> X_ref
    array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]])
    >>> librosa.util.softmask(X, X_ref, power=1)
    array([[ 1.   ,  1.   ,  0.667],
           [ 0.667,  0.667,  0.667],
           [ 0.333,  0.5  ,  0.667]])
    >>> librosa.util.softmask(X_ref, X, power=1)
    array([[ 0.   ,  0.   ,  0.333],
           [ 0.333,  0.333,  0.333],
           [ 0.667,  0.5  ,  0.333]])
    >>> librosa.util.softmask(X, X_ref, power=2)
    array([[ 1. ,  1. ,  0.8],
           [ 0.8,  0.8,  0.8],
           [ 0.2,  0.5,  0.8]])
    >>> librosa.util.softmask(X, X_ref, power=4)
    array([[ 1.   ,  1.   ,  0.941],
           [ 0.941,  0.941,  0.941],
           [ 0.059,  0.5  ,  0.941]])
    >>> librosa.util.softmask(X, X_ref, power=100)
    array([[  1.000e+00,   1.000e+00,   1.000e+00],
           [  1.000e+00,   1.000e+00,   1.000e+00],
           [  7.889e-31,   5.000e-01,   1.000e+00]])
    >>> librosa.util.softmask(X, X_ref, power=np.inf)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [False, False,  True]], dtype=bool)
    """
    if X.shape != X_ref.shape:
        raise ParameterError(f"Shape mismatch: {X.shape}!={X_ref.shape}")

    if np.any(X < 0) or np.any(X_ref < 0):
        raise ParameterError("X and X_ref must be non-negative")

    if power <= 0:
        raise ParameterError("power must be strictly positive")

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = Z < np.finfo(dtype).tiny
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    mask: np.ndarray

    if np.isfinite(power):
        mask = (X / Z) ** power
        ref_mask = (X_ref / Z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask


def tiny(x: Union[float, np.ndarray]) -> _FloatLike_co:
    """Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in ``x.dtype``
    (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is ``x.dtype``

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of ``x``.
        If ``x`` is integer-typed, then the tiny value for ``np.float32``
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------
    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    """
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny


def fill_off_diagonal(x: np.ndarray, *, radius: float, value: float = 0) -> None:
    """Set all cells of a matrix to a given ``value``
    if they lie outside a constraint region.

    In this case, the constraint region is the
    Sakoe-Chiba band which runs with a fixed ``radius``
    along the main diagonal.

    When ``x.shape[0] != x.shape[1]``, the radius will be
    expanded so that ``x[-1, -1] = 1`` always.

    ``x`` will be modified in place.

    Parameters
    ----------
    x : np.ndarray [shape=(N, M)]
        Input matrix, will be modified in place.
    radius : float
        The band radius (1/2 of the width) will be
        ``int(radius*min(x.shape))``
    value : float
        ``x[n, m] = value`` when ``(n, m)`` lies outside the band.

    Examples
    --------
    >>> x = np.ones((8, 8))
    >>> librosa.util.fill_off_diagonal(x, radius=0.25)
    >>> x
    array([[1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> x = np.ones((8, 12))
    >>> librosa.util.fill_off_diagonal(x, radius=0.25)
    >>> x
    array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    """
    nx, ny = x.shape

    # Calculate the radius in indices, rather than proportion
    radius = int(np.round(radius * np.min(x.shape)))

    nx, ny = x.shape
    offset = np.abs((x.shape[0] - x.shape[1]))

    if nx < ny:
        idx_u = np.triu_indices_from(x, k=radius + offset)
        idx_l = np.tril_indices_from(x, k=-radius)
    else:
        idx_u = np.triu_indices_from(x, k=radius)
        idx_l = np.tril_indices_from(x, k=-radius - offset)

    # modify input matrix
    x[idx_u] = value
    x[idx_l] = value


def cyclic_gradient(
    data: np.ndarray, *, edge_order: Literal[1, 2] = 1, axis: int = -1
) -> np.ndarray:
    """Estimate the gradient of a function over a uniformly sampled,
    periodic domain.

    This is essentially the same as `np.gradient`, except that edge effects
    are handled by wrapping the observations (i.e. assuming periodicity)
    rather than extrapolation.

    Parameters
    ----------
    data : np.ndarray
        The function values observed at uniformly spaced positions on
        a periodic domain
    edge_order : {1, 2}
        The order of the difference approximation used for estimating
        the gradient
    axis : int
        The axis along which gradients are calculated.

    Returns
    -------
    grad : np.ndarray like ``data``
        The gradient of ``data`` taken along the specified axis.

    See Also
    --------
    numpy.gradient

    Examples
    --------
    This example estimates the gradient of cosine (-sine) from 64
    samples using direct (aperiodic) and periodic gradient
    calculation.

    >>> import matplotlib.pyplot as plt
    >>> x = 2 * np.pi * np.linspace(0, 1, num=64, endpoint=False)
    >>> y = np.cos(x)
    >>> grad = np.gradient(y)
    >>> cyclic_grad = librosa.util.cyclic_gradient(y)
    >>> true_grad = -np.sin(x) * 2 * np.pi / len(x)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, true_grad, label='True gradient', linewidth=5,
    ...          alpha=0.35)
    >>> ax.plot(x, cyclic_grad, label='cyclic_gradient')
    >>> ax.plot(x, grad, label='np.gradient', linestyle=':')
    >>> ax.legend()
    >>> # Zoom into the first part of the sequence
    >>> ax.set(xlim=[0, np.pi/16], ylim=[-0.025, 0.025])
    """
    # Wrap-pad the data along the target axis by `edge_order` on each side
    padding = [(0, 0)] * data.ndim
    padding[axis] = (edge_order, edge_order)
    data_pad = np.pad(data, padding, mode="wrap")

    # Compute the gradient
    grad = np.gradient(data_pad, edge_order=edge_order, axis=axis)

    # Remove the padding
    slices = [slice(None)] * data.ndim
    slices[axis] = slice(edge_order, -edge_order)
    grad_slice: np.ndarray = grad[tuple(slices)]
    return grad_slice


@numba.jit(nopython=True, cache=True)  # type: ignore
def __shear_dense(X: np.ndarray, *, factor: int = +1, axis: int = -1) -> np.ndarray:
    """Numba-accelerated shear for dense (ndarray) arrays"""
    if axis == 0:
        X = X.T

    X_shear = np.empty_like(X)

    for i in range(X.shape[1]):
        X_shear[:, i] = np.roll(X[:, i], factor * i)

    if axis == 0:
        X_shear = X_shear.T

    return X_shear


def __shear_sparse(
    X: scipy.sparse.spmatrix, *, factor: int = +1, axis: int = -1
) -> scipy.sparse.spmatrix:
    """Fast shearing for sparse matrices

    Shearing is performed using CSC array indices,
    and the result is converted back to whatever sparse format
    the data was originally provided in.
    """
    fmt = X.format
    if axis == 0:
        X = X.T

    # Now we're definitely rolling on the correct axis
    X_shear = X.tocsc(copy=True)

    # The idea here is to repeat the shear amount (factor * range)
    # by the number of non-zeros for each column.
    # The number of non-zeros is computed by diffing the index pointer array
    roll = np.repeat(factor * np.arange(X_shear.shape[1]), np.diff(X_shear.indptr))

    # In-place roll
    np.mod(X_shear.indices + roll, X_shear.shape[0], out=X_shear.indices)

    if axis == 0:
        X_shear = X_shear.T

    # And convert back to the input format
    return X_shear.asformat(fmt)


_ArrayOrSparseMatrix = TypeVar(
    "_ArrayOrSparseMatrix", bound=Union[np.ndarray, scipy.sparse.spmatrix]
)


@overload
def shear(X: np.ndarray, *, factor: int = ..., axis: int = ...) -> np.ndarray:
    ...


@overload
def shear(
    X: scipy.sparse.spmatrix, *, factor: int = ..., axis: int = ...
) -> scipy.sparse.spmatrix:
    ...


def shear(
    X: _ArrayOrSparseMatrix, *, factor: int = 1, axis: int = -1
) -> _ArrayOrSparseMatrix:
    """Shear a matrix by a given factor.

    The column ``X[:, n]`` will be displaced (rolled)
    by ``factor * n``

    This is primarily useful for converting between lag and recurrence
    representations: shearing with ``factor=-1`` converts the main diagonal
    to a horizontal.  Shearing with ``factor=1`` converts a horizontal to
    a diagonal.

    Parameters
    ----------
    X : np.ndarray [ndim=2] or scipy.sparse matrix
        The array to be sheared
    factor : integer
        The shear factor: ``X[:, n] -> np.roll(X[:, n], factor * n)``
    axis : integer
        The axis along which to shear

    Returns
    -------
    X_shear : same type as ``X``
        The sheared matrix

    Examples
    --------
    >>> E = np.eye(3)
    >>> librosa.util.shear(E, factor=-1, axis=-1)
    array([[1., 1., 1.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> librosa.util.shear(E, factor=-1, axis=0)
    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.]])
    >>> librosa.util.shear(E, factor=1, axis=-1)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    if not np.issubdtype(type(factor), np.integer):
        raise ParameterError(f"factor={factor} must be integer-valued")

    # Suppress type checks because mypy doesn't like numba jitting
    # or scipy sparse conversion
    if scipy.sparse.isspmatrix(X):
        return __shear_sparse(X, factor=factor, axis=axis)  # type: ignore
    else:
        return __shear_dense(X, factor=factor, axis=axis)  # type: ignore


def stack(arrays: List[np.ndarray], *, axis: int = 0) -> np.ndarray:
    """Stack one or more arrays along a target axis.

    This function is similar to `np.stack`, except that memory contiguity is
    retained when stacking along the first dimension.

    This is useful when combining multiple monophonic audio signals into a
    multi-channel signal, or when stacking multiple feature representations
    to form a multi-dimensional array.

    Parameters
    ----------
    arrays : list
        one or more `np.ndarray`
    axis : integer
        The target axis along which to stack.  ``axis=0`` creates a new first axis,
        and ``axis=-1`` creates a new last axis.

    Returns
    -------
    arr_stack : np.ndarray [shape=(len(arrays), array_shape) or shape=(array_shape, len(arrays))]
        The input arrays, stacked along the target dimension.

        If ``axis=0``, then ``arr_stack`` will be F-contiguous.
        Otherwise, ``arr_stack`` will be C-contiguous by default, as computed by
        `np.stack`.

    Raises
    ------
    ParameterError
        - If ``arrays`` do not all have the same shape
        - If no ``arrays`` are given

    See Also
    --------
    numpy.stack
    numpy.ndarray.flags
    frame

    Examples
    --------
    Combine two buffers into a contiguous arrays

    >>> y_left = np.ones(5)
    >>> y_right = -np.ones(5)
    >>> y_stereo = librosa.util.stack([y_left, y_right], axis=0)
    >>> y_stereo
    array([[ 1.,  1.,  1.,  1.,  1.],
           [-1., -1., -1., -1., -1.]])
    >>> y_stereo.flags
      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
      UPDATEIFCOPY : False

    Or along the trailing axis

    >>> y_stereo = librosa.util.stack([y_left, y_right], axis=-1)
    >>> y_stereo
    array([[ 1., -1.],
           [ 1., -1.],
           [ 1., -1.],
           [ 1., -1.],
           [ 1., -1.]])
    >>> y_stereo.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      OWNDATA : True
      WRITEABLE : True
      ALIGNED : True
      WRITEBACKIFCOPY : False
      UPDATEIFCOPY : False
    """
    shapes = {arr.shape for arr in arrays}
    if len(shapes) > 1:
        raise ParameterError("all input arrays must have the same shape")
    elif len(shapes) < 1:
        raise ParameterError("at least one input array must be provided for stack")

    shape_in = shapes.pop()

    if axis != 0:
        return np.stack(arrays, axis=axis)
    else:
        # If axis is 0, enforce F-ordering
        shape = tuple([len(arrays)] + list(shape_in))

        # Find the common dtype for all inputs
        dtype = np.result_type(*arrays) 

        # Allocate an empty array of the right shape and type
        result = np.empty(shape, dtype=dtype, order="F")

        # Stack into the preallocated buffer
        np.stack(arrays, axis=axis, out=result)

        return result


def dtype_r2c(d: DTypeLike, *, default: Optional[type] = np.complex64) -> DTypeLike:
    """Find the complex numpy dtype corresponding to a real dtype.

    This is used to maintain numerical precision and memory footprint
    when constructing complex arrays from real-valued data
    (e.g. in a Fourier transform).

    A `float32` (single-precision) type maps to `complex64`,
    while a `float64` (double-precision) maps to `complex128`.

    Parameters
    ----------
    d : np.dtype
        The real-valued dtype to convert to complex.
        If ``d`` is a complex type already, it will be returned.
    default : np.dtype, optional
        The default complex target type, if ``d`` does not match a
        known dtype

    Returns
    -------
    d_c : np.dtype
        The complex dtype

    See Also
    --------
    dtype_c2r
    numpy.dtype

    Examples
    --------
    >>> librosa.util.dtype_r2c(np.float32)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.int16)
    dtype('complex64')

    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('complex128')
    """
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def dtype_c2r(d: DTypeLike, *, default: Optional[type] = np.float32) -> DTypeLike:
    """Find the real numpy dtype corresponding to a complex dtype.

    This is used to maintain numerical precision and memory footprint
    when constructing real arrays from complex-valued data
    (e.g. in an inverse Fourier transform).

    A `complex64` (single-precision) type maps to `float32`,
    while a `complex128` (double-precision) maps to `float64`.

    Parameters
    ----------
    d : np.dtype
        The complex-valued dtype to convert to real.
        If ``d`` is a real (float) type already, it will be returned.
    default : np.dtype, optional
        The default real target type, if ``d`` does not match a
        known dtype

    Returns
    -------
    d_r : np.dtype
        The real dtype

    See Also
    --------
    dtype_r2c
    numpy.dtype

    Examples
    --------
    >>> librosa.util.dtype_r2c(np.complex64)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.float32)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.int16)
    dtype('float32')

    >>> librosa.util.dtype_r2c(np.complex128)
    dtype('float64')
    """
    mapping: Dict[DTypeLike, type] = {
        np.dtype(np.complex64): np.float32,
        np.dtype(np.complex128): np.float64,
        np.dtype(complex): np.dtype(float).type,
    }

    # If we're given a real type already, return it
    dt = np.dtype(d)
    if dt.kind == "f":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


@numba.jit(nopython=True, cache=True)
def __count_unique(x):
    """Count the number of unique values in an array.

    This function is a helper for `count_unique` and is not
    to be called directly.
    """
    uniques = np.unique(x)
    return uniques.shape[0]


def count_unique(data: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Count the number of unique values in a multi-dimensional array
    along a given axis.

    Parameters
    ----------
    data : np.ndarray
        The input array
    axis : int
        The target axis to count

    Returns
    -------
    n_uniques
        The number of unique values.
        This array will have one fewer dimension than the input.

    See Also
    --------
    is_unique

    Examples
    --------
    >>> x = np.vander(np.arange(5))
    >>> x
    array([[  0,   0,   0,   0,   1],
       [  1,   1,   1,   1,   1],
       [ 16,   8,   4,   2,   1],
       [ 81,  27,   9,   3,   1],
       [256,  64,  16,   4,   1]])
    >>> # Count unique values along rows (within columns)
    >>> librosa.util.count_unique(x, axis=0)
    array([5, 5, 5, 5, 1])
    >>> # Count unique values along columns (within rows)
    >>> librosa.util.count_unique(x, axis=-1)
    array([2, 1, 5, 5, 5])
    """
    return np.apply_along_axis(__count_unique, axis, data)


@numba.jit(nopython=True, cache=True)
def __is_unique(x):
    """Determine if the input array has all unique values.

    This function is a helper for `is_unique` and is not
    to be called directly.
    """
    uniques = np.unique(x)
    return uniques.shape[0] == x.size


def is_unique(data: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Determine if the input array consists of all unique values
    along a given axis.

    Parameters
    ----------
    data : np.ndarray
        The input array
    axis : int
        The target axis

    Returns
    -------
    is_unique
        Array of booleans indicating whether the data is unique along the chosen
        axis.
        This array will have one fewer dimension than the input.

    See Also
    --------
    count_unique

    Examples
    --------
    >>> x = np.vander(np.arange(5))
    >>> x
    array([[  0,   0,   0,   0,   1],
       [  1,   1,   1,   1,   1],
       [ 16,   8,   4,   2,   1],
       [ 81,  27,   9,   3,   1],
       [256,  64,  16,   4,   1]])
    >>> # Check uniqueness along rows
    >>> librosa.util.is_unique(x, axis=0)
    array([ True,  True,  True,  True, False])
    >>> # Check uniqueness along columns
    >>> librosa.util.is_unique(x, axis=-1)
    array([False, False,  True,  True,  True])
    """
    return np.apply_along_axis(__is_unique, axis, data)


@numba.vectorize(
    ["float32(complex64)", "float64(complex128)"], nopython=True, cache=True, identity=0
)  # type: ignore
def _cabs2(x: _ComplexLike_co) -> _FloatLike_co:  # pragma: no cover
    """Efficiently compute abs2 on complex inputs"""
    return x.real**2 + x.imag**2


_Number = Union[complex, "np.number[Any]"]
_NumberOrArray = TypeVar("_NumberOrArray", bound=Union[_Number, np.ndarray])


def abs2(x: _NumberOrArray, dtype: Optional[DTypeLike] = None) -> _NumberOrArray:
    """Compute the squared magnitude of a real or complex array.

    This function is equivalent to calling `np.abs(x)**2` but it
    is slightly more efficient.

    Parameters
    ----------
    x : np.ndarray or scalar, real or complex typed
        The input data, either real (float32, float64) or complex (complex64, complex128) typed
    dtype : np.dtype, optional
        The data type of the output array.
        If not provided, it will be inferred from `x`

    Returns
    -------
    p : np.ndarray or scale, real
        squared magnitude of `x`

    Examples
    --------
    >>> librosa.util.abs2(3 + 4j)
    25.0

    >>> librosa.util.abs2((0.5j)**np.arange(8))
    array([1.000e+00, 2.500e-01, 6.250e-02, 1.562e-02, 3.906e-03, 9.766e-04,
       2.441e-04, 6.104e-05])
    """
    if np.iscomplexobj(x):
        # suppress type check, mypy doesn't like vectorization
        y = _cabs2(x)
        if dtype is None:
            return y  # type: ignore
        else:
            return y.astype(dtype)  # type: ignore
    else:
        # suppress type check, mypy doesn't know this is real
        return np.square(x, dtype=dtype)  # type: ignore


@numba.vectorize(
    ["complex64(float32)", "complex128(float64)"], nopython=True, cache=True, identity=1
)  # type: ignore
def _phasor_angles(x) -> np.complexfloating[Any, Any]:
    return np.cos(x) + 1j * np.sin(x)  # type: ignore


_Real = Union[float, "np.integer[Any]", "np.floating[Any]"]


@overload
def phasor(angles: np.ndarray, *, mag: Optional[np.ndarray] = ...) -> np.ndarray:
    ...


@overload
def phasor(angles: _Real, *, mag: Optional[_Number] = ...) -> np.complexfloating[Any, Any]:
    ...


def phasor(
    angles: Union[np.ndarray, _Real],
    *,
    mag: Optional[Union[np.ndarray, _Number]] = None,
) -> Union[np.ndarray, np.complexfloating[Any, Any]]:
    """Construct a complex phasor representation from angles.

    When `mag` is not provided, this is equivalent to:

        z = np.cos(angles) + 1j * np.sin(angles)

    or by Euler's formula:

        z = np.exp(1j * angles)

    When `mag` is provided, this is equivalent to:

        z = mag * np.exp(1j * angles)

    This function should be more efficient (in time and memory) than the equivalent'
    formulations above, but produce numerically identical results.

    Parameters
    ----------
    angles : np.ndarray or scalar, real-valued
        Angle(s), measured in radians

    mag : np.ndarray or scalar, optional
        If provided, phasor(s) will be scaled by `mag`.

        If not provided (default), phasors will have unit magnitude.

        `mag` must be of compatible shape to multiply with `angles`.

    Returns
    -------
    z : np.ndarray or scalar, complex-valued
        Complex number(s) z corresponding to the given angle(s)
        and optional magnitude(s).

    Examples
    --------
    Construct unit phasors at angles 0, pi/2, and pi:

    >>> librosa.util.phasor([0, np.pi/2, np.pi])
    array([ 1.000e+00+0.000e+00j,  6.123e-17+1.000e+00j,
           -1.000e+00+1.225e-16j])

    Construct a phasor with magnitude 1/2:

    >>> librosa.util.phasor(np.pi/2, mag=0.5)
    (3.061616997868383e-17+0.5j)

    Or arrays of angles and magnitudes:

    >>> librosa.util.phasor(np.array([0, np.pi/2]), mag=np.array([0.5, 1.5]))
    array([5.000e-01+0.j , 9.185e-17+1.5j])
    """
    z = _phasor_angles(angles)

    if mag is not None:
        z *= mag

    return z  # type: ignore
