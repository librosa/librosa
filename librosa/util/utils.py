#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions"""

import numpy as np
import scipy.ndimage
import scipy.sparse
import six

from numpy.lib.stride_tricks import as_strided

from .. import cache
from . import decorators
from .exceptions import ParameterError

# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

SMALL_FLOAT = 1e-20

__all__ = ['MAX_MEM_BLOCK', 'SMALL_FLOAT',
           'frame', 'pad_center', 'fix_length',
           'valid_audio', 'valid_int', 'valid_intervals',
           'fix_frames',
           'axis_sort', 'localmax', 'normalize',
           'match_intervals', 'match_events',
           'peak_pick',
           'sparsify_rows',
           'buf_to_float',
           # Deprecated functions
           'buf_to_int']


def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    This implementation uses low-level stride manipulation to avoid
    redundant copies of the time series data.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `y`:
        `y_frames[i, j] == y[j * hop_length + i]`

    Raises
    ------
    ParameterError
        If `y` is not contiguous in memory, framing is invalid.
        See `np.ascontiguous()` for details.

        If `hop_length < 1`, frames cannot advance.

    Examples
    --------
    Extract 2048-sample frames from `y` with a hop of 64 samples per frame

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.frame(y, frame_length=2048, hop_length=64)
    array([[ -9.216e-06,   7.710e-06, ...,  -2.117e-06,  -4.362e-07],
           [  2.518e-06,  -6.294e-06, ...,  -1.775e-05,  -6.365e-06],
           ..., 
           [ -7.429e-04,   5.173e-03, ...,   1.105e-05,  -5.074e-06],
           [  2.169e-03,   4.867e-03, ...,   3.666e-06,  -5.571e-06]], dtype=float32)

    '''

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    valid_audio(y)

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    if n_frames < 1:
        raise ParameterError('Buffer is too short (n={:d})'
                                    ' for frame_length={:d}'.format(len(y),
                                                                    frame_length))
    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


@cache
def valid_audio(y, mono=True):
    '''Validate whether a variable contains valid, mono audio data.


    Parameters
    ----------
    y : np.ndarray
      The input data to validate

    mono : bool
      Whether or not to force monophonic audio

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        If `y` fails to meet the following criteria:
            - `type(y)` is `np.ndarray`
            - `mono == True` and `y.ndim` is not 1
            - `mono == False` and `y.ndim` is not 1 or 2
            - `np.isfinite(y).all()` is not True

    Examples
    --------
    >>> # Only allow monophonic signals
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.util.valid_audio(y)
    True

    >>> # If we want to allow stereo signals
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    >>> librosa.util.valid_audio(y, mono=False)
    True
    '''

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                                    'ndim={:d}, shape={}'.format(y.ndim,
                                                                 y.shape))
    elif y.ndim > 2:
        raise ParameterError('Invalid shape for audio: '
                                    'ndim={:d}, shape={}'.format(y.ndim,
                                                                 y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True


def valid_int(x, cast=None):
    '''Ensure that an input value is integer-typed.
    This is primarily useful for ensuring integrable-valued
    array indices.

    Parameters
    ----------
    x : number
        A scalar value to be cast to int

    cast : function [optional]
        A function to modify `x` before casting.
        Default: `np.floor`

    Returns
    -------
    x_int : int
        `x_int = int(cast(x))`

    Raises
    ------
    ParameterError
        If `cast` is provided and is not callable.
    '''

    if cast is None:
        cast = np.floor

    if not six.callable(cast):
        raise ParameterError('cast parameter must be callable')

    return int(cast(x))


def valid_intervals(intervals):
    '''Ensure that an array is a valid representation of time intervals:

        - intervals.ndim == 2
        - intervals.shape[1] == 2

    Parameters
    ----------
    intervals : np.ndarray [shape=(n, 2)]
        set of time intervals

    Returns
    -------
    valid : bool
        True if `intervals` passes validation.
    '''

    if intervals.ndim != 2 or intervals.shape[-1] != 2:
        raise ParameterError('intervals must have shape (n, 2)')

    return True


@cache
def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, size - n - lpad)

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                                     'at least input size ({:d})').format(size,
                                                                          n))

    return np.pad(data, lengths, **kwargs)


@cache
def fix_length(data, size, axis=-1, **kwargs):
    '''Fix the length an array `data` to exactly `size`.

    If `data.shape[axis] < n`, pad according to the provided kwargs.
    By default, `data` is padded with trailing zeros.

    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted

    size : int >= 0 [scalar]
      desired length of the array

    axis : int, <= data.ndim
      axis along which to fix length

    kwargs : additional keyword arguments
        Parameters to `np.pad()`

    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        `data` either trimmed or padded to length `size`
        along the specified axis.

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def fix_frames(frames, x_min=0, x_max=None, pad=True):
    '''Fix a list of frames to lie within [x_min, x_max]

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
        If `True`, then `frames` is expanded to span the full range
        `[x_min, x_max]`

    Returns
    -------
    fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
        Fixed frame indices, flattened and sorted

    Raises
    ------
    ParameterError
        If `frames` contains negative values
    '''

    if np.any(frames < 0):
        raise ParameterError('Negative frame index detected')

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)


@cache
def axis_sort(S, axis=-1, index=False, value=None):
    '''Sort an array along its rows or columns.

    Examples
    --------
    Visualize NMF output for a spectrogram S

    >>> # Sort the columns of W by peak frequency bin
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> W, H = librosa.decompose.decompose(S, n_components=32)
    >>> W_sort = librosa.util.axis_sort(W)

    Or sort by the lowest frequency bin

    >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)

    Or sort the rows instead of the columns

    >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)

    Get the sorting index also, and use it to permute the rows of H

    >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
    >>> H_sort = H[idx, :]

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 2, 1)
    >>> librosa.display.specshow(librosa.logamplitude(W**2, ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.title('W')
    >>> plt.subplot(2, 2, 2)
    >>> librosa.display.specshow(H, x_axis='time')
    >>> plt.title('H')
    >>> plt.subplot(2, 2, 3)
    >>> librosa.display.specshow(librosa.logamplitude(W_sort**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.title('W sorted')
    >>> plt.subplot(2, 2, 4)
    >>> librosa.display.specshow(H_sort, x_axis='time')
    >>> plt.title('H sorted')
    >>> plt.tight_layout()


    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        Array to be sorted

    axis : int [scalar]
        The axis along which to compute the sorting values

        - `axis=0` to sort rows by peak column index
        - `axis=1` to sort columns by peak row index

    index : boolean [scalar]
        If true, returns the index array as well as the permuted data.

    value : function
        function to return the index corresponding to the sort order.
        Default: `np.argmax`.

    Returns
    -------
    S_sort : np.ndarray [shape=(d, n)]
        `S` with the columns or rows permuted in sorting order

    idx : np.ndarray (optional) [shape=(d,) or (n,)]
        If `index == True`, the sorting index used to permute `S`.
        Length of `idx` corresponds to the selected `axis`.

    Raises
    ------
    ParameterError
        If `S` does not have exactly 2 dimensions (`S.ndim != 2`)
    '''

    if value is None:
        value = np.argmax

    if S.ndim != 2:
        raise ParameterError('axis_sort is only defined for 2D arrays')

    bin_idx = value(S, axis=np.mod(1-axis, S.ndim))
    idx = np.argsort(bin_idx)

    sort_slice = [slice(None)] * S.ndim
    sort_slice[axis] = idx

    if index:
        return S[sort_slice], idx
    else:
        return S[sort_slice]


@cache
def normalize(S, norm=np.inf, axis=0):
    '''Normalize the columns or rows of a matrix

    .. note::
         Columns/rows with length 0 will be left as zeros.

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

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        The matrix to normalize

    norm : {inf, -inf, 0, float > 0, None}
        - `inf`  : maximum absolute value
        - `-inf` : mininum absolute value
        - `0`    : number of non-zeros
        - float  : corresponding l_p norm.
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.
        `axis=0` will normalize columns, `axis=1` will normalize rows.
        `axis=None` will normalize according to the entire matrix.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized matrix

    Raises
    ------
    ParameterError
        If `norm` is not among the valid types defined above
    '''

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S)

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        length = np.sum(mag > 0, axis=axis, keepdims=True)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True)**(1./norm)

    elif norm is None:
        return S

    else:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Avoid div-by-zero
    length[length < SMALL_FLOAT] = 1.0

    return S / length


@cache
def match_intervals(intervals_from, intervals_to):
    '''Match one set of time intervals to another.

    This can be useful for tasks such as mapping beat timings
    to segments.

    .. note:: A target interval may be matched to multiple source
      intervals.

    Parameters
    ----------
    intervals_from : np.ndarray [shape=(n, 2)]
        The time range for source intervals.
        The `i` th interval spans time `intervals_from[i, 0]`
        to `intervals_from[i, 1]`.
        `intervals_from[0, 0]` should be 0, `intervals_from[-1, 1]`
        should be the track duration.

    intervals_to : np.ndarray [shape=(m, 2)]
        Analogous to `intervals_from`.

    Returns
    -------
    interval_mapping : np.ndarray [shape=(n,)]
        For each interval in `intervals_from`, the
        corresponding interval in `intervals_to`.

    See Also
    --------
    match_events

    Raises
    ------
    ParameterError
        If either array of input intervals is not the correct shape
    '''

    if len(intervals_from) == 0 or len(intervals_to) == 0:
        raise ParameterError('Attempting to match empty interval list')

    # Verify that the input intervals has correct shape and size
    valid_intervals(intervals_from)
    valid_intervals(intervals_to)

    # The overlap score of a beat with a segment is defined as
    #   max(0, min(beat_end, segment_end) - max(beat_start, segment_start))
    output = np.empty(len(intervals_from), dtype=np.int)

    n_rows = int(MAX_MEM_BLOCK / (len(intervals_to) * intervals_to.itemsize))
    n_rows = max(1, n_rows)

    for bl_s in range(0, len(intervals_from), n_rows):
        bl_t = min(bl_s + n_rows, len(intervals_from))
        tmp_from = intervals_from[bl_s:bl_t]

        starts = np.maximum.outer(tmp_from[:, 0], intervals_to[:, 0])
        ends = np.minimum.outer(tmp_from[:, 1], intervals_to[:, 1])
        score = np.maximum(0, ends - starts)

        output[bl_s:bl_t] = np.argmax(score, axis=-1)

    return output


@cache
def match_events(events_from, events_to):
    '''Match one set of events to another.

    This is useful for tasks such as matching beats to the nearest
    detected onsets, or frame-aligned events to the nearest zero-crossing.

    .. note:: A target event may be matched to multiple source events.

    Examples
    --------
    >>> # Sources are multiples of 7
    >>> s_from = np.arange(0, 100, 7)
    >>> s_from
    array([ 0,  7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91,
           98])
    >>> # Targets are multiples of 10
    >>> s_to = np.arange(0, 100, 10)
    >>> s_to
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> # Find the matching
    >>> idx = librosa.util.match_events(s_from, s_to)
    >>> idx
    array([0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9])
    >>> # Print each source value to its matching target
    >>> zip(s_from, s_to[idx])
    [(0, 0), (7, 10), (14, 10), (21, 20), (28, 30), (35, 30),
     (42, 40), (49, 50), (56, 60), (63, 60), (70, 70), (77, 80),
     (84, 80), (91, 90), (98, 90)]

    Parameters
    ----------
    events_from : ndarray [shape=(n,)]
      Array of events (eg, times, sample or frame indices) to match from.

    events_to : ndarray [shape=(m,)]
      Array of events (eg, times, sample or frame indices) to
      match against.

    Returns
    -------
    event_mapping : np.ndarray [shape=(n,)]
        For each event in `events_from`, the corresponding event
        index in `events_to`.

        `event_mapping[i] == arg min |events_from[i] - events_to[:]|`

    See Also
    --------
    match_intervals

    Raises
    ------
    ParameterError
        If either array of input events is not the correct shape
    '''

    if len(events_from) == 0 or len(events_to) == 0:
        raise ParameterError('Attempting to match empty event list')

    output = np.empty_like(events_from, dtype=np.int)

    n_rows = int(MAX_MEM_BLOCK / (np.prod(output.shape[1:]) * len(events_to)
                                  * events_from.itemsize))

    # Make sure we can at least make some progress
    n_rows = max(1, n_rows)

    # Iterate over blocks of the data
    for bl_s in range(0, len(events_from), n_rows):
        bl_t = min(bl_s + n_rows, len(events_from))

        event_block = events_from[bl_s:bl_t]
        output[bl_s:bl_t] = np.argmin(np.abs(np.subtract.outer(event_block,
                                                               events_to)),
                                      axis=-1)

    return output


@cache
def localmax(x, axis=0):
    """Find local maxima in an array `x`.

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
    x     : np.ndarray [shape=(d1,d2,...)]
      input vector or array

    axis : int
      axis along which to compute local maximality

    Returns
    -------
    m     : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along `axis`

    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


@cache
def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    '''Uses a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding x[n]
    fulfills the following three conditions:

    1. `x[n] == max(x[n - pre_max:n + post_max])`
    2. `x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta`
    3. `n - previous_n > wait`

    where `previous_n` is the last sample picked as a peak (greedily).

    This implementation is based on [1]_ and [2]_.

    .. [1] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [2] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py


    Parameters
    ----------
    x         : np.ndarray [shape=(n,)]
        input signal to peak picks from

    pre_max   : int >= 0 [scalar]
        number of samples before `n` over which max is computed

    post_max  : int >= 1 [scalar]
        number of samples after `n` over which max is computed

    pre_avg   : int >= 0 [scalar]
        number of samples before `n` over which mean is computed

    post_avg  : int >= 1 [scalar]
        number of samples after `n` over which mean is computed

    delta     : float >= 0 [scalar]
        threshold offset for mean

    wait      : int >= 0 [scalar]
        number of samples to wait after picking a peak

    Returns
    -------
    peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
        indices of peaks in `x`

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    >>> peaks
    array([  4,  23,  73, 102, 142, 162, 182, 211, 261, 301, 320,
           331, 348, 368, 382, 396, 411, 431, 446, 461, 476, 491,
           510, 525, 536, 555, 570, 590, 609, 625, 639])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.plot(onset_env[:30 * sr // 512], alpha=0.8, label='Onset strength')
    >>> plt.vlines(peaks[peaks < 30 * sr // 512], 0,
    ...            onset_env.max(), color='r', alpha=0.8,
    ...            label='Selected peaks')
    >>> plt.legend(frameon=True, framealpha=0.8)
    >>> plt.axis('tight')
    >>> plt.axis('off')
    >>> plt.subplot(2, 1, 2)
    >>> D = np.abs(librosa.stft(y))**2
    >>> librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.tight_layout()
    '''

    if pre_max < 0:
        raise ParameterError('pre_max must be non-negative')
    if pre_avg < 0:
        raise ParameterError('pre_avg must be non-negative')
    if delta < 0:
        raise ParameterError('delta must be non-negative')
    if wait < 0:
        raise ParameterError('wait must be non-negative')

    if post_max <= 0:
        raise ParameterError('post_max must be positive')

    if post_avg <= 0:
        raise ParameterError('post_avg must be positive')

    if x.ndim != 1:
        raise ParameterError('input array must be one-dimensional')

    # Ensure valid index types
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)

    # Get the maximum of the signal over a sliding window
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max))
    # Using mode='constant' and cval=x.min() effectively truncates
    # the sliding window at the boundaries
    mov_max = scipy.ndimage.filters.maximum_filter1d(x, int(max_length),
                                                     mode='constant',
                                                     origin=int(max_origin),
                                                     cval=x.min())

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    # Here, there is no mode which results in the behavior we want,
    # so we'll correct below.
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, int(avg_length),
                                                     mode='nearest',
                                                     origin=int(avg_origin))

    # Correct sliding average at the beginning
    n = 0
    # Only need to correct in the range where the window needs to be truncated
    while n - pre_avg < 0 and n < x.shape[0]:
        # This just explicitly does mean(x[n - pre_avg:n + post_avg])
        # with truncation
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1
    # Correct sliding average at the end
    n = x.shape[0] - post_avg
    # When post_avg > x.shape[0] (weird case), reset to 0
    n = n if n > 0 else 0
    while n < x.shape[0]:
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1

    # First mask out all entries not equal to the local max
    detections = x * (x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)


@cache
def sparsify_rows(x, quantile=0.01):
    '''
    Return a row-sparse matrix approximating the input `x`.

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

    Parameters
    ----------
    x : np.ndarray [ndim <= 2]
        The input matrix to sparsify.

    quantile : float in [0, 1.0)
        Percentage of magnitude to discard in each row of `x`

    Returns
    -------
    x_sparse : scipy.sparse.csr_matrix [shape=x.shape]
        Row-sparsified approximation of `x`

        If `x.ndim == 1`, then `x` is interpreted as a row vector,
        and `x_sparse.shape == (1, len(x))`.

    Raises
    ------
    ParameterError
        If `x.ndim > 2`

        If `quantile` lies outside `[0, 1.0)`
    '''

    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError('Input must have 2 or fewer dimensions.  '
                                    'Provided x.shape={}.'.format(x.shape))

    if not (0.0 <= quantile < 1):
        raise ParameterError('Invalid quantile {:.2f}'.format(quantile))

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


# Deprecated functions

@decorators.deprecated('0.4', '0.5')
def buf_to_int(x, n_bytes=2):  # pragma: no cover
    """Convert a floating point buffer into integer values.
    This is primarily useful as an intermediate step in wav output.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=float]
        Floating point data buffer

    n_bytes : int [1, 2, 4]
        Number of bytes per output sample

    Returns
    -------
    x_int : np.ndarray [dtype=int]
        The original buffer cast to integer type.
    """

    if n_bytes not in [1, 2, 4]:
        raise ParameterError('n_bytes must be one of {1, 2, 4}')

    # What is the scale of the input data?
    scale = float(1 << ((8 * n_bytes) - 1))

    # Construct a format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and cast the data
    return (x * scale).astype(fmt)
