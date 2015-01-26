#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature manipulation utilities"""

import numpy as np
import scipy.signal

from .. import cache
from .. import util

__all__ = ['delta', 'stack_memory', 'sync']


@cache
def delta(data, width=9, order=1, axis=-1, trim=True):
    '''Compute delta features.

    Parameters
    ----------
    data      : np.ndarray [shape=(d, T)]
        the input data matrix (eg, spectrogram)

    width     : int > 0, odd [scalar]
        Number of frames over which to compute the delta feature

    order     : int > 0 [scalar]
        the order of the difference operator.
        1 for first derivative, 2 for second, etc.

    axis      : int [scalar]
        the axis along which to compute deltas.
        Default is -1 (columns).

    trim      : bool
        set to `True` to trim the output matrix to the original size.

    Returns
    -------
    delta_data   : np.ndarray [shape=(d, t) or (d, t + window)]
        delta matrix of `data`.

    Examples
    --------
    Compute MFCC deltas, delta-deltas

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)
    >>> librosa.feature.delta(mfccs)
    array([[ -4.250e+03,  -3.060e+03, ...,  -4.547e-13,  -4.547e-13],
           [  5.673e+02,   6.931e+02, ...,   0.000e+00,   0.000e+00],
    ...,
           [ -5.986e+01,  -5.018e+01, ...,   0.000e+00,   0.000e+00],
           [ -3.112e+01,  -2.908e+01, ...,   0.000e+00,   0.000e+00]])
    >>> librosa.feature.delta(mfccs, order=2)
    array([[ -4.297e+04,  -3.207e+04, ...,  -8.185e-11,  -5.275e-11],
           [  5.736e+03,   5.420e+03, ...,  -7.390e-12,  -4.547e-12],
    ...,
           [ -6.053e+02,  -4.801e+02, ...,   0.000e+00,   0.000e+00],
           [ -3.146e+02,  -2.615e+02, ...,  -4.619e-13,  -2.842e-13]])

    '''

    half_length = 1 + int(np.floor(width / 2.0))
    window = np.arange(half_length - 1, -half_length, -1)

    # Pad out the data by repeating the border values (delta=0)
    padding = [(0, 0)] * data.ndim
    padding[axis] = (half_length, half_length)
    delta_x = np.pad(data, padding, mode='edge')

    for _ in range(order):
        delta_x = scipy.signal.lfilter(window, 1, delta_x, axis=axis)

    if trim:
        idx = [Ellipsis] * delta_x.ndim
        idx[axis] = slice(half_length, -half_length)
        delta_x = delta_x[idx]

    return delta_x


@cache
def stack_memory(data, n_steps=2, delay=1, **kwargs):
    """Short-term history embedding: vertically concatenate a data
    vector or matrix with delayed copies of itself.

    Each column `data[:, i]` is mapped to::

        data[:, i] ->  [data[:, i],
                        data[:, i - delay],
                        ...
                        data[:, i - (n_steps-1)*delay]]

    For columns `i < (n_steps - 1) * delay` , the data will be padded.
    By default, the data is padded with zeros, but this behavior can be
    overridden by supplying additional keyword arguments which are passed
    to `np.pad()`.


    Parameters
    ----------
    data : np.ndarray [shape=(t,) or (d, t)]
        Input data matrix.  If `data` is a vector (`data.ndim == 1`),
        it will be interpreted as a row matrix and reshaped to `(1, t)`.

    n_steps : int > 0 [scalar]
        embedding dimension, the number of steps back in time to stack

    delay : int > 0 [scalar]
        the number of columns to step

    kwargs : additional keyword arguments
      Additional arguments to pass to `np.pad`.

    Returns
    -------
    data_history : np.ndarray [shape=(m * d, t)]
        data augmented with lagged copies of itself,
        where `m == n_steps - 1`.

    Examples
    --------
    Keep two steps (current and previous)

    >>> data = np.arange(-3, 3)
    >>> librosa.feature.stack_memory(data)
    array([[-3, -2, -1,  0,  1,  2],
           [ 0, -3, -2, -1,  0,  1]])

    Or three steps

    >>> librosa.feature.stack_memory(data, n_steps=3)
    array([[-3, -2, -1,  0,  1,  2],
           [ 0, -3, -2, -1,  0,  1],
           [ 0,  0, -3, -2, -1,  0]])

    Use reflection padding instead of zero-padding

    >>> librosa.feature.stack_memory(data, n_steps=3, mode='reflect')
    array([[-3, -2, -1,  0,  1,  2],
           [-2, -3, -2, -1,  0,  1],
           [-1, -2, -3, -2, -1,  0]])

    Or pad with edge-values, and delay by 2

    >>> librosa.feature.stack_memory(data, n_steps=3, delay=2, mode='edge')
    array([[-3, -2, -1,  0,  1,  2],
           [-3, -3, -3, -2, -1,  0],
           [-3, -3, -3, -3, -3, -2]])

    Stack time-lagged beat-synchronous chroma edge padding

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> chroma = librosa.feature.chromagram(y=y, sr=sr)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> chroma_sync = librosa.feature.sync(chroma, beats)
    >>> chroma_lag = librosa.feature.stack_memory(chroma_sync, n_steps=3,
    ...                                           mode='edge')

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(chroma_lag)
    >>> librosa.display.time_ticks(librosa.frames_to_time(beats, sr=sr))
    >>> plt.title('Time-lagged chroma')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    # If we're given a vector, interpret as a matrix
    if data.ndim == 1:
        data = data.reshape((1, -1))

    t = data.shape[1]
    kwargs.setdefault('mode', 'constant')

    if kwargs['mode'] == 'constant':
        kwargs.setdefault('constant_values', [0])

    # Pad the end with zeros, which will roll to the front below
    data = np.pad(data, [(0, 0), ((n_steps - 1) * delay, 0)], **kwargs)

    history = data

    for i in range(1, n_steps):
        history = np.vstack([np.roll(data, -i * delay, axis=1), history])

    # Trim to original width
    history = history[:, :t]

    # Make contiguous
    return np.ascontiguousarray(history.T).T


@cache
def sync(data, frames, aggregate=None, pad=True):
    """Synchronous aggregation of a feature matrix

    .. note::
        In order to ensure total coverage, boundary points may be added
        to `frames`.

        If synchronizing a feature matrix against beat tracker output, ensure
        that frame numbers are properly aligned and use the same hop length.

    Parameters
    ----------
    data      : np.ndarray [shape=(d, T) or shape=(T,)]
        matrix of features

    frames    : np.ndarray [shape=(m,)]
        ordered array of frame segment boundaries

    aggregate : function
        aggregation function (default: `np.mean`)

    pad : boolean
        If `True`, `frames` is padded to span the full range `[0, T]`

    Returns
    -------
    Y         : ndarray [shape=(d, M)]
        `Y[:, i] = aggregate(data[:, F[i-1]:F[i]], axis=1)`

    Raises
    ------
    ValueError
        If `data.ndim` is not 1 or 2

    Examples
    --------
    Beat-synchronous MFCCs

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)

    By default, use mean aggregation

    >>> mfcc_avg = librosa.feature.sync(mfcc, beats)

    Use median-aggregation instead of mean

    >>> mfcc_med = librosa.feature.sync(mfcc, beats,
    ...                                 aggregate=np.median)

    Or max aggregation

    >>> mfcc_max = librosa.feature.sync(mfcc, beats,
    ...                                 aggregate=np.max)

    """

    if data.ndim < 2:
        data = np.asarray([data])

    elif data.ndim > 2:
        raise ValueError('Synchronized data has ndim={:d},'
                         ' must be 1 or 2.'.format(data.ndim))

    if aggregate is None:
        aggregate = np.mean

    dimension, n_frames = data.shape

    frames = util.fix_frames(frames, 0, n_frames, pad=pad)

    data_agg = np.empty((dimension, len(frames)-1), order='F')

    start = frames[0]

    for (i, end) in enumerate(frames[1:]):
        data_agg[:, i] = aggregate(data[:, start:end], axis=1)
        start = end

    return data_agg
