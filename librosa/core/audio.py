#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core IO, DSP and utility functions."""

import os
import warnings
import six

import audioread
import numpy as np
import scipy.signal

from .. import cache
from .. import util

__all__ = ['load', 'to_mono', 'resample', 'get_duration',
           'autocorrelate', 'zero_crossings',
           # Deprecated functions
           'peak_pick', 'localmax']

# Resampling bandwidths as percentage of Nyquist
# http://www.mega-nerd.com/SRC/api_misc.html#Converters
BW_BEST = 0.97
BW_MEDIUM = 0.9
BW_FASTEST = 0.8

# Do we have scikits.samplerate?
try:
    # Pylint won't handle dynamic imports, so we suppress this warning
    import scikits.samplerate as samplerate  # pylint: disable=import-error
    _HAS_SAMPLERATE = True
except ImportError:
    warnings.warn('Could not import scikits.samplerate. ' +
                  'Falling back to scipy.signal')
    _HAS_SAMPLERATE = False


# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32):
    """Load an audio file as a floating point time series.

    Parameters
    ----------
    path : string
        path to the input file.

        Any format supported by `audioread` will work.

    sr   : int > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    offset : float
        start reading after this time (in seconds)

    duration : float
        only load up to this much audio (in seconds)

    dtype : numeric type
        data type of `y`


    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series

    sr   : int > 0 [scalar]
        sampling rate of `y`


    Examples
    --------
    >>> # Load a wav file
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename)
    >>> y
    array([ -4.756e-06,  -6.020e-06, ...,  -1.040e-06,   0.000e+00], dtype=float32)
    >>> sr
    22050

    >>> # Load a wav file and resample to 11 KHz
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([ -2.077e-06,  -2.928e-06, ...,  -4.395e-06,   0.000e+00], dtype=float32)
    >>> sr
    11025

    >>> # Load 5 seconds of a wav file, starting 15 seconds in
    >>> filename = librosa.util.example_audio_file()
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([ 0.069,  0.1  , ..., -0.101,  0.   ], dtype=float32)
    >>> sr
    22050

    """

    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate

        s_start = int(np.floor(sr_native * offset)) * input_file.channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.ceil(sr_native * duration))
                               * input_file.channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start < n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if len(y):
        y = np.concatenate(y)

        if input_file.channels > 1:
            y = y.reshape((-1, 2)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            if y.ndim > 1:
                y = np.vstack([resample(yi, sr_native, sr) for yi in y])
            else:
                y = resample(y, sr_native, sr)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


@cache
def to_mono(y):
    '''Force an audio signal down to mono.

    Parameters
    ----------
    y : np.ndarray [shape=(2,n) or shape=(n,)]
        audio time series, either stereo or mono

    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        `y` as a monophonic time-series

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    >>> y.shape
    (2, 1355168)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (1355168,)

    '''

    # Validate the buffer.  Stereo is ok here.
    util.valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


@cache
def resample(y, orig_sr, target_sr, res_type='sinc_best', fix=True, **kwargs):
    """Resample a time series from orig_sr to target_sr

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    orig_sr : int > 0 [scalar]
        original sampling rate of `y`

    target_sr : int > 0 [scalar]
        target sampling rate

    res_type : str
        resample type (see note)

        .. note::
            If `scikits.samplerate` is installed, `resample`
            will use `res_type`.

            Otherwise, fall back on `scipy.signal.resample`.

            To force use of `scipy.signal.resample`, set `res_type='scipy'`.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        `ceil(target_sr * len(y) / orig_sr)`

    kwargs : additional keyword arguments
        If `fix==True`, additional keyword arguments to pass to
        `librosa.util.fix_length`.

    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        `y` resampled from `orig_sr` to `target_sr`


    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((1355168,), (491671,))

    """

    # First, validate the audio buffer
    util.valid_audio(y)

    if orig_sr == target_sr:
        return y

    n_samples = int(np.ceil(y.shape[-1] * float(target_sr) / orig_sr))

    scipy_resample = (res_type == 'scipy')

    if _HAS_SAMPLERATE and not scipy_resample:
        y_hat = samplerate.resample(y.T,
                                    float(target_sr) / orig_sr,
                                    res_type).T
    else:
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                 center=True):
    """Compute the duration (in seconds) of an audio time series or STFT matrix.

    Examples
    --------
    >>> # Load the example audio file
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.get_duration(y=y, sr=sr)
    61.44

    >>> # Or compute duration from an STFT matrix
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = librosa.stft(y)
    >>> librosa.get_duration(S=S, sr=sr)
    61.44

    >>> # Or a non-centered STFT matrix
    >>> S_left = librosa.stft(y, center=False)
    >>> librosa.get_duration(S=S_left, sr=sr)
    61.3471201814059

    Parameters
    ----------
    y : np.ndarray [shape=(n,), (2, n)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        STFT matrix, or any STFT-derived matrix (e.g., chromagram
        or mel spectrogram).

    n_fft       : int > 0 [scalar]
        FFT window size for `S`

    hop_length  : int > 0 [ scalar]
        number of audio samples between columns of `S`

    center  : boolean
        - If `True`, `S[:, t]` is centered at `y[t * hop_length]`
        - If `False`, then `S[:, t]` begins at `y[t * hop_length]`

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.
    """

    if y is None:
        assert S is not None

        n_frames = S.shape[1]
        n_samples = n_fft + hop_length * (n_frames - 1)

        # If centered, we lose half a window from each end of S
        if center:
            n_samples = n_samples - 2 * int(n_fft / 2)

    else:
        # Validate the audio buffer.  Stereo is okay here.
        util.valid_audio(y, mono=False)
        if y.ndim == 1:
            n_samples = len(y)
        else:
            n_samples = y.shape[-1]

    return float(n_samples) / sr


@cache
def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        vector to autocorrelate

    max_size  : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to `len(y)` (unbounded)

    Returns
    -------
    z : np.ndarray [shape=(n,) or (max_size,)]
        truncated autocorrelation `y*y`

    Examples
    --------
    Compute full autocorrelation of y

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.autocorrelate(y)
    array([  1.584e+04,   1.580e+04, ...,  -1.154e-10,  -2.725e-13])


    Compute onset strength auto-correlation up to 4 seconds

    >>> import matplotlib.pyplot as plt
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    >>> ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
    >>> plt.plot(ac)
    >>> plt.title('Auto-correlation')
    >>> plt.xlabel('Lag (frames)')

    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[int(len(result)/2):]

    if max_size is None:
        return result
    else:
        max_size = int(max_size)

    return result[:max_size]


@cache
def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True,
                   zero_pos=True, axis=-1):
    '''Find the zero-crossings of a signal `y`: indices `i` such that
    `sign(y[i]) != sign(y[j])`.

    If `y` is multi-dimensional, then zero-crossings are computed along
    the specified `axis`.

    Examples
    --------
    >>> # Generate a time-series
    >>> y = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
    >>> y
    array([  0.000e+00,   9.694e-01,   4.759e-01,  -7.357e-01,
            -8.372e-01,   3.247e-01,   9.966e-01,   1.646e-01,
            -9.158e-01,  -6.142e-01,   6.142e-01,   9.158e-01,
            -1.646e-01,  -9.966e-01,  -3.247e-01,   8.372e-01,
             7.357e-01,  -4.759e-01,  -9.694e-01,  -9.797e-16])
    >>> # Compute zero-crossings
    >>> z = librosa.zero_crossings(y)
    >>> z
    array([ True, False, False,  True, False,  True, False, False,
            True, False,  True, False,  True, False, False,  True,
           False,  True, False,  True], dtype=bool)
    >>> # Stack y against the zero-crossing indicator
    >>> np.vstack([y, z]).T
    array([[  0.000e+00,   1.000e+00],
           [  9.694e-01,   0.000e+00],
           [  4.759e-01,   0.000e+00],
           [ -7.357e-01,   1.000e+00],
           [ -8.372e-01,   0.000e+00],
           [  3.247e-01,   1.000e+00],
           [  9.966e-01,   0.000e+00],
           [  1.646e-01,   0.000e+00],
           [ -9.158e-01,   1.000e+00],
           [ -6.142e-01,   0.000e+00],
           [  6.142e-01,   1.000e+00],
           [  9.158e-01,   0.000e+00],
           [ -1.646e-01,   1.000e+00],
           [ -9.966e-01,   0.000e+00],
           [ -3.247e-01,   0.000e+00],
           [  8.372e-01,   1.000e+00],
           [  7.357e-01,   0.000e+00],
           [ -4.759e-01,   1.000e+00],
           [ -9.694e-01,   0.000e+00],
           [ -9.797e-16,   1.000e+00]])
    >>> # Find the indices of zero-crossings
    >>> np.nonzero(z)
    (array([ 0,  3,  5,  8, 10, 12, 15, 17, 19]),)


    Parameters
    ----------
    y : np.ndarray
        The input array

    threshold : float > 0 or None
        If specified, values where `-threshold <= y <= threshold` are
        clipped to 0.

    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to `ref_magnitude`.

        If callable, the threshold is scaled relative to
        `ref_magnitude(np.abs(y))`.

    pad : boolean
        If `True`, then `y[0]` is considered a valid zero-crossing.

    zero_pos : boolean
        If `True` then the value 0 is interpreted as having positive sign.

        If `False`, then 0, -1, and +1 all have distinct signs.

    axis : int
        Axis along which to compute zero-crossings.

    Returns
    -------
    zero_crossings : np.ndarray [shape=y.shape, dtype=boolean]
        Indicator array of zero-crossings in `y` along the selected axis.
    '''

    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if six.callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    return np.pad((y_sign[slice_post] != y_sign[slice_pre]),
                  padding,
                  mode='constant',
                  constant_values=pad)


# Moved/deprecated functions
peak_pick = util.decorators.moved('librosa.core.peak_pick',
                                  '0.4', '0.5')(util.peak_pick)

localmax = util.decorators.moved('librosa.core.localmax',
                                 '0.4', '0.5')(util.localmax)
