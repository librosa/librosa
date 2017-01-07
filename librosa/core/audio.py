#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core IO, DSP and utility functions."""

import os
import six

import audioread
import numpy as np
import scipy.signal
import scipy.fftpack as fft
import resampy

from .time_frequency import frames_to_samples, time_to_samples
from .. import cache
from .. import util
from ..util.exceptions import ParameterError

__all__ = ['load', 'to_mono', 'resample', 'get_duration',
           'autocorrelate', 'zero_crossings', 'clicks']

# Resampling bandwidths as percentage of Nyquist
BW_BEST = resampy.filters.get_filter('kaiser_best')[2]
BW_FASTEST = resampy.filters.get_filter('kaiser_fast')[2]


# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best'):
    """Load an audio file as a floating point time series.

    Parameters
    ----------
    path : string
        path to the input file.

        Any format supported by `audioread` will work.

    sr   : number > 0 [scalar]
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

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            To use a faster method, set `res_type='kaiser_fast'`.

            To use `scipy.signal.resample`, set `res_type='scipy'`.


    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]
        audio time series

    sr   : number > 0 [scalar]
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
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

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

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, 2)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


@cache(level=20)
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

    Notes
    -----
    This function caches at level 20.

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


@cache(level=20)
def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    """Resample a time series from orig_sr to target_sr

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.

    orig_sr : number > 0 [scalar]
        original sampling rate of `y`

    target_sr : number > 0 [scalar]
        target sampling rate

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            To use a faster method, set `res_type='kaiser_fast'`.

            To use `scipy.signal.resample`, set `res_type='scipy'`.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        `ceil(target_sr * len(y) / orig_sr)`

    scale : bool
        Scale the resampled signal so that `y` and `y_hat` have approximately
        equal total energy.

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
    resampy.resample

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
    >>> y_8k = librosa.resample(y, sr, 8000)
    >>> y.shape, y_8k.shape
    ((1355168,), (491671,))

    """

    # First, validate the audio buffer
    util.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type == 'scipy':
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                 center=True, filename=None):
    """Compute the duration (in seconds) of an audio time series,
    feature matrix, or filename.

    Examples
    --------
    >>> # Load the example audio file
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.get_duration(y=y, sr=sr)
    61.44

    >>> # Or directly from an audio file
    >>> librosa.get_duration(filename=librosa.util.example_audio_file())
    61.4

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

    sr : number > 0 [scalar]
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

    filename : str
        If provided, all other parameters are ignored, and the
        duration is calculated directly from the audio file.
        Note that this avoids loading the contents into memory,
        and is therefore useful for querying the duration of
        long files.

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.
    """

    if filename is not None:
        with audioread.audio_open(filename) as fdesc:
            return fdesc.duration

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


@cache(level=20)
def autocorrelate(y, max_size=None, axis=-1):
    """Bounded auto-correlation

    Parameters
    ----------
    y : np.ndarray
        array to autocorrelate

    max_size  : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to `y.shape[axis]` (unbounded)

    axis : int
        The axis along which to autocorrelate.
        By default, the last axis (-1) is taken.

    Returns
    -------
    z : np.ndarray
        truncated autocorrelation `y*y` along the specified axis.
        If `max_size` is specified, then `z.shape[axis]` is bounded
        to `max_size`.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Compute full autocorrelation of y

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=20, duration=10)
    >>> librosa.autocorrelate(y)
    array([  3.226e+03,   3.217e+03, ...,   8.277e-04,   3.575e-04], dtype=float32)

    Compute onset strength auto-correlation up to 4 seconds

    >>> import matplotlib.pyplot as plt
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    >>> ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
    >>> plt.plot(ac)
    >>> plt.title('Auto-correlation')
    >>> plt.xlabel('Lag (frames)')

    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis))**2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis, overwrite_x=True)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[subslice]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr


@cache(level=20)
def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True,
                   zero_pos=True, axis=-1):
    '''Find the zero-crossings of a signal `y`: indices `i` such that
    `sign(y[i]) != sign(y[j])`.

    If `y` is multi-dimensional, then zero-crossings are computed along
    the specified `axis`.


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

    Notes
    -----
    This function caches at level 20.

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


def clicks(times=None, frames=None, sr=22050, hop_length=512,
           click_freq=1000.0, click_duration=0.1, click=None, length=None):
    """Returns a signal with the signal `click` placed at each specified time

    Parameters
    ----------
    times : np.ndarray or None
        times to place clicks, in seconds

    frames : np.ndarray or None
        frame indices to place clicks

    sr : number > 0
        desired sampling rate of the output signal

    hop_length : int > 0
        if positions are specified by `frames`, the number of samples between frames.

    click_freq : float > 0
        frequency (in Hz) of the default click signal.  Default is 1KHz.

    click_duration : float > 0
        duration (in seconds) of the default click signal.  Default is 100ms.

    click : np.ndarray or None
        optional click signal sample to use instead of the default blip.

    length : int > 0
        desired number of samples in the output signal


    Returns
    -------
    click_signal : np.ndarray
        Synthesized click signal


    Raises
    ------
    ParameterError
        - If neither `times` nor `frames` are provided.
        - If any of `click_freq`, `click_duration`, or `length` are out of range.


    Examples
    --------
    >>> # Sonify detected beat events
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> y_beats = librosa.clicks(frames=beats, sr=sr)

    >>> # Or generate a signal of the same length as y
    >>> y_beats = librosa.clicks(frames=beats, sr=sr, length=len(y))

    >>> # Or use timing instead of frame indices
    >>> times = librosa.frames_to_time(beats, sr=sr)
    >>> y_beat_times = librosa.clicks(times=times, sr=sr)

    >>> # Or with a click frequency of 880Hz and a 500ms sample
    >>> y_beat_times880 = librosa.clicks(times=times, sr=sr,
    ...                                  click_freq=880, click_duration=0.5)

    Display click waveform next to the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr)
    >>> ax = plt.subplot(2,1,2)
    >>> librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                          x_axis='time', y_axis='mel')
    >>> plt.subplot(2,1,1, sharex=ax)
    >>> librosa.display.waveplot(y_beat_times, sr=sr, label='Beat clicks')
    >>> plt.legend()
    >>> plt.xlim(15, 30)
    >>> plt.tight_layout()
    """

    # Compute sample positions from time or frames
    if times is None:
        if frames is None:
            raise ParameterError('either "times" or "frames" must be provided')

        positions = frames_to_samples(frames, hop_length=hop_length)
    else:
        # Convert times to positions
        positions = time_to_samples(times, sr=sr)

    if click is not None:
        # Check that we have a well-formed audio buffer
        util.valid_audio(click, mono=True)

    else:
        # Create default click signal
        if click_duration <= 0:
            raise ParameterError('click_duration must be strictly positive')

        if click_freq <= 0:
            raise ParameterError('click_freq must be strictly positive')

        angular_freq = 2 * np.pi * click_freq / float(sr)

        click = np.logspace(0, -10,
                            num=int(np.round(sr * click_duration)),
                            base=2.0)

        click *= np.sin(angular_freq * np.arange(len(click)))

    # Set default length
    if length is None:
        length = positions.max() + click.shape[0]
    else:
        if length < 1:
            raise ParameterError('length must be a positive integer')

        # Filter out any positions past the length boundary
        positions = positions[positions < length]

    # Pre-allocate click signal
    click_signal = np.zeros(length, dtype=np.float32)

    # Place clicks
    for start in positions:
        # Compute the end-point of this click
        end = start + click.shape[0]

        if end >= length:
            click_signal[start:] += click[:length - start]
        else:
            # Normally, just add a click here
            click_signal[start:end] += click

    return click_signal
