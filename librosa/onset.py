#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Onset detection
===============
.. autosummary::
    :toctree: generated/

    onset_detect
    onset_strength
    onset_strength_multi
"""

import warnings

import numpy as np
import scipy

from . import cache
from . import core
from . import util
from .util.exceptions import ParameterError

from .feature.spectral import melspectrogram

__all__ = ['onset_detect', 'onset_strength', 'onset_strength_multi']


def onset_detect(y=None, sr=22050, onset_envelope=None, hop_length=512,
                 **kwargs):
    """Basic onset detector.  Locate note onset events by picking peaks in an
    onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [1]_.

    .. [1] https://github.com/CPJKU/onset_db


    Parameters
    ----------
    y          : np.ndarray [shape=(n,)]
        audio time series

    sr         : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope     : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.


    Returns
    -------

    onsets : np.ndarray [shape=(n_onsets,)]
        estimated frame numbers of onsets

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.


    Raises
    ------
    ParameterError
        if neither `y` nor `onsets` are provided


    See Also
    --------
    onset_strength : compute onset strength per-frame
    librosa.util.peak_pick : pick peaks from a time series


    Examples
    --------
    Get onset times from a signal

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=10.0)
    >>> onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    >>> librosa.frames_to_time(onset_frames[:20], sr=sr)
    array([ 0.07 ,  0.279,  0.511,  0.859,  1.091,  1.207,  1.463,
            1.672,  1.904,  2.159,  2.368,  2.601,  2.949,  3.065,
            3.297,  3.529,  3.762,  3.994,  4.203,  4.69 ])


    Or use a pre-computed onset envelope

    >>> o_env = librosa.onset.onset_strength(y, sr=sr)
    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)


    >>> import matplotlib.pyplot as plt
    >>> D = np.abs(librosa.stft(y))**2
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
    ...                          x_axis='time', y_axis='log')
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(o_env, label='Onset strength')
    >>> plt.vlines(onset_frames, 0, o_env.max(), color='r', alpha=0.9,
    ...            linestyle='--', label='Onsets')
    >>> plt.xticks([])
    >>> plt.axis('tight')
    >>> plt.legend(frameon=True, framealpha=0.75)

    """

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    onset_envelope -= onset_envelope.min()

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)

    # Normalize onset strength function to [0, 1] range
    onset_envelope /= onset_envelope.max()

    # These parameter settings found by large-scale search
    kwargs.setdefault('pre_max', 0.03*sr//hop_length)       # 30ms
    kwargs.setdefault('post_max', 0.00*sr//hop_length + 1)  # 0ms
    kwargs.setdefault('pre_avg', 0.10*sr//hop_length)       # 100ms
    kwargs.setdefault('post_avg', 0.10*sr//hop_length + 1)  # 100ms
    kwargs.setdefault('wait', 0.03*sr//hop_length)          # 30ms
    kwargs.setdefault('delta', 0.07)

    # Peak pick the onset envelope
    return util.peak_pick(onset_envelope, **kwargs)


def onset_strength(y=None, sr=22050, S=None, lag=1, max_size=1,
                   detrend=False, center=True,
                   feature=None, aggregate=None,
                   centering=None,
                   **kwargs):
    """Compute a spectral flux onset strength envelope.

    Onset strength at time `t` is determined by:

    `mean_f max(0, S[f, t] - ref_S[f, t - lag])`

    where `ref_S` is `S` after local max filtering along the frequency
    axis [1]_.

    By default, if a time series `y` is provided, S will be the
    log-power Mel spectrogram.

    .. [1] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    Parameters
    ----------
    y        : np.ndarray [shape=(n,)]
        audio time-series

    sr       : number > 0 [scalar]
        sampling rate of `y`

    S        : np.ndarray [shape=(d, m)]
        pre-computed (log-power) spectrogram

    lag      : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
    centering : bool [scalar] (deprecated)
        Shift the onset function by `n_fft / (2 * hop_length)` frames

        .. note:: The `centering` parameter is deprecated as of 0.4.1,
        and has been replaced by the `center` parameter.

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with `fmax=11025.0`

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    kwargs : additional keyword arguments
        Additional parameters to `feature()`, if `S` is not provided.


    Returns
    -------
    onset_envelope   : np.ndarray [shape=(m,)]
        vector containing the onset strength envelope


    Raises
    ------
    ParameterError
        if neither `(y, sr)` nor `S` are provided

        or if `lag` or `max_size` are not positive integers


    See Also
    --------
    onset_detect
    onset_strength_multi


    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=10.0)
    >>> D = np.abs(librosa.stft(y))**2
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.title('Power spectrogram')

    Construct a standard onset function

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(2 + onset_env / onset_env.max(), alpha=0.8,
    ...          label='Mean aggregation (mel)')


    Median aggregation, and custom mel options

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median,
    ...                                          fmax=8000, n_mels=256)
    >>> plt.plot(1 + onset_env / onset_env.max(), alpha=0.8,
    ...          label='Median aggregation (custom mel)')


    Constant-Q spectrogram instead of Mel

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          feature=librosa.cqt)
    >>> plt.plot(onset_env / onset_env.max(), alpha=0.8,
    ...          label='Mean aggregation (CQT)')

    >>> plt.legend(frameon=True, framealpha=0.75)
    >>> librosa.display.time_ticks(librosa.frames_to_time(np.arange(len(onset_env))))
    >>> plt.ylabel('Normalized strength')
    >>> plt.yticks([])
    >>> plt.axis('tight')
    >>> plt.tight_layout()

    """

    if centering is not None:
        center = centering
        warnings.warn("The 'centering=' parameter of onset_strength is "
                      "deprecated as of librosa version 0.4.1."
                      "\n\tIt will be removed in librosa version 0.5.0."
                      "\n\tPlease use 'center=' instead.",
                      category=DeprecationWarning)

    odf_all = onset_strength_multi(y=y,
                                   sr=sr,
                                   S=S,
                                   lag=lag,
                                   max_size=max_size,
                                   detrend=detrend,
                                   center=center,
                                   feature=feature,
                                   aggregate=aggregate,
                                   channels=None,
                                   **kwargs)

    return odf_all[0]


@cache
def onset_strength_multi(y=None, sr=22050, S=None, lag=1, max_size=1,
                         detrend=False, center=True, feature=None,
                         aggregate=None, channels=None, **kwargs):
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel `i` at time `t` is determined by:

    `mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])`


    Parameters
    ----------
    y        : np.ndarray [shape=(n,)]
        audio time-series

    sr       : number > 0 [scalar]
        sampling rate of `y`

    S        : np.ndarray [shape=(d, m)]
        pre-computed (log-power) spectrogram

    lag      : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by `n_fft / (2 * hop_length)` frames

    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with `fmax=11025.0`

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    kwargs : additional keyword arguments
        Additional parameters to `feature()`, if `S` is not provided.


    Returns
    -------
    onset_envelope   : np.ndarray [shape=(n_channels, m)]
        array containing the onset strength envelope for each specified channel.


    Raises
    ------
    ParameterError
        if neither `(y, sr)` nor `S` are provided


    See Also
    --------
    onset_strength


    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=10.0)
    >>> D = np.abs(librosa.stft(y))**2
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(D, ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.title('Power spectrogram')

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(onset_subbands, x_axis='time')
    >>> plt.title('Sub-band onset strength')

    """

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault('fmax', 11025.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, int):
        raise ParameterError('lag must be a positive integer')

    if max_size < 1 or not isinstance(max_size, int):
        raise ParameterError('max_size must be a positive integer')

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, **kwargs))

        # Convert to dBs
        S = core.logamplitude(S)

    # Retrieve the n_fft and hop_length,
    # or default values for onsets if not provided
    n_fft = kwargs.get('n_fft', 2048)
    hop_length = kwargs.get('hop_length', 512)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if max_size == 1:
        ref_spec = S
    else:
        ref_spec = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref_spec[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    onset_env = util.sync(onset_env, channels,
                          aggregate=aggregate,
                          pad=pad,
                          axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]), mode='constant')

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, :S.shape[1]]

    return onset_env
