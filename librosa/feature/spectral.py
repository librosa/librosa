#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

import numpy as np

from .. import cache
from .. import util
from .. import filters

from ..core.time_frequency import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import stft, logamplitude
from ..core.pitch import estimate_tuning


# -- Spectral features -- #
@cache
def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                      freq=None):
    '''Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    Examples
    --------
    >>> # From time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_centroid(y=y, sr=sr)
    array([[  545.929,   400.609, ...,  1621.184,  1591.604]])

    >>> # From spectrogram input
    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[  545.929,   400.609, ...,  1621.184,  1591.604]])

    >>> # Using variable bin center frequencies
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    array([[  545.069,   400.764, ...,  1621.139,  1590.362]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        :func:`librosa.core.ifgram`

    Returns
    -------
    centroid : np.ndarray [shape=(1, t)]
        centroid frequencies
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    if not np.isrealobj(S):
        raise ValueError('Spectral centroid is only defined '
                         'with real-valued input')
    elif np.any(S < 0):
        raise ValueError('Spectral centroid is only defined '
                         'with non-negative energies')

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    # Column-normalize S
    return np.sum(freq * util.normalize(S, norm=1, axis=0),
                  axis=0, keepdims=True)


@cache
def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                       freq=None, centroid=None, norm=True, p=2):
    '''Compute p'th-order spectral bandwidth:

        (sum_k S[k] * (freq[k] - centroid)**p)**(1/p)

    Examples
    --------
    >>> # From time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_bandwidth(y=y, sr=sr)
    array([[ 1201.067,   920.588, ...,  2218.177,  2211.325]])

    >>> # From spectrogram input
    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_bandwidth(S=S)
    array([[ 1201.067,   920.588, ...,  2218.177,  2211.325]])

    >>> # Using variable bin center frequencies
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
    array([[ 1202.514,   920.453, ...,  2218.172,  2213.157]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        :func:`librosa.core.ifgram`

    centroid : None or np.ndarray [shape=(1, t)]
        pre-computed centroid frequencies

    norm : bool
        Normalize per-frame spectral energy (sum to one)

    p : float > 0
        Power to raise deviation from spectral centroid.

    Returns
    -------
    bandwidth : np.ndarray [shape=(1, t)]
        frequency bandwidth for each frame
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    if not np.isrealobj(S):
        raise ValueError('Spectral bandwidth is only defined '
                         'with real-valued input')
    elif np.any(S < 0):
        raise ValueError('Spectral bandwidth is only defined '
                         'with non-negative energies')

    if centroid is None:
        centroid = spectral_centroid(y=y, sr=sr, S=S,
                                     n_fft=n_fft,
                                     hop_length=hop_length,
                                     freq=freq)

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        deviation = np.abs(np.subtract.outer(freq, np.squeeze(centroid)))
    else:
        deviation = np.abs(freq - np.squeeze(centroid))

    # Column-normalize S
    if norm:
        S = util.normalize(S, norm=1, axis=0)

    return np.sum(S * deviation**p, axis=0, keepdims=True)**(1./p)


@cache
def spectral_contrast(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                      freq=None, n_bands=6):
    '''Compute spectral contrast

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        :func:`librosa.core.ifgram`

    n_bands : int > 1
        number of frequency bands

    Returns
    -------
    contrast : np.ndarray [shape=(n_bands + 1, t)]
        each row of spectral contrast values corresponds to a given
        octave-based frequency
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    #     TODO:   2014-12-31 12:48:36 by Brian McFee <brian.mcfee@nyu.edu>
    #   shouldn't this be scaled relative to the max frequency?
    octa = np.zeros(n_bands + 2)
    octa[1:] = 200 * (2.0**np.arange(0, n_bands + 1))

    valley = np.zeros((n_bands + 1, S.shape[1]))
    peak = np.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)

        idx = np.flatnonzero(current_band)

        if k > 0:
            current_band[idx[0] - 1] = True

        if k == n_bands:
            current_band[idx[-1] + 1:] = True

        sub_band = S[current_band]

        if k < n_bands:
            sub_band = sub_band[:-1]

        # FIXME:  2014-12-31 13:06:49 by Brian McFee <brian.mcfee@nyu.edu>
        # why 50?  what is this?
        alph = int(max(1, np.rint(0.02 * np.sum(current_band))))

        sortedr = np.sort(sub_band, axis=0)

        valley[k] = np.mean(sortedr[:alph], axis=0)
        peak[k] = np.mean(sortedr[-alph:], axis=0)

    return peak - valley


@cache
def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                     freq=None, roll_percent=0.85):
    '''Compute roll-off frequency

    Examples
    --------
    >>> # From time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr)
    array([[  936.694,   635.229, ...,  3983.643,  3886.743]])

    >>> # With a higher roll percentage:
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    array([[ 2637.817,  1496.558, ...,  6955.225,  6933.691]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,

    roll_percent : float [0 < roll_percent < 1]
        Roll-off percentage.

    Returns
    -------
    rolloff : np.ndarray [shape=(1, t)]
        roll-off frequency for each frame
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    if not np.isrealobj(S):
        raise ValueError('Spectral centroid is only defined '
                         'with real-valued input')
    elif np.any(S < 0):
        raise ValueError('Spectral centroid is only defined '
                         'with non-negative energies')

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    total_energy = np.cumsum(S, axis=0)

    threshold = roll_percent * total_energy[-1]

    ind = np.where(total_energy < threshold, np.nan, 1)

    return np.nanmin(ind * freq, axis=0, keepdims=True)


@cache
def rms(y=None, S=None, n_fft=2048, hop_length=512):
    '''Compute root-mean-square (RMS) energy for each frame.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.rms(y=y)
    array([[  1.204e-01,   6.263e-01, ...,   1.413e-04,   2.191e-05]],
          dtype=float32)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    Returns
    -------
    rms : np.ndarray [shape=(1, t)]
        RMS value for each frame
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = stft(y, n_fft=n_fft, hop_length=hop_length)

    return np.sqrt(np.mean(np.abs(S)**2, axis=0, keepdims=True))


@cache
def poly_features(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                  order=1, freq=None):
    '''Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> # Line features
    >>> librosa.feature.poly_features(y=y, sr=sr)
    array([[ -9.454e-06,  -4.322e-05, ...,  -1.640e-08,  -2.626e-09],
           [  7.144e-02,   3.241e-01, ...,   1.332e-04,   2.127e-05]])
    >>> # Quadratic features
    >>> librosa.feature.poly_features(y=y, sr=sr, order=2)
    array([[  3.742e-09,   1.753e-08, ...,   5.145e-12,   8.343e-13],
           [ -5.071e-05,  -2.364e-04, ...,  -7.312e-08,  -1.182e-08],
           [  1.472e-01,   6.788e-01, ...,   2.373e-04,   3.816e-05]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    order : int > 0
        order of the polynomial to fit

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        :func:`librosa.core.ifgram`

    Returns
    -------
    coefficients : np.ndarray [shape=(order+1, t)]
        polynomial coefficients for each frame
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # If frequencies are constant over frames, then we only need to fit once
    if freq.ndim == 1:
        coefficients = np.polyfit(freq, S, order)
    else:
        # Else, fit each frame independently and stack the results
        coefficients = np.concatenate([[np.polyfit(freq_t, S_t, order)]
                                       for (freq_t, S_t) in zip(freq.T, S.T)],
                                      axis=0).T

    return coefficients


@cache
def zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True,
                       **kwargs):
    '''Compute the zero-crossing rate of an audio time series.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.zero_crossing_rate(y)
    array([[ 0.072,  0.074, ...,  0.091,  0.038]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Audio time series

    frame_length : int > 0
        Length of the frame over which to compute zero crossing rates

    hop_length : int > 0
        Number of samples to advance for each frame

    center : bool
        If `True`, frames are centered by padding the edges of `y`.
        This is similar to the padding in :func:`librosa.core.stft`,
        but uses edge-value copies instead of reflection.

    kwargs : additional keyword arguments
        See :func:`librosa.core.zero_crossings`

        .. note:: By default, the `pad` parameter is set to `False`, which
            differs from the default specified by
            :func:`librosa.core.zero_crossings`.

    Returns
    -------
    zcr : np.ndarray [shape=(1, t)]
        `zcr[0, i]` is the fraction of zero crossings in the
        `i` th frame

    See Also
    --------
    :func:`librosa.core.zero_crossings`
    '''

    util.valid_audio(y)

    if center:
        y = np.pad(y, int(frame_length / 2), mode='edge')

    y_framed = util.frame(y, frame_length, hop_length)

    kwargs['axis'] = 0
    kwargs.setdefault('pad', False)

    crossings = zero_crossings(y_framed, **kwargs)

    return np.mean(crossings, axis=0, keepdims=True)


# -- Chroma --#
@cache
def logfsgram(y=None, sr=22050, S=None, n_fft=4096, hop_length=512, **kwargs):
    '''Compute a log-frequency spectrogram (piano roll) using a fixed-window STFT.

    Examples
    --------
    >>> # From time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.logfsgram(y=y, sr=sr)
    array([[  9.255e-01,   1.649e+00, ...,   9.232e-07,   8.588e-07],
           [  1.152e-22,   2.052e-22, ...,   1.149e-28,   1.069e-28],
           ...,
           [  1.919e-04,   2.465e-04, ...,   1.740e-08,   1.128e-08],
           [  4.007e-04,   4.216e-04, ...,   4.577e-09,   1.997e-09]])

    >>> # Convert to (unnormalized) chroma
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S_log = librosa.feature.logfsgram(y=y, sr=sr)
    >>> chroma_map = librosa.filters.cq_to_chroma(S_log.shape[0])
    >>> chroma_map.dot(S_log)
    array([[  2.524e+02,   2.484e+02, ...,   6.902e-06,   7.132e-06],
           [  3.245e+00,   3.303e+02, ...,   9.670e-06,   4.191e-06],
           ...,
           [  4.675e+01,   6.706e+01, ...,   5.044e-06,   3.260e-06],
           [  2.253e+02,   2.426e+02, ...,   8.244e-06,   6.981e-06]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        (optional) power spectrogram

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See :func:`librosa.core.stft` for details.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave. Defaults to 12.

    tuning : float in `[-0.5,  0.5)` [scalar]
        Deviation (in fractions of a bin) from A440 tuning.

        If not provided, it will be automatically estimated.

    kwargs : additional keyword arguments
        See :func:`librosa.filters.logfrequency()`

    Returns
    -------
    P : np.ndarray [shape=(n_pitches, t)]
        `P[f, t]` contains the energy at pitch bin `f`, frame `t`.
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a power spectrogram
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**2

    else:
        n_fft = (S.shape[0] - 1) * 2

    # If we don't have tuning already, grab it from S
    if 'tuning' not in kwargs:
        bins_per_oct = kwargs.get('bins_per_octave', 12)
        kwargs['tuning'] = estimate_tuning(S=S, sr=sr,
                                           bins_per_octave=bins_per_oct)

    # Build the CQ basis
    cq_basis = filters.logfrequency(sr, n_fft=n_fft, **kwargs)

    return cq_basis.dot(S)


@cache
def chromagram(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048,
               hop_length=512, tuning=None, **kwargs):
    """Compute a chromagram from a spectrogram or waveform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.chromagram(y=y, sr=sr)
    array([[ 0.548,  0.293, ...,  0.698,  0.677],
           [ 0.984,  0.369, ...,  0.945,  0.48 ],
           ...,
           [ 0.424,  0.466, ...,  0.747,  0.616],
           [ 0.568,  0.33 , ...,  0.652,  0.565]])

    >>> # Use a pre-computed spectrogram with a larger frame
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y, n_fft=4096))
    >>> librosa.feature.chromagram(S=S, sr=sr)
    array([[ 0.591,  0.336, ...,  0.821,  0.831],
           [ 0.677,  0.46 , ...,  0.961,  0.963],
           ...,
           [ 0.499,  0.276, ...,  0.914,  0.881],
           [ 0.593,  0.388, ...,  0.819,  0.764]])


    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : int > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        power spectrogram

    norm : float or None
        Column-wise normalization.
        See :func:`librosa.util.normalize` for details.

        If `None`, no normalization is performed.

    n_fft : int  > 0 [scalar]
        FFT window size if provided `y, sr` instead of `S`

    hop_length : int > 0 [scalar]
        hop length if provided `y, sr` instead of `S`

    tuning : float in `[-0.5, 0.5)` [scalar] or None.
        Deviation from A440 tuning in fractional bins (cents).
        If `None`, it is automatically estimated.

    kwargs : additional keyword arguments
        Arguments to parameterize chroma filters.
        See :func:`librosa.filters.chroma()` for details.

    Returns
    -------
    chromagram  : np.ndarray [shape=(n_chroma, t)]
        Normalized energy for each chroma bin at each frame.
    """

    n_chroma = kwargs.get('n_chroma', 12)

    # Build the power spectrogram if unspecified
    if S is None:
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**2
    else:
        n_fft = (S.shape[0] - 1) * 2

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = filters.chroma(sr, n_fft, **kwargs)

    # Compute raw chroma
    raw_chroma = np.dot(chromafb, S)

    # Compute normalization factor for each frame
    if norm is None:
        return raw_chroma

    return util.normalize(raw_chroma, norm=norm, axis=0)


# -- Mel spectrogram and MFCCs -- #
@cache
def mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients

    Examples
    --------
    >>> # Generate mfccs from a time series
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[ -4.722e+02,  -4.107e+02, ...,  -5.234e+02,  -5.234e+02],
           [  6.304e+01,   1.260e+02, ...,   2.753e-14,   2.753e-14],
           ...,
           [ -6.652e+00,  -7.556e+00, ...,   1.865e-14,   1.865e-14],
           [ -3.458e+00,  -4.677e+00, ...,   3.020e-14,   3.020e-14]])

    >>> # Use a pre-computed log-power Mel spectrogram
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.logamplitude(S))
    array([[ -4.659e+02,  -3.988e+02, ...,  -5.212e+02,  -5.212e+02],
           [  6.631e+01,   1.305e+02, ...,  -2.842e-14,  -2.842e-14],
           ...,
           [ -1.608e+00,  -3.963e+00, ...,   1.421e-14,   1.421e-14],
           [ -1.480e+00,  -4.348e+00, ...,   2.487e-14,   2.487e-14]])

    >>> # Get more components
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Parameters
    ----------
    y     : np.ndarray [shape=(n,)] or None
        audio time series

    sr    : int > 0 [scalar]
        sampling rate of `y`

    S     : np.ndarray [shape=(d, t)] or None
        log-power Mel spectrogram

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    kwargs : additional keyword arguments
        Arguments to :func:`librosa.feature.melspectrogram`, if operating
        on time series input

    Returns
    -------
    M     : np.ndarray [shape=(n_mfcc, t)]
        MFCC sequence
    """

    if S is None:
        S = logamplitude(melspectrogram(y=y, sr=sr, **kwargs))

    return np.dot(filters.dct(n_mfcc, S.shape[0]), S)


@cache
def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   **kwargs):
    """Compute a Mel-scaled power spectrogram.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[  1.223e-02,   2.988e-02, ...,   1.354e-08,   1.497e-09],
           [  4.341e-02,   2.063e+00, ...,   9.532e-08,   2.233e-09],
           ...,
           [  2.473e-11,   1.167e-10, ...,   1.130e-15,   3.280e-17],
           [  1.477e-13,   6.739e-13, ...,   4.292e-17,   1.718e-18]])

    >>> # Using a pre-computed power spectrogram
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D)

    >>> # Passing through arguments to the Mel filters
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                           fmax=8000)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : int > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        magnitude or power spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See :func:`librosa.core.stft()`

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See :func:`librosa.filters.mel()` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel power spectrogram
    """

    # Compute the STFT
    if S is None:
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**2
    else:
        n_fft = 2 * (S.shape[0] - 1)

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)
