#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature extraction routines."""

import numpy as np
import scipy.signal

import librosa.core
import librosa.util
from . import cache


# -- Spectral features -- #
@cache
def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                      freq=None):
    '''Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    :usage:
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

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
          Center frequencies for spectrogram bins.
          If `None`, then FFT bin center frequencies are used.
          Otherwise, it can be a single array of `d` center frequencies,
          or a matrix of center frequencies as constructed by
          :func:`librosa.core.ifgram`

    :returns:
      - centroid : np.ndarray [shape=(1, t)]
          centroid frequencies
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
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
        freq = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    # Column-normalize S
    return np.sum(freq * librosa.util.normalize(S, norm=1, axis=0),
                  axis=0, keepdims=True)


@cache
def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                       freq=None, centroid=None, norm=True, p=2):
    '''Compute p'th-order spectral bandwidth:

        (sum_k S[k] * (freq[k] - centroid)**p)**(1/p)

    :usage:
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

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
          Center frequencies for spectrogram bins.
          If `None`, then FFT bin center frequencies are used.
          Otherwise, it can be a single array of `d` center frequencies,
          or a matrix of center frequencies as constructed by
          :func:`librosa.core.ifgram`

      - centroid : None or np.ndarray [shape=(1, t)]
          pre-computed centroid frequencies

      - norm : bool
          Normalize per-frame spectral energy (sum to one)

      - p : float > 0
          Power to raise deviation from spectral centroid.

    :returns:
      - bandwidth : np.ndarray [shape=(1, t)]
          frequency bandwidth for each frame
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
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
        freq = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        deviation = np.abs(np.subtract.outer(freq, np.squeeze(centroid)))
    else:
        deviation = np.abs(freq - np.squeeze(centroid))

    # Column-normalize S
    if norm:
        S = librosa.util.normalize(S, norm=1, axis=0)

    return np.sum(S * deviation**p, axis=0, keepdims=True)**(1./p)


@cache
def spectral_contrast(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                      freq=None, n_bands=6):
    '''Compute spectral contrast

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
          Center frequencies for spectrogram bins.
          If `None`, then FFT bin center frequencies are used.
          Otherwise, it can be a single array of `d` center frequencies,
          or a matrix of center frequencies as constructed by
          :func:`librosa.core.ifgram`

      - n_bands : int > 1
          number of frequency bands

    :returns:
      - contrast : np.ndarray [shape=(n_bands + 1, t)]
          each row of spectral contrast values corresponds to a given
          octave-based frequency
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    # Compute the center frequencies of each bin
    if freq is None:
        freq = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

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

    :usage:
        >>> # From time-series input
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.spectral_rolloff(y=y, sr=sr)
        array([[  936.694,   635.229, ...,  3983.643,  3886.743]])

        >>> # With a higher roll percentage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
        array([[ 2637.817,  1496.558, ...,  6955.225,  6933.691]])

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
          Center frequencies for spectrogram bins.
          If `None`, then FFT bin center frequencies are used.
          Otherwise, it can be a single array of `d` center frequencies,

      - roll_percent : float [0 < roll_percent < 1]
          Roll-off percentage.

    :returns:
      - rolloff : np.ndarray [shape=(1, t)]
          roll-off frequency for each frame
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
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
        freq = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    total_energy = np.cumsum(S, axis=0)

    threshold = roll_percent * total_energy[-1]

    ind = np.where(total_energy < threshold, np.nan, 1)

    return np.nanmin(ind * freq, axis=0, keepdims=True)


@cache
def rms(y=None, sr=22050, S=None, n_fft=2048, hop_length=512):
    '''Compute root-mean-square (RMS) energy for each frame.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.rms(y=y, sr=sr)
        array([[  1.204e-01,   6.263e-01, ...,   1.413e-04,   2.191e-05]],
              dtype=float32)

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

    :returns:
      - rms : np.ndarray [shape=(1, t)]
          RMS value for each frame
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    return np.sqrt(np.mean(np.abs(S)**2, axis=0, keepdims=True))


@cache
def poly_features(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                  order=1, freq=None):
    '''Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.


    :usage:
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

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) spectrogram magnitude

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - order : int > 0
          order of the polynomial to fit

      - freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
          Center frequencies for spectrogram bins.
          If `None`, then FFT bin center frequencies are used.
          Otherwise, it can be a single array of `d` center frequencies,
          or a matrix of center frequencies as constructed by
          :func:`librosa.core.ifgram`

    :returns:
      - coefficients : np.ndarray [shape=(order+1, t)]
          polynomial coefficients for each frame
    '''
    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a magnitude spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        # Infer n_fft from spectrogram shape
        n_fft = (S.shape[0] - 1) * 2

    # Compute the center frequencies of each bin
    if freq is None:
        freq = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

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

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.zero_crossing_rate(y)
        array([[ 0.072,  0.074, ...,  0.091,  0.038]])

    :parameters:
        - y : np.ndarray [shape=(n,)]
          Audio time series

        - frame_length : int > 0
          Length of the frame over which to compute zero crossing rates

        - hop_length : int > 0
          Number of samples to advance for each frame

        - center : bool
          If true, frames are centered by padding the edges of ``y``.
          .. seealso:: :func:`librosa.stft`

        - kwargs : additional keyword arguments
          See :func:`librosa.zero_crossings`
          .. note:: By default, the ``pad`` parameter is set to ``False``

    :returns:
        - zcr : np.ndarray [shape=(1, t)]
          ``zcr[0, i]`` is the fraction of zero crossings in the ``i``th frame
    '''

    librosa.util.valid_audio(y)

    if center:
        y = np.pad(y, int(frame_length / 2), mode='edge')

    y_framed = librosa.util.frame(y, frame_length, hop_length)

    kwargs['axis'] = 0
    kwargs.setdefault('pad', False)

    crossings = librosa.core.zero_crossings(y_framed, **kwargs)

    return np.mean(crossings, axis=0, keepdims=True)


# -- Chroma --#
@cache
def logfsgram(y=None, sr=22050, S=None, n_fft=4096, hop_length=512, **kwargs):
    '''Compute a log-frequency spectrogram (piano roll) using a fixed-window STFT.

    :usage:
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

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)] or None
          (optional) power spectrogram

      - n_fft : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length for STFT. See :func:`librosa.core.stft` for details.

      - bins_per_octave : int > 0 [scalar]
          Number of bins per octave. Defaults to 12.

      - tuning : float in ``[-0.5,  0.5)`` [scalar]
          Deviation (in fractions of a bin) from A440 tuning.

          If not provided, it will be automatically estimated.

      - *kwargs*
          Additional keyword arguments.
          See :func:`librosa.filters.logfrequency()`

    :returns:
      - P : np.ndarray [shape=(n_pitches, t)]
          P(f, t) contains the energy at pitch bin f, frame t.

    .. note:: One of either ``S`` or ``y`` must be provided.
          If ``y`` is provided, the power spectrogram is computed
          automatically given the parameters ``n_fft`` and ``hop_length``.

          If ``S`` is provided, it is used as the input spectrogram, and
          ``n_fft`` is inferred from its shape.
    '''

    # If we don't have a spectrogram, build one
    if S is None:
        # By default, use a power spectrogram
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    else:
        n_fft = (S.shape[0] - 1) * 2

    # If we don't have tuning already, grab it from S
    if 'tuning' not in kwargs:
        bins_per_oct = kwargs.get('bins_per_octave', 12)
        kwargs['tuning'] = estimate_tuning(S=S, sr=sr,
                                           bins_per_octave=bins_per_oct)

    # Build the CQ basis
    cq_basis = librosa.filters.logfrequency(sr, n_fft=n_fft, **kwargs)

    return cq_basis.dot(S)


@cache
def chromagram(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048,
               hop_length=512, tuning=None, **kwargs):
    """Compute a chromagram from a spectrogram or waveform

    :usage:
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


    :parameters:
      - y          : np.ndarray [shape=(n,)] or None
          audio time series

      - sr         : int > 0 [scalar]
          sampling rate of ``y``

      - S          : np.ndarray [shape=(d, t)] or None
          power spectrogram

      - norm       : float or None
          Column-wise normalization.
          See :func:`librosa.util.normalize` for details.

          If ``None``, no normalization is performed.

      - n_fft      : int  > 0 [scalar]
          FFT window size if provided ``y, sr`` instead of ``S``

      - hop_length : int > 0 [scalar]
          hop length if provided ``y, sr`` instead of ``S``

      - tuning : float in ``[-0.5, 0.5)`` [scalar] or None.
          Deviation from A440 tuning in fractional bins (cents).
          If ``None``, it is automatically estimated.

      - *kwargs*
          Additional keyword arguments to parameterize chroma filters.
          See :func:`librosa.filters.chroma()` for details.

    .. note:: One of either ``S`` or ``y`` must be provided.
          If ``y`` is provided, the magnitude spectrogram is computed
          automatically given the parameters ``n_fft`` and ``hop_length``.
          If ``S`` is provided, it is used as the input spectrogram, and
          ``n_fft`` is inferred from its shape.

    :returns:
      - chromagram  : np.ndarray [shape=(n_chroma, t)]
          Normalized energy for each chroma bin at each frame.

    :raises:
      - ValueError
          if an improper value is supplied for norm
    """

    n_chroma = kwargs.get('n_chroma', 12)

    # Build the power spectrogram if unspecified
    if S is None:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    else:
        n_fft = (S.shape[0] - 1) * 2

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = librosa.filters.chroma(sr, n_fft, **kwargs)

    # Compute raw chroma
    raw_chroma = np.dot(chromafb, S)

    # Compute normalization factor for each frame
    if norm is None:
        return raw_chroma

    return librosa.util.normalize(raw_chroma, norm=norm, axis=0)


# -- Pitch and tuning -- #
@cache
def estimate_tuning(resolution=0.01, bins_per_octave=12, **kwargs):
    '''Estimate the tuning of an audio time series or spectrogram input.

    :usage:
        >>> # With time-series input
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.estimate_tuning(y=y, sr=sr)
        0.070000000000000062

        >>> # In tenths of a cent
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.estimate_tuning(y=y, sr=sr, resolution=1e-3))
        0.071000000000000063

        >>> # Using spectrogram input
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> S = np.abs(librosa.stft(y))
        >>> librosa.feature.estimate_tuning(S=S, sr=sr)
        0.089999999999999969

        >>> # Using pass-through arguments to ``librosa.feature.piptrack``
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.feature.estimate_tuning(y=y, sr=sr, n_fft=8192,
                                            fmax=librosa.midi_to_hz(128))
        0.070000000000000062

    :parameters:
      - resolution : float in ``(0, 1)``
          Resolution of the tuning as a fraction of a bin.
          0.01 corresponds to cents.

      - bins_per_octave : int > 0 [scalar]
          How many frequency bins per octave

      - *kwargs*
          Additional keyword arguments.  See :func:`librosa.feature.piptrack`

    :returns:
      - tuning: float in ``[-0.5, 0.5)``
          estimated tuning deviation (fractions of a bin)
    '''

    pitch, mag = librosa.feature.piptrack(**kwargs)

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return librosa.feature.pitch_tuning(pitch[(mag > threshold) & pitch_mask],
                                        resolution=resolution,
                                        bins_per_octave=bins_per_octave)


@cache
def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    '''Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.

    :usage:
        >>> # Generate notes at +25 cents
        >>> freqs = librosa.cqt_frequencies(24, 55, tuning=0.25)
        >>> librosa.feature.pitch_tuning(freqs)
        0.25

        >>> # Track frequencies from a real spectrogram
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> pitches, magnitudes, stft = librosa.feature.ifptrack(y, sr)
        >>> # Select out pitches with high energy
        >>> pitches = pitches[magnitudes > np.median(magnitudes)]
        >>> librosa.feature.pitch_tuning(pitches)
        0.089999999999999969

    :parameters:
      - frequencies : array-like, float
          A collection of frequencies detected in the signal.
          See :func:`librosa.feature.piptrack`

      - resolution : float in ``(0, 1)``
          Resolution of the tuning as a fraction of a bin.
          0.01 corresponds to cents.

      - bins_per_octave : int > 0 [scalar]
          How many frequency bins per octave

    :returns:
      - tuning: float in ``[-0.5, 0.5)``
          estimated tuning deviation (fractions of a bin)

    .. seealso::
      - :func:`librosa.feature.estimate_tuning`
        For estimating tuning from time-series or spectrogram input
    '''

    frequencies = np.asarray([frequencies], dtype=float).flatten()

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave * librosa.core.hz_to_octs(frequencies),
                      1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, np.ceil(1./resolution), endpoint=False)

    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]


@cache
def ifptrack(y, sr=22050, n_fft=4096, hop_length=None, fmin=None,
             fmax=None, threshold=0.75):
    '''Instantaneous pitch frequency tracking.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> pitches, magnitudes, D = librosa.feature.ifptrack(y, sr=sr)

    :parameters:
      - y: np.ndarray [shape=(n,)]
          audio signal

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - n_fft: int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar] or None
          Hop size for STFT.  Defaults to ``n_fft / 4``.
          See :func:`librosa.core.stft()` for details.

      - threshold : float in ``(0, 1)``
          Maximum fraction of expected frequency increment to tolerate

      - fmin : float or tuple of float
          Ramp parameter for lower frequency cutoff.

          If scalar, the ramp has 0 width.

          If tuple, a linear ramp is applied from ``fmin[0]`` to ``fmin[1]``

          Default: (150.0, 300.0)

      - fmax : float or tuple of float
          Ramp parameter for upper frequency cutoff.

          If scalar, the ramp has 0 width.

          If tuple, a linear ramp is applied from ``fmax[0]`` to ``fmax[1]``

          Default: (2000.0, 4000.0)

    :returns:
      - pitches : np.ndarray [shape=(d, t)]
      - magnitudes : np.ndarray [shape=(d, t)]
          Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.

          ``pitches[i, t]`` contains instantaneous frequencies at time ``t``

          ``magnitudes[i, t]`` contains their magnitudes.

      - D : np.ndarray [shape=(d, t), dtype=complex]
          STFT matrix
    '''

    if fmin is None:
        fmin = (150.0, 300.0)

    if fmax is None:
        fmax = (2000.0, 4000.0)

    fmin = np.asarray([fmin]).squeeze()
    fmax = np.asarray([fmax]).squeeze()

    # Truncate to feasible region
    fmin = np.maximum(0, fmin)
    fmax = np.minimum(fmax, float(sr) / 2)

    # What's our DFT bin resolution?
    fft_res = float(sr) / n_fft

    # Only look at bins up to 2 kHz
    max_bin = int(round(fmax[-1] / fft_res))

    if hop_length is None:
        hop_length = int(n_fft / 4)

    # Calculate the inst freq gram
    if_gram, D = librosa.core.ifgram(y, sr=sr,
                                     n_fft=n_fft,
                                     win_length=int(n_fft/2),
                                     hop_length=hop_length)

    # Find plateaus in ifgram - stretches where delta IF is < thr:
    # ie, places where the same frequency is spread across adjacent bins
    idx_above = list(range(1, max_bin)) + [max_bin - 1]
    idx_below = [0] + list(range(0, max_bin - 1))

    # expected increment per bin = sr/w, threshold at 3/4 that
    matches = (abs(if_gram[idx_above] - if_gram[idx_below])
               < (threshold * fft_res))

    # mask out any singleton bins (where both above and below are zero)
    matches = matches * ((matches[idx_above] > 0) | (matches[idx_below] > 0))

    pitches = np.zeros_like(matches, dtype=float)
    magnitudes = np.zeros_like(matches, dtype=float)

    # For each frame, extract all harmonic freqs & magnitudes
    for t in range(matches.shape[1]):

        # find nonzero regions in this vector
        # The mask selects out constant regions + active borders
        mask = ~np.pad(matches[:, t], 1, mode='constant')

        starts = np.argwhere(matches[:, t] & mask[:-2]).astype(int)
        ends = 1 + np.argwhere(matches[:, t] & mask[2:]).astype(int)

        # Set up inner loop
        frqs = np.zeros_like(starts, dtype=float)
        mags = np.zeros_like(starts, dtype=float)

        for i, (start_i, end_i) in enumerate(zip(starts, ends)):

            start_i = np.asscalar(start_i)
            end_i = np.asscalar(end_i)

            # Weight frequencies by energy
            weights = np.abs(D[start_i:end_i, t])
            mags[i] = weights.sum()

            # Compute the weighted average frequency.
            # FIXME: is this the right thing to do?
            # These are frequencies... shouldn't this be a
            # weighted geometric average?
            frqs[i] = weights.dot(if_gram[start_i:end_i, t])
            if mags[i] > 0:
                frqs[i] /= mags[i]

        # Clip outside the ramp zones
        idx = (fmax[-1] < frqs) | (frqs < fmin[0])
        mags[idx] = 0
        frqs[idx] = 0

        # Ramp down at the high end
        idx = (fmax[-1] > frqs) & (frqs > fmax[0])
        mags[idx] *= (fmax[-1] - frqs[idx]) / (fmax[-1] - fmax[0])

        # Ramp up from the bottom end
        idx = (fmin[-1] > frqs) & (frqs > fmin[0])
        mags[idx] *= (frqs[idx] - fmin[0]) / (fmin[-1] - fmin[0])

        # Assign pitch and magnitude to their center bin
        bins = (starts + ends) / 2
        pitches[bins, t] = frqs
        magnitudes[bins, t] = mags

    return pitches, magnitudes, D


@cache
def piptrack(y=None, sr=22050, S=None, n_fft=4096, fmin=150.0,
             fmax=4000.0, threshold=.1):
    '''Pitch tracking on thresholded parabolically-interpolated STFT

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> pitches, magnitudes = librosa.feature.piptrack(y=y, sr=sr)

    :parameters:
      - y: np.ndarray [shape=(n,)] or None
          audio signal

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - S: np.ndarray [shape=(d, t)] or None
          magnitude or power spectrogram

      - n_fft : int > 0 [scalar] or None
          number of fft bins to use, if ``y`` is provided.

      - threshold : float in ``(0, 1)``
          A bin in spectrum X is considered a pitch when it is greater than
          ``threshold*X.max()``

      - fmin : float > 0 [scalar]
          lower frequency cutoff.

      - fmax : float > 0 [scalar]
          upper frequency cutoff.

    .. note::
        One of ``S`` or ``y`` must be provided.

        If ``S`` is not given, it is computed from ``y`` using
        the default parameters of ``stft``.

    :returns:
      - pitches : np.ndarray [shape=(d, t)]
      - magnitudes : np.ndarray [shape=(d,t)]
          Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.

          ``pitches[f, t]`` contains instantaneous frequency at bin
          ``f``, time ``t``

          ``magnitudes[f, t]`` contains the corresponding magnitudes.

          .. note:: Both ``pitches`` and ``magnitudes`` take value 0 at bins
            of non-maximal magnitude.

    .. note::
      https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html
    '''

    # Check that we received an audio time series or STFT
    if S is None:
        if y is None:
            raise ValueError('Either "y" or "S" must be provided')
        S = np.abs(librosa.core.stft(y, n_fft=n_fft))

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    # Pre-compute FFT frequencies
    n_fft = 2 * (S.shape[0] - 1)
    fft_freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]
    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (shift < librosa.SMALL_FLOAT))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode='constant')
    shift = np.pad(shift, ([1, 1], [0, 0]), mode='constant')

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    idx = np.argwhere(freq_mask &
                      librosa.core.localmax(S * (S > (threshold
                                                      * S.max(axis=0)))))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]])
                                     * float(sr) / n_fft)

    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]]
                                  + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags


# -- Mel spectrogram and MFCCs -- #
@cache
def mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients

    :usage:
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

    :parameters:
      - y     : np.ndarray [shape=(n,)] or None
          audio time series

      - sr    : int > 0 [scalar]
          sampling rate of ``y``

      - S     : np.ndarray [shape=(d, t)] or None
          log-power Mel spectrogram

      - n_mfcc: int > 0 [scalar]
          number of MFCCs to return

      - *kwargs*
          Additional keyword arguments for
          :func:`librosa.feature.melspectrogram`, if operating on time series

    .. note::
        One of ``S`` or ``y, sr`` must be provided.

        If ``S`` is not given, it is computed from ``y, sr`` using
        the default parameters of ``melspectrogram``.

    :returns:
      - M     : np.ndarray [shape=(n_mfcc, t)]
          MFCC sequence
    """

    if S is None:
        S = librosa.logamplitude(melspectrogram(y=y, sr=sr, **kwargs))

    return np.dot(librosa.filters.dct(n_mfcc, S.shape[0]), S)


@cache
def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   **kwargs):
    """Compute a Mel-scaled power spectrogram.

    :usage:
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

    :parameters:
      - y : np.ndarray [shape=(n,)] or None
          audio time-series

      - sr : int > 0 [scalar]
          sampling rate of ``y``

      - S : np.ndarray [shape=(d, t)]
          magnitude or power spectrogram

      - n_fft : int > 0 [scalar]
          length of the FFT window

      - hop_length : int > 0 [scalar]
          number of samples between successive frames.

          See :func:`librosa.core.stft()`

      - *kwargs*
          Additional keyword arguments for mel filterbank parameters.
          See :func:`librosa.filters.mel()` for details.

    .. note:: One of either ``S`` or ``y, sr`` must be provided.
        If the pair ``y, sr`` is provided, the power spectrogram is computed.

        If ``S`` is provided, it is used as the spectrogram, and the
        parameters ``y, n_fft, hop_length`` are ignored.

    :returns:
      - S : np.ndarray [shape=(n_mels, t)]
          Mel power spectrogram
    """

    # Compute the STFT
    if S is None:
        S = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    else:
        n_fft = 2 * (S.shape[0] - 1)

    # Build a Mel filter
    mel_basis = librosa.filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


# -- miscellaneous utilities -- #
@cache
def delta(data, width=9, order=1, axis=-1, trim=True):
    '''Compute delta features.

    :usage:
        >>> # Compute MFCC deltas, delta-deltas
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

    :parameters:
      - data      : np.ndarray [shape=(d, T)]
          the input data matrix (eg, spectrogram)

      - width     : int > 0, odd [scalar]
          Number of frames over which to compute the delta feature

      - order     : int > 0 [scalar]
          the order of the difference operator.
          1 for first derivative, 2 for second, etc.

      - axis      : int [scalar]
          the axis along which to compute deltas.
          Default is -1 (columns).

      - trim      : bool
          set to True to trim the output matrix to the original size.

    :returns:
      - delta_data   : np.ndarray [shape=(d, t) or (d, t + window)]
          delta matrix of ``data``.
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

    Each column ``data[:, i]`` is mapped to::

        data[:, i] ->  [ data[:, i],                        ...
                         data[:, i - delay],                ...
                         ...
                         data[:, i - (n_steps-1)*delay],    ...
                       ]

    For columns ``i < (n_steps - 1) * delay`` , the data will be padded.
    By default, the data is padded with zeros, but this behavior can be
    overridden by supplying additional keyword arguments which are passed
    to ``np.pad()``.

    :usage:
        >>> # Generate a data vector
        >>> data = np.arange(-3, 3)
        >>> # Keep two steps (current and previous)
        >>> librosa.feature.stack_memory(data)
        array([[-3, -2, -1,  0,  1,  2],
               [ 0, -3, -2, -1,  0,  1]])

        >>> # Or three steps
        >>> librosa.feature.stack_memory(data, n_steps=3)
        array([[-3, -2, -1,  0,  1,  2],
               [ 0, -3, -2, -1,  0,  1],
               [ 0,  0, -3, -2, -1,  0]])

        >>> # Use reflection padding instead of zero-padding
        >>> librosa.feature.stack_memory(data, n_steps=3, mode='reflect')
        array([[-3, -2, -1,  0,  1,  2],
               [-2, -3, -2, -1,  0,  1],
               [-1, -2, -3, -2, -1,  0]])

        >>> # Or pad with edge-values, and delay by 2
        >>> librosa.feature.stack_memory(data, n_steps=3, delay=2, mode='edge')
        array([[-3, -2, -1,  0,  1,  2],
               [-3, -3, -3, -2, -1,  0],
               [-3, -3, -3, -3, -3, -2]])

    :parameters:
      - data : np.ndarray [shape=(t,) or (d, t)]
          Input data matrix.  If ``data`` is a vector (``data.ndim == 1``),
          it will be interpreted as a row matrix and reshaped to ``(1, t)``.

      - n_steps : int > 0 [scalar]
          embedding dimension, the number of steps back in time to stack

      - delay : int > 0 [scalar]
          the number of columns to step

      - *kwargs*
          Additional arguments to pass to ``np.pad``.

    :returns:
      - data_history : np.ndarray [shape=(m * d, t)]
          data augmented with lagged copies of itself,
          where ``m == n_steps - 1``.
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

    :usage:
        >>> # Beat-synchronous MFCCs
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
        >>> # By default, use mean aggregation
        >>> mfcc_avg = librosa.feature.sync(mfcc, beats)
        >>> # Use median-aggregation instead of mean
        >>> mfcc_med = librosa.feature.sync(mfcc, beats,
                                            aggregate=np.median)
        >>> # Or max aggregation
        >>> mfcc_max = librosa.feature.sync(mfcc, beats,
                                            aggregate=np.max)

    :parameters:
      - data      : np.ndarray [shape=(d, T)]
          matrix of features

      - frames    : np.ndarray [shape=(m,)]
          ordered array of frame segment boundaries

      - aggregate : function
          aggregation function (defualt: ``np.mean``)

      - pad : boolean
          If true, `frames` is padded to span the full range [0, T]

    :returns:
      - Y         : ndarray [shape=(d, M)]
          ``Y[:, i] = aggregate(data[:, F[i-1]:F[i]], axis=1)``

    .. note::
        In order to ensure total coverage, boundary points may be added
        to ``frames``.

        If synchronizing a feature matrix against beat tracker output, ensure
        that frame numbers are properly aligned and use the same hop length.
    """

    if data.ndim < 2:
        data = np.asarray([data])

    elif data.ndim > 2:
        raise ValueError('Synchronized data has ndim={:d},'
                         ' must be 1 or 2.'.format(data.ndim))

    if aggregate is None:
        aggregate = np.mean

    (dimension, n_frames) = data.shape

    frames = librosa.util.fix_frames(frames, 0, n_frames, pad=pad)

    if min(frames) < 0:
        raise ValueError('Negative frame index.')

    elif max(frames) > n_frames:
        raise ValueError('Frame index exceeds data length.')

    data_agg = np.empty((dimension, len(frames)-1), order='F')

    start = frames[0]

    for (i, end) in enumerate(frames[1:]):
        data_agg[:, i] = aggregate(data[:, start:end], axis=1)
        start = end

    return data_agg
