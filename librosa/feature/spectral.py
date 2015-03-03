#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

import numpy as np

from .. import cache
from .. import util
from .. import filters

from ..core.time_frequency import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import logamplitude, _spectrogram
from ..core.constantq import cqt, hybrid_cqt
from ..core.pitch import estimate_tuning


__all__ = ['spectral_centroid',
           'spectral_bandwidth',
           'spectral_contrast',
           'spectral_rolloff',
           'poly_features',
           'rmse',
           'zero_crossing_rate',
           'chroma_stft',
           'chroma_cqt',
           'melspectrogram',
           'mfcc',
           # Deprecated functions
           'logfsgram',
           'chromagram',
           'tonnetz']


# -- Spectral features -- #
@cache
def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                      freq=None):
    '''Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

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
        hop length for STFT. See `librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.core.ifgram`

    Returns
    -------
    centroid : np.ndarray [shape=(1, t)]
        centroid frequencies

    See Also
    --------
    librosa.core.stft
        Short-time Fourier Transform

    librosa.core.ifgram
        Instantaneous-frequency spectrogram

    Examples
    --------
    From time-series input:

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    >>> cent
    array([[  545.929,   400.609, ...,  1621.184,  1591.604]])

    From spectrogram input:

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[  545.929,   400.609, ...,  1621.184,  1591.604]])

    Using variable bin center frequencies:

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    array([[  545.069,   400.764, ...,  1621.139,  1590.362]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(cent.T, label='Spectral centroid')
    >>> plt.ylabel('Hz')
    >>> plt.xticks([])
    >>> plt.xlim([0, cent.shape[-1]])
    >>> plt.legend()
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()
    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

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
        hop length for STFT. See `librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.core.ifgram`

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


    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    >>> spec_bw
    array([[ 1201.067,   920.588, ...,  2218.177,  2211.325]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_bandwidth(S=S)
    array([[ 1201.067,   920.588, ...,  2218.177,  2211.325]])

    Using variable bin center frequencies

    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
    array([[ 1202.514,   920.453, ...,  2218.172,  2213.157]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(spec_bw.T, label='Spectral bandwidth')
    >>> plt.ylabel('Hz')
    >>> plt.xticks([])
    >>> plt.xlim([0, spec_bw.shape[-1]])
    >>> plt.legend()
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

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
                      freq=None, fmin=200.0, n_bands=6, quantile=0.02,
                      linear=False):
    '''Compute spectral contrast [1]_

    .. [1] Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao,
           and Lian-Hong Cai.
           "Music type classification by spectral contrast feature."
           In Multimedia and Expo, 2002. ICME'02. Proceedings.
           2002 IEEE International Conference on, vol. 1, pp. 113-116.
           IEEE, 2002.

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
        hop length for STFT. See `librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies.

    fmin : float > 0
        Frequency cutoff for the first bin `[0, fmin]`
        Subsequent bins will cover `[fmin, 2*fmin]`, `[2*fmin, 4*fmin]`, etc.

    n_bands : int > 1
        number of frequency bands

    quantile : float in (0, 1)
        quantile for determining peaks and valleys

    linear : bool
        If `True`, return the linear difference of magnitudes:
        `peaks - valleys`.

        If `False`, return the logarithmic difference:
        `log(peaks) - log(valleys)`.


    Returns
    -------
    contrast : np.ndarray [shape=(n_bands + 1, t)]
        each row of spectral contrast values corresponds to a given
        octave-based frequency


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(S ** 2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(contrast, x_axis='time')
    >>> plt.colorbar()
    >>> plt.ylabel('Frequency bands')
    >>> plt.title('Spectral contrast')
    >>> plt.tight_layout()
    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    freq = np.atleast_1d(freq)

    if freq.ndim != 1 or len(freq) != S.shape[0]:
        raise ValueError('freq.shape mismatch: expected '
                         '({:d},)'.format(S.shape[0]))

    if n_bands < 1 or not isinstance(n_bands, int):
        raise ValueError('n_bands must be a positive integer')

    if not (0.0 < quantile < 1.0):
        raise ValueError('quantile must lie in the range (0, 1)')

    if fmin <= 0:
        raise ValueError('fmin must be a positive number')

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0**np.arange(0, n_bands + 1))

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

        # Always take at least one bin from each side
        idx = np.rint(quantile * np.sum(current_band))
        idx = int(np.maximum(idx, 1))

        sortedr = np.sort(sub_band, axis=0)

        valley[k] = np.mean(sortedr[:idx], axis=0)
        peak[k] = np.mean(sortedr[-idx:], axis=0)

    if linear:
        return peak - valley
    else:
        return logamplitude(peak) - logamplitude(valley)


@cache
def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                     freq=None, roll_percent=0.85):
    '''Compute roll-off frequency

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
        hop length for STFT. See `librosa.core.stft` for details.

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,

        .. note:: `freq` is assumed to be sorted in increasing order

    roll_percent : float [0 < roll_percent < 1]
        Roll-off percentage.

    Returns
    -------
    rolloff : np.ndarray [shape=(1, t)]
        roll-off frequency for each frame


    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    >>> rolloff
    array([[  936.694,   635.229, ...,  3983.643,  3886.743]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_rolloff(S=S, sr=sr)
    array([[  936.694,   635.229, ...,  3983.643,  3886.743]])

    >>> # With a higher roll percentage:
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    array([[ 2637.817,  1496.558, ...,  6955.225,  6933.691]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(rolloff.T, label='Roll-off frequency')
    >>> plt.ylabel('Hz')
    >>> plt.xticks([])
    >>> plt.xlim([0, rolloff.shape[-1]])
    >>> plt.legend()
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    '''

    if not (0.0 < roll_percent < 1.0):
        raise ValueError('roll_percent must lie in the range (0, 1)')

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    if not np.isrealobj(S):
        raise ValueError('Spectral rolloff is only defined '
                         'with real-valued input')
    elif np.any(S < 0):
        raise ValueError('Spectral rolloff is only defined '
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
def rmse(y=None, S=None, n_fft=2048, hop_length=512):
    '''Compute root-mean-square (RMS) energy for each frame.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.core.stft` for details.


    Returns
    -------
    rms : np.ndarray [shape=(1, t)]
        RMS value for each frame


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.rmse(y=y)
    array([[  1.204e-01,   6.263e-01, ...,   1.413e-04,   2.191e-05]],
          dtype=float32)

    Or from spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> rms = librosa.feature.rmse(S=S)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(rms.T, label='RMS Energy')
    >>> plt.xticks([])
    >>> plt.xlim([0, rms.shape[-1]])
    >>> plt.legend(loc='best')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    '''

    S, _ = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    return np.sqrt(np.mean(np.abs(S)**2, axis=0, keepdims=True))


@cache
def poly_features(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                  order=1, freq=None):
    '''Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.

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
        hop length for STFT. See `librosa.core.stft` for details.

    order : int > 0
        order of the polynomial to fit

    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of `d` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.core.ifgram`

    Returns
    -------
    coefficients : np.ndarray [shape=(order+1, t)]
        polynomial coefficients for each frame

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))

    Line features

    >>> line = librosa.feature.poly_features(S=S, sr=sr)
    >>> line
    array([[ -9.454e-06,  -4.322e-05, ...,  -1.640e-08,  -2.626e-09],
           [  7.144e-02,   3.241e-01, ...,   1.332e-04,   2.127e-05]])

    Quadratic features

    >>> quad = librosa.feature.poly_features(S=S, order=2)
    >>> quad
    array([[  3.742e-09,   1.753e-08, ...,   5.145e-12,   8.343e-13],
           [ -5.071e-05,  -2.364e-04, ...,  -7.312e-08,  -1.182e-08],
           [  1.472e-01,   6.788e-01, ...,   2.373e-04,   3.816e-05]])

    Plot the results for comparison

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(line)
    >>> plt.colorbar()
    >>> plt.title('Line coefficients')
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(quad)
    >>> plt.colorbar()
    >>> plt.title('Quadratic coefficients')
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # If frequencies are constant over frames, then we only need to fit once
    if freq.ndim == 1:
        coefficients = np.polyfit(freq, S, order)
    else:
        # Else, fit each frame independently and stack the results
        coefficients = np.concatenate([[np.polyfit(freq[:, i], S[:, i], order)]
                                       for i in range(S.shape[1])], axis=0).T

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
        This is similar to the padding in `librosa.core.stft`,
        but uses edge-value copies instead of reflection.

    kwargs : additional keyword arguments
        See `librosa.core.zero_crossings`

        .. note:: By default, the `pad` parameter is set to `False`, which
            differs from the default specified by
            `librosa.core.zero_crossings`.

    Returns
    -------
    zcr : np.ndarray [shape=(1, t)]
        `zcr[0, i]` is the fraction of zero crossings in the
        `i` th frame

    See Also
    --------
    librosa.core.zero_crossings
        Compute zero-crossings in a time-series
    '''

    util.valid_audio(y)

    if center:
        y = np.pad(y, int(frame_length // 2), mode='edge')

    y_framed = util.frame(y, frame_length, hop_length)

    kwargs['axis'] = 0
    kwargs.setdefault('pad', False)

    crossings = zero_crossings(y_framed, **kwargs)

    return np.mean(crossings, axis=0, keepdims=True)


# -- Chroma --#
@cache
def chroma_stft(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048,
               hop_length=512, tuning=None, **kwargs):
    """Compute a chromagram from an STFT spectrogram or waveform

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
        See `librosa.util.normalize` for details.

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
        See `librosa.filters.chroma` for details.

    Returns
    -------
    chromagram  : np.ndarray [shape=(n_chroma, t)]
        Normalized energy for each chroma bin at each frame.

    See Also
    --------
    librosa.filters.chroma
        Chroma filter bank construction
    librosa.util.normalize
        Vector normalization

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.chroma_stft(y=y, sr=sr)
    array([[ 0.548,  0.293, ...,  0.698,  0.677],
           [ 0.984,  0.369, ...,  0.945,  0.48 ],
    ...,
           [ 0.424,  0.466, ...,  0.747,  0.616],
           [ 0.568,  0.33 , ...,  0.652,  0.565]])

    Use a pre-computed spectrogram with a larger frame

    >>> S = np.abs(librosa.stft(y, n_fft=4096))
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[ 0.591,  0.336, ...,  0.821,  0.831],
           [ 0.677,  0.46 , ...,  0.961,  0.963],
    ...,
           [ 0.499,  0.276, ...,  0.914,  0.881],
           [ 0.593,  0.388, ...,  0.819,  0.764]])

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Chromagram')
    >>> plt.tight_layout()

    """

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=2)

    n_chroma = kwargs.get('n_chroma', 12)

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(float(tuning) / n_chroma)

    chromafb = filters.chroma(sr, n_fft, **kwargs)

    # Compute raw chroma
    raw_chroma = np.dot(chromafb, S)

    # Compute normalization factor for each frame
    return util.normalize(raw_chroma, norm=norm, axis=0)


@cache
def chroma_cqt(y=None, sr=22050, C=None, hop_length=512, fmin=None, norm=np.inf,
               tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=None,
               mode='full'):
    r'''Constant-Q chromagram

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : int > 0
        sampling rate of `y`

    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram

    hop_length : int > 0
        number of samples between successive chroma frames

    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: 'C2' ~ 32.7 Hz

    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.

    tuning : float
        Deviation (in cents) from A440 tuning

    n_chroma : int > 0
        Number of chroma bins to produce

    n_octaves : int > 0
        Number of octaves to analyze above `fmin`

    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: matches `n_chroma`

    mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    chromagram : np.ndarray [shape=(n_chroma, t)]
        The output chromagram

    See Also
    --------
    util.normalize
    core.cqt
    core.hybrid_cqt
    chroma_stft

    Examples
    --------
    Compare a long-window STFT chromagram to the CQT chromagram


    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10, duration=15)
    >>> chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr,
    ...                                           n_chroma=12, n_fft=4096)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(chroma_stft, y_axis='chroma')
    >>> plt.title('chroma_stft')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
    >>> plt.title('chroma_cqt')
    >>> plt.colorbar()
    >>> plt.tight_layout()

    '''

    cqt_func = {'full': cqt, 'hybrid': hybrid_cqt}

    if bins_per_octave is None:
        bins_per_octave = n_chroma

    # Build the CQT if we don't have one already
    if C is None:
        C = cqt_func[mode](y, sr=sr,
                           hop_length=hop_length,
                           fmin=fmin,
                           n_bins=n_octaves * bins_per_octave,
                           bins_per_octave=bins_per_octave,
                           tuning=tuning)

    # Map to chroma
    cq_to_chr = filters.cq_to_chroma(C.shape[0],
                                     bins_per_octave=bins_per_octave,
                                     n_chroma=n_chroma,
                                     fmin=fmin)
    chroma = cq_to_chr.dot(C)

    # Normalize
    if norm is not None:
        chroma = util.normalize(chroma, norm=norm, axis=0)

    return chroma


# -- Mel spectrogram and MFCCs -- #
@cache
def mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs):
    """Mel-frequency cepstral coefficients

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
        Arguments to `melspectrogram`, if operating
        on time series input

    Returns
    -------
    M     : np.ndarray [shape=(n_mfcc, t)]
        MFCC sequence

    See Also
    --------
    melspectrogram

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[ -4.722e+02,  -4.107e+02, ...,  -5.234e+02,  -5.234e+02],
           [  6.304e+01,   1.260e+02, ...,   2.753e-14,   2.753e-14],
    ...,
           [ -6.652e+00,  -7.556e+00, ...,   1.865e-14,   1.865e-14],
           [ -3.458e+00,  -4.677e+00, ...,   3.020e-14,   3.020e-14]])

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.logamplitude(S))
    array([[ -4.659e+02,  -3.988e+02, ...,  -5.212e+02,  -5.212e+02],
           [  6.631e+01,   1.305e+02, ...,  -2.842e-14,  -2.842e-14],
    ...,
           [ -1.608e+00,  -3.963e+00, ...,   1.421e-14,   1.421e-14],
           [ -1.480e+00,  -4.348e+00, ...,   2.487e-14,   2.487e-14]])

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(mfccs, x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('MFCC')
    >>> plt.tight_layout()


    """

    if S is None:
        S = logamplitude(melspectrogram(y=y, sr=sr, **kwargs))

    return np.dot(filters.dct(n_mfcc, S.shape[0]), S)


@cache
def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   **kwargs):
    """Compute a Mel-scaled power spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : int > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        power spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel power spectrogram

    See Also
    --------
    librosa.filters.mel
        Mel filter bank construction

    librosa.core.stft
        Short-time Fourier Transform


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[  1.223e-02,   2.988e-02, ...,   1.354e-08,   1.497e-09],
           [  4.341e-02,   2.063e+00, ...,   9.532e-08,   2.233e-09],
    ...,
           [  2.473e-11,   1.167e-10, ...,   1.130e-15,   3.280e-17],
           [  1.477e-13,   6.739e-13, ...,   4.292e-17,   1.718e-18]])

    Using a pre-computed power spectrogram

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D)

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.logamplitude(S,
    ...                                               ref_power=np.max),
    ...                          y_axis='mel', fmax=8000,
    ...                          x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Mel spectrogram')
    >>> plt.tight_layout()


    """

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=2)

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


# Deprecated functions
@util.decorators.deprecated('0.4', '0.5')
@cache
def logfsgram(y=None, sr=22050, S=None, n_fft=4096,
              hop_length=512, **kwargs):  # pragma: no cover
    '''Compute a log-frequency spectrogram using a
    fixed-window STFT.

    .. warning:: Deprecated in librosa 0.4
              Functionality is superseded by `librosa.core.pseudo_cqt`

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
        hop length for STFT. See `librosa.core.stft` for details.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave. Defaults to 12.

    tuning : float in `[-0.5,  0.5)` [scalar]
        Deviation (in fractions of a bin) from A440 tuning.

        If not provided, it will be automatically estimated.

    kwargs : additional keyword arguments
        See `librosa.filters.logfrequency`

    Returns
    -------
    P : np.ndarray [shape=(n_pitches, t)]
        `P[f, t]` contains the energy at pitch bin `f`, frame `t`.


    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> L = librosa.feature.logfsgram(y=y, sr=sr)
    >>> L
    array([[  9.255e-01,   1.649e+00, ...,   9.232e-07,   8.588e-07],
           [  1.152e-22,   2.052e-22, ...,   1.149e-28,   1.069e-28],
    ...,
           [  1.919e-04,   2.465e-04, ...,   1.740e-08,   1.128e-08],
           [  4.007e-04,   4.216e-04, ...,   4.577e-09,   1.997e-09]])

    Plot the pseudo CQT

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(librosa.logamplitude(L,
    ...                                               ref_power=np.max),
    ...                          y_axis='cqt_hz', x_axis='time')
    >>> plt.title('Log-frequency power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=2)

    # If we don't have tuning already, grab it from S
    if 'tuning' not in kwargs:
        bins_per_oct = kwargs.get('bins_per_octave', 12)
        kwargs['tuning'] = estimate_tuning(S=S, sr=sr,
                                           bins_per_octave=bins_per_oct)

    # Build the CQ basis
    cq_basis = filters.logfrequency(sr, n_fft=n_fft, **kwargs)

    return cq_basis.dot(S)


@cache
def tonnetz(chromagram, norm=np.inf):
    '''Computes the tonal centroid features (tonnetz) from a 12-dimensional
    chromagram.

    Parameters
    ----------
    chromagram : np.ndarray [shape=(12, t)]
        Normalized energy for each chroma bin at each frame.

    norm : float or None
        Column-wise normalization.
        See `librosa.util.normalize` for details.

        If `None`, no normalization is performed.

    Returns
    -------
    tonnetz : np.ndarray [shape(6, t)]
        Tonal centroid features for each frame.

    See Also
    --------
    librosa.feature.chroma_cqt
        Compute a chromagram from a constat-Q transform.
    librosa.feature.chroma_stft
        Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    >>> tonnetz = librosa.feature.chroma_to_tonnetz(chroma)
    >>> tonnetz
    array([[ 0.79506535,  0.17179337,  0.30261799, ...,  0.05586615,
         0.14303219,  0.187812  ],
       [-0.44264622,  0.41896667,  0.21596437, ...,  0.11552605,
         0.13349843,  0.92497935],
       [-1.        ,  0.46310843, -1.        , ...,  0.4240766 ,
         0.62088668,  0.74145399],
       [ 0.50550166, -1.        ,  0.33409652, ..., -1.        ,
        -1.        , -1.        ],
       [-0.34921771,  0.52859071, -0.14055977, ...,  0.05132072,
        -0.05028974, -0.22788206],
       [-0.01232449, -0.19507053, -0.07335521, ..., -0.43501825,
        -0.49050545, -0.36520804]])

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(tonnetz, x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Tonal Centroids (Tonnetz)')
    >>> plt.tight_layout()
    '''

    if chromagram.shape[0] != 12:
        raise ValueError('Tonnetz can only be obtained from 12-dimensional '
                         'chromagrams.')

    n_chroma, t = chromagram.shape
    tonnetz = np.zeros((6, t))

    r1 = 1      # Fifths
    r2 = 1      # Minor
    r3 = 0.5    # Major

    # Generate Transformation matrix
    phi = np.zeros((6, n_chroma))
    j = np.linspace(0, 12, num=n_chroma, endpoint=False)
    scale = np.pi * np.asarray([7./6, 7./6, 3./2, 3./2, 2./3, 2./3])

    V = np.multiply.outer(scale, j)
    V[::2] -= np.pi / 2
    R = np.array([r1, r1, r2, r2, r3, r3])
    phi = R[:, np.newaxis] * np.cos(V)

    # Do the transform to tonnetz
    tonnetz = phi.dot(util.normalize(chromagram, norm=1, axis=0))

    return util.normalize(tonnetz, norm=norm, axis=0)


# Moved functions
chromagram = util.decorators.moved('librosa.feature.chromagram',
                                   '0.4', '0.5')(chroma_stft)
