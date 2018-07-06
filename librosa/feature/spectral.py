#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

import numpy as np
import scipy
import scipy.signal
import scipy.fftpack

from .. import util
from .. import filters
from ..util.exceptions import ParameterError

from ..core.time_frequency import fft_frequencies
from ..core.audio import zero_crossings, to_mono
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt
from ..core.pitch import estimate_tuning


__all__ = ['spectral_centroid',
           'spectral_bandwidth',
           'spectral_contrast',
           'spectral_rolloff',
           'spectral_flatness',
           'poly_features',
           'rmse',
           'zero_crossing_rate',
           'chroma_stft',
           'chroma_cqt',
           'chroma_cens',
           'melspectrogram',
           'mfcc',
           'tonnetz']


# -- Spectral features -- #
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

    sr : number > 0 [scalar]
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
    array([[ 4382.894,   626.588, ...,  5037.07 ,  5413.398]])

    From spectrogram input:

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[ 4382.894,   626.588, ...,  5037.07 ,  5413.398]])

    Using variable bin center frequencies:

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    array([[ 4420.719,   625.769, ...,  5011.86 ,  5221.492]])

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
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()
    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    if not np.isrealobj(S):
        raise ParameterError('Spectral centroid is only defined '
                             'with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral centroid is only defined '
                             'with non-negative energies')

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    # Column-normalize S
    return np.sum(freq * util.normalize(S, norm=1, axis=0),
                  axis=0, keepdims=True)


def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                       freq=None, centroid=None, norm=True, p=2):
    '''Compute p'th-order spectral bandwidth:

        (sum_k S[k] * (freq[k] - centroid)**p)**(1/p)

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
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
    array([[ 3379.878,  1429.486, ...,  3235.214,  3080.148]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_bandwidth(S=S)
    array([[ 3379.878,  1429.486, ...,  3235.214,  3080.148]])

    Using variable bin center frequencies

    >>> if_gram, D = librosa.ifgram(y)
    >>> librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
    array([[ 3380.011,  1429.11 , ...,  3235.22 ,  3080.148]])

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
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    '''

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    if not np.isrealobj(S):
        raise ParameterError('Spectral bandwidth is only defined '
                             'with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral bandwidth is only defined '
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
        deviation = np.abs(np.subtract.outer(freq, centroid[0]))
    else:
        deviation = np.abs(freq - centroid[0])

    # Column-normalize S
    if norm:
        S = util.normalize(S, norm=1, axis=0)

    return np.sum(S * deviation**p, axis=0, keepdims=True)**(1./p)


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

    sr : number  > 0 [scalar]
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
    >>> librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                  ref=np.max),
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
        raise ParameterError('freq.shape mismatch: expected '
                             '({:d},)'.format(S.shape[0]))

    if n_bands < 1 or not isinstance(n_bands, int):
        raise ParameterError('n_bands must be a positive integer')

    if not 0.0 < quantile < 1.0:
        raise ParameterError('quantile must lie in the range (0, 1)')

    if fmin <= 0:
        raise ParameterError('fmin must be a positive number')

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0**np.arange(0, n_bands + 1))

    if np.any(octa[:-1] >= 0.5 * sr):
        raise ParameterError('Frequency band exceeds Nyquist. '
                             'Reduce either fmin or n_bands.')

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
        return power_to_db(peak) - power_to_db(valley)


def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                     freq=None, roll_percent=0.85):
    '''Compute roll-off frequency.

    The roll-off frequency is defined for each frame as the center frequency
    for a spectrogram bin such that at least roll_percent (0.85 by default)
    of the energy of the spectrum in this frame is contained in this bin and
    the bins below. This can be used to, e.g., approximate the maximum (or
    minimum) frequency by setting roll_percent to a value close to 1 (or 0).

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
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
    >>> # Approximate maximum frequencies with roll_percent=0.85 (default)
    >>> rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    >>> rolloff
    array([[ 8376.416,   968.994, ...,  8925.513,  9108.545]])
    >>> # Approximate minimum frequencies with roll_percent=0.1
    >>> rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.1)
    >>> rolloff
    array([[ 75.36621094,  64.59960938,  64.59960938, ...,  75.36621094,
         75.36621094,  64.59960938]])


    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_rolloff(S=S, sr=sr)
    array([[ 8376.416,   968.994, ...,  8925.513,  9108.545]])

    >>> # With a higher roll percentage:
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    array([[ 10012.939,   3003.882, ...,  10034.473,  10077.539]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> plt.semilogy(rolloff.T, label='Roll-off frequency')
    >>> plt.ylabel('Hz')
    >>> plt.xticks([])
    >>> plt.xlim([0, rolloff.shape[-1]])
    >>> plt.legend()
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    '''

    if not 0.0 < roll_percent < 1.0:
        raise ParameterError('roll_percent must lie in the range (0, 1)')

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    if not np.isrealobj(S):
        raise ParameterError('Spectral rolloff is only defined '
                             'with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral rolloff is only defined '
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


def spectral_flatness(y=None, S=None, n_fft=2048, hop_length=512,
                      amin=1e-10, power=2.0):
    '''Compute spectral flatness

    Spectral flatness (or tonality coefficient) is a measure to
    quantify how much noise-like a sound is, as opposed to being
    tone-like [1]_. A high spectral flatness (closer to 1.0)
    indicates the spectrum is similar to white noise.
    It is often converted to decibel.

    .. [1] Dubnov, Shlomo  "Generalization of spectral flatness
           measure for non-gaussian linear processes"
           IEEE Signal Processing Letters, 2004, Vol. 11.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    S : np.ndarray [shape=(d, t)] or None
        (optional) pre-computed spectrogram magnitude

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.core.stft` for details.

    amin : float > 0 [scalar]
        minimum threshold for `S` (=added noise floor for numerical stability)

    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
        Power spectrogram is usually used for computing spectral flatness.

    Returns
    -------
    flatness : np.ndarray [shape=(1, t)]
        spectral flatness for each frame.
        The returned value is in [0, 1] and often converted to dB scale.


    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> flatness = librosa.feature.spectral_flatness(y=y)
    >>> flatness
    array([[  1.00000e+00,   5.82299e-03,   5.64624e-04, ...,   9.99063e-01,
          1.00000e+00,   1.00000e+00]], dtype=float32)

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_flatness(S=S)
    array([[  1.00000e+00,   5.82299e-03,   5.64624e-04, ...,   9.99063e-01,
          1.00000e+00,   1.00000e+00]], dtype=float32)

    From power spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> S_power = S ** 2
    >>> librosa.feature.spectral_flatness(S=S_power, power=1.0)
    array([[  1.00000e+00,   5.82299e-03,   5.64624e-04, ...,   9.99063e-01,
          1.00000e+00,   1.00000e+00]], dtype=float32)

    '''
    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=1.)

    if not np.isrealobj(S):
        raise ParameterError('Spectral flatness is only defined '
                             'with real-valued input')
    elif np.any(S < 0):
        raise ParameterError('Spectral flatness is only defined '
                             'with non-negative energies')

    S_thresh = np.maximum(amin, S ** power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=0, keepdims=True))
    amean = np.mean(S_thresh, axis=0, keepdims=True)
    return gmean / amean


def rmse(y=None, S=None, frame_length=2048, hop_length=512,
         center=True, pad_mode='reflect'):
    '''Compute root-mean-square (RMS) energy for each frame, either from the
    audio samples `y` or from a spectrogram `S`.

    Computing the energy from audio samples is faster as it doesn't require a
    STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed,
    thus prefer using `S` if it's already available.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        (optional) audio time series. Required if `S` is not input.

    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude. Required if `y` is not input.

    frame_length : int > 0 [scalar]
        length of analysis frame (in samples) for energy calculation

    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.core.stft` for details.

    center : bool
        If `True` and operating on time-domain input (`y`), pad the signal
        by `frame_length//2` on either side.

        If operating on spectrogram input, this has no effect.

    pad_mode : str
        Padding mode for centered analysis.  See `np.pad` for valid
        values.

    Returns
    -------
    rms : np.ndarray [shape=(1, t)]
        RMS value for each frame


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.rmse(y=y)
    array([[ 0.   ,  0.056, ...,  0.   ,  0.   ]], dtype=float32)

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
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('log Power spectrogram')
    >>> plt.tight_layout()

    Use a STFT window of constant ones and no frame centering to get consistent
    results with the RMS energy computed from the audio samples `y`

    >>> S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    >>> librosa.feature.rmse(S=S)

    '''
    if y is not None and S is not None:
        raise ValueError('Either `y` or `S` should be input.')
    if y is not None:
        y = to_mono(y)
        if center:
            y = np.pad(y, int(frame_length // 2), mode=pad_mode)

        x = util.frame(y,
                       frame_length=frame_length,
                       hop_length=hop_length)
    elif S is not None:
        x, _ = _spectrogram(y=y, S=S,
                            n_fft=frame_length,
                            hop_length=hop_length)
    else:
        raise ValueError('Either `y` or `S` must be input.')
    return np.sqrt(np.mean(np.abs(x)**2, axis=0, keepdims=True))


def poly_features(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                  order=1, freq=None):
    '''Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
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
        polynomial coefficients for each frame.

        `coeffecients[0]` corresponds to the highest degree (`order`),

        `coefficients[1]` corresponds to the next highest degree (`order-1`),

        down to the constant term `coefficients[order]`.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))

    Fit a degree-0 polynomial (constant) to each frame

    >>> p0 = librosa.feature.poly_features(S=S, order=0)

    Fit a linear polynomial to each frame

    >>> p1 = librosa.feature.poly_features(S=S, order=1)

    Fit a quadratic to each frame

    >>> p2 = librosa.feature.poly_features(S=S, order=2)

    Plot the results for comparison

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 8))
    >>> ax = plt.subplot(4,1,1)
    >>> plt.plot(p2[2], label='order=2', alpha=0.8)
    >>> plt.plot(p1[1], label='order=1', alpha=0.8)
    >>> plt.plot(p0[0], label='order=0', alpha=0.8)
    >>> plt.xticks([])
    >>> plt.ylabel('Constant')
    >>> plt.legend()
    >>> plt.subplot(4,1,2, sharex=ax)
    >>> plt.plot(p2[1], label='order=2', alpha=0.8)
    >>> plt.plot(p1[0], label='order=1', alpha=0.8)
    >>> plt.xticks([])
    >>> plt.ylabel('Linear')
    >>> plt.subplot(4,1,3, sharex=ax)
    >>> plt.plot(p2[0], label='order=2', alpha=0.8)
    >>> plt.xticks([])
    >>> plt.ylabel('Quadratic')
    >>> plt.subplot(4,1,4, sharex=ax)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log')
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


def zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True,
                       **kwargs):
    '''Compute the zero-crossing rate of an audio time series.

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

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.feature.zero_crossing_rate(y)
    array([[ 0.134,  0.139, ...,  0.387,  0.322]])

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
def chroma_stft(y=None, sr=22050, S=None, norm=np.inf, n_fft=2048,
                hop_length=512, tuning=None, **kwargs):
    """Compute a chromagram from a waveform or power spectrogram.

    This implementation is derived from `chromagram_E` [1]_

    .. [1] Ellis, Daniel P.W.  "Chroma feature analysis and synthesis"
           2007/04/21
           http://labrosa.ee.columbia.edu/matlab/chroma-ansyn/

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series

    sr : number > 0 [scalar]
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
    array([[ 0.974,  0.881, ...,  0.925,  1.   ],
           [ 1.   ,  0.841, ...,  0.882,  0.878],
           ...,
           [ 0.658,  0.985, ...,  0.878,  0.764],
           [ 0.969,  0.92 , ...,  0.974,  0.915]])

    Use an energy (magnitude) spectrum instead of power spectrogram

    >>> S = np.abs(librosa.stft(y))
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[ 0.884,  0.91 , ...,  0.861,  0.858],
           [ 0.963,  0.785, ...,  0.968,  0.896],
           ...,
           [ 0.871,  1.   , ...,  0.928,  0.829],
           [ 1.   ,  0.982, ...,  0.93 ,  0.878]])

    Use a pre-computed power spectrogram with a larger frame

    >>> S = np.abs(librosa.stft(y, n_fft=4096))**2
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[ 0.685,  0.477, ...,  0.961,  0.986],
           [ 0.674,  0.452, ...,  0.952,  0.926],
           ...,
           [ 0.844,  0.575, ...,  0.934,  0.869],
           [ 0.793,  0.663, ...,  0.964,  0.972]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 4))
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


def chroma_cqt(y=None, sr=22050, C=None, hop_length=512, fmin=None,
               norm=np.inf, threshold=0.0, tuning=None, n_chroma=12,
               n_octaves=7, window=None, bins_per_octave=None, cqt_mode='full'):
    r'''Constant-Q chromagram

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0
        sampling rate of `y`

    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram

    hop_length : int > 0
        number of samples between successive chroma frames

    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: 'C1' ~= 32.7 Hz

    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.

    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.

    tuning : float
        Deviation (in cents) from A440 tuning

    n_chroma : int > 0
        Number of chroma bins to produce

    n_octaves : int > 0
        Number of octaves to analyze above `fmin`

    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`

    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: matches `n_chroma`

    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    chromagram : np.ndarray [shape=(n_chroma, t)]
        The output chromagram

    See Also
    --------
    librosa.util.normalize
    librosa.core.cqt
    librosa.core.hybrid_cqt
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
        C = np.abs(cqt_func[cqt_mode](y, sr=sr,
                                      hop_length=hop_length,
                                      fmin=fmin,
                                      n_bins=n_octaves * bins_per_octave,
                                      bins_per_octave=bins_per_octave,
                                      tuning=tuning))

    # Map to chroma
    cq_to_chr = filters.cq_to_chroma(C.shape[0],
                                     bins_per_octave=bins_per_octave,
                                     n_chroma=n_chroma,
                                     fmin=fmin,
                                     window=window)
    chroma = cq_to_chr.dot(C)

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    # Normalize
    if norm is not None:
        chroma = util.normalize(chroma, norm=norm, axis=0)

    return chroma


def chroma_cens(y=None, sr=22050, C=None, hop_length=512, fmin=None,
                tuning=None, n_chroma=12,
                n_octaves=7, bins_per_octave=None, cqt_mode='full', window=None,
                norm=2, win_len_smooth=41):
    r'''Computes the chroma variant "Chroma Energy Normalized" (CENS), following [1]_.

    .. [1] Meinard Müller and Sebastian Ewert
           Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features
           In Proceedings of the International Conference on Music Information Retrieval (ISMIR), 2011.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0
        sampling rate of `y`

    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram

    hop_length : int > 0
        number of samples between successive chroma frames

    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: 'C1' ~= 32.7 Hz

    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.

    tuning : float
        Deviation (in cents) from A440 tuning

    n_chroma : int > 0
        Number of chroma bins to produce

    n_octaves : int > 0
        Number of octaves to analyze above `fmin`

    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`

    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: matches `n_chroma`

    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    win_len_smooth : int > 0
        Length of temporal smoothing window.
        Default: 41

    Returns
    -------
    chroma_cens : np.ndarray [shape=(n_chroma, t)]
        The output cens-chromagram

    See Also
    --------
    chroma_cqt
        Compute a chromagram from a constant-Q transform.

    chroma_stft
        Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    Compare standard cqt chroma to CENS.


    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10, duration=15)
    >>> chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(chroma_cq, y_axis='chroma')
    >>> plt.title('chroma_cq')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
    >>> plt.title('chroma_cens')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    '''
    chroma = chroma_cqt(y=y, C=C, sr=sr,
                        hop_length=hop_length,
                        fmin=fmin,
                        bins_per_octave=bins_per_octave,
                        tuning=tuning,
                        norm=None,
                        n_chroma=n_chroma,
                        n_octaves=n_octaves,
                        cqt_mode=cqt_mode,
                        window=window)

    # L1-Normalization
    chroma = util.normalize(chroma, norm=1, axis=0)

    # Quantize amplitudes
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    chroma_quant = np.zeros_like(chroma)

    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]

    # Apply temporal smoothing
    win = filters.get_window('hann', win_len_smooth + 2, fftbins=False)
    win /= np.sum(win)
    win = np.atleast_2d(win)

    cens = scipy.signal.convolve2d(chroma_quant, win,
                                   mode='same', boundary='fill')

    # L2-Normalization
    return util.normalize(cens, norm=norm, axis=0)


def tonnetz(y=None, sr=22050, chroma=None):
    '''Computes the tonal centroid features (tonnetz), following the method of
    [1]_.

    .. [1] Harte, C., Sandler, M., & Gasser, M. (2006). "Detecting Harmonic
           Change in Musical Audio." In Proceedings of the 1st ACM Workshop
           on Audio and Music Computing Multimedia (pp. 21-26).
           Santa Barbara, CA, USA: ACM Press. doi:10.1145/1178723.1178727.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        Audio time series.

    sr : number > 0 [scalar]
        sampling rate of `y`

    chroma : np.ndarray [shape=(n_chroma, t)] or None
        Normalized energy for each chroma bin at each frame.

        If `None`, a cqt chromagram is performed.

    Returns
    -------
    tonnetz : np.ndarray [shape(6, t)]
        Tonal centroid features for each frame.

        Tonnetz dimensions:
            - 0: Fifth x-axis
            - 1: Fifth y-axis
            - 2: Minor x-axis
            - 3: Minor y-axis
            - 4: Major x-axis
            - 5: Major y-axis

    See Also
    --------
    chroma_cqt
        Compute a chromagram from a constant-Q transform.

    chroma_stft
        Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    Compute tonnetz features from the harmonic component of a song

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y = librosa.effects.harmonic(y)
    >>> tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    >>> tonnetz
    array([[-0.073, -0.053, ..., -0.054, -0.073],
           [ 0.001,  0.001, ..., -0.054, -0.062],
           ...,
           [ 0.039,  0.034, ...,  0.044,  0.064],
           [ 0.005,  0.002, ...,  0.011,  0.017]])

    Compare the tonnetz features to `chroma_cqt`

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(tonnetz, y_axis='tonnetz')
    >>> plt.colorbar()
    >>> plt.title('Tonal Centroids (Tonnetz)')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.feature.chroma_cqt(y, sr=sr),
    ...                          y_axis='chroma', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Chroma')
    >>> plt.tight_layout()

    '''

    if y is None and chroma is None:
        raise ParameterError('Either the audio samples or the chromagram must be '
                             'passed as an argument.')

    if chroma is None:
        chroma = chroma_cqt(y=y, sr=sr)

    # Generate Transformation matrix
    dim_map = np.linspace(0, 12, num=chroma.shape[0], endpoint=False)

    scale = np.asarray([7. / 6, 7. / 6,
                        3. / 2, 3. / 2,
                        2. / 3, 2. / 3])

    V = np.multiply.outer(scale, dim_map)

    # Even rows compute sin()
    V[::2] -= 0.5

    R = np.array([1, 1,         # Fifths
                  1, 1,         # Minor
                  0.5, 0.5])    # Major

    phi = R[:, np.newaxis] * np.cos(np.pi * V)

    # Do the transform to tonnetz
    return phi.dot(util.normalize(chroma, norm=1, axis=0))


# -- Mel spectrogram and MFCCs -- #
def mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', **kwargs):
    """Mel-frequency cepstral coefficients (MFCCs)

    Parameters
    ----------
    y     : np.ndarray [shape=(n,)] or None
        audio time series

    sr    : number > 0 [scalar]
        sampling rate of `y`

    S     : np.ndarray [shape=(d, t)] or None
        log-power Mel spectrogram

    n_mfcc: int > 0 [scalar]
        number of MFCCs to return

    dct_type : None, or {1, 2, 3}
        Discrete cosine transform (DCT) type.
        By default, DCT type-2 is used.

    norm : None or 'ortho'
        If `dct_type` is `2 or 3`, setting `norm='ortho'` uses an ortho-normal
        DCT basis.

        Normalization is not supported for `dct_type=1`.

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
    scipy.fftpack.dct

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
           [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
           ...,
           [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
           [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.power_to_db(S))
    array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
           [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
           ...,
           [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
           [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 4))
    >>> librosa.display.specshow(mfccs, x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('MFCC')
    >>> plt.tight_layout()

    Compare different DCT bases

    >>> m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    >>> m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(m_slaney, x_axis='time')
    >>> plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
    >>> plt.colorbar()
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(m_htk, x_axis='time')
    >>> plt.title('HTK-style (dct_type=3)')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if S is None:
        S = power_to_db(melspectrogram(y=y, sr=sr, **kwargs))

    return scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
    """Compute a mel-scaled spectrogram.

    If a spectrogram input `S` is provided, then it is mapped directly onto
    the mel basis `mel_f` by `mel_f.dot(S)`.

    If a time-series input `y, sr` is provided, then its magnitude spectrogram
    `S` is first computed, and then mapped onto the mel scale by
    `mel_f.dot(S**power)`.  By default, `power=2` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time-series

    sr : number > 0 [scalar]
        sampling rate of `y`

    S : np.ndarray [shape=(d, t)]
        spectrogram

    n_fft : int > 0 [scalar]
        length of the FFT window

    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.core.stft`

    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power, etc.

    kwargs : additional keyword arguments
      Mel filter bank parameters.
      See `librosa.filters.mel` for details.

    Returns
    -------
    S : np.ndarray [shape=(n_mels, t)]
        Mel spectrogram

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
    array([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
           [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
           ...,
           [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
           [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])

    Using a pre-computed power spectrogram

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D)

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 4))
    >>> librosa.display.specshow(librosa.power_to_db(S,
    ...                                              ref=np.max),
    ...                          y_axis='mel', fmax=8000,
    ...                          x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Mel spectrogram')
    >>> plt.tight_layout()


    """

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length,
                            power=power)

    # Build a Mel filter
    mel_basis = filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)
