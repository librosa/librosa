#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Effects and filters for audio buffer data"""

import librosa.core
import librosa.decompose
import librosa.util


def hpss(y):
    '''Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform ``y``.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    :parameters:
      - y : np.ndarray [shape=(n,)]
        audio time series

    :returns:
      - y_harmonic : np.ndarray, shape=``y.shape``
        audio time series of the harmonic elements

      - y_percussive : np.ndarray, shape=``y.shape``
        audio time series of the percussive elements

    .. seealso:: ``librosa.decompose.hpss``
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Decompose into harmonic and percussives
    D_harm, D_perc = librosa.decompose.hpss(D)

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = librosa.util.fix_length(librosa.istft(D_harm), len(y))
    y_perc = librosa.util.fix_length(librosa.istft(D_perc), len(y))

    return y_harm, y_perc


def harmonic(y):
    '''Extract harmonic elements from an audio time-series.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> y_harmonic = librosa.effects.harmonic(y)

    :parameters:
      - y : np.ndarray
        audio time series

    :returns:
      - y_harmonic : np.ndarray, shape=``y.shape``
        audio time series of just the harmonic portion

    .. seealso:: ``librosa.decompose.hpss``, ``librosa.effects.hpss``,
        ``librosa.effects.percussive``
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Remove percussives
    D_harm = librosa.decompose.hpss(D)[0]

    # Invert the STFTs
    y_harm = librosa.util.fix_length(librosa.istft(D_harm), len(y))

    return y_harm


def percussive(y):
    '''Extract percussive elements from an audio time-series.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> y_percussive = librosa.effects.percussive(y)

    :parameters:
      - y : np.ndarray
        audio time series

    :returns:
      - y_percussive : np.ndarray, shape=``y.shape``
        audio time series of just the percussive portion

    .. seealso:: ``librosa.decompose.hpss``, ``librosa.effects.hpss``,
      ``librosa.effects.percussive``
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Remove harmonics
    D_perc = librosa.decompose.hpss(D)[1]

    # Invert the STFT
    y_perc = librosa.util.fix_length(librosa.istft(D_perc), len(y))

    return y_perc


def time_stretch(y, rate):
    '''Time-stretch an audio series by a fixed rate.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> # Compress to be twice as fast
        >>> y_fast = librosa.effects.time_stretch(y, 2.0)
        >>> # Or half the original speed
        >>> y_slow = librosa.effects.time_stretch(y, 0.5)

    :parameters:
      - y : np.ndarray
        audio time series

      - rate : float > 0
        Stretch factor.  If ``rate > 1``, then the signal is sped up.
        If ``rate < 1``, then the signal is slowed down.

    :returns:
      - y_stretch : np.ndarray
        audio time series stretched by the specified rate

    .. seealso:: ``librosa.core.phase_vocoder``,
      ``librosa.effects.pitch_shift``
    '''

    # Construct the stft
    D = librosa.stft(y)

    # Stretch by phase vocoding
    D_stretch = librosa.phase_vocoder(D, rate)

    # Invert the stft
    y_stretch = librosa.istft(D_stretch)

    return y_stretch


def pitch_shift(y, sr, n_steps, bins_per_octave=12):
    '''Pitch-shift the waveform by ``n_steps`` half-steps.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> # Shift up by a major third (four half-steps)
        >>> y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)
        >>> # Shift down by a tritone (six half-steps)
        >>> y_tritone = librosa.effects.pitch_shift(y, sr, n_steps=-6)
        >>> # Shift up by 3 quarter-tones
        >>> y_three_qt = librosa.effects.pitch_shift(y, sr, n_steps=3,
                                                     bins_per_octave=24)


    :parameters:
      - y : np.ndarray
        audio time-series

      - sr : int > 0
        audio sampling rate of ``y``

      - n_steps : float
        how many (fractional) half-steps tp shift ``y``

      - bins_per_octave : float > 0
        how many steps per octave

    :returns:
      - y_shift : np.ndarray, shape=``y.shape``
        The pitch-shifted audio time-series

    .. seealso:: ``librosa.core.phase_vocoder``,
      ``librosa.effects.time_stretch``
    '''

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = librosa.resample(time_stretch(y, rate),
                               sr / rate,
                               sr)

    # Crop to the same dimension as the input
    return librosa.util.fix_length(y_shift, len(y))
