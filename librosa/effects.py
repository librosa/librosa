#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Effects and filters for audio buffer data"""

import numpy as np

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
      - y_harmonic : np.ndarray [shape=(n,)]
          audio time series of the harmonic elements

      - y_percussive : np.ndarray [shape=(n,)]
          audio time series of the percussive elements

    .. seealso:: :func:`librosa.decompose.hpss`
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Decompose into harmonic and percussives
    D_harm, D_perc = librosa.decompose.hpss(D)

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = librosa.util.fix_length(librosa.istft(D_harm, dtype=y.dtype),
                                     len(y))
    y_perc = librosa.util.fix_length(librosa.istft(D_perc, dtype=y.dtype),
                                     len(y))

    return y_harm, y_perc


def harmonic(y):
    '''Extract harmonic elements from an audio time-series.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> y_harmonic = librosa.effects.harmonic(y)

    :parameters:
      - y : np.ndarray [shape=(n,)]
          audio time series

    :returns:
      - y_harmonic : np.ndarray [shape=(n,)]
          audio time series of just the harmonic portion

    .. seealso:: :func:`librosa.decompose.hpss`, :func:`librosa.effects.hpss`,
        :func:`librosa.effects.percussive`
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Remove percussives
    D_harm = librosa.decompose.hpss(D)[0]

    # Invert the STFTs
    y_harm = librosa.util.fix_length(librosa.istft(D_harm, dtype=y.dtype),
                                     len(y))

    return y_harm


def percussive(y):
    '''Extract percussive elements from an audio time-series.

    :usage:
        >>> # Load a waveform
        >>> y, sr = librosa.load('file.mp3')
        >>> y_percussive = librosa.effects.percussive(y)

    :parameters:
      - y : np.ndarray [shape=(n,)]
          audio time series

    :returns:
      - y_percussive : np.ndarray [shape=(n,)]
          audio time series of just the percussive portion

    .. seealso:: :func:`librosa.decompose.hpss`, :func:`librosa.effects.hpss`,
        :func:`librosa.effects.percussive`
    '''

    # Compute the STFT matrix
    D = librosa.core.stft(y)

    # Remove harmonics
    D_perc = librosa.decompose.hpss(D)[1]

    # Invert the STFT
    y_perc = librosa.util.fix_length(librosa.istft(D_perc, dtype=y.dtype),
                                     len(y))

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
      - y : np.ndarray [shape=(n,)]
          audio time series

      - rate : float > 0 [scalar]
          Stretch factor.  If ``rate > 1``, then the signal is sped up.
          If ``rate < 1``, then the signal is slowed down.

    :returns:
      - y_stretch : np.ndarray [shape=(rate * n,)]
          audio time series stretched by the specified rate

    .. seealso:: :func:`librosa.core.phase_vocoder`,
      :func:`librosa.effects.pitch_shift`
    '''

    # Construct the stft
    D = librosa.stft(y)

    # Stretch by phase vocoding
    D_stretch = librosa.phase_vocoder(D, rate)

    # Invert the stft
    y_stretch = librosa.istft(D_stretch, dtype=y.dtype)

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
      - y : np.ndarray [shape=(n,)]
          audio time-series

      - sr : int > 0 [scalar]
          audio sampling rate of ``y``

      - n_steps : float [scalar]
          how many (fractional) half-steps to shift ``y``

      - bins_per_octave : float > 0 [scalar]
          how many steps per octave

    :returns:
      - y_shift : np.ndarray [shape=(n,)]
          The pitch-shifted audio time-series

    .. seealso:: :func:`librosa.core.phase_vocoder`,
      :func:`librosa.effects.time_stretch`
    '''

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = librosa.resample(time_stretch(y, rate),
                               float(sr) / rate,
                               sr)

    # Crop to the same dimension as the input
    return librosa.util.fix_length(y_shift, len(y))
