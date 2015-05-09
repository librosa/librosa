#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Effects
=======

Harmonic-percussive source separation
-------------------------------------
.. autosummary::
    :toctree: generated/

    hpss
    harmonic
    percussive

Time and frequency
------------------
.. autosummary::
    :toctree: generated/

    time_stretch
    pitch_shift

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    remix
"""

import numpy as np

from . import core
from . import decompose
from . import util
from .util.exceptions import ParameterError

__all__ = ['hpss', 'harmonic', 'percussive',
           'time_stretch', 'pitch_shift',
           'remix']


def hpss(y):
    '''Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform `y`.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    Returns
    -------
    y_harmonic : np.ndarray [shape=(n,)]
        audio time series of the harmonic elements

    y_percussive : np.ndarray [shape=(n,)]
        audio time series of the percussive elements

    See Also
    --------
    harmonic : Extract only the harmonic component
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS on spectrograms


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Decompose into harmonic and percussives
    stft_harm, stft_perc = decompose.hpss(stft)

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = util.fix_length(core.istft(stft_harm, dtype=y.dtype), len(y))
    y_perc = util.fix_length(core.istft(stft_perc, dtype=y.dtype), len(y))

    return y_harm, y_perc


def harmonic(y):
    '''Extract harmonic elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    Returns
    -------
    y_harmonic : np.ndarray [shape=(n,)]
        audio time series of just the harmonic portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_harmonic = librosa.effects.harmonic(y)

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Remove percussives
    stft_harm = decompose.hpss(stft)[0]

    # Invert the STFTs
    y_harm = util.fix_length(core.istft(stft_harm, dtype=y.dtype), len(y))

    return y_harm


def percussive(y):
    '''Extract percussive elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    Returns
    -------
    y_percussive : np.ndarray [shape=(n,)]
        audio time series of just the percussive portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    harmonic : Extract only the harmonic component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_percussive = librosa.effects.percussive(y)

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Remove harmonics
    stft_perc = decompose.hpss(stft)[1]

    # Invert the STFT
    y_perc = util.fix_length(core.istft(stft_perc, dtype=y.dtype), len(y))

    return y_perc


def time_stretch(y, rate):
    '''Time-stretch an audio series by a fixed rate.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    rate : float > 0 [scalar]
        Stretch factor.  If `rate > 1`, then the signal is sped up.

        If `rate < 1`, then the signal is slowed down.

    Returns
    -------
    y_stretch : np.ndarray [shape=(rate * n,)]
        audio time series stretched by the specified rate

    See Also
    --------
    pitch_shift : pitch shifting
    librosa.core.phase_vocoder : spectrogram phase vocoder


    Examples
    --------
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_fast = librosa.effects.time_stretch(y, 2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, 0.5)

    '''

    if rate <= 0:
        raise ParameterError('rate must be a positive number')

    # Construct the stft
    stft = core.stft(y)

    # Stretch by phase vocoding
    stft_stretch = core.phase_vocoder(stft, rate)

    # Invert the stft
    y_stretch = core.istft(stft_stretch, dtype=y.dtype)

    return y_stretch


def pitch_shift(y, sr, n_steps, bins_per_octave=12):
    '''Pitch-shift the waveform by `n_steps` half-steps.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time-series

    sr : int > 0 [scalar]
        audio sampling rate of `y`

    n_steps : float [scalar]
        how many (fractional) half-steps to shift `y`

    bins_per_octave : float > 0 [scalar]
        how many steps per octave


    Returns
    -------
    y_shift : np.ndarray [shape=(n,)]
        The pitch-shifted audio time-series


    See Also
    --------
    time_stretch : time stretching
    librosa.core.phase_vocoder : spectrogram phase vocoder


    Examples
    --------
    Shift up by a major third (four half-steps)

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)

    Shift down by a tritone (six half-steps)

    >>> y_tritone = librosa.effects.pitch_shift(y, sr, n_steps=-6)

    Shift up by 3 quarter-tones

    >>> y_three_qt = librosa.effects.pitch_shift(y, sr, n_steps=3,
    ...                                          bins_per_octave=24)
    '''

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.int):
        raise ParameterError('bins_per_octave must be a positive integer.')

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = core.resample(time_stretch(y, rate), float(sr) / rate, sr)

    # Crop to the same dimension as the input
    return util.fix_length(y_shift, len(y))


def remix(y, intervals, align_zeros=True):
    '''Remix an audio signal by re-ordering time intervals.


    Parameters
    ----------
    y : np.ndarray [shape=(t,) or (2, t)]
        Audio time series

    intervals : iterable of tuples (start, end)
        An iterable (list-like or generator) where the `i`th item
        `intervals[i]` indicates the start and end (in samples)
        of a slice of `y`.

    align_zeros : boolean
        If `True`, interval boundaries are mapped to the closest
        zero-crossing in `y`.  If `y` is stereo, zero-crossings
        are computed after converting to mono.


    Returns
    -------
    y_remix : np.ndarray [shape=(d,) or (2, d)]
        `y` remixed in the order specified by `intervals`


    Examples
    --------
    Load in the example track and reverse the beats

    >>> y, sr = librosa.load(librosa.util.example_audio_file())


    Compute beats

    >>> _, beat_frames = librosa.beat.beat_track(y=y, sr=sr,
    ...                                          hop_length=512)


    Convert from frames to sample indices

    >>> beat_samples = librosa.frames_to_samples(beat_frames)


    Generate intervals from consecutive events

    >>> intervals = librosa.util.frame(beat_samples, frame_length=2,
    ...                                hop_length=1).T


    Reverse the beat intervals

    >>> y_out = librosa.effects.remix(y, intervals[::-1])
    '''

    # Validate the audio buffer
    util.valid_audio(y, mono=False)

    y_out = []

    if align_zeros:
        y_mono = core.to_mono(y)
        zeros = np.nonzero(core.zero_crossings(y_mono))[-1]
        # Force end-of-signal onto zeros
        zeros = np.append(zeros, [len(y_mono)])

    clip = [slice(None)] * y.ndim

    for interval in intervals:

        if align_zeros:
            interval = zeros[util.match_events(interval, zeros)]

        clip[-1] = slice(interval[0], interval[1])

        y_out.append(y[clip])

    return np.concatenate(y_out, axis=-1)
