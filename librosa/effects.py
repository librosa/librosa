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
    trim
    split
"""

import numpy as np

from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError

__all__ = ['hpss', 'harmonic', 'percussive',
           'time_stretch', 'pitch_shift',
           'remix', 'trim', 'split']


def hpss(y, **kwargs):
    '''Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform `y`.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series
    kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.


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
    >>> # Extract harmonic and percussive components
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    >>> # Get a more isolated percussive component by widening its margin
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Decompose into harmonic and percussives
    stft_harm, stft_perc = decompose.hpss(stft, **kwargs)

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = util.fix_length(core.istft(stft_harm, dtype=y.dtype), len(y))
    y_perc = util.fix_length(core.istft(stft_perc, dtype=y.dtype), len(y))

    return y_harm, y_perc


def harmonic(y, **kwargs):
    '''Extract harmonic elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series
    kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.

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
    >>> # Extract harmonic component
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_harmonic = librosa.effects.harmonic(y)

    >>> # Use a margin > 1.0 for greater harmonic separation
    >>> y_harmonic = librosa.effects.harmonic(y, margin=3.0)

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Remove percussives
    stft_harm = decompose.hpss(stft, **kwargs)[0]

    # Invert the STFTs
    y_harm = util.fix_length(core.istft(stft_harm, dtype=y.dtype), len(y))

    return y_harm


def percussive(y, **kwargs):
    '''Extract percussive elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series
    kwargs : additional keyword arguments.
        See `librosa.decompose.hpss` for details.

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
    >>> # Extract percussive component
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_percussive = librosa.effects.percussive(y)

    >>> # Use a margin > 1.0 for greater percussive separation
    >>> y_percussive = librosa.effects.percussive(y, margin=3.0)

    '''

    # Compute the STFT matrix
    stft = core.stft(y)

    # Remove harmonics
    stft_perc = decompose.hpss(stft, **kwargs)[1]

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

    sr : number > 0 [scalar]
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

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer):
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

        y_out.append(y[tuple(clip)])

    return np.concatenate(y_out, axis=-1)


def _signal_to_frame_nonsilent(y, frame_length=2048, hop_length=512, top_db=60,
                               ref=np.max):
    '''Frame-wise non-silent indicator for audio input.

    This is a helper function for `trim` and `split`.

    Parameters
    ----------
    y : np.ndarray, shape=(n,) or (2,n)
        Audio signal, mono or stereo

    frame_length : int > 0
        The number of samples per frame

    hop_length : int > 0
        The number of samples between frames

    top_db : number > 0
        The threshold (in decibels) below reference to consider as
        silence

    ref : callable or float
        The reference power

    Returns
    -------
    non_silent : np.ndarray, shape=(m,), dtype=bool
        Indicator of non-silent frames
    '''
    # Convert to mono
    y_mono = core.to_mono(y)

    # Compute the MSE for the signal
    mse = feature.rmse(y=y_mono,
                       frame_length=frame_length,
                       hop_length=hop_length)**2

    return (core.power_to_db(mse.squeeze(),
                             ref=ref,
                             top_db=None) > - top_db)


def trim(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    '''Trim leading and trailing silence from an audio signal.

    Parameters
    ----------
    y : np.ndarray, shape=(n,) or (2,n)
        Audio signal, can be mono or stereo

    top_db : number > 0
        The threshold (in decibels) below reference to consider as
        silence

    ref : number or callable
        The reference power.  By default, it uses `np.max` and compares
        to the peak power in the signal.

    frame_length : int > 0
        The number of samples per analysis frame

    hop_length : int > 0
        The number of samples between analysis frames

    Returns
    -------
    y_trimmed : np.ndarray, shape=(m,) or (2, m)
        The trimmed signal

    index : np.ndarray, shape=(2,)
        the interval of `y` corresponding to the non-silent region:
        `y_trimmed = y[index[0]:index[1]]` (for mono) or
        `y_trimmed = y[:, index[0]:index[1]]` (for stereo).


    Examples
    --------
    >>> # Load some audio
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> # Trim the beginning and ending silence
    >>> yt, index = librosa.effects.trim(y)
    >>> # Print the durations
    >>> print(librosa.get_duration(y), librosa.get_duration(yt))
    61.45886621315193 60.58086167800454
    '''

    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(core.frames_to_samples(nonzero[0], hop_length))
        end = min(y.shape[-1],
                  int(core.frames_to_samples(nonzero[-1] + 1, hop_length)))
    else:
        # The signal only contains zeros
        start, end = 0, 0

    # Build the mono/stereo index
    full_index = [slice(None)] * y.ndim
    full_index[-1] = slice(start, end)

    return y[tuple(full_index)], np.asarray([start, end])


def split(y, top_db=60, ref=np.max, frame_length=2048, hop_length=512):
    '''Split an audio signal into non-silent intervals.

    Parameters
    ----------
    y : np.ndarray, shape=(n,) or (2, n)
        An audio signal

    top_db : number > 0
        The threshold (in decibels) below reference to consider as
        silence

    ref : number or callable
        The reference power.  By default, it uses `np.max` and compares
        to the peak power in the signal.

    frame_length : int > 0
        The number of samples per analysis frame

    hop_length : int > 0
        The number of samples between analysis frames

    Returns
    -------
    intervals : np.ndarray, shape=(m, 2)
        `intervals[i] == (start_i, end_i)` are the start and end time
        (in samples) of non-silent interval `i`.

    '''

    non_silent = _signal_to_frame_nonsilent(y,
                                            frame_length=frame_length,
                                            hop_length=hop_length,
                                            ref=ref,
                                            top_db=top_db)

    # Interval slicing, adapted from
    # https://stackoverflow.com/questions/2619413/efficiently-finding-the-interval-with-non-zeros-in-scipy-numpy-in-python
    # Find points where the sign flips
    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if non_silent[0]:
        edges.insert(0, [0])

    # Likewise for the last frame
    if non_silent[-1]:
        edges.append([len(non_silent)])

    # Convert from frames to samples
    edges = core.frames_to_samples(np.concatenate(edges),
                                   hop_length=hop_length)

    # Clip to the signal duration
    edges = np.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    return edges.reshape((-1, 2))
