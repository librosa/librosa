#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pitch-tracking and tuning estimation'''

import warnings
import numpy as np

from .spectrum import _spectrogram
from . import time_frequency
from .. import cache
from .. import util

__all__ = ['estimate_tuning', 'pitch_tuning', 'piptrack']


def estimate_tuning(y=None, sr=22050, S=None, n_fft=2048,
                    resolution=0.01, bins_per_octave=12, **kwargs):
    '''Estimate the tuning of an audio time series or spectrogram input.

    Parameters
    ----------
    y: np.ndarray [shape=(n,)] or None
        audio signal

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if `y` is provided.

    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to measurements in cents.

    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    kwargs : additional keyword arguments
        Additional arguments passed to `piptrack`

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    piptrack
        Pitch tracking by parabolic interpolation

    Examples
    --------
    >>> # With time-series input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr)
    0.089999999999999969

    >>> # In tenths of a cent
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)
    0.093999999999999972

    >>> # Using spectrogram input
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.estimate_tuning(S=S, sr=sr)
    0.089999999999999969

    >>> # Using pass-through arguments to `librosa.piptrack`
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> librosa.estimate_tuning(y=y, sr=sr, n_fft=8192,
    ...                         fmax=librosa.note_to_hz('G#9'))
    0.070000000000000062

    '''

    pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return pitch_tuning(pitch[(mag >= threshold) & pitch_mask],
                        resolution=resolution,
                        bins_per_octave=bins_per_octave)


def pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    '''Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.

    Parameters
    ----------
    frequencies : array-like, float
        A collection of frequencies detected in the signal.
        See `piptrack`

    resolution : float in `(0, 1)`
        Resolution of the tuning as a fraction of a bin.
        0.01 corresponds to cents.

    bins_per_octave : int > 0 [scalar]
        How many frequency bins per octave

    Returns
    -------
    tuning: float in `[-0.5, 0.5)`
        estimated tuning deviation (fractions of a bin)

    See Also
    --------
    estimate_tuning
        Estimating tuning from time-series or spectrogram input

    Examples
    --------
    >>> # Generate notes at +25 cents
    >>> freqs = librosa.cqt_frequencies(24, 55, tuning=0.25)
    >>> librosa.pitch_tuning(freqs)
    0.25

    >>> # Track frequencies from a real spectrogram
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> pitches, magnitudes, stft = librosa.ifptrack(y, sr)
    >>> # Select out pitches with high energy
    >>> pitches = pitches[magnitudes > np.median(magnitudes)]
    >>> librosa.pitch_tuning(pitches)
    0.089999999999999969

    '''

    frequencies = np.atleast_1d(frequencies)

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        warnings.warn('Trying to estimate tuning from empty frequency set.')
        return 0.0

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave *
                      time_frequency.hz_to_octs(frequencies), 1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1./resolution)), endpoint=False)

    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]


@cache(level=30)
def piptrack(y=None, sr=22050, S=None, n_fft=2048, hop_length=None,
             fmin=150.0, fmax=4000.0, threshold=0.1):
    '''Pitch tracking on thresholded parabolically-interpolated STFT

    .. [1] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    y: np.ndarray [shape=(n,)] or None
        audio signal

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if `y` is provided.

    hop_length : int > 0 [scalar] or None
        number of samples to hop

    threshold : float in `(0, 1)`
        A bin in spectrum X is considered a pitch when it is greater than
        `threshold*X.max()`

    fmin : float > 0 [scalar]
        lower frequency cutoff.

    fmax : float > 0 [scalar]
        upper frequency cutoff.

    .. note::
        One of `S` or `y` must be provided.

        If `S` is not given, it is computed from `y` using
        the default parameters of `librosa.core.stft`.

    Returns
    -------
    pitches : np.ndarray [shape=(d, t)]
    magnitudes : np.ndarray [shape=(d,t)]
        Where `d` is the subset of FFT bins within `fmin` and `fmax`.

        `pitches[f, t]` contains instantaneous frequency at bin
        `f`, time `t`

        `magnitudes[f, t]` contains the corresponding magnitudes.

        Both `pitches` and `magnitudes` take value 0 at bins
        of non-maximal magnitude.

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    '''

    # Check that we received an audio time series or STFT
    if hop_length is None:
        hop_length = int(n_fft // 4)

    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length)

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = time_frequency.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

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
                      util.localmax(S * (S > (threshold * S.max(axis=0)))))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = ((idx[:, 0] + shift[idx[:, 0], idx[:, 1]])
                                     * float(sr) / n_fft)

    mags[idx[:, 0], idx[:, 1]] = (S[idx[:, 0], idx[:, 1]]
                                  + dskew[idx[:, 0], idx[:, 1]])

    return pitches, mags
