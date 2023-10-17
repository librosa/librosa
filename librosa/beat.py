#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beat and tempo
==============
.. autosummary::
   :toctree: generated/

   beat_track
   plp
"""

import numpy as np
import scipy
import scipy.stats
import numba

from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved, vectorize
from typing import Any, Callable, Optional, Tuple

__all__ = ["beat_track", "tempo", "plp"]


tempo = moved(moved_from="librosa.beat.tempo", version="0.10.0", version_removed="1.0")(
    _tempo
)


def beat_track(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = True,
    bpm: Optional[float] = None,
    prior: Optional[scipy.stats.rv_continuous] = None,
    units: str = "frames",
) -> Tuple[float, np.ndarray]:
    r"""Dynamic programming beat tracker.

    Beats are detected in three stages, following the method of [#]_:

      1. Measure onset strength
      2. Estimate tempo from onset correlation
      3. Pick peaks in onset strength approximately consistent with estimated
         tempo

    .. [#] Ellis, Daniel PW. "Beat tracking by dynamic programming."
           Journal of New Music Research 36.1 (2007): 51-60.
           http://labrosa.ee.columbia.edu/projects/beattrack/

    Parameters
    ----------
    y : np.ndarray [shape=(n,)] or None
        audio time series
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(n,)] or None
        (optional) pre-computed onset strength envelope.
    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values
    start_bpm : float > 0 [scalar]
        initial guess for the tempo estimator (in beats per minute)
    tightness : float [scalar]
        tightness of beat distribution around tempo
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets
    bpm : float [scalar]
        (optional) If provided, use ``bpm`` as the tempo instead of
        estimating it from ``onsets``.
    prior : scipy.stats.rv_continuous [optional]
        An optional prior distribution over tempo.
        If provided, ``start_bpm`` will be ignored.
    units : {'frames', 'samples', 'time'}
        The units to encode detected beat events in.
        By default, 'frames' are used.

    Returns
    -------
    tempo : float [scalar, non-negative]
        estimated global tempo (in beats per minute)
    beats : np.ndarray [shape=(m,)]
        estimated beat event locations in the specified units
        (default is frame indices)
    .. note::
        If no onset strength could be detected, beat_tracker estimates 0 BPM
        and returns an empty list.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided,
        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    librosa.onset.onset_strength

    Examples
    --------
    Track beats using time series input

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)

    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> tempo
    135.99917763157896

    Print the frames corresponding to beats

    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Or print them as timestamps

    >>> librosa.frames_to_time(beats, sr=sr)
    array([0.07 , 0.488, 0.929, 1.37 , 1.811, 2.229, 2.694, 3.135,
           3.576, 4.017, 4.458, 4.899, 5.341, 5.782, 6.223, 6.664,
           7.105, 7.546, 7.988, 8.429])

    Track beats using a pre-computed onset envelope

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          aggregate=np.median)
    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,
    ...                                        sr=sr)
    >>> tempo
    135.99917763157896
    >>> beats
    array([  3,  21,  40,  59,  78,  96, 116, 135, 154, 173, 192, 211,
           230, 249, 268, 287, 306, 325, 344, 363])

    Plot the beat events against the onset strength envelope

    >>> import matplotlib.pyplot as plt
    >>> hop_length = 512
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    >>> M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    >>> librosa.display.specshow(librosa.power_to_db(M, ref=np.max),
    ...                          y_axis='mel', x_axis='time', hop_length=hop_length,
    ...                          ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[1].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[1].legend()
    """
    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.median
        )

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return (0, np.array([], dtype=int))

    # Estimate BPM if one was not provided
    if bpm is None:
        bpm = _tempo(
            onset_envelope=onset_envelope,
            sr=sr,
            hop_length=hop_length,
            start_bpm=start_bpm,
            prior=prior,
        )[..., 0]

    # Then, run the tracker
    beats = __beat_tracker(onset_envelope, bpm, float(sr) / hop_length, tightness, trim)

    if units == "frames":
        return (bpm, beats)
    elif units == "samples":
        return (bpm, core.frames_to_samples(beats, hop_length=hop_length))
    elif units == "time":
        return (bpm, core.frames_to_time(beats, hop_length=hop_length, sr=sr))
    else:
        raise ParameterError(f"Invalid unit type: {units}")


def plp(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    tempo_min: Optional[float] = 30,
    tempo_max: Optional[float] = 300,
    prior: Optional[scipy.stats.rv_continuous] = None,
) -> np.ndarray:
    """Predominant local pulse (PLP) estimation. [#]_

    The PLP method analyzes the onset strength envelope in the frequency domain
    to find a locally stable tempo for each frame.  These local frames_per_beaticities
    are used to synthesize local half-waves, which are combined such that peaks
    coincide with rhythmically salient frames (e.g. onset events on a musical time grid).
    The local maxima of the pulse curve can be taken as estimated beat positions.

    This method may be preferred over the dynamic programming method of `beat_track`
    when the tempo is expected to vary significantly over time.  Additionally,
    since `plp` does not require the entire signal to make predictions, it may be
    preferable when beat-tracking long recordings in a streaming setting.

    .. [#] Grosche, P., & Muller, M. (2011).
        "Extracting predominant local pulse information from music recordings."
        IEEE Transactions on Audio, Speech, and Language Processing, 19(6), 1688-1701.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., n)] or None
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        number of audio samples between successive ``onset_envelope`` values

    win_length : int > 0 [scalar]
        number of frames to use for tempogram analysis.
        By default, 384 frames (at ``sr=22050`` and ``hop_length=512``) corresponds
        to about 8.9 seconds.

    tempo_min, tempo_max : numbers > 0 [scalar], optional
        Minimum and maximum permissible tempo values.  ``tempo_max`` must be at least
        ``tempo_min``.

        Set either (or both) to `None` to disable this constraint.

    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a uniform prior over ``[tempo_min, tempo_max]`` is used.

    Returns
    -------
    pulse : np.ndarray, shape=[(..., n)]
        The estimated pulse curve.  Maxima correspond to rhythmically salient
        points of time.

        If input is multi-channel, one pulse curve per channel is computed.

    See Also
    --------
    beat_track
    librosa.onset.onset_strength
    librosa.feature.fourier_tempogram

    Examples
    --------
    Visualize the PLP compared to an onset strength envelope.
    Both are normalized here to make comparison easier.

    >>> y, sr = librosa.load(librosa.ex('brahms'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    >>> # Or compute pulse with an alternate prior, like log-normal
    >>> import scipy.stats
    >>> prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> pulse_lognorm = librosa.beat.plp(onset_envelope=onset_env, sr=sr,
    ...                                  prior=prior)
    >>> melspec = librosa.feature.melspectrogram(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.power_to_db(melspec,
    ...                                              ref=np.max),
    ...                          x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Mel spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[1].plot(librosa.times_like(pulse),
    ...          librosa.util.normalize(pulse),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[1].set(title='Uniform tempo prior [30, 300]')
    >>> ax[1].label_outer()
    >>> ax[2].plot(librosa.times_like(onset_env),
    ...          librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[2].plot(librosa.times_like(pulse_lognorm),
    ...          librosa.util.normalize(pulse_lognorm),
    ...          label='Predominant local pulse (PLP)')
    >>> ax[2].set(title='Log-normal tempo prior, mean=120', xlim=[5, 20])
    >>> ax[2].legend()

    PLP local maxima can be used as estimates of beat positions.

    >>> tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    >>> beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> times = librosa.times_like(onset_env, sr=sr)
    >>> ax[0].plot(times, librosa.util.normalize(onset_env),
    ...          label='Onset strength')
    >>> ax[0].vlines(times[beats], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='Beats')
    >>> ax[0].legend()
    >>> ax[0].set(title='librosa.beat.beat_track')
    >>> ax[0].label_outer()
    >>> # Limit the plot to a 15-second window
    >>> times = librosa.times_like(pulse, sr=sr)
    >>> ax[1].plot(times, librosa.util.normalize(pulse),
    ...          label='PLP')
    >>> ax[1].vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
    ...            linestyle='--', label='PLP Beats')
    >>> ax[1].legend()
    >>> ax[1].set(title='librosa.beat.plp', xlim=[5, 20])
    >>> ax[1].xaxis.set_major_formatter(librosa.display.TimeFormatter())
    """
    # Step 1: get the onset envelope
    if onset_envelope is None:
        onset_envelope = onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length, aggregate=np.median
        )

    if tempo_min is not None and tempo_max is not None and tempo_max <= tempo_min:
        raise ParameterError(
            f"tempo_max={tempo_max} must be larger than tempo_min={tempo_min}"
        )

    # Step 2: get the fourier tempogram
    ftgram = fourier_tempogram(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Step 3: pin to the feasible tempo range
    tempo_frequencies = core.fourier_tempo_frequencies(
        sr=sr, hop_length=hop_length, win_length=win_length
    )

    if tempo_min is not None:
        ftgram[..., tempo_frequencies < tempo_min, :] = 0
    if tempo_max is not None:
        ftgram[..., tempo_frequencies > tempo_max, :] = 0

    # reshape lengths to match dimension properly
    tempo_frequencies = util.expand_to(tempo_frequencies, ndim=ftgram.ndim, axes=-2)

    # Step 3: Discard everything below the peak
    ftmag = np.log1p(1e6 * np.abs(ftgram))
    if prior is not None:
        ftmag += prior.logpdf(tempo_frequencies)

    peak_values = ftmag.max(axis=-2, keepdims=True)
    ftgram[ftmag < peak_values] = 0

    # Normalize to keep only phase information
    ftgram /= util.tiny(ftgram) ** 0.5 + np.abs(ftgram.max(axis=-2, keepdims=True))

    # Step 5: invert the Fourier tempogram to get the pulse
    pulse = core.istft(
        ftgram, hop_length=1, n_fft=win_length, length=onset_envelope.shape[-1]
    )

    # Step 6: retain only the positive part of the pulse cycle
    pulse = np.clip(pulse, 0, None, pulse)

    # Return the normalized pulse
    return util.normalize(pulse, axis=-1)


def __beat_tracker(
    onset_envelope: np.ndarray, bpm: float, frame_rate: float, tightness: float, trim: bool
) -> np.ndarray:
    """Tracks beats in an onset strength envelope.

    Parameters
    ----------
    onset_envelope : np.ndarray [shape=(n,)]
        onset strength envelope
    bpm : float [scalar]
        tempo estimate
    frame_rate : float [scalar]
        frame rate of the spectrogram (sr / hop_length, frames per second)
    tightness : float [scalar, positive]
        how closely do we adhere to bpm?
    trim : bool [scalar]
        trim leading/trailing beats with weak onsets?

    Returns
    -------
    beats : np.ndarray [shape=(n,)]
        frame numbers of beat events
    """
    if np.any(bpm <= 0):
        raise ParameterError(f"bpm={bpm} must be strictly positive")

    if tightness <= 0:
        raise ParameterError("tightness must be strictly positive")

    # convert bpm to frames per beat (rounded)
    # [frames / sec] * [60 sec / min] / [beat / min] = [frames / beat]
    frames_per_beat = np.round(frame_rate * 60.0 / bpm)

    # localscore is a smoothed version of AGC'd onset envelope
    localscore = __beat_local_score(__normalize_onsets(onset_envelope), frames_per_beat)

    # run the DP
    backlink, cumscore = __beat_track_dp(localscore, frames_per_beat, tightness)

    # Reconstruct the beat path from backlinks
    tail = __last_beat(cumscore)
    beats = np.zeros_like(onset_envelope, dtype=bool)
    __dp_backtrack(backlink, tail, beats)

    # Convert into an array of frame numbers
    beats = np.flatnonzero(beats)

    # Discard spurious trailing beats
    if trim:
        beats = __trim_beats(localscore, beats)

    return beats


# -- Helper functions for beat tracking
def __normalize_onsets(onsets):
    """Normalize onset strength by its standard deviation"""
    norm = onsets.std(ddof=1, axis=-1, keepdims=True)
    return onsets / (norm + util.tiny(onsets))


@vectorize(signature="(t),()->(t)")
def __beat_local_score(onset_envelope, frames_per_beat):
    """Construct the local score for an onset envlope and given frames_per_beat"""
    # Smoothing occurs over a ±1 beat time window
    # magic number 32 makes the window extremely sharp
    window = np.exp(-0.5 * (np.arange(-frames_per_beat, frames_per_beat + 1) * 32.0 / frames_per_beat) ** 2)
    return scipy.signal.convolve(onset_envelope, window, mode="same")


@numba.guvectorize(
        [
            "void(float32[:], float32, float32, int32[:], float32[:])",
            "void(float64[:], float64, float32, int32[:], float64[:])",
        ],
        "(t),(),()->(t),(t)",
        nopython=True, cache=True)
def __beat_track_dp(localscore, frames_per_beat, tightness, backlink, cumscore):
    """Core dynamic program for beat tracking"""
    # Search range for previous beat:
    window = np.arange(-2 * frames_per_beat, -np.round(frames_per_beat / 2) + 1, dtype=np.int32)

    # Make a score window, which begins biased toward start_bpm and skewed
    txwt = -tightness * (np.log(-window / frames_per_beat) ** 2)
    
    # Threshold for the first beat to exceed
    score_thresh = 0.01 * localscore.max()

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):
        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(-window[0] - i, len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] += cumscore[i + window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumscore[i] = score_i + candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < score_thresh:
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location] + i
            first_beat = False


def __trim_beats(localscore: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Remove spurious leading and trailing beats"""
    smooth_boe = scipy.ndimage.convolve1d(localscore[beats],
                                          scipy.signal.hann(5),
                                          mode="constant",
                                          axis=-1)

    # FIXME: support target axes here
    threshold = 0.5 * ((smooth_boe**2).mean() ** 0.5)

    # FIXME:  rewrite this to operate on boolean array instead of indices
    valid = np.argwhere(smooth_boe > threshold)

    return beats[valid.min() : valid.max()]


def __last_beat(cumscore):

    # Use a masked array to support multidimensional statistics
    # We negate the mask here because of numpy masked array semantics
    mask = ~util.localmax(cumscore, axis=-1)
    masked_scores = np.ma.masked_array(data=cumscore, mask=mask)  # type: ignore
    medians = np.ma.median(masked_scores, axis=-1, keepdims=True)
    thresholds = 0.5 * np.ma.getdata(medians)

    # Also find the last beat positions
    tail = np.empty(shape=cumscore.shape[:-1], dtype=int)
    __last_beat_selector(cumscore, mask, thresholds, tail)
    return tail


@numba.guvectorize(
        [
            "void(float32[:], bool_[:], float32, int64[:])",
            "void(float64[:], bool_[:], float64, int64[:])",
        ],
        "(t),(t),()->()",
        nopython=True, cache=True
        )
def __last_beat_selector(cumscore, mask, threshold, out):
    """Vectorized helper to identify the last valid beat position:

    cumscore[n] > threshold and not mask[n]
    """
    n = len(cumscore) - 1

    out[0] = n
    while n >= 0:
        if not mask[n] and cumscore[n] >= threshold:
            out[0] = n
            break
        else:
            n -= 1


@numba.guvectorize(
        [
            "void(int32[:], int32, bool_[:])",
            "void(int64[:], int64, bool_[:])"
        ],
        "(t),()->(t)",
        nopython=True, cache=True
        )
def __dp_backtrack(backlinks, tail, beats):

    n = tail
    while n >= 0:
        beats[n] = True
        n = backlinks[n]
