#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rhythmic feature extraction"""

import numpy as np
import scipy

from .. import util

from .._cache import cache
from ..core.audio import autocorrelate
from ..core.spectrum import stft
from ..core.convert import tempo_frequencies, time_to_frames
from ..core.harmonic import f0_harmonics
from ..util.exceptions import ParameterError
from ..filters import get_window
from typing import Optional, Callable, Any
from .._typing import _WindowSpec

__all__ = ["tempogram", "fourier_tempogram", "tempo", "tempogram_ratio"]


# -- Rhythmic features -- #
def tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = "hann",
    norm: Optional[float] = np.inf,
) -> np.ndarray:
    """Compute the tempogram: local autocorrelation of the onset strength envelope. [#]_

    .. [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series.  Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., n) or (..., m, n)] or None
        Optional pre-computed onset strength envelope as provided by
        `librosa.onset.onset_strength`.

        If multi-dimensional, tempograms are computed independently for each
        band (first dimension).

    hop_length : int > 0
        number of audio samples between successive onset measurements

    win_length : int > 0
        length of the onset autocorrelation window (in frames/onset measurements)
        The default settings (384) corresponds to ``384 * hop_length / sr ~= 8.9s``.

    center : bool
        If `True`, onset autocorrelation windows are centered.
        If `False`, windows are left-aligned.

    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.

    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode.  Set to `None` to disable normalization.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length, n)]
        Localized autocorrelation of the onset strength envelope.

        If given multi-band input (``onset_envelope.shape==(m,n)``) then
        ``tempogram[i]`` is the tempogram of ``onset_envelope[i]``.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided

        if ``win_length < 1``

    See Also
    --------
    fourier_tempogram
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.stft

    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                       hop_length=hop_length)
    >>> # Compute global onset autocorrelation
    >>> ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    >>> ac_global = librosa.util.normalize(ac_global)
    >>> # Estimate the global tempo for display purposes
    >>> tempo = librosa.feature.tempo(onset_envelope=oenv, sr=sr,
    ...                               hop_length=hop_length)[0]

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
    >>> times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    >>> ax[0].plot(times, oenv, label='Onset strength')
    >>> ax[0].label_outer()
    >>> ax[0].legend(frameon=True)
    >>> librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo', cmap='magma',
    ...                          ax=ax[1])
    >>> ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> ax[1].legend(loc='upper right')
    >>> ax[1].set(title='Tempogram')
    >>> x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
    ...                 num=tempogram.shape[0])
    >>> ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    >>> ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    >>> ax[2].set(xlabel='Lag (seconds)')
    >>> ax[2].legend(frameon=True)
    >>> freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    >>> ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
    ...              label='Mean local autocorrelation', base=2)
    >>> ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
    ...              label='Global autocorrelation', base=2)
    >>> ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
    ...             label='Estimated tempo={:g}'.format(tempo))
    >>> ax[3].legend(frameon=True)
    >>> ax[3].set(xlabel='BPM')
    >>> ax[3].grid(True)
    """

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError("win_length must be a positive integer")

    ac_window = get_window(window, win_length, fftbins=True)

    if onset_envelope is None:
        if y is None:
            raise ParameterError("Either y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Center the autocorrelation windows
    n = onset_envelope.shape[-1]

    if center:
        padding = [(0, 0) for _ in onset_envelope.shape]
        padding[-1] = (int(win_length // 2),) * 2
        onset_envelope = np.pad(
            onset_envelope, padding, mode="linear_ramp", end_values=[0, 0]
        )

    # Carve onset envelope into frames
    odf_frame = util.frame(onset_envelope, frame_length=win_length, hop_length=1)

    # Truncate to the length of the original signal
    if center:
        odf_frame = odf_frame[..., :n]

    # explicit broadcast of ac_window
    ac_window = util.expand_to(ac_window, ndim=odf_frame.ndim, axes=-2)

    # Window, autocorrelate, and normalize
    return util.normalize(
        autocorrelate(odf_frame * ac_window, axis=-2), norm=norm, axis=-2
    )


def fourier_tempogram(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    center: bool = True,
    window: _WindowSpec = "hann",
) -> np.ndarray:
    """Compute the Fourier tempogram: the short-time Fourier transform of the
    onset strength envelope. [#]_

    .. [#] Grosche, Peter, Meinard Müller, and Frank Kurth.
        "Cyclic tempogram - A mid-level tempo representation for music signals."
        ICASSP, 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        Audio time series.  Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    onset_envelope : np.ndarray [shape=(..., n)] or None
        Optional pre-computed onset strength envelope as provided by
        ``librosa.onset.onset_strength``.
        Multi-channel is supported.
    hop_length : int > 0
        number of audio samples between successive onset measurements
    win_length : int > 0
        length of the onset window (in frames/onset measurements)
        The default settings (384) corresponds to ``384 * hop_length / sr ~= 8.9s``.
    center : bool
        If `True`, onset windows are centered.
        If `False`, windows are left-aligned.
    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.

    Returns
    -------
    tempogram : np.ndarray [shape=(..., win_length // 2 + 1, n)]
        Complex short-time Fourier transform of the onset envelope.

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onset_envelope`` are provided

        if ``win_length < 1``

    See Also
    --------
    tempogram
    librosa.onset.onset_strength
    librosa.util.normalize
    librosa.stft

    Examples
    --------
    >>> # Compute local onset autocorrelation
    >>> y, sr = librosa.load(librosa.ex('nutcracker'))
    >>> hop_length = 512
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    >>> tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
    ...                                               hop_length=hop_length)
    >>> # Compute the auto-correlation tempogram, unnormalized to make comparison easier
    >>> ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
    ...                                          hop_length=hop_length, norm=None)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> ax[0].plot(librosa.times_like(oenv), oenv, label='Onset strength')
    >>> ax[0].legend(frameon=True)
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='fourier_tempo', cmap='magma',
    ...                          ax=ax[1])
    >>> ax[1].set(title='Fourier tempogram')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length,
    >>>                          x_axis='time', y_axis='tempo', cmap='magma',
    ...                          ax=ax[2])
    >>> ax[2].set(title='Autocorrelation tempogram')
    """

    from ..onset import onset_strength

    if win_length < 1:
        raise ParameterError("win_length must be a positive integer")

    if onset_envelope is None:
        if y is None:
            raise ParameterError("Either y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Generate the short-time Fourier transform
    return stft(
        onset_envelope, n_fft=win_length, hop_length=1, center=center, window=window
    )


@cache(level=30)
def tempo(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    tg: Optional[np.ndarray] = None,
    hop_length: int = 512,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    ac_size: float = 8.0,
    max_tempo: Optional[float] = 320.0,
    aggregate: Optional[Callable[..., Any]] = np.mean,
    prior: Optional[scipy.stats.rv_continuous] = None,
) -> np.ndarray:
    """Estimate the tempo (beats per minute)

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of the time series
    onset_envelope : np.ndarray [shape=(..., n)]
        pre-computed onset strength envelope
    tg : np.ndarray
        pre-computed tempogram.  If provided, then `y` and
        `onset_envelope` are ignored, and `win_length` is
        inferred from the shape of the tempogram.
    hop_length : int > 0 [scalar]
        hop length of the time series
    start_bpm : float [scalar]
        initial guess of the BPM
    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution
    ac_size : float > 0 [scalar]
        length (in seconds) of the auto-correlation window
    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold
    aggregate : callable [optional]
        Aggregation function for estimating global tempo.
        If `None`, then tempo is estimated independently for each frame.
    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a pseudo-log-normal prior is used.
        If given, ``start_bpm`` and ``std_bpm`` will be ignored.

    Returns
    -------
    tempo : np.ndarray
        estimated tempo (beats per minute).
        If input is multi-channel, one tempo estimate per channel is provided.

    See Also
    --------
    librosa.onset.onset_strength
    librosa.feature.tempogram

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Estimate a static tempo
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    >>> tempo
    array([143.555])

    >>> # Or a static tempo with a uniform prior instead
    >>> import scipy.stats
    >>> prior = scipy.stats.uniform(30, 300)  # uniform over 30-300 BPM
    >>> utempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, prior=prior)
    >>> utempo
    array([161.499])

    >>> # Or a dynamic tempo
    >>> dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
    ...                                aggregate=None)
    >>> dtempo
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    >>> # Dynamic tempo with a proper log-normal prior
    >>> prior_lognorm = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    >>> dtempo_lognorm = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
    ...                                        aggregate=None,
    ...                                        prior=prior_lognorm)
    >>> dtempo_lognorm
    array([ 89.103,  89.103,  89.103, ..., 123.047, 123.047, 123.047])

    Plot the estimated tempo against the onset autocorrelation

    >>> import matplotlib.pyplot as plt
    >>> # Convert to scalar
    >>> tempo = tempo.item()
    >>> utempo = utempo.item()
    >>> # Compute 2-second windowed autocorrelation
    >>> hop_length = 512
    >>> ac = librosa.autocorrelate(onset_env, max_size=2 * sr // hop_length)
    >>> freqs = librosa.tempo_frequencies(len(ac), sr=sr,
    ...                                   hop_length=hop_length)
    >>> # Plot on a BPM axis.  We skip the first (0-lag) bin.
    >>> fig, ax = plt.subplots()
    >>> ax.semilogx(freqs[1:], librosa.util.normalize(ac)[1:],
    ...              label='Onset autocorrelation', base=2)
    >>> ax.axvline(tempo, 0, 1, alpha=0.75, linestyle='--', color='r',
    ...             label='Tempo (default prior): {:.2f} BPM'.format(tempo))
    >>> ax.axvline(utempo, 0, 1, alpha=0.75, linestyle=':', color='g',
    ...             label='Tempo (uniform prior): {:.2f} BPM'.format(utempo))
    >>> ax.set(xlabel='Tempo (BPM)', title='Static tempo estimation')
    >>> ax.grid(True)
    >>> ax.legend()

    Plot dynamic tempo estimates over a tempogram

    >>> fig, ax = plt.subplots()
    >>> tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
    ...                                hop_length=hop_length)
    >>> librosa.display.specshow(tg, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)
    >>> ax.plot(librosa.times_like(dtempo), dtempo,
    ...          color='c', linewidth=1.5, label='Tempo estimate (default prior)')
    >>> ax.plot(librosa.times_like(dtempo_lognorm), dtempo_lognorm,
    ...          color='c', linewidth=1.5, linestyle='--',
    ...          label='Tempo estimate (lognorm prior)')
    >>> ax.set(title='Dynamic tempo estimation')
    >>> ax.legend()
    """

    if start_bpm <= 0:
        raise ParameterError("start_bpm must be strictly positive")

    if tg is None:
        win_length = time_to_frames(ac_size, sr=sr, hop_length=hop_length).item()

        tg = tempogram(
            y=y,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            win_length=win_length,
        )
    else:
        # Override window length by what's actually given
        win_length = tg.shape[-2]

    # Eventually, we want this to work for time-varying tempo
    if aggregate is not None:
        tg = aggregate(tg, axis=-1, keepdims=True)

    assert tg is not None

    # Get the BPM values for each bin, skipping the 0-lag bin
    bpms = tempo_frequencies(win_length, hop_length=hop_length, sr=sr)

    # Weight the autocorrelation by a log-normal distribution
    if prior is None:
        logprior = -0.5 * ((np.log2(bpms) - np.log2(start_bpm)) / std_bpm) ** 2
    else:
        logprior = prior.logpdf(bpms)

    # Kill everything above the max tempo
    if max_tempo is not None:
        max_idx = int(np.argmax(bpms < max_tempo))
        logprior[:max_idx] = -np.inf
    # explicit axis expansion
    logprior = util.expand_to(logprior, ndim=tg.ndim, axes=-2)

    # Get the maximum, weighted by the prior
    # Using log1p here for numerical stability
    best_period = np.argmax(np.log1p(1e6 * tg) + logprior, axis=-2)

    tempo_est: np.ndarray = np.take(bpms, best_period)
    return tempo_est


@cache(level=40)
def tempogram_ratio(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    onset_envelope: Optional[np.ndarray] = None,
    tg: Optional[np.ndarray] = None,
    bpm: Optional[np.ndarray] = None,
    hop_length: int = 512,
    win_length: int = 384,
    start_bpm: float = 120,
    std_bpm: float = 1.0,
    max_tempo: Optional[float] = 320.0,
    freqs: Optional[np.ndarray] = None,
    factors: Optional[np.ndarray] = None,
    aggregate: Optional[Callable[..., Any]] = None,
    prior: Optional[scipy.stats.rv_continuous] = None,
    center: bool = True,
    window: _WindowSpec = "hann",
    kind: str = "linear",
    fill_value: float = 0,
    norm: Optional[float] = np.inf,
) -> np.ndarray:
    """Tempogram ratio features, also known as spectral rhythm patterns. [1]_

    This function summarizes the energy at metrically important multiples
    of the tempo.  For example, if the tempo corresponds to the quarter-note
    period, the tempogram ratio will measure the energy at the eighth note,
    sixteenth note, half note, whole note, etc. periods, as well as dotted
    and triplet ratios.

    By default, the multiplicative factors used here are as specified by
    [2]_.  If the estimated tempo corresponds to a quarter note, these factors
    will measure relative energy at the following metrical subdivisions:

    +-------+--------+------------------+
    | Index | Factor | Description      |
    +=======+========+==================+
    |     0 |    4   | Sixteenth note   |
    +-------+--------+------------------+
    |     1 |    8/3 | Dotted sixteenth |
    +-------+--------+------------------+
    |     2 |    3   | Eighth triplet   |
    +-------+--------+------------------+
    |     3 |    2   | Eighth note      |
    +-------+--------+------------------+
    |     4 |    4/3 | Dotted eighth    |
    +-------+--------+------------------+
    |     5 |    3/2 | Quarter triplet  |
    +-------+--------+------------------+
    |     6 |    1   | Quarter note     |
    +-------+--------+------------------+
    |     7 |    2/3 | Dotted quarter   |
    +-------+--------+------------------+
    |     8 |    3/4 | Half triplet     |
    +-------+--------+------------------+
    |     9 |    1/2 | Half note        |
    +-------+--------+------------------+
    |    10 |    1/3 | Dotted half note |
    +-------+--------+------------------+
    |    11 |    3/8 | Whole triplet    |
    +-------+--------+------------------+
    |    12 |    1/4 | Whole note       |
    +-------+--------+------------------+

    .. [1] Peeters, Geoffroy.
        "Rhythm Classification Using Spectral Rhythm Patterns."
        In ISMIR, pp. 644-647. 2005.

    .. [2] Prockup, Matthew, Andreas F. Ehmann, Fabien Gouyon, Erik M. Schmidt, and Youngmoo E. Kim.
        "Modeling musical rhythm at scale with the music genome project."
        In 2015 IEEE workshop on applications of signal processing to audio and acoustics (WASPAA), pp. 1-5. IEEE, 2015.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series
    sr : number > 0 [scalar]
        sampling rate of the time series
    onset_envelope : np.ndarray [shape=(..., n)]
        pre-computed onset strength envelope
    tg : np.ndarray
        pre-computed tempogram.  If provided, then `y` and
        `onset_envelope` are ignored, and `win_length` is
        inferred from the shape of the tempogram.
    bpm : np.ndarray
        pre-computed tempo estimate.  This must be a per-frame
        estimate, and have dimension compatible with `tg`.
    hop_length : int > 0 [scalar]
        hop length of the time series
    win_length : int > 0 [scalar]
        window length of the autocorrelation window for tempogram
        calculation
    start_bpm : float [scalar]
        initial guess of the BPM if `bpm` is not provided
    std_bpm : float > 0 [scalar]
        standard deviation of tempo distribution
    max_tempo : float > 0 [scalar, optional]
        If provided, only estimate tempo below this threshold
    freqs : np.ndarray
        Frequencies (in BPM) of the tempogram axis.
    factors : np.ndarray
        Multiples of the fundamental tempo (bpm) to estimate.
        If not provided, the factors are as specified above.
    prior : scipy.stats.rv_continuous [optional]
        A prior distribution over tempo (in beats per minute).
        By default, a pseudo-log-normal prior is used.
        If given, ``start_bpm`` and ``std_bpm`` will be ignored.
    center : bool
        If `True`, onset windows are centered.
        If `False`, windows are left-aligned.
    aggregate : callable [optional]
        Aggregation function for estimating global tempogram ratio.
        If `None`, then ratios are estimated independently for each frame.
    window : string, function, number, tuple, or np.ndarray [shape=(win_length,)]
        A window specification as in `stft`.
    kind : str
        Interpolation mode for measuring tempogram ratios
    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.
    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode.  Set to `None` to disable normalization.

    Returns
    -------
    tgr : np.ndarray
        The tempogram ratio for the specified factors.
        If `aggregate` is provided, the trailing time axis
        will be removed.
        If `aggregate` is not provided (default), ratios
        will be estimated for each frame.

    See Also
    --------
    tempogram
    tempo
    librosa.f0_harmonics
    librosa.tempo_frequencies

    Examples
    --------
    Compute tempogram ratio features using the default factors
    for a waltz (3/4 time)

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('sweetwaltz'))
    >>> tempogram = librosa.feature.tempogram(y=y, sr=sr)
    >>> tgr = librosa.feature.tempogram_ratio(tg=tempogram, sr=sr)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(tempogram, x_axis='time', y_axis='tempo',
    ...                          ax=ax[0])
    >>> librosa.display.specshow(tgr, x_axis='time', ax=ax[1])
    >>> ax[0].label_outer()
    >>> ax[0].set(title="Tempogram")
    >>> ax[1].set(title="Tempogram ratio")
    """

    # Get a tempogram and time-varying tempo estimate
    if tg is None:
        tg = tempogram(
            y=y,
            sr=sr,
            onset_envelope=onset_envelope,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            window=window,
            norm=norm,
        )

    if freqs is None:
        freqs = tempo_frequencies(sr=sr, n_bins=len(tg), hop_length=hop_length)

    # Estimate tempo per-frame, no aggregation yet
    if bpm is None:
        bpm = tempo(
            sr=sr,
            tg=tg,
            hop_length=hop_length,
            start_bpm=start_bpm,
            std_bpm=std_bpm,
            max_tempo=max_tempo,
            aggregate=None,
            prior=prior,
        )

    if factors is None:
        # metric multiples from Prockup'15
        factors = np.array(
            [4, 8 / 3, 3, 2, 4 / 3, 3 / 2, 1, 2 / 3, 3 / 4, 1 / 2, 1 / 3, 3 / 8, 1 / 4]
        )

    tgr = f0_harmonics(
        tg, freqs=freqs, f0=bpm, harmonics=factors, kind=kind, fill_value=fill_value
    )

    if aggregate is not None:
        return aggregate(tgr, axis=-1)  # type: ignore

    return tgr
