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
    preemphasis
    deemphasis
"""

import numpy as np
import scipy.signal

from . import core
from . import decompose
from . import feature
from . import util
from .util.exceptions import ParameterError
from typing import Any, Callable, Iterable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from numpy.typing import ArrayLike
from ._typing import (
    _WindowSpec,
    _PadModeSTFT,
    _IntLike_co,
    _FloatLike_co,
)

__all__ = [
    "hpss",
    "harmonic",
    "percussive",
    "time_stretch",
    "pitch_shift",
    "remix",
    "trim",
    "split",
]


def hpss(
    y: np.ndarray,
    *,
    kernel_size: Union[
        _IntLike_co, Tuple[_IntLike_co, _IntLike_co], List[_IntLike_co]
    ] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[
        _FloatLike_co, Tuple[_FloatLike_co, _FloatLike_co], List[_FloatLike_co]
    ] = 1.0,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose an audio time series into harmonic and percussive components.

    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform ``y``.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    kernel_size
    power
    mask
    margin
        See `librosa.decompose.hpss`
    n_fft
    hop_length
    win_length
    window
    center
    pad_mode
        See `librosa.stft`

    Returns
    -------
    y_harmonic : np.ndarray [shape=(..., n)]
        audio time series of the harmonic elements
    y_percussive : np.ndarray [shape=(..., n)]
        audio time series of the percussive elements

    See Also
    --------
    harmonic : Extract only the harmonic component
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS on spectrograms

    Examples
    --------
    >>> # Extract harmonic and percussive components
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y)

    >>> # Get a more isolated percussive component by widening its margin
    >>> y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0,5.0))
    """
    # Compute the STFT matrix
    stft = core.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        pad_mode=pad_mode,
    )

    # Decompose into harmonic and percussives
    stft_harm, stft_perc = decompose.hpss(
        stft, kernel_size=kernel_size, power=power, mask=mask, margin=margin
    )

    # Invert the STFTs.  Adjust length to match the input.
    y_harm = core.istft(
        stft_harm,
        dtype=y.dtype,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        length=y.shape[-1],
    )
    y_perc = core.istft(
        stft_perc,
        dtype=y.dtype,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        length=y.shape[-1],
    )

    return y_harm, y_perc


def harmonic(
    y: np.ndarray,
    *,
    kernel_size: Union[
        _IntLike_co, Tuple[_IntLike_co, _IntLike_co], List[_IntLike_co]
    ] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[
        _FloatLike_co, Tuple[_FloatLike_co, _FloatLike_co], List[_FloatLike_co]
    ] = 1.0,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> np.ndarray:
    """Extract harmonic elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    kernel_size
    power
    mask
    margin
        See `librosa.decompose.hpss`
    n_fft
    hop_length
    win_length
    window
    center
    pad_mode
        See `librosa.stft`

    Returns
    -------
    y_harmonic : np.ndarray [shape=(..., n)]
        audio time series of just the harmonic portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    percussive : Extract only the percussive component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> # Extract harmonic component
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_harmonic = librosa.effects.harmonic(y)

    >>> # Use a margin > 1.0 for greater harmonic separation
    >>> y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    """
    # Compute the STFT matrix
    stft = core.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        pad_mode=pad_mode,
    )

    # Remove percussives
    stft_harm = decompose.hpss(
        stft, kernel_size=kernel_size, power=power, mask=mask, margin=margin
    )[0]

    # Invert the STFTs
    y_harm = core.istft(
        stft_harm,
        dtype=y.dtype,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        length=y.shape[-1],
    )

    return y_harm


def percussive(
    y: np.ndarray,
    *,
    kernel_size: Union[
        _IntLike_co, Tuple[_IntLike_co, _IntLike_co], List[_IntLike_co]
    ] = 31,
    power: float = 2.0,
    mask: bool = False,
    margin: Union[
        _FloatLike_co, Tuple[_FloatLike_co, _FloatLike_co], List[_FloatLike_co]
    ] = 1.0,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> np.ndarray:
    """Extract percussive elements from an audio time-series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    kernel_size
    power
    mask
    margin
        See `librosa.decompose.hpss`
    n_fft
    hop_length
    win_length
    window
    center
    pad_mode
        See `librosa.stft`

    Returns
    -------
    y_percussive : np.ndarray [shape=(..., n)]
        audio time series of just the percussive portion

    See Also
    --------
    hpss : Separate harmonic and percussive components
    harmonic : Extract only the harmonic component
    librosa.decompose.hpss : HPSS for spectrograms

    Examples
    --------
    >>> # Extract percussive component
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_percussive = librosa.effects.percussive(y)

    >>> # Use a margin > 1.0 for greater percussive separation
    >>> y_percussive = librosa.effects.percussive(y, margin=3.0)
    """
    # Compute the STFT matrix
    stft = core.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        pad_mode=pad_mode,
    )

    # Remove harmonics
    stft_perc = decompose.hpss(
        stft, kernel_size=kernel_size, power=power, mask=mask, margin=margin
    )[1]

    # Invert the STFT
    y_perc = core.istft(
        stft_perc,
        dtype=y.dtype,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
        length=y.shape[-1],
    )

    return y_perc


def time_stretch(y: np.ndarray, *, rate: float, **kwargs: Any) -> np.ndarray:
    """Time-stretch an audio series by a fixed rate.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    rate : float > 0 [scalar]
        Stretch factor.  If ``rate > 1``, then the signal is sped up.
        If ``rate < 1``, then the signal is slowed down.
    **kwargs : additional keyword arguments.
        See `librosa.decompose.stft` for details.

    Returns
    -------
    y_stretch : np.ndarray [shape=(..., round(n/rate))]
        audio time series stretched by the specified rate

    See Also
    --------
    pitch_shift :
        pitch shifting
    librosa.phase_vocoder :
        spectrogram phase vocoder
    pyrubberband.pyrb.time_stretch :
        high-quality time stretching using RubberBand

    Examples
    --------
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_fast = librosa.effects.time_stretch(y, rate=2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, rate=0.5)
    """
    if rate <= 0:
        raise ParameterError("rate must be a positive number")

    # Construct the short-term Fourier transform (STFT)
    stft = core.stft(y, **kwargs)

    # Stretch by phase vocoding
    stft_stretch = core.phase_vocoder(
        stft,
        rate=rate,
        hop_length=kwargs.get("hop_length", None),
        n_fft=kwargs.get("n_fft", None),
    )

    # Predict the length of y_stretch
    len_stretch = int(round(y.shape[-1] / rate))

    # Invert the STFT
    y_stretch = core.istft(stft_stretch, dtype=y.dtype, length=len_stretch, **kwargs)

    return y_stretch


def pitch_shift(
    y: np.ndarray,
    *,
    sr: float,
    n_steps: float,
    bins_per_octave: int = 12,
    res_type: str = "soxr_hq",
    scale: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Shift the pitch of a waveform by ``n_steps`` steps.

    A step is equal to a semitone if ``bins_per_octave`` is set to 12.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    n_steps : float [scalar]
        how many (fractional) steps to shift ``y``

    bins_per_octave : int > 0 [scalar]
        how many steps per octave

    res_type : string
        Resample type. By default, 'soxr_hq' is used.

        See `librosa.resample` for more information.

    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.

    **kwargs : additional keyword arguments.
        See `librosa.decompose.stft` for details.

    Returns
    -------
    y_shift : np.ndarray [shape=(..., n)]
        The pitch-shifted audio time-series

    See Also
    --------
    time_stretch :
        time stretching
    librosa.phase_vocoder :
        spectrogram phase vocoder
    pyrubberband.pyrb.pitch_shift :
        high-quality pitch shifting using RubberBand

    Examples
    --------
    Shift up by a major third (four steps if ``bins_per_octave`` is 12)

    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> y_third = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)

    Shift down by a tritone (six steps if ``bins_per_octave`` is 12)

    >>> y_tritone = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)

    Shift up by 3 quarter-tones

    >>> y_three_qt = librosa.effects.pitch_shift(y, sr=sr, n_steps=3,
    ...                                          bins_per_octave=24)
    """
    if not util.is_positive_int(bins_per_octave):
        raise ParameterError(
            f"bins_per_octave={bins_per_octave} must be a positive integer."
        )

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = core.resample(
        time_stretch(y, rate=rate, **kwargs),
        orig_sr=float(sr) / rate,
        target_sr=sr,
        res_type=res_type,
        scale=scale,
    )

    # Crop to the same dimension as the input
    return util.fix_length(y_shift, size=y.shape[-1])


def remix(
    y: np.ndarray, intervals: Iterable[Tuple[int, int]], *, align_zeros: bool = True
) -> np.ndarray:
    """Remix an audio signal by re-ordering time intervals.

    Parameters
    ----------
    y : np.ndarray [shape=(..., t)]
        Audio time series. Multi-channel is supported.
    intervals : iterable of tuples (start, end)
        An iterable (list-like or generator) where the ``i``th item
        ``intervals[i]`` indicates the start and end (in samples)
        of a slice of ``y``.
    align_zeros : boolean
        If ``True``, interval boundaries are mapped to the closest
        zero-crossing in ``y``.  If ``y`` is stereo, zero-crossings
        are computed after converting to mono.

    Returns
    -------
    y_remix : np.ndarray [shape=(..., d)]
        ``y`` remixed in the order specified by ``intervals``

    Examples
    --------
    Load in the example track and reverse the beats

    >>> y, sr = librosa.load(librosa.ex('choice'))

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
    """
    y_out = []

    if align_zeros:
        y_mono = core.to_mono(y)
        zeros = np.nonzero(core.zero_crossings(y_mono))[-1]
        # Force end-of-signal onto zeros
        zeros = np.append(zeros, [len(y_mono)])

    for interval in intervals:
        if align_zeros:
            interval = zeros[util.match_events(interval, zeros)]

        y_out.append(y[..., interval[0] : interval[1]])

    return np.concatenate(y_out, axis=-1)


def _signal_to_frame_nonsilent(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    top_db: float = 60,
    ref: Union[Callable, float] = np.max,
    aggregate: Callable = np.max,
) -> np.ndarray:
    """Frame-wise non-silent indicator for audio input.

    This is a helper function for `trim` and `split`.

    Parameters
    ----------
    y : np.ndarray
        Audio signal, mono or stereo

    frame_length : int > 0
        The number of samples per frame

    hop_length : int > 0
        The number of samples between frames

    top_db : number
        The threshold (in decibels) below reference to consider as
        silence.
        You can also use a negative value for `top_db` to treat any value
        below `ref + |top_db|` as silent.  This will only make sense if
        `ref` is not `np.max`.

    ref : callable or float
        The reference amplitude

    aggregate : callable [default: np.max]
        Function to aggregate dB measurements across channels (if y.ndim > 1)

        Note: for multiple leading axes, this is performed using ``np.apply_over_axes``.

    Returns
    -------
    non_silent : np.ndarray, shape=(m,), dtype=bool
        Indicator of non-silent frames
    """
    # Compute the MSE for the signal
    mse = feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Convert to decibels and slice out the mse channel
    db: np.ndarray = core.amplitude_to_db(mse[..., 0, :], ref=ref, top_db=None)

    # Aggregate everything but the time dimension
    if db.ndim > 1:
        db = np.apply_over_axes(aggregate, db, range(db.ndim - 1))
        # Squeeze out leading singleton dimensions here
        # We always want to keep the trailing dimension though
        db = np.squeeze(db, axis=tuple(range(db.ndim - 1)))

    return db > -top_db


def trim(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref: Union[float, Callable] = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable = np.max,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trim leading and trailing silence from an audio signal.

    Silence is defined as segments of the audio signal that are `top_db`
    decibels (or more) quieter than a reference level, `ref`.
    By default, `ref` is set to the signal's maximum RMS value.
    It's important to note that if the entire signal maintains a uniform
    RMS value, there will be no segments considered quieter than the maximum,
    leading to no trimming.
    This implies that a completely silent signal will remain untrimmed with the default `ref` setting.
    In these situations, an explicit value for `ref` (in decibels) should be used instead.

    Parameters
    ----------
    y : np.ndarray, shape=(..., n)
        Audio signal. Multi-channel is supported.
    top_db : number
        The threshold (in decibels) below reference to consider as
        silence.
        You can also use a negative value for `top_db` to treat any value
        below `ref + |top_db|` as silent.  This will only make sense if
        `ref` is not `np.max`.
    ref : number or callable
        The reference amplitude.  By default, it uses `np.max` and compares
        to the peak amplitude in the signal.
    frame_length : int > 0
        The number of samples per analysis frame
    hop_length : int > 0
        The number of samples between analysis frames
    aggregate : callable [default: np.max]
        Function to aggregate across channels (if y.ndim > 1)

    Returns
    -------
    y_trimmed : np.ndarray, shape=(..., m)
        The trimmed signal
    index : np.ndarray, shape=(2,)
        the interval of ``y`` corresponding to the non-silent region:
        ``y_trimmed = y[index[0]:index[1]]`` (for mono) or
        ``y_trimmed = y[:, index[0]:index[1]]`` (for stereo).

    Examples
    --------
    >>> # Load some audio
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> # Trim the beginning and ending silence
    >>> yt, index = librosa.effects.trim(y)
    >>> # Print the durations
    >>> print(librosa.get_duration(y, sr=sr), librosa.get_duration(yt, sr=sr))
    25.025986394557822 25.007891156462584
    """
    non_silent = _signal_to_frame_nonsilent(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        ref=ref,
        top_db=top_db,
        aggregate=aggregate,
    )

    nonzero = np.flatnonzero(non_silent)

    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(core.frames_to_samples(nonzero[0], hop_length=hop_length))
        end = min(
            y.shape[-1],
            int(core.frames_to_samples(nonzero[-1] + 1, hop_length=hop_length)),
        )
    else:
        # The entire signal is trimmed here: nothing is above the threshold
        start, end = 0, 0

    # Slice the buffer and return the corresponding interval
    return y[..., start:end], np.asarray([start, end])


def split(
    y: np.ndarray,
    *,
    top_db: float = 60,
    ref: Union[float, Callable] = np.max,
    frame_length: int = 2048,
    hop_length: int = 512,
    aggregate: Callable = np.max,
) -> np.ndarray:
    """Split an audio signal into non-silent intervals.

    Parameters
    ----------
    y : np.ndarray, shape=(..., n)
        An audio signal. Multi-channel is supported.
    top_db : number > 0
        The threshold (in decibels) below reference to consider as
        silence
    ref : number or callable
        The reference amplitude.  By default, it uses `np.max` and compares
        to the peak amplitude in the signal.
    frame_length : int > 0
        The number of samples per analysis frame
    hop_length : int > 0
        The number of samples between analysis frames
    aggregate : callable [default: np.max]
        Function to aggregate across channels (if y.ndim > 1)

    Returns
    -------
    intervals : np.ndarray, shape=(m, 2)
        ``intervals[i] == (start_i, end_i)`` are the start and end time
        (in samples) of non-silent interval ``i``.
    """
    non_silent = _signal_to_frame_nonsilent(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        ref=ref,
        top_db=top_db,
        aggregate=aggregate,
    )

    # Interval slicing, adapted from
    # https://stackoverflow.com/questions/2619413/efficiently-finding-the-interval-with-non-zeros-in-scipy-numpy-in-python
    # Find points where the sign flips
    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if non_silent[0]:
        edges.insert(0, np.array([0]))

    # Likewise for the last frame
    if non_silent[-1]:
        edges.append(np.array([len(non_silent)]))

    # Convert from frames to samples
    edges = core.frames_to_samples(np.concatenate(edges), hop_length=hop_length)

    # Clip to the signal duration
    edges = np.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    edges = edges.reshape((-1, 2))  # type: np.ndarray
    return edges


@overload
def preemphasis(
    y: np.ndarray,
    *,
    coef: float = ...,
    zi: Optional[ArrayLike] = ...,
    return_zf: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def preemphasis(
    y: np.ndarray,
    *,
    coef: float = ...,
    zi: Optional[ArrayLike] = ...,
    return_zf: Literal[True],
) -> Tuple[np.ndarray, np.ndarray]: ...


@overload
def preemphasis(
    y: np.ndarray,
    *,
    coef: float = ...,
    zi: Optional[ArrayLike] = ...,
    return_zf: bool,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...


def preemphasis(
    y: np.ndarray,
    *,
    coef: float = 0.97,
    zi: Optional[ArrayLike] = None,
    return_zf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Pre-emphasize an audio signal with a first-order differencing filter:

        y[n] -> y[n] - coef * y[n-1]

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Audio signal. Multi-channel is supported.

    coef : positive number
        Pre-emphasis coefficient.  Typical values of ``coef`` are between 0 and 1.

        At the limit ``coef=0``, the signal is unchanged.

        At ``coef=1``, the result is the first-order difference of the signal.

        The default (0.97) matches the pre-emphasis filter used in the HTK
        implementation of MFCCs [#]_.

        .. [#] https://htk.eng.cam.ac.uk/

    zi : number
        Initial filter state.  When making successive calls to non-overlapping
        frames, this can be set to the ``zf`` returned from the previous call.
        (See example below.)

        By default ``zi`` is initialized as ``2*y[0] - y[1]``.

    return_zf : boolean
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray
        pre-emphasized signal
    zf : number
        if ``return_zf=True``, the final filter state is also returned

    Examples
    --------
    Apply a standard pre-emphasis filter

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> y_filt = librosa.effects.preemphasis(y)
    >>> # and plot the results for comparison
    >>> S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max, top_db=None)
    >>> S_preemph = librosa.amplitude_to_db(np.abs(librosa.stft(y_filt)), ref=np.max, top_db=None)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(S_orig, y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Original signal')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(S_preemph, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Pre-emphasized signal')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")

    Apply pre-emphasis in pieces for block streaming.  Note that the second block
    initializes ``zi`` with the final state ``zf`` returned by the first call.

    >>> y_filt_1, zf = librosa.effects.preemphasis(y[:1000], return_zf=True)
    >>> y_filt_2, zf = librosa.effects.preemphasis(y[1000:], zi=zf, return_zf=True)
    >>> np.allclose(y_filt, np.concatenate([y_filt_1, y_filt_2]))
    True

    See Also
    --------
    deemphasis
    """
    b = np.asarray([1.0, -coef], dtype=y.dtype)
    a = np.asarray([1.0], dtype=y.dtype)

    if zi is None:
        # Initialize the filter to implement linear extrapolation
        zi = 2 * y[..., 0:1] - y[..., 1:2]

    zi = np.atleast_1d(zi)

    y_out: np.ndarray
    z_f: np.ndarray

    y_out, z_f = scipy.signal.lfilter(b, a, y, zi=np.asarray(zi, dtype=y.dtype))

    if return_zf:
        return y_out, z_f

    return y_out


@overload
def deemphasis(
    y: np.ndarray,
    *,
    coef: float = ...,
    zi: Optional[ArrayLike] = ...,
    return_zf: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def deemphasis(
    y: np.ndarray,
    *,
    coef: float = ...,
    zi: Optional[ArrayLike] = ...,
    return_zf: Literal[True],
) -> Tuple[np.ndarray, np.ndarray]: ...


def deemphasis(
    y: np.ndarray,
    *,
    coef: float = 0.97,
    zi: Optional[ArrayLike] = None,
    return_zf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """De-emphasize an audio signal with the inverse operation of preemphasis():

    If y = preemphasis(x, coef=coef, zi=zi), the deemphasis is:

    >>> x[i] = y[i] + coef * x[i-1]
    >>> x = deemphasis(y, coef=coef, zi=zi)

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Audio signal. Multi-channel is supported.

    coef : positive number
        Pre-emphasis coefficient.  Typical values of ``coef`` are between 0 and 1.

        At the limit ``coef=0``, the signal is unchanged.

        At ``coef=1``, the result is the first-order difference of the signal.

        The default (0.97) matches the pre-emphasis filter used in the HTK
        implementation of MFCCs [#]_.

        .. [#] https://htk.eng.cam.ac.uk/

    zi : number
        Initial filter state. If inverting a previous preemphasis(), the same value should be used.

        By default ``zi`` is initialized as
        ``((2 - coef) * y[0] - y[1]) / (3 - coef)``. This
        value corresponds to the transformation of the default initialization of ``zi`` in ``preemphasis()``,
        ``2*x[0] - x[1]``.

    return_zf : boolean
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray
        de-emphasized signal
    zf : number
        if ``return_zf=True``, the final filter state is also returned

    Examples
    --------
    Apply a standard pre-emphasis filter and invert it with de-emphasis

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> y_filt = librosa.effects.preemphasis(y)
    >>> y_deemph = librosa.effects.deemphasis(y_filt)
    >>> np.allclose(y, y_deemph)
    True

    See Also
    --------
    preemphasis
    """
    b = np.array([1.0, -coef], dtype=y.dtype)
    a = np.array([1.0], dtype=y.dtype)

    y_out: np.ndarray
    zf: np.ndarray
    if zi is None:
        # initialize with all zeros
        zi = np.zeros(list(y.shape[:-1]) + [1], dtype=y.dtype)
        y_out, zf = scipy.signal.lfilter(a, b, y, zi=zi)

        # factor in the linear extrapolation
        y_out -= (
            ((2 - coef) * y[..., 0:1] - y[..., 1:2])
            / (3 - coef)
            * (coef ** np.arange(y.shape[-1]))
        )

    else:
        zi = np.atleast_1d(zi)
        y_out, zf = scipy.signal.lfilter(a, b, y, zi=zi.astype(y.dtype))

    if return_zf:
        return y_out, zf
    else:
        return y_out
