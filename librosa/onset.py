#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
librosa.onset contains functions for computing onset strength envelopes
and estimating onset events from audio signals.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from . import core, util
from ._cache import cache
from .feature.spectral import melspectrogram
from .util.exceptions import (
    ParameterShapeError,
    ParameterValueError,
)

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Sequence

__all__ = ["onset_detect", "onset_strength", "onset_strength_multi", "onset_backtrack"]


def _call_with_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function with positional arguments and signature-filtered keyword arguments."""
    sig = inspect.signature(func)
    params = sig.parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return func(*args, **kwargs)


def onset_detect(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    onset_envelope: np.ndarray | None = None,
    hop_length: int = 512,
    backtrack: bool = False,
    energy: np.ndarray | None = None,
    units: str = "frames",
    normalize: bool = True,
    sparse: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Locate note onset events by picking peaks in an onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [#]_.

    .. [#] https://github.com/CPJKU/onset_db

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    onset_envelope : np.ndarray [shape=(..., m)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    backtrack : bool
        If ``True``, detected onset events are backtracked to the nearest
        preceding minimum of ``energy``.

        This is primarily useful when using onsets as slice points for segmentation.

        .. note:: backtracking is only supported if ``sparse=True``.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then ``onset_envelope`` is used.

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    normalize : bool
        If ``True`` (default), normalize the onset envelope to have minimum of 0 and
        maximum of 1 prior to detection.  This is helpful for standardizing the
        parameters of `librosa.util.peak_pick`.

        Otherwise, the onset envelope is left unnormalized.

    sparse : bool
        If ``True`` (default), detections are returned as an array of frames,
        samples, or time indices (as specified by ``units=``).

        If ``False``, detections are encoded as a dense boolean array where
        ``onsets[n]`` is True if there's an onset at frame index ``n``.

        .. note:: multi-channel input is only supported if ``sparse=False``.

    **kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.

    Returns
    -------
    onsets : np.ndarray [shape=(n_onsets,) or onset_envelope.shape]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        If `sparse=False`, `onsets[..., n]` indicates an onset
        detection at frame index `n`.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty array (sparse=True) or all-False array (sparse=False).

    Raises
    ------
    ParameterError
        if neither ``y`` nor ``onsets`` are provided

        or if ``units`` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : Compute onset strength per-frame
    onset_backtrack : Backtracking onset events
    librosa.util.peak_pick : Pick peaks from a time series

    Examples
    --------
    Get onset times from a signal

    >>> y, sr = librosa.loadx('trumpet')
    >>> librosa.onset.onset_detect(y=y, sr=sr, units='time')
    array([0.07 , 0.232, 0.395, 0.604, 0.743, 0.929, 1.045, 1.115,
           1.416, 1.672, 1.881, 2.043, 2.206, 2.368, 2.554, 3.019])

    Or use a pre-computed onset envelope

    >>> o_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(o_env, sr=sr)
    >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(D, vscale='dBFS',
    ...                          x_axis='time', y_axis='log', ax=ax[0], sr=sr)
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, o_env, label='Onset strength')
    >>> ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    ...            linestyle='--', label='Onsets')
    >>> ax[1].legend()
    """
    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterValueError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    if normalize:
        # Normalize onset strength function to [0, 1] range
        # Normalization is performed over the trailing axis
        onset_envelope = onset_envelope - np.min(onset_envelope, keepdims=True, axis=-1)

        # Max-scale with safe division
        onset_envelope /= np.max(onset_envelope, keepdims=True, axis=-1) + util.tiny(onset_envelope)

    # help out mypy
    assert onset_envelope is not None

    # Do we have any onsets to grab?
    if not onset_envelope.any() or not np.all(np.isfinite(onset_envelope)):
        if sparse:
            onsets = np.array([], dtype=int)
        else:
            onsets = np.zeros_like(onset_envelope, dtype=bool)

    else:
        # These parameter settings found by large-scale search
        kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
        kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
        kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
        kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
        kwargs.setdefault("delta", 0.07)

        # Peak pick the onset envelope
        onsets = util.peak_pick(onset_envelope, sparse=sparse, axis=-1, **kwargs)

        # Optionally backtrack the events
        if backtrack:
            if not sparse:
                raise ParameterValueError("onset backtracking is only supported if sparse=True")

            if energy is None:
                energy = onset_envelope
            assert energy is not None
            onsets = onset_backtrack(onsets, energy)

    if sparse:
        if units == "frames":
            pass
        elif units == "samples":
            onsets = core.frames_to_samples(onsets, hop_length=hop_length)
        elif units == "time":
            onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
        else:
            raise ParameterValueError(f"Invalid unit type: {units}")

    return onsets


def onset_strength(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    lag: int = 1,
    max_size: int = 1,
    ref: np.ndarray | None = None,
    detrend: bool = False,
    center: bool = True,
    feature: Callable | None = None,
    aggregate: Callable | bool | None = None,
    top_db: float | None = 80.0,
    mode: Literal["flux", "complex"] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope.

    In standard spectral flux mode (``mode="flux"``), onset strength at time ``t``
    is determined by::

        mean_f max(0, S[f, t] - ref[f, t - lag])

    where ``ref`` is ``S`` after local max filtering along the frequency
    axis [#]_.

    In complex-domain mode (``mode="complex"``), onset strength models both energy
    dynamics and phase trajectory continuity [#]_ [#]_. The complex phasor at
    frame ``t`` is predicted by extrapolating the phase advance from reference frame
    ``t - lag``, and the onset envelope measures the Euclidean residual between the
    target phasor and predicted phasor under non-decreasing magnitude.

    By default, if a time series ``y`` is provided:

    - In ``"flux"`` mode: ``S`` will be the log-power Mel spectrogram.
    - In ``"complex"`` mode: ``S`` will be the complex STFT.

    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.

    .. [#] Bello, Juan Pablo, Laurent Daudet, Samer Abdallah, Chris Duxbury, Mike Davies, and Mark B. Sandler.
           "A tutorial on onset detection in music signals."
           IEEE Transactions on Speech and Audio Processing 13, no. 5 (2005): 1035-1047.

    .. [#] Dixon, Simon.
           "Onset detection revisited."
           In Proceedings of the 9th International Conference on Digital Audio Effects (DAFx-06),
           pp. 133-137. 2006.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed spectrogram.
        Can be real-valued (magnitude/power) or complex-valued.

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.
        Must be `1` when ``mode="complex"``.

    ref : None or np.ndarray [shape=(..., d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.
        Must be `None` when ``mode="complex"``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function, optional
        Function for computing time-series features (spectrograms).
        By default:

        - In ``"flux"`` mode: uses `librosa.feature.melspectrogram` with ``fmax=sr/2``
          and log-amplitude scaling (`power_to_db`).
        - In ``"complex"`` mode: uses `librosa.core.stft` to preserve complex phase.

        If a custom callable is provided, arguments are filtered to match its signature,
        allowing functions like `librosa.stft`, `librosa.cqt`, `librosa.vqt`, or custom
        callables to be passed directly.

    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.

        Default: `np.mean`

    top_db : float or None
        Threshold the spectrogram at ``top_db`` below its peak before the
        dB conversion in ``"flux"`` mode, as in `power_to_db`. Pass ``None`` to
        disable thresholding. Ignored when ``mode="complex"``.

    mode : {"flux", "complex"} or None, default=None
        Computation mode for onset strength:

        - ``None``: infer automatically from ``S`` if provided (``"complex"`` if
          `numpy.iscomplexobj(S)` else ``"flux"``); defaults to ``"flux"`` if
          ``(y, sr)`` is provided.
        - ``"flux"``: standard spectral flux (rectified spectral difference).
        - ``"complex"``: complex-domain spectral flux.

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., m,)]
        vector containing the onset strength envelope.
        If the input contains multiple channels, then onset envelope is computed for each channel.

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

        or if ``lag`` or ``max_size`` are not positive integers

        or if ``mode`` is not one of ``None``, ``"flux"``, or ``"complex"``

        or if ``mode="complex"`` and ``max_size > 1``

        or if ``mode="complex"`` and ``ref is not None``

        or if ``mode="complex"`` and ``S`` (or ``feature(...)``) is real-valued

    See Also
    --------
    onset_detect
    onset_strength_multi

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('trumpet', duration=3)
    >>> D = np.abs(librosa.stft(y))
    >>> times = librosa.times_like(D, sr=sr)
    >>> fig, ax = plt.subplots(nrows=4, sharex=True, height_ratios=[3, 1, 1, 1])
    >>> librosa.display.specshow(D, vscale='dBFS',
    ...                          y_axis='log', x_axis='time', ax=ax[0], sr=sr)
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()

    Construct a standard onset function

    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
    ...            label='Mean (mel)', color='C0')
    >>> ax[1].label_outer()

    Compute complex-domain onset strength

    >>> onset_complex = librosa.onset.onset_strength(y=y, sr=sr, mode='complex')
    >>> ax[2].plot(times, onset_complex / onset_complex.max(), alpha=0.8,
    ...            label='Complex domain (STFT)', color='C1')
    >>> ax[2].label_outer()

    Constant-Q spectrogram instead of Mel

    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> onset_cqt = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    >>> ax[3].plot(times, onset_cqt / onset_cqt.max(), alpha=0.8,
    ...            label='Mean (CQT)', color='C2')
    >>> ax[3].label_outer()
    >>> fig.legend(loc='center right', ncols=3)
    """
    if aggregate is False:
        raise ParameterValueError(
            "aggregate parameter cannot be False when computing full-spectrum onset strength."
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        top_db=top_db,
        mode=mode,
        **kwargs,
    )

    return odf_all[..., 0, :]


def onset_backtrack(events: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Backtrack detected onset events to the nearest preceding local minimum of an energy function.

    This function can be used to roll back the timing of detected onsets
    from a detected peak amplitude to the preceding minimum.

    This is most useful when using onsets to determine slice points for
    segmentation, as described by [#]_.

    .. [#] Jehan, Tristan.
           "Creating music by listening"
           Doctoral dissertation
           Massachusetts Institute of Technology, 2005.

    Parameters
    ----------
    events : np.ndarray, dtype=int
        List of onset event frame indices, as computed by `onset_detect`
    energy : np.ndarray, shape=(m,)
        An energy function

    Returns
    -------
    events_backtracked : np.ndarray, shape=events.shape
        The input events matched to nearest preceding minima of ``energy``.

    Examples
    --------
    Backtrack the events using the onset envelope

    >>> y, sr = librosa.loadx('trumpet', duration=3)
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(oenv, sr=sr)
    >>> # Detect events without backtracking
    >>> onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
    ...                                        backtrack=False)
    >>> onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)

    Backtrack the events using the RMS values

    >>> S = np.abs(librosa.stft(y=y))
    >>> rms = librosa.feature.rms(S=S)
    >>> onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(S, vscale='dBFS',
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, oenv, label='Onset strength')
    >>> ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label='Raw onsets')
    >>> ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label='Backtracked', color='r')
    >>> ax[1].legend()
    >>> ax[1].label_outer()
    >>> ax[2].plot(times, rms[0], label='RMS')
    >>> ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label='Backtracked (RMS)', color='r')
    >>> ax[2].legend()
    """
    # Find points where energy is non-increasing
    # all points:  energy[i] <= energy[i-1]
    # tail points: energy[i] < energy[i+1]
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))

    # Pad on a 0, just in case we have onsets with no preceding minimum
    # Shift by one to account for slicing in minima detection
    minima = util.fix_frames(1 + minima, x_min=0)

    # Only match going left from the detected events
    results: np.ndarray = minima[util.match_events(events, minima, right=False)]
    return results


@cache(level=30)
def onset_strength_multi(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    lag: int = 1,
    max_size: int = 1,
    ref: np.ndarray | None = None,
    detrend: bool = False,
    center: bool = True,
    feature: Callable | None = None,
    aggregate: Callable | bool | None = None,
    channels: Sequence[int] | Sequence[slice] | None = None,
    top_db: float | None = 80.0,
    mode: Literal["flux", "complex"] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope across multiple channels.

    Onset strength for channel ``i`` at time ``t`` is determined by::

        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time-series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, m)]
        pre-computed spectrogram.
        Can be real-valued (magnitude/power) or complex-valued.

    n_fft : int > 0 [scalar]
        length of the FFT frame for use in ``feature()`` if ``S`` is not provided.

    hop_length : int > 0 [scalar]
        hop length for use in ``feature()`` if ``S`` is not provided.

    lag : int > 0
        time lag for computing differences

    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.
        Must be `1` when ``mode="complex"``.

    ref : None or np.ndarray [shape=(d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.
        Must be `None` when ``mode="complex"``.

    detrend : bool [scalar]
        Filter the onset strength to remove the DC component

    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.

    feature : function, optional
        Function for computing time-series features (spectrograms).
        By default:

        - In ``"flux"`` mode: uses `librosa.feature.melspectrogram` with ``fmax=sr/2``
          and log-amplitude scaling (`power_to_db`).
        - In ``"complex"`` mode: uses `librosa.core.stft` to preserve complex phase.

        If a custom callable is provided, arguments are filtered to match its signature,
        allowing functions like `librosa.stft`, `librosa.cqt`, `librosa.vqt`, or custom
        callables to be passed directly.

    aggregate : function or False
        Aggregation function to use when combining onsets
        at different frequency bins.

        If ``False``, then no aggregation is performed.

        Default: `np.mean`

    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.

    top_db : float or None
        Threshold the spectrogram at ``top_db`` below its peak before the
        dB conversion in ``"flux"`` mode, as in `power_to_db`. Pass ``None`` to
        disable thresholding. Ignored when ``mode="complex"``.

    mode : {"flux", "complex"} or None, default=None
        Computation mode for onset strength:

        - ``None``: infer automatically from ``S`` if provided (``"complex"`` if
          `numpy.iscomplexobj(S)` else ``"flux"``); defaults to ``"flux"`` if
          ``(y, sr)`` is provided.
        - ``"flux"``: standard spectral flux (rectified spectral difference).
        - ``"complex"``: complex-domain spectral flux.

    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.

    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., n_channels, m)]
        array containing the onset strength envelope for each specified channel

    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided

        or if ``lag`` or ``max_size`` are not positive integers

        or if ``mode`` is not one of ``None``, ``"flux"``, or ``"complex"``

        or if ``mode="complex"`` and ``max_size > 1``

        or if ``mode="complex"`` and ``ref is not None``

        or if ``mode="complex"`` and ``S`` (or ``feature(...)``) is real-valued

    See Also
    --------
    onset_strength

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    First, load some audio and plot the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('choice', duration=5)
    >>> D = np.abs(librosa.stft(y))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(D, vscale='dBFS',
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> librosa.display.colorbar_db(img1, ax=ax[0])

    Construct a standard onset function over four sub-bands

    >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    ...                                                     channels=[0, 32, 64, 96, 128])
    >>> img2 = librosa.display.specshow(onset_subbands, x_axis='time', ax=ax[1])
    >>> ax[1].set(ylabel='Sub-bands', title='Sub-band onset strength')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """
    if aggregate is None:
        aggregate = np.mean

    if not util.is_positive_int(lag):
        raise ParameterValueError(f"lag={lag} must be a positive integer")

    if not util.is_positive_int(max_size):
        raise ParameterValueError(f"max_size={max_size} must be a positive integer")

    if mode is not None and mode not in ("flux", "complex"):
        raise ParameterValueError(f"Invalid mode='{mode}'. Must be one of ['flux', 'complex']")

    # First, resolve S and mode
    if S is not None:
        if mode is None:
            mode = "complex" if np.iscomplexobj(S) else "flux"
        elif mode == "complex" and not np.iscomplexobj(S):
            raise ParameterValueError(
                "mode='complex' was specified, but input spectrogram S is real-valued"
            )
        elif mode == "flux" and np.iscomplexobj(S):
            S = np.abs(S)
    else:
        if y is None:
            raise ParameterValueError("One of y or S must be provided")

        if mode is None:
            mode = "flux"

        if mode == "flux":
            if feature is None:
                feature = melspectrogram
                kwargs.setdefault("fmax", 0.5 * sr)

            S = _call_with_kwargs(
                feature, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs
            )
            S = np.abs(S)
            S = core.power_to_db(S, top_db=top_db)
        else:  # mode == "complex"
            if feature is None:
                feature = core.stft

            S = _call_with_kwargs(
                feature, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs
            )
            if not np.iscomplexobj(S):
                raise ParameterValueError(
                    f"mode='complex' requires a complex-valued spectrogram, "
                    f"but feature={feature} returned a real-valued array"
                )

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    if mode == "complex":
        if max_size > 1:
            raise ParameterValueError(
                "max_size > 1 is not supported for complex-valued spectrograms S"
            )
        if ref is not None:
            raise ParameterValueError(
                "ref is not supported for complex-valued spectrograms S"
            )

        pad_width = lag + 1
        if S.shape[-1] <= pad_width:
            onset_env = np.zeros((*S.shape[:-1], 0), dtype=S.real.dtype)
        else:
            mag, phase = core.magphase(S)
            phi = np.angle(phase)
            dphi = np.angle(phase[..., 1:] * np.conj(phase[..., :-1]))

            ref_mag = mag[..., 1:-lag]
            pred_phase = phi[..., 1:-lag] + lag * dphi[..., :-lag]
            pred_S = ref_mag * np.exp(1j * pred_phase)

            onset_env = np.where(
                mag[..., 1 + lag:] >= ref_mag,
                np.abs(S[..., 1 + lag:] - pred_S),
                0.0,
            )
    else:
        pad_width = lag
        if ref is None:
            if max_size == 1:
                ref = S
            else:
                import scipy.ndimage

                ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
        elif ref.shape != S.shape:
            raise ParameterShapeError(
                f"Reference spectrum shape {ref.shape} must match input spectrum {S.shape}"
            )

        # Compute difference to the reference, spaced by lag
        onset_env = S[..., lag:] - ref[..., :-lag]

        # Discard negatives (decreasing amplitude)
        onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if callable(aggregate):
        onset_env = util.sync(
            onset_env, channels, aggregate=aggregate, pad=pad, axis=-2
        )

    # compensate for lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode="constant")

    # remove the DC component
    if detrend:
        import scipy.signal

        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[..., : S.shape[-1]]

    return onset_env
