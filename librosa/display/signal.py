#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-domain signal visualization
================================

This module contains tools for visualizing time-domain audio signals,
waveforms, and their overlays. It includes functions like `waveshow`,
`wavebars`, and `wavef0` which handle adaptive plotting based on
zoom levels and sequence lengths.
"""
# mypy: disable-error-code="attr-defined"

# Standard library imports for type checking and future compatibility
from __future__ import annotations
from typing import TYPE_CHECKING, cast

# Third-party imports for plotting and numerical operations
import matplotlib.axes as mplaxes
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

# Core and utility imports for audio processing
from .. import core, util
from ..util.exceptions import ParameterError

# Signal visualization utilities for time-domain waveforms
if TYPE_CHECKING:
    from typing import Any, Literal, Sequence

    import matplotlib
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as MplPath
    
    from .._typing import ArrayLike

# Module imports for waveform formatting and adaptive plotting
from .formatting import (
    AdaptiveWaveplot,
    Transformf0,
    _check_axes,
    _decorate_axis,
    _WAVESHOW_ADAPTORS,
)


def _envelope(x, hop):
    """Compute the max-envelope of non-overlapping frames of x at length hop

    x is assumed to be multi-channel, of shape (n_channels, n_samples).
    """
    x_frame = np.abs(util.frame(x, frame_length=hop, hop_length=hop))
    return x_frame.max(axis=1)


def waveshow(
    y: np.ndarray,
    *,
    sr: float = 22050,
    max_points: int = 11025,
    axis: str | None = "time",
    offset: float = 0.0,
    marker: str | MplPath | MarkerStyle = "",
    where: Literal["pre", "post", "mid"] = "post",
    label: str | None = None,
    transpose: bool = False,
    mask: ArrayLike | None = None,
    ax: mplaxes.Axes | None = None,
    invert: bool = False,
    invert_color: str | tuple | None = None,
    **kwargs: Any,
) -> AdaptiveWaveplot:
    """Visualize a waveform in the time domain.

    This function constructs a plot which adaptively switches between a raw
    samples-based view of the signal (`matplotlib.pyplot.step`) and an
    amplitude-envelope view of the signal (`matplotlib.pyplot.fill_between`)
    depending on the time extent of the plot's viewport.

    More specifically, when the plot spans a time interval of less than ``max_points /
    sr`` (by default, 1/2 second), the samples-based view is used, and otherwise a
    downsampled amplitude envelope is used.
    This is done to limit the complexity of the visual elements to guarantee an
    efficient, visually interpretable plot.

    When using interactive rendering (e.g., in a Jupyter notebook or IPython
    console), the plot will automatically update as the view-port is changed, either
    through widget controls or programmatic updates.

    .. note:: When visualizing stereo waveforms, the amplitude envelope will be generated
        so that the upper limits derive from the left channel, and the lower limits derive
        from the right channel, which can produce a vertically asymmetric plot.

        When zoomed in to the sample view, only the first channel will be shown.
        If you want to visualize both channels at the sample level, it is recommended to
        plot each signal independently.

        To visualize stereo waveforms as two separate signal displays, see `multiplot`.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)
        If stereo, the left channel's amplitude envelope will be used for the top of the plot,,
        and the right channel's amplitude envelope (negated) will be used for the bottom of the plot.
        If mono, the signal's envelope is mirrored across the axis.

    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)

    max_points : int > 0
        Maximum number of samples to draw.  When the plot covers a time extent
        smaller than ``max_points / sr`` (default: 1/2 second), samples are drawn.

        If drawing raw samples would exceed `max_points`, then a downsampled
        amplitude envelope extracted from non-overlapping windows of `y` is
        visualized instead.  The parameters of the amplitude envelope are defined so
        that the resulting plot cannot produce more than `max_points` frames.

    axis : str or None
        Display style of the axis ticks and tick markers. Accepted values are:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
                    Values are plotted in units of seconds.

        - 'h' : markers are shown as hours, minutes, and seconds.

        - 'm' : markers are shown as minutes and seconds.

        - 's' : markers are shown as seconds.

        - 'ms' : markers are shown as milliseconds.

        - 'lag' : like time, but past the halfway point counts as negative values.

        - 'lag_h' : same as lag, but in hours.

        - 'lag_m' : same as lag, but in minutes.

        - 'lag_s' : same as lag, but in seconds.

        - 'lag_ms' : same as lag, but in milliseconds.

        - `None`, 'none', or 'off': ticks and tick markers are hidden.

    offset : float
        Offset (in seconds) to start the waveform plot

    marker : str
        Marker symbol to use for sample values. (default: no markers)

        See Also: `matplotlib.markers`.

    where : {'pre', 'mid', 'post'}
        This setting determines how both waveform and envelope plots interpolate
        between observations.

        See `matplotlib.pyplot.step` for details.

        Default: 'post'

    label : str or None
        The label string applied to this plot.
        Note that the label

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

    mask : np.ndarray [shape=(n,)] or None
        If provided, this mask will be used to determine which samples to display.
        The mask should be a 1D boolean array of the same length as `y` (`y.shape[-1]`),
        where `True` indicates that the sample should be displayed, and `False` indicates
        that it should be ignored.

        .. note:: This mask is only used directly by the envelope display, and a raw sample
            display will not be masked.  The `mask` parameter is intended to be used by the
            `wavef0` function, and it is not recommended to be used directly by the user.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    invert : bool
        If `True`, invert the foreground and background of the display, so that the axes background
        is colored.
        If `False` (default), the waveform display is colored and the background is unchanged.

        .. note:: This option should only be used if the wave display is the only element in the axes.

    invert_color : str, tuple, None
        If `invert` is `True`, this parameter specifies the color to use for the inverted
        waveform display.
        If `None` (default), the color is set to the current axes background color.

    **kwargs
        Additional keyword arguments to `matplotlib.pyplot.fill_between` and
        `matplotlib.pyplot.step`.

        Note that only those arguments which are common to both functions will be
        supported.

    Returns
    -------
    librosa.display.AdaptiveWaveplot
        An object of type `librosa.display.AdaptiveWaveplot`

    See Also
    --------
    wavebars
    AdaptiveWaveplot
    multiplot
    matplotlib.pyplot.step
    matplotlib.pyplot.fill_between
    matplotlib.pyplot.fill_betweenx
    matplotlib.markers

    Examples
    --------
    Plot a monophonic waveform with an envelope view

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('choice', duration=10)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[0])
    >>> ax[0].set(title='Envelope view, mono')
    >>> ax[0].label_outer()

    Or a stereo waveform

    >>> y, sr = librosa.loadx('choice', mono=False, duration=10)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[1])
    >>> ax[1].set(title='Envelope view, stereo')
    >>> ax[1].label_outer()

    Or harmonic and percussive components with transparency

    >>> y, sr = librosa.loadx('choice', duration=10)
    >>> y_harm, y_perc = librosa.effects.hpss(y)
    >>> librosa.display.waveshow(y_harm, sr=sr, color='C1', alpha=0.75, ax=ax[2], label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='C2', alpha=0.75, ax=ax[2], label='Percussive')
    >>> ax[2].set(title='Multiple waveforms')
    >>> ax[2].legend()
    >>> plt.show()

    Zooming in on a plot to show raw sample values

    >>> fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    >>> ax.set(xlim=[6.1, 6.25], title='Sample view')
    >>> librosa.display.waveshow(y, sr=sr, ax=ax, label='Full signal')
    >>> librosa.display.waveshow(y_harm, sr=sr, color='C1', alpha=0.75, ax=ax2, label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='C2', alpha=0.75, ax=ax2, label='Percussive')
    >>> ax.label_outer()
    >>> ax.legend()
    >>> ax2.legend(ncols=2)
    >>> plt.show()

    Plotting a transposed wave along with a self-similarity matrix

    >>> fig, ax = plt.subplot_mosaic("hSSS;hSSS;hSSS;.vvv", layout='compressed')
    >>> y, sr = librosa.loadx('trumpet')
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> sim = librosa.segment.recurrence_matrix(chroma, mode='affinity')
    >>> librosa.display.specshow(sim, ax=ax['S'], sr=sr,
    ...                          x_axis='time', y_axis='time',
    ...                          auto_aspect=False)
    >>> ax['S'].label_outer()
    >>> ax['S'].sharex(ax['v'])
    >>> ax['S'].sharey(ax['h'])
    >>> ax['S'].set(title='Self-similarity')
    >>> librosa.display.waveshow(y, ax=ax['v'])
    >>> ax['v'].label_outer()
    >>> ax['v'].set(title='transpose=False')
    >>> librosa.display.waveshow(y, ax=ax['h'], transpose=True)
    >>> ax['h'].label_outer()
    >>> ax['h'].set(title='transpose=True')
    >>> plt.show()
    """
    util.valid_audio(y)

    # Pad an extra channel dimension, if necessary
    if y.ndim == 1:
        y = y[np.newaxis, :]

    if max_points <= 0:
        raise ParameterError(f"max_points={max_points} must be strictly positive")

    # Create the adaptive drawing object
    axes = _check_axes(ax)

    # Reduce by envelope calculation
    # this choice of hop ensures that the envelope has at most max_points values
    hop_length = max(1, y.shape[-1] // max_points)
    y_env = _envelope(y, hop_length)

    # Split the envelope into top and bottom
    y_bottom, y_top = -y_env[-1], y_env[0]

    times = offset + core.times_like(y, sr=sr, hop_length=1)

    # Only plot up to max_points worth of data here
    xdata, ydata = times[:max_points], y[0, :max_points]
    dec_axis: matplotlib.axis.Axis
    if transpose:
        ydata, xdata = xdata, ydata
        filler = axes.fill_betweenx
        signal = "ylim_changed"
        dec_axis = axes.yaxis
    else:
        filler = axes.fill_between
        signal = "xlim_changed"
        dec_axis = axes.xaxis

    if mask is not None:
        mask = cast(
            "Sequence[bool]",
            np.asarray(mask, dtype=bool)[: len(y_top) * hop_length : hop_length]
        )

    (steps,) = axes.step(xdata, ydata, marker=marker, where=where, **kwargs)

    # Pull color property from the steps object, if we don't already have it
    if "color" not in kwargs:
        kwargs.setdefault("color", steps.get_color())

    envelope = filler(
        times[: len(y_top) * hop_length : hop_length],
        y_bottom,
        y_top,
        step=where,
        where=mask,
        **kwargs,
    )
    adaptor = AdaptiveWaveplot(
        times,
        y[0],
        steps,
        envelope,
        sr=sr,
        max_samples=max_points,
        transpose=transpose,
        label=label,
    )

    # Register adaptor to keep it alive as long as Axes exists
    bucket = _WAVESHOW_ADAPTORS.get(axes)
    if bucket is None:
        bucket = set()
        _WAVESHOW_ADAPTORS[axes] = bucket
    bucket.add(adaptor)

    adaptor.connect(axes, signal=signal)

    # Force an initial update to ensure the state is consistent
    adaptor.update(axes)

    # Handle color inversion if needed
    if invert:
        # If no inverted color is given, just swap it from the axes face
        if invert_color is None:
            invert_color = axes.patch.get_facecolor()

        # Get the fg color from the steps plot
        color = steps.get_color()

        # Set the axes facecolor to our wave color
        axes.patch.set_facecolor(color)
        steps.set_color(invert_color)
        envelope.set_color(invert_color)

    # Construct tickers and locators
    _decorate_axis(dec_axis, axis)

    return adaptor


def wavebars(
    y: np.ndarray,
    *,
    sr: float = 22050,
    n_bars: int = 100,
    gap_ratio: float = 0.4,
    rounding_ratio: float = 0.5,
    axis: str | None = "time",
    offset: float = 0.0,
    invert: bool = False,
    invert_color: str | tuple | None = None,
    transpose: bool = False,
    label: str | None = None,
    ax: mplaxes.Axes | None = None,
    **patch_kwargs: Any,
) -> mcollections.PatchCollection:
    """Visualize a waveform as a series of bars representing the amplitude envelope.

    This visualization is appropriate for displaying a simplified view of the
    signal, and is best suited for small figures where simplicity is desired.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)
        If stereo, the left channel's amplitude envelope will be used for the top of the bars,
        and the right channel's amplitude envelope (negated) will be used for the bottom of the bars.
        If mono, the signal's envelope is mirrored across the axis.
    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)
    n_bars : int > 0
        Number of bars to display in the waveform plot.
        The total time extent of the plot will be divided into `n_bars` segments,
        and the amplitude envelope of each segment will be represented as a bar.
    gap_ratio : float in [0, 1]
        The fraction of the bar width that will be left as a gap between adjacent bars.
    rounding_ratio : float in [0, 1]
        The fraction of the bar width that will be used for rounding the corners of the bars.
        A value of 0.5 will produce bars with rounded corners, while a value of 0 will produce
        rectangular bars.
    axis : str or None
        Display style of the axis ticks and tick markers. Accepted values are:
            - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
            - 'h' : markers are shown as hours, minutes, and seconds.
            - 'm' : markers are shown as minutes and seconds.
            - 's' : markers are shown as seconds.
            - 'ms' : markers are shown as milliseconds.
            - 'lag' : like time, but past the halfway point counts as negative values.
            - 'lag_h' : same as lag, but in hours.
            - 'lag_m' : same as lag, but in minutes.
            - 'lag_s' : same as lag, but in seconds.
            - 'lag_ms' : same as lag, but in milliseconds.
            - `None`, 'none', or 'off': ticks and tick markers are hidden.
    offset : float
        Offset (in seconds) to start the waveform plot.
    invert : bool
        If `True`, invert the foreground and background of the display, so that the axes background
        is colored.
        If `False` (default), the envelope display is colored and the background is unchanged.
    invert_color : str, tuple, None
        If `invert` is `True`, this parameter specifies the color to use for the inverted
        waveform display.
        If `None` (default), the color is set to the current axes background color.
    transpose : bool
        If `True`, display the wave vertically instead of horizontally.
    label : str or None
        The label string applied to this plot.
        If `None`, no label is applied.
    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.
    **patch_kwargs : dict
        Additional keyword arguments to pass to `matplotlib.patches.FancyBboxPatch`

    Returns
    -------
    matplotlib.collections.PatchCollection
        A collection of patches representing the amplitude envelope of the waveform.

    See Also
    --------
    waveshow

    Examples
    --------
    Plot a waveform as bars, compared to the `waveshow` version of the plot

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('libri1', duration=10)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.waveshow(y=y, sr=sr, ax=ax[0], label='waveshow()')
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.wavebars(y=y, sr=sr, ax=ax[1], label='wavebars()')
    >>> ax[1].legend()
    >>> plt.show()

    Make plots with varying amounts of detail, squared corners, and inverted colors.

    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.wavebars(y=y, sr=sr, n_bars=100, rounding_ratio=0,
    ...                          invert=True, ax=ax[0], label='100 bars')
    >>> librosa.display.wavebars(y=y, sr=sr, n_bars=200, rounding_ratio=0,
    ...                          color='C1', invert=True, ax=ax[1], label='200 bars')
    >>> librosa.display.wavebars(y=y, sr=sr, n_bars=50, rounding_ratio=0,
    ...                          color='C2', invert=True, ax=ax[2], label='50 bars')
    >>> ax[0].legend()
    >>> ax[1].legend()
    >>> ax[2].legend()
    >>> ax[0].label_outer()
    >>> ax[1].label_outer()
    >>> plt.show()
    """
    util.valid_audio(y)

    if y.ndim == 1:
        y = y[np.newaxis, :]

    patch_kwargs.setdefault("linewidth", 0)

    axes = _check_axes(ax)

    hop = max(1, y.shape[-1] // n_bars)
    env = _envelope(y, hop)
    env_bottom, env_top = env[-1], env[0]

    bar_width = (hop / sr) * (1 - gap_ratio)
    rounding_size = bar_width * rounding_ratio

    times = offset + core.times_like(env, sr=sr, hop_length=hop)

    patches = []
    boxstyle = f"round,pad=0,rounding_size={rounding_size}"
    for t, a0, a1 in zip(times, env_bottom, env_top, strict=True):
        base = min(-rounding_size, -a0)
        top = max(rounding_size, a1)
        if transpose:
            xy, width, height = (base, t), top - base, bar_width
        else:
            xy, width, height = (t, base), bar_width, top - base

        p = mpatches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=boxstyle,
        )
        patches.append(p)

    patch_kwargs.setdefault("transform", axes.transData)
    coll = mcollections.PatchCollection(patches, **patch_kwargs)
    axes.add_collection(coll)

    # Create a proxy artist if we have a label to set
    # Even if we don't have a label, we'll still need it for handling inversion later on
    proxy = mpatches.FancyBboxPatch(
        (np.nan, np.nan), 1, 1, boxstyle=boxstyle, label=label, **patch_kwargs
    )
    proxy.set_in_layout(False)
    if label is not None:
        axes.add_patch(proxy)

    axes.autoscale_view()

    if invert:
        # If no inverted color is given, just swap it from the axes face
        if invert_color is None:
            invert_color = axes.patch.get_facecolor()

        # Get the fg color from the steps plot
        color = coll.get_facecolor()

        # Set the axes facecolor to our wave color
        axes.patch.set_facecolor(color)  # type: ignore[arg-type]
        proxy.set_facecolor(color)  # type: ignore[arg-type]
        coll.set_facecolor(invert_color)

    if transpose:
        _decorate_axis(axes.yaxis, axis)
    else:
        _decorate_axis(axes.xaxis, axis)

    return coll


def wavef0(
    y: np.ndarray,
    *,
    f0: np.ndarray,
    sr: float = 22050,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    time_axis: str = "time",
    freq_axis: str = "cqt_note",
    offset: float = 0.0,
    key: str = "C:maj",
    Sa: float | None = None,
    mela: str | int | None = None,
    thaat: str | None = None,
    unicode: bool = True,
    ax: mplaxes.Axes | None = None,
    method: str = "waveshow",
    transpose: bool = False,
    **kwargs: Any,
) -> AdaptiveWaveplot | mcollections.PatchCollection:
    """Visualize a waveform with an f0-displacement.

    This can be used to simultaneously visualize the fundamental frequency (f0)
    estimates and the waveform or amplitude envelope of an audio signal in one
    compact display.

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)
        If stereo, the left channel's amplitude envelope will be used for the top of
        the plot,
        and the right channel's amplitude envelope (negated) will be used for the
        bottom of the plot.
        If mono, the signal's envelope is mirrored across the axis.

    f0 : np.ndarray [shape=(m,)]
        Fundamental frequency (f0) estimates in Hz.
        This should be computed using a pitch estimation algorithm such as
        `librosa.pyin` or `librosa.yin`.

    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)

    hop_length : int > 0
        Hop length (in samples) between successive f0 estimates.
        This should match the hop length used to compute `f0`.

    bins_per_octave : int > 0
        Number of frequency bins per octave used to scale the waveform's
        amplitude displacement around f0.  Combined with the waveform's peak
        amplitude (used as the displacement norm), this controls how many bins
        of vertical displacement correspond to one octave above or below f0.

    time_axis : str
        Display style of the time axis ticks and tick markers.
        Accepted values are:

          - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
          - 'h' : markers are shown as hours, minutes, and seconds.
          - 'm' : markers are shown as minutes and seconds.
          - 's' : markers are shown as seconds.
          - 'ms' : markers are shown as milliseconds.
          - 'lag' : like time, but past the halfway point counts as negative values.
          - 'lag_h' : same as lag, but in hours.
          - 'lag_m' : same as lag, but in minutes.
          - 'lag_s' : same as lag, but in seconds.
          - 'lag_ms' : same as lag, but in milliseconds.
          - `None`, 'none', or 'off': ticks and tick markers are hidden.

    freq_axis : str
        Display style of the frequency axis ticks and tick markers.
        Accepted values are:

          - 'cqt_note' : markers are shown as note names.
          - 'cqt_hz' : markers are shown as frequencies in Hz.
          - 'cqt_oct3' : markers are shown in Hz using 1/3-octave intervals.
          - 'cqt_svara' : markers are shown as Indian classical music svara names.

    offset : float
        Offset (in seconds) to start the waveform plot.

    key : str
        Key signature for the frequency axis.
        This is used to determine the note names for the frequency axis when using
        `cqt_note` mode.

    Sa : float or None
        Sa (tonic) frequency in Hz for the frequency axis.
        Required for `cqt_svara` mode.

    mela : str or int or None
        Mela (scale) name or index for the frequency axis.
        This is used to determine the svara names for the frequency axis when using
        `cqt_svara` mode.

    thaat : str or None
        Thaat (scale) name for the frequency axis.
        This is used to determine the svara names for the frequency axis when using
        `cqt_svara` mode.

    unicode : bool
        If `True`, use Unicode characters for frequency axis labels.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    method : str
        Method to use for visualizing the waveform with f0 displacement.
        Accepted values are:

          - 'waveshow' : Use `librosa.display.waveshow` to visualize the waveform with an f0 displacement.
          - 'wavebars' : Use `librosa.display.wavebars` to visualize the waveform as bars with an f0 displacement.

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

    **kwargs : dict
        Additional keyword arguments forwarded to the plotting function selected
        by `method`.

        If `method='waveshow'`, these must be keyword arguments supported by
        `librosa.display.waveshow` (for example, `max_points`).

        If `method='wavebars'`, these must be keyword arguments supported by
        `librosa.display.wavebars` (for example, `n_bars`, `gap_ratio`,
        `rounding_ratio`, `invert`, and `invert_color`).

        Keyword arguments for one method are not valid when using the other.

    Returns
    -------
    AdaptiveWaveplot or PatchCollection
        An object of type `librosa.display.AdaptiveWaveplot` if `method='waveshow'`,
        or a `matplotlib.collections.PatchCollection` if `method='wavebars'`.

    See Also
    --------
    waveshow
    wavebars

    Examples
    --------
    Visualize a waveform with an f0 displacement using `waveshow`

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('trumpet')
    >>> f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
    ...                         fmax=librosa.note_to_hz('C7'),
    ...                         sr=sr, hop_length=512)
    >>> fig, ax = plt.subplots()
    >>> librosa.display.wavef0(y=y, f0=f0, sr=sr, ax=ax,
    ...                        method='waveshow')
    >>> ax.set(title='Waveform with f0 displacement (waveshow)')
    >>> plt.show()

    Visualize a waveform with an f0 displacement using `wavebars`, and Hz
    labels instead of note names.
    Using a larger number of bars shows more detail here.

    >>> fig, ax = plt.subplots()
    >>> librosa.display.wavef0(y=y, f0=f0, sr=sr, ax=ax, n_bars=256,
    ...                        method='wavebars', freq_axis='cqt_hz')
    >>> ax.set(title='Waveform with f0 displacement (wavebars, cqt_hz)')
    >>> plt.show()

    Overlay a displaced waveform on a CQT plot via `specshow`.

    >>> fig, ax = plt.subplots()
    >>> C = librosa.cqt(y, sr=sr)
    >>> librosa.display.specshow(C, ax=ax, sr=sr, x_axis='time', y_axis='cqt_note',
    ...                          vscale='dBFS', cmap='gray_r')
    >>> hl = librosa.display.highlight(ax=ax)
    >>> librosa.display.wavef0(y=y, f0=f0, sr=sr, ax=ax, path_effects=hl)
    """
    # Create the adaptive drawing object
    axes = _check_axes(ax)

    if method not in ("waveshow", "wavebars"):
        raise ParameterError(f"Invalid display method={method}.")

    # Force norm to be strictly positive and handle empty arrays
    norm = float(util.tiny(y))
    if y.size > 0:
        norm += max(y.max(), -y.min())

    trans = Transformf0(
        f0,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        norm=norm,
        offset=offset,
        transpose=transpose,
    )

    # and transposed mode here
    if transpose:
        _decorate_axis(
            axes.xaxis,
            freq_axis,
            key=key,
            Sa=Sa,
            mela=mela,
            thaat=thaat,
            unicode=unicode,
        )
    else:
        _decorate_axis(
            axes.yaxis,
            freq_axis,
            key=key,
            Sa=Sa,
            mela=mela,
            thaat=thaat,
            unicode=unicode,
        )

    if method == "waveshow":
        times = offset + np.arange(y.shape[-1]) / sr
        mask = np.isfinite(trans.f0_interp(times))

        adaptor = waveshow(
            y=y,
            sr=sr,
            axis=time_axis,
            offset=offset,
            mask=mask,
            ax=axes,
            transform=trans + axes.transData,
            transpose=transpose,
            **kwargs,
        )

        # Kludge the data limits because the fill_between collection does not automatically
        # update the data limits
        assert adaptor.envelope is not None
        xy = adaptor.envelope.get_datalim(trans + axes.transData).get_points()

        f0min = np.nanmin(f0)
        f0max = np.nanmax(f0)

        if transpose:
            handle = mlines.Line2D([xy[0, 0] + f0min, xy[1, 0] + f0max], xy[:, 1])
        else:
            handle = mlines.Line2D(xy[:, 0], [xy[0, 1] + f0min, xy[1, 1] + f0max])

        axes.add_line(handle)
        axes.autoscale_view()
        handle.remove()
        # end kludge
        return adaptor

    else:
        return wavebars(
            y=y,
            sr=sr,
            axis=time_axis,
            offset=offset,
            ax=axes,
            transform=trans + axes.transData,
            transpose=transpose,
            **kwargs,
        )
