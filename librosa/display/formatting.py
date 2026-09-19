#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tick formatters and axis decoration
===================================

This module provides specialized Matplotlib formatters and tick locators for
musical and time/frequency axes. It includes utilities for formatting time,
frequencies (linear and log), musical notes, chords, and chromatic scales.
"""
# mypy: disable-error-code="attr-defined"

# Standard library imports for type checking and future compatibility
from __future__ import annotations

import warnings
import weakref

# Third-party imports for plotting and numerical operations
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING, cast

import matplotlib.axes as mplaxes
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import colormaps as mcm
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerLine2D, HandlerPatch

# Core and utility imports for audio processing
from .. import core, util
from ..util.decorators import moved
from ..util.exceptions import ParameterError

# Type checking imports for development and static analysis
if TYPE_CHECKING:
    from typing import Any, Collection, Sequence

    import matplotlib
    import matplotlib.axes
    import matplotlib.figure
    import scipy.interpolate
    from matplotlib.artist import Artist
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Colormap
    from matplotlib.lines import Line2D

    from .._typing import ArrayLike, _Array1D, _FloatLike_co


# Keeps adaptors alive as long as their Axes exists, preventing GC
_WAVESHOW_ADAPTORS: weakref.WeakKeyDictionary[mplaxes.Axes, set["AdaptiveWaveplot"]] = (
    weakref.WeakKeyDictionary()
)

# Nominal center frequencies for oct3 bands
_OCT3_FREQUENCIES = np.array(
    [
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
        16000,
        20000,
        25000,
        # --- ultrasonic up to 800KHz
        31500,
        40000,
        50000,
        63000,
        80000,
        100000,
        125000,
        160000,
        200000,
        250000,
        315000,
        400000,
        500000,
        630000,
        800000,
    ]
)


class TimeFormatter(mplticker.Formatter):
    """A tick formatter for time axes.

    Automatically switches between seconds, minutes:seconds,
    or hours:minutes:seconds.

    Parameters
    ----------
    lag : bool
        If ``True``, then the time axis is interpreted in lag coordinates.
        Anything past the midpoint will be converted to negative time.

    unit : str or None
        Abbreviation of the string representation for axis labels and ticks.
        List of supported units:
        * `"h"`: hour-based format (`H:MM:SS`)
        * `"m"`: minute-based format (`M:SS`)
        * `"s"`: second-based format (`S.sss` in scientific notation)
        * `"ms"`: millisecond-based format (`s.µµµ` in scientific notation)
        * `None`: adaptive to the duration of the underlying time range: similar
        to `"h"` above 3600 seconds; to `"m"` between 60 and 3600 seconds; to
        `"s"` between 1 and 60 seconds; and to `"ms"` below 1 second.

    See Also
    --------
    matplotlib.ticker.Formatter

    Examples
    --------
    For normal time

    >>> import matplotlib.pyplot as plt
    >>> times = np.arange(30)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    >>> ax.set(xlabel='Time')
    >>> plt.show()

    Manually set the physical time unit of the x-axis to milliseconds

    >>> times = np.arange(100)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
    >>> ax.set(xlabel='Time (ms)')
    >>> plt.show()

    For lag plots

    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set(xlabel='Lag')
    >>> plt.show()
    """

    unit: str | None
    lag: bool

    def __init__(self, lag: bool = False, unit: str | None = None):
        if unit not in ["h", "m", "s", "ms", None]:
            raise ParameterError(f"Unknown time unit: {unit}")

        super().__init__()
        self.unit = unit
        self.lag = lag

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Return the time format as pos"""
        assert self.axis is not None

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ""
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = "-"
        else:
            value = x
            sign = ""

        if self.unit == "h" or ((self.unit is None) and (vmax - vmin > 3600)):
            s = "{:d}:{:02d}:{:02d}".format(
                int(value / 3600.0),
                int(np.mod(value / 60.0, 60)),
                int(np.mod(value, 60)),
            )
        elif self.unit == "m" or ((self.unit is None) and (vmax - vmin > 60)):
            s = "{:d}:{:02d}".format(int(value / 60.0), int(np.mod(value, 60)))
        elif self.unit == "s":
            s = f"{value:.3g}"
        elif self.unit is None and (vmax - vmin >= 1):
            s = f"{value:.2g}"
        elif self.unit == "ms":
            s = "{:.3g}".format(value * 1000)
        elif self.unit is None and (vmax - vmin < 1):
            s = f"{value:.3f}"

        return f"{sign:s}{s:s}"


class AdaptiveFormatterBase(mplticker.Formatter):
    """Base formatter handling 2-octave span suppression.

    Subclasses must implement `_format_tick`.

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves.
    """

    major: bool
    vmin: float | None
    vmax: float | None

    def __init__(self, major: bool = True):
        super().__init__()
        self.major = major
        self.vmin = None
        self.vmax = None

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Apply the bounds check, then delegate to subclass formatting."""
        if x <= 0:
            return ""

        assert self.axis is not None
        vmin, vmax = self.axis.get_view_interval()

        # Handle inverted axes
        self.vmin, self.vmax = (vmin, vmax) if vmin <= vmax else (vmax, vmin)

        if not self.major and self.vmax > 4 * max(1, self.vmin):
            return ""

        return self._format_tick(x, pos)

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        raise NotImplementedError


class NoteFormatter(AdaptiveFormatterBase):
    """Ticker formatter for Notes

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    key : str
        Key for determining pitch spelling.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    LogHzFormatter
    matplotlib.ticker.Formatter
    """

    octave: bool
    major: bool
    key: str
    unicode: bool

    def __init__(
        self,
        octave: bool = True,
        major: bool = True,
        key: str = "C:maj",
        unicode: bool = True,
    ):
        super().__init__(major=major)

        self.octave = octave
        self.key = key
        self.unicode = unicode

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        """Apply the formatter to position"""
        # Only use cent precision if our vspan is less than an octave
        assert self.vmax is not None and self.vmin is not None
        cents = self.vmax <= 2 * max(1, self.vmin)

        return core.hz_to_note(
            x, octave=self.octave, cents=cents, key=self.key, unicode=self.unicode
        )


class SvaraFormatter(AdaptiveFormatterBase):
    """Ticker formatter for Svara

    Parameters
    ----------
    Sa : number > 0
        Frequency (in Hz) of Sa

    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    abbr : bool
        If ``True``, use abbreviated svara names.

        If ``False``, use full svara names.

    mela : str or int
        For Carnatic svara, the index or name of the melakarta raga in question

        To use Hindustani svara, set ``mela=None``

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter
    librosa.hz_to_svara_c
    librosa.hz_to_svara_h
    """

    def __init__(
        self,
        Sa: float,
        octave: bool = True,
        major: bool = True,
        abbr: bool = False,
        mela: str | int | None = None,
        unicode: bool = True,
    ):
        if Sa is None:
            raise ParameterError(
                "Sa frequency is required for svara display formatting"
            )

        super().__init__(major=major)
        self.Sa = Sa
        self.octave = octave
        self.abbr = abbr
        self.mela = mela
        self.unicode = unicode

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        if self.mela is None:
            return core.hz_to_svara_h(
                x, Sa=self.Sa, octave=self.octave, abbr=self.abbr, unicode=self.unicode
            )
        else:
            return core.hz_to_svara_c(
                x,
                Sa=self.Sa,
                mela=self.mela,
                octave=self.octave,
                abbr=self.abbr,
                unicode=self.unicode,
            )


class FJSFormatter(AdaptiveFormatterBase):
    """Ticker formatter for Functional Just System (FJS) notation

    Parameters
    ----------
    fmin : float
        The unison frequency for this axis

    n_bins : int > 0
        The number of frequency bins.

    bins_per_octave : int > 0
        The number of bins per octave.

    intervals : str or array of float in [1, 2)
        The interval specification for the frequency axis.

        See `core.interval_frequencies` for supported values.

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    unison : str
        The unison note name.  If not provided, it will be inferred from fmin.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    NoteFormatter
    hz_to_fjs
    matplotlib.ticker.Formatter
    """

    fmin: float
    unison: str | None
    unicode: bool
    intervals: str | Collection[float]
    n_bins: int
    bins_per_octave: int
    frequencies_: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(
        self,
        *,
        fmin: float,
        n_bins: int,
        bins_per_octave: int,
        intervals: str | Collection[float],
        major: bool = True,
        unison: str | None = None,
        unicode: bool = True,
    ):
        super().__init__(major=major)
        self.fmin = fmin
        self.unison = unison
        self.unicode = unicode
        self.intervals = intervals
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.frequencies_ = core.interval_frequencies(
            n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave
        )

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        """Apply the formatter to position"""
        # Map the given frequency to the nearest JI interval
        idx = util.match_events(np.atleast_1d(x), self.frequencies_)[0]

        label: str = core.hz_to_fjs(
            self.frequencies_[idx],
            fmin=self.fmin,
            unison=self.unison,
            unicode=self.unicode,
        )
        return label


class LogHzFormatter(AdaptiveFormatterBase):
    """Ticker formatter for logarithmic frequency

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    See Also
    --------
    NoteFormatter
    matplotlib.ticker.Formatter
    """

    def __init__(self, major: bool = True):
        super().__init__(major=major)

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        """Apply the formatter to position"""
        return f"{x:g}"


class AdaptiveEngFormatter(AdaptiveFormatterBase):
    """Engineering formatter that limits tick labels to a 2-octave span.

    Parameters
    ----------
    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    **kwargs : keyword arguments
        Additional keyword arguments are passed to `matplotlib.ticker.EngFormatter`.
    """

    def __init__(self, major: bool = True, **kwargs):
        super().__init__(major=major)
        self._formatter = mplticker.EngFormatter(**kwargs)

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        # Delegate string conversion to the wrapped matplotlib formatter
        return self._formatter(x, pos)


class ChromaFormatter(mplticker.Formatter):
    """A formatter for chroma axes

    Parameters
    ----------
    key : str
        The key in which to display pitch class labels.
        See `core.midi_to_note` for supported values.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    matplotlib.ticker.Formatter
    """

    key: str
    unicode: bool

    def __init__(self, key: str = "C:maj", unicode: bool = True):
        super().__init__()
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format for chroma positions"""
        return core.midi_to_note(
            int(x), octave=False, cents=False, key=self.key, unicode=self.unicode
        )


class ChromaSvaraFormatter(mplticker.Formatter):
    """A formatter for chroma axes with svara instead of notes.

    If mela is given, Carnatic svara names will be used.
    Otherwise, Hindustani svara names will be used.
    If `Sa` is not given, it will default to 0 (equivalent to `C`).

    Parameters
    ----------
    Sa : float or None
        The MIDI note number corresponding to Sa. If ``None``, defaults to 0 (C).

    mela : str, int, or None
        For Carnatic svara, the index or name of the melakarta raga.
        If ``None``, Hindustani svara names are used.

    abbr : bool
        If ``True``, use abbreviated svara names.

        If ``False``, use full svara names.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    See Also
    --------
    ChromaFormatter
    """

    Sa: float
    mela: int | str | None
    abbr: bool
    unicode: bool

    def __init__(
        self,
        Sa: float | None = None,
        mela: int | str | None = None,
        abbr: bool = True,
        unicode: bool = True,
    ):
        super().__init__()
        if Sa is None:
            Sa = 0
        self.Sa = Sa
        self.mela = mela
        self.abbr = abbr
        self.unicode = unicode

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format for chroma positions"""
        if self.mela is not None:
            return core.midi_to_svara_c(
                int(x),
                Sa=self.Sa,
                mela=self.mela,
                octave=False,
                abbr=self.abbr,
                unicode=self.unicode,
            )
        else:
            return core.midi_to_svara_h(
                int(x), Sa=self.Sa, octave=False, abbr=self.abbr, unicode=self.unicode
            )


class ChromaFJSFormatter(mplticker.Formatter):
    """A formatter for chroma axes with functional just notation

    Parameters
    ----------
    intervals : str or array of float in [1, 2)
        The interval specification for the chroma axis.
        See `core.interval_frequencies` for supported values.

    unison : str
        The unison (tonic) note name.

    unicode : bool
        If ``True``, use unicode symbols for accidentals.

        If ``False``, use ASCII symbols for accidentals.

    bins_per_octave : int or None
        The number of bins per octave. If ``None``, inferred from ``intervals``.

    See Also
    --------
    matplotlib.ticker.Formatter
    """

    unison: str
    unicode: bool
    intervals: str | Collection[float]
    bins_per_octave: int
    intervals_: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(
        self,
        *,
        intervals: str | Collection[float],
        unison: str = "C",
        unicode: bool = True,
        bins_per_octave: int | None = None,
    ):
        super().__init__()
        self.unison = unison
        self.unicode = unicode
        self.intervals = intervals
        try:
            if not isinstance(intervals, str):
                bins_per_octave = len(intervals)
            if not isinstance(bins_per_octave, int):
                raise ParameterError(
                    f"bins_per_octave={bins_per_octave} must be integer-valued"
                )
            self.bins_per_octave = bins_per_octave
            # Construct the explicit interval set
            self.intervals_ = core.interval_frequencies(
                self.bins_per_octave,
                fmin=1,
                intervals=intervals,
                bins_per_octave=self.bins_per_octave,
            )
        except TypeError as exc:
            raise ParameterError(
                f"intervals={intervals} must be of type str or a collection of numbers between 1 and 2"
            ) from exc

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format for chroma positions"""
        lab: str = core.interval_to_fjs(
            self.intervals_[int(x) % self.bins_per_octave],
            unison=self.unison,
            unicode=self.unicode,
        )
        return lab


class TonnetzFormatter(mplticker.Formatter):
    """A formatter for tonnetz axes

    See Also
    --------
    matplotlib.ticker.Formatter
    """

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format for tonnetz positions"""
        return [r"5$_y$", r"5$_x$", r"m3$_y$", r"m3$_x$", r"M3$_y$", r"M3$_x$"][int(x)]


class AdaptiveWaveplot:
    """A helper class for managing adaptive wave visualizations.

    This object is used to dynamically switch between sample-based and envelope-based
    visualizations of waveforms.
    When the display is zoomed in such that no more than `max_samples` would be
    visible, the sample-based display is used.
    When displaying the raw samples would require more than `max_samples`, an
    envelope-based plot is used instead.

    You should never need to instantiate this object directly, as it is constructed
    automatically by `waveshow`.

    Parameters
    ----------
    times : np.ndarray
        An array containing the time index (in seconds) for each sample.

    y : np.ndarray
        An array containing the (monophonic) wave samples.

    steps : matplotlib.lines.Line2D
        The matplotlib artist used for the sample-based visualization.
        This is constructed by `matplotlib.pyplot.step`.

    envelope : matplotlib.collections.PolyCollection
        The matplotlib artist used for the envelope-based visualization.
        This is constructed by `matplotlib.pyplot.fill_between`.

    sr : number > 0
        The sampling rate of the audio

    max_samples : int > 0
        The maximum number of samples to use for sample-based display.

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

    label : str or None
        An optional label for the waveplot, used in legend entries.

    See Also
    --------
    waveshow
    """

    times: np.ndarray
    samples: np.ndarray
    sr: float
    max_samples: int
    transpose: bool
    cid: int | None
    label_proxy_: _WaveplotDecoy

    def __init__(
        self,
        times: np.ndarray,
        y: np.ndarray,
        steps: Line2D,
        envelope: PolyCollection,
        sr: float = 22050,
        max_samples: int = 11025,
        transpose: bool = False,
        label: str | None = None,
    ):
        self.times = times
        self.samples = y
        self._steps_ref = weakref.ref(steps)
        self._envelope_ref = weakref.ref(envelope)
        self.sr = sr
        self.max_samples = max_samples
        self.transpose = transpose
        self.cid = None
        self._ax_ref: weakref.ref[mplaxes.Axes] | None = None

        # This creates an invisible proxy artist to contain the label
        self.label_proxy_ = _WaveplotDecoy(self)
        self.label_proxy_.set_in_layout(False)

        if label is not None:
            self.label_proxy_.set_label(label)

    # Preserve the old attribute API by exposing properties with same names
    @property
    def steps(self) -> Line2D | None:
        """The step plot artist (Line2D), or None if garbage collected.

        Returns
        -------
        Line2D or None
            The step plot artist, or ``None`` if it has been garbage collected.
        """
        return self._steps_ref()

    @property
    def envelope(self) -> PolyCollection | None:
        """The envelope artist (PolyCollection), or None if garbage collected.

        Returns
        -------
        PolyCollection or None
            The envelope artist, or ``None`` if it has been garbage collected.
        """
        return self._envelope_ref()

    @property
    def ax(self) -> mplaxes.Axes | None:
        """The connected Axes, or None if not connected or garbage collected.

        Returns
        -------
        matplotlib.axes.Axes or None
            The connected axes, or ``None`` if not connected or garbage collected.
        """
        return None if self._ax_ref is None else self._ax_ref()

    def __del__(self) -> None:
        """Disconnect callback methods on delete"""
        self.disconnect(strict=True)

    def connect(
        self,
        ax: mplaxes.Axes,
        *,
        signal: str = "xlim_changed",
    ) -> None:
        """Connect the adaptor to a signal on an axes object.

        Note that if the adaptor has already been connected to an axes object,
        that connect is first broken and then replaced by a new callback.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to connect with this adaptor's `update`
        signal : string, {"xlim_changed", "ylim_changed"}
            The signal to connect

        See Also
        --------
        disconnect
        """
        # Disconnect any existing callback first
        self.disconnect()

        # Attach to axes and store the connection id
        self._ax_ref = weakref.ref(ax)
        ax.add_artist(self.label_proxy_)
        self.cid = ax.callbacks.connect(signal, self.update)

    def disconnect(self, *, strict: bool = False) -> None:
        """Disconnect the adaptor's update callback.

        Parameters
        ----------
        strict : bool
            If `True`, remove references to the connected axes.
            If `False` (default), only disconnect the callback.

            This functionality is intended primarily for internal use,
            and should have no observable effects for users.

        See Also
        --------
        connect
        """
        ax = self.ax
        if ax is not None and self.cid is not None:
            ax.callbacks.disconnect(self.cid)
            self.cid = None
        if strict:
            self._ax_ref = None

    def update(self, ax: mplaxes.Axes) -> None:
        """Update the matplotlib display according to the current viewport limits.

        This is a callback function, and should not be used directly.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to update
        """
        # Deref artists and bail if they've been garbage collected
        steps = self.steps
        envelope = self.envelope
        if steps is None or envelope is None:
            return

        lims = ax.viewLim

        if self.transpose:
            dim = lims.height * self.sr
            start, end = lims.y0, lims.y1
            xdata, ydata = self.samples, self.times
            data = steps.get_ydata()
        else:
            dim = lims.width * self.sr
            start, end = lims.x0, lims.x1
            xdata, ydata = self.times, self.samples
            data = steps.get_xdata()
        # Does our width cover fewer than max_samples?
        # If so, then use the sample-based plot
        if dim <= self.max_samples:
            envelope.set_visible(False)
            steps.set_visible(True)

            # Now check our viewport
            # we have to squash mypy errors on operand compatibility
            # here because the type annotations from matplotlib are too
            # loose.
            if start <= data[0] or end >= data[-1]:  # type: ignore[index]
                # Viewport expands beyond current data in steps; update
                # we want to cover a window of self.max_samples centered on the current viewport
                midpoint_time = (start + end) / 2
                idx_start = np.searchsorted(
                    self.times, midpoint_time - 0.5 * self.max_samples / self.sr
                )
                steps.set_data(
                    xdata[idx_start : idx_start + self.max_samples],
                    ydata[idx_start : idx_start + self.max_samples],
                )
        else:
            # Otherwise, use the envelope plot
            envelope.set_visible(True)
            steps.set_visible(False)

        ax.figure.canvas.draw_idle()


class _WaveplotDecoy(mlines.Line2D):
    waveplot: AdaptiveWaveplot

    def __init__(self, parent_waveplot: AdaptiveWaveplot, *args: Any, **kwargs: Any):
        # We'll never actually set the color on this decoy at construction time
        kwargs["color"] = "none"
        super().__init__([], [], *args, **kwargs)
        self.waveplot = parent_waveplot  # Store reference to the parent wrapper


class _AdaptiveWaveplotHandler(HandlerBase):
    def create_artists(self, legend: Legend,
        orig_handle: Artist,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: mtransforms.Transform
    ) -> list[Artist]:
        """
        Matplotlib automatically passes the exact dimensions and coordinate
        transform (`trans`) needed to paint safely inside the legend key box.
        """
        orig_handle = cast("_WaveplotDecoy", orig_handle)
        waveplot = orig_handle.waveplot
        ax = waveplot.ax
        if ax is not None:
            bgcolor = ax.get_facecolor()
        else:
            bgcolor = "none"
        bg_rect = mpatches.Rectangle((0, 0), 1, 1, facecolor=bgcolor, edgecolor="none")
        bg_artists = HandlerPatch().create_artists(
            legend, bg_rect, xdescent, ydescent, width, height, fontsize, trans
        )

        proxy_line = mlines.Line2D([], [])
        if waveplot.steps is not None:
            proxy_line.update_from(waveplot.steps)
        proxy_line.set_data([], [])
        proxy_line.set(visible=True)
        line_artists = HandlerLine2D().create_artists(
            legend, proxy_line, xdescent, ydescent, width, height,  fontsize, trans
        )

        return [*bg_artists, *line_artists]


# Add our custom handler to the default legend handler map
if _WaveplotDecoy not in Legend.get_default_handler_map():
    Legend.update_default_handler_map({_WaveplotDecoy: _AdaptiveWaveplotHandler()})


class Transformf0(mtransforms.Transform):
    """A utility class to handle f0-displacement for waveform visualizations.

    Parameters
    ----------
    f0 : np.ndarray
        Array of fundamental frequency values (in Hz), one per frame.
        Values may be NaN for unvoiced frames.

    sr : number > 0
        Audio sampling rate, used with ``hop_length`` to compute time stamps.

    hop_length : int > 0
        Number of audio samples between successive f0 frames.

    bins_per_octave : int > 0
        Number of bins per octave used for the pitch axis.

    norm : float
        Normalization factor applied to the pitch axis.

    offset : float
        Time offset (in seconds) applied to the frame time stamps.

    transpose : bool
        If ``True``, the time axis is the second dimension instead of the first.

    is_inverted : bool
        If ``True``, apply the inverse of the f0-displacement transformation.
    """

    f0_interp: scipy.interpolate.interp1d
    norm: float
    bins_per_octave: int
    f0: np.ndarray
    sr: float
    hop_length: int
    offset: float
    transpose: bool
    input_dims: int
    output_dims: int
    is_separable: bool
    is_inverted: bool

    def __init__(
        self,
        f0: np.ndarray,
        *,
        sr: float = 22050,
        hop_length: int = 512,
        bins_per_octave: int = 12,
        norm: float = 1,
        offset: float = 0,
        transpose: bool = False,
        is_inverted: bool = False,
    ):
        super().__init__(shorthand_name="Transformf0")

        if not np.any(np.isfinite(f0)) or np.nanmin(f0) <= 0:
            raise ParameterError("f0 must be strictly positive (or NaN) and contain at least one finite value")

        times = offset + core.times_like(f0, sr=sr, hop_length=hop_length)
        import scipy.interpolate

        self.f0_interp = scipy.interpolate.interp1d(
            times,
            f0,
            kind="previous",
            copy=False,
            bounds_error=False,
            assume_sorted=True,
        )

        self.norm = norm
        self.bins_per_octave = bins_per_octave
        self.f0 = f0
        self.sr = sr
        self.hop_length = hop_length
        self.offset = offset
        self.transpose = transpose

        self.input_dims = 2
        self.output_dims = 2
        self.is_separable = False
        self.is_inverted = is_inverted

    def transform_non_affine(self, values: ArrayLike) -> np.ndarray:
        """Apply the f0 displacement transformation to the given values.

        Parameters
        ----------
        values : np.ndarray
            An array of shape (..., 2) containing time and sample values to be
            transformed.  The order of time and sample values is determined by
            the `transpose` parameter of this class.

        Returns
        -------
        output : np.ndarray
            An array of the same shape as `values`, containing the transformed
            time and sample values.
        """
        values = np.asarray(values)

        if self.transpose:
            idx = (1, 0)
        else:
            idx = (0, 1)
        times = values[:, idx[0]]
        samples = values[:, idx[1]]

        output = np.empty_like(values)
        output[:, idx[0]] = times
        if self.is_inverted:
            output[:, idx[1]] = (
                (np.log2(samples) - np.log2(self.f0_interp(times)))
                * self.norm
                * self.bins_per_octave
            )
        else:
            output[:, idx[1]] = 2.0 ** (
                samples / self.norm / self.bins_per_octave
            ) * self.f0_interp(times)

        return output

    def inverted(self) -> Transformf0:
        """Return the inverse of this transformation.

        Returns
        -------
        Transformf0
            A new ``Transformf0`` with ``is_inverted`` toggled.
        """
        return Transformf0(
            f0=self.f0,
            sr=self.sr,
            hop_length=self.hop_length,
            bins_per_octave=self.bins_per_octave,
            norm=self.norm,
            offset=self.offset,
            transpose=self.transpose,
            is_inverted=not self.is_inverted,
        )


def infer_cmap(
    data: np.ndarray,
    *,
    robust: bool = True,
    cmap_seq: str | colors.Colormap = "magma",
    cmap_bool: str | colors.Colormap = "gray_r",
    cmap_div: str | colors.Colormap = "coolwarm",
    div_thresh: float = 0.0,
) -> Colormap:
    """Get a default colormap from the given data.

    If the data is boolean, use a black and white colormap.

    If the data has both positive and negative values,
    use a diverging colormap.

    Otherwise, use a sequential colormap.

    Parameters
    ----------
    data : np.ndarray
        Input data
    robust : bool
        If True, discard the top and bottom 2% of data when calculating
        range.
    cmap_seq : str or matplotlib.colors.Colormap
        The sequential colormap
    cmap_bool : str or matplotlib.colors.Colormap
        The boolean colormap
    cmap_div : str or matplotlib.colors.Colormap
        The diverging colormap
    div_thresh : float
        The threshold for determining whether to use a diverging colormap.
        If the data has values both above and below this threshold, then
        a diverging colormap is used.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap to use for ``data``

    See Also
    --------
    matplotlib.pyplot.colormaps
    """
    data = np.atleast_1d(data)

    if not isinstance(cmap_seq, colors.Colormap):
        cmap_seq = mcm[cmap_seq]

    if not isinstance(cmap_bool, colors.Colormap):
        cmap_bool = mcm[cmap_bool]

    if not isinstance(cmap_div, colors.Colormap):
        cmap_div = mcm[cmap_div]

    if data.dtype.kind == "b":
        return cmap_bool

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    min_val, max_val = np.percentile(data, [min_p, max_p])

    if min_val >= div_thresh or max_val <= div_thresh:
        return cmap_seq

    return cmap_div


# Deprecation rename of cmap -> infer_cmap for 1.0
cmap = moved(moved_from="librosa.display.cmap", version="1.0", version_removed="1.1")(
    infer_cmap
)




_chroma_ax_types = (
    "chroma",
    "chroma_h",
    "chroma_c",
    "chroma_fjs",
)
_cqt_ax_types = (
    "cqt_hz",
    "cqt_note",
    "cqt_svara",
    "cqt_oct3",
)
_vqt_ax_types = (
    "vqt_hz",
    "vqt_note",
    "vqt_oct3",
    "vqt_fjs",
)
_freq_ax_types = (
    "linear",
    "fft",
    "hz",
    "fft_note",
    "fft_svara",
    "oct3",
)
_time_ax_types = (
    "time",
    "h",
    "m",
    "s",
    "ms",
)
_lag_ax_types = (
    "lag",
    "lag_h",
    "lag_m",
    "lag_s",
    "lag_ms",
)
_misc_ax_types = (
    "tempo",
    "fourier_tempo",
    "mel",
    "mel_oct3",
    "log",
    "tonnetz",
    "frames",
)

_AXIS_COMPAT = set(
    [(t, t) for t in _misc_ax_types]
    + [t for t in product(_chroma_ax_types, _chroma_ax_types)]
    + [t for t in product(_cqt_ax_types, _cqt_ax_types)]
    + [t for t in product(_vqt_ax_types, _vqt_ax_types)]
    + [t for t in product(_freq_ax_types, _freq_ax_types)]
    + [t for t in product(_time_ax_types, _time_ax_types)]
    + [t for t in product(_lag_ax_types, _lag_ax_types)]
)




def _check_axes(axes: mplaxes.Axes | None) -> mplaxes.Axes:
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        axes = plt.gca()
    elif not isinstance(axes, mplaxes.Axes):
        raise ParameterError(
            "`axes` must be an instance of matplotlib.axes.Axes. "
            "Found type(axes)={}".format(type(axes))
        )
    return axes


def _scale_axes(axes, ax_type, which, tempo_min, tempo_max):
    """Set the axis scaling"""
    kwargs = dict()
    thresh = "linthresh"
    base = "base"
    scale = "linscale"

    if which == "x":
        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type in ["mel", "mel_oct3"]:
        mode = "symlog"
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type in [
        "cqt",
        "cqt_hz",
        "cqt_note",
        "cqt_svara",
        "cqt_oct3",
        "vqt_hz",
        "vqt_note",
        "vqt_fjs",
        "vqt_oct3",
    ]:
        mode = "log"
        kwargs[base] = 2

    elif ax_type in ["log", "fft_note", "fft_svara", "log_oct3"]:
        mode = "symlog"
        kwargs[base] = 2
        kwargs[thresh] = float(core.note_to_hz("C2"))
        kwargs[scale] = 0.5

    elif ax_type in ["tempo", "fourier_tempo"]:
        mode = "log"
        kwargs[base] = 2
        limit(tempo_min, tempo_max)
    else:
        return

    scaler(mode, **kwargs)


def _decorate_axis(
    axis,
    ax_type,
    key="C:maj",
    Sa=None,
    mela=None,
    thaat=None,
    unicode=True,
    fmin=None,
    unison=None,
    intervals=None,
    bins_per_octave=None,
    n_bins=None,
):
    """Configure axis tickers, locators, and labels"""
    time_units = {"h": "hours", "m": "minutes", "s": "seconds", "ms": "milliseconds"}

    if ax_type == "tonnetz":
        axis.set_major_formatter(TonnetzFormatter())
        axis.set_major_locator(mplticker.FixedLocator([0, 1, 2, 3, 4, 5]))
        axis.set_label_text("Tonnetz")

    elif ax_type == "chroma":
        axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
        degrees = core.key_to_degrees(key)
        axis.set_major_locator(
            mplticker.FixedLocator(
                cast(
                    "Sequence[float]",
                    np.add.outer(12 * np.arange(10), degrees, dtype=float).ravel(),
                )
            )
        )
        axis.set_label_text("Pitch class")

    elif ax_type == "chroma_h":
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(ChromaSvaraFormatter(Sa=Sa, unicode=unicode))
        if thaat is None:
            # If no thaat is given, show all svara
            degrees = np.arange(12)
        else:
            degrees = core.thaat_to_degrees(thaat)
        # Rotate degrees relative to Sa
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(
            mplticker.FixedLocator(
                cast(
                    "Sequence[float]",
                    np.add.outer(12 * np.arange(10), degrees, dtype=float).ravel(),
                )
            )
        )
        axis.set_label_text("Svara")

    elif ax_type == "chroma_c":
        if Sa is None:
            Sa = 0
        axis.set_major_formatter(
            ChromaSvaraFormatter(Sa=Sa, mela=mela, unicode=unicode)
        )
        degrees = core.mela_to_degrees(mela)
        # Rotate degrees relative to Sa
        degrees = np.mod(degrees + Sa, 12)
        axis.set_major_locator(
            mplticker.FixedLocator(
                cast(
                    "Sequence[float]",
                    np.add.outer(12 * np.arange(10), degrees, dtype=float).ravel(),
                )
            )
        )
        axis.set_label_text("Svara")

    elif ax_type == "chroma_fjs":
        if fmin is None:
            fmin = core.note_to_hz("C1")

        if unison is None:
            unison = core.hz_to_note(fmin, octave=False, cents=False)

        axis.set_major_formatter(
            ChromaFJSFormatter(
                intervals=intervals,
                unison=unison,
                unicode=unicode,
                bins_per_octave=bins_per_octave,
            )
        )

        if isinstance(intervals, str) and bins_per_octave > 7:
            # If intervals are implicit, generate the first 7 and identify
            # them in the sorted set
            tick_intervals = core.interval_frequencies(
                7,
                fmin=1,
                intervals=intervals,
                bins_per_octave=bins_per_octave,
                sort=False,
            )

            all_intervals = core.interval_frequencies(
                bins_per_octave,
                fmin=1,
                intervals=intervals,
                bins_per_octave=bins_per_octave,
                sort=True,
            )

            degrees = util.match_events(tick_intervals, all_intervals)
        else:
            # If intervals are explicit, tick them all
            degrees = np.arange(bins_per_octave)

        axis.set_major_locator(mplticker.FixedLocator(degrees))  # type: ignore[arg-type]
        axis.set_label_text("Pitch class")

    elif ax_type in ["tempo", "fourier_tempo"]:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_major_locator(mplticker.LogLocator(base=2.0))
        axis.set_label_text("BPM")

    elif ax_type == "time":
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Time")

    elif ax_type in time_units:
        axis.set_major_formatter(TimeFormatter(unit=ax_type, lag=False))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Time ({:s})".format(time_units[ax_type]))

    elif ax_type == "lag":
        axis.set_major_formatter(TimeFormatter(unit=None, lag=True))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Lag")

    elif isinstance(ax_type, str) and ax_type.startswith("lag_"):
        unit = ax_type[4:]
        axis.set_major_formatter(TimeFormatter(unit=unit, lag=True))
        axis.set_major_locator(
            mplticker.MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10])
        )
        axis.set_label_text("Lag ({:s})".format(time_units[unit]))

    elif ax_type == "cqt_note":
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        # Where is C1 relative to 2**k hz?
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)
            )
        )
        axis.set_label_text("Note")

    elif ax_type == "cqt_svara":
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        # Find the offset of Sa relative to 2**k Hz
        sa_offset = 2.0 ** (np.log2(Sa) - np.floor(np.log2(Sa)))

        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(sa_offset,)))
        axis.set_minor_formatter(
            SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
        )
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)
            )
        )
        axis.set_label_text("Svara")

    elif ax_type == "vqt_fjs":
        if fmin is None:
            fmin = float(core.note_to_hz("C1"))
        axis.set_major_formatter(
            FJSFormatter(
                intervals=intervals,
                fmin=fmin,
                unison=unison,
                unicode=unicode,
                bins_per_octave=bins_per_octave,
                n_bins=n_bins,
            )
        )
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))

        axis.set_minor_formatter(
            FJSFormatter(
                intervals=intervals,
                fmin=fmin,
                unison=unison,
                unicode=unicode,
                bins_per_octave=bins_per_octave,
                n_bins=n_bins,
                major=False,
            )
        )
        axis.set_minor_locator(
            mplticker.FixedLocator(
                core.interval_frequencies(
                    n_bins * 12 // bins_per_octave,
                    fmin=fmin,
                    intervals=intervals,
                    bins_per_octave=12,
                )  # type: ignore[arg-type]
            )
        )
        axis.set_label_text("Note")

    elif ax_type == "vqt_hz":
        if fmin is None:
            fmin = core.note_to_hz("C1")
        axis.set_major_formatter(LogHzFormatter())
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0,
                subs=core.interval_frequencies(
                    12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12
                ),  # type: ignore[arg-type]
            )
        )
        axis.set_label_text("Hz")

    elif ax_type == "vqt_note":
        if fmin is None:
            fmin = core.note_to_hz("C1")
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        log_fmin = np.log2(fmin)
        fmin_offset = 2.0 ** (log_fmin - np.floor(log_fmin))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(fmin_offset,)))
        axis.set_minor_formatter(NoteFormatter(key=key, unicode=unicode, major=False))
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0,
                subs=core.interval_frequencies(
                    12, fmin=fmin_offset, intervals=intervals, bins_per_octave=12
                ),  # type: ignore[arg-type]
            )
        )
        axis.set_label_text("Note")

    elif ax_type in ["cqt_hz"]:
        axis.set_major_formatter(LogHzFormatter())
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.LogLocator(base=2.0, subs=(C_offset,)))
        axis.set_major_locator(mplticker.LogLocator(base=2.0))
        axis.set_minor_formatter(LogHzFormatter(major=False))
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0, subs=C_offset * 2.0 ** (np.arange(1, 12) / 12.0)
            )
        )
        axis.set_label_text("Hz")

    elif ax_type == "fft_note":
        axis.set_major_formatter(NoteFormatter(key=key, unicode=unicode))
        # Where is C1 relative to 2**k hz?
        log_C1 = np.log2(core.note_to_hz("C1"))
        C_offset = 2.0 ** (log_C1 - np.floor(log_C1))
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        axis.set_minor_formatter(NoteFormatter(key=key, major=False, unicode=unicode))
        axis.set_minor_locator(
            mplticker.LogLocator(base=2.0, subs=2.0 ** (np.arange(1, 12) / 12.0))  # type: ignore[arg-type]
        )
        axis.set_label_text("Note")

    elif ax_type == "fft_svara":
        axis.set_major_formatter(SvaraFormatter(Sa=Sa, mela=mela, unicode=unicode))
        # Find the offset of Sa relative to 2**k Hz
        log_Sa = np.log2(Sa)
        sa_offset = 2.0 ** (log_Sa - np.floor(log_Sa))

        axis.set_major_locator(
            mplticker.SymmetricalLogLocator(
                axis.get_transform(), base=2.0, subs=[sa_offset]
            )
        )
        axis.set_minor_formatter(
            SvaraFormatter(Sa=Sa, mela=mela, major=False, unicode=unicode)
        )
        axis.set_minor_locator(
            mplticker.LogLocator(
                base=2.0, subs=sa_offset * 2.0 ** (np.arange(1, 12) / 12.0)
            )
        )
        axis.set_label_text("Svara")

    elif ax_type in ["mel", "log"]:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_major_locator(mplticker.SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text("Hz")

    elif ax_type in ["linear", "hz", "fft"]:
        axis.set_major_formatter(mplticker.ScalarFormatter())
        axis.set_label_text("Hz")

    elif ax_type in ["oct3", "cqt_oct3", "vqt_oct3", "log_oct3", "mel_oct3"]:
        # Label once per octave
        if ax_type == "mel_oct3":
            # Suppress major ticks for frequencies below 100 Hz in mel mode
            axis.set_major_locator(mplticker.FixedLocator(_OCT3_FREQUENCIES[5::3]))  # type: ignore[arg-type]
        else:
            axis.set_major_locator(mplticker.FixedLocator(_OCT3_FREQUENCIES[::3]))  # type: ignore[arg-type]
        axis.set_major_formatter(AdaptiveEngFormatter(major=True, unit="Hz"))
        axis.set_label_text("Frequency")
        # Minor ticks at the 1/3 octaves
        axis.set_minor_locator(mplticker.FixedLocator(_OCT3_FREQUENCIES, nbins=None))  # type: ignore[arg-type]
        axis.set_minor_formatter(AdaptiveEngFormatter(major=False, unit="Hz"))

    elif ax_type in ["frames"]:
        axis.set_label_text("Frames")

    elif ax_type in ["off", "none", None]:
        axis.set_label_text("")
        axis.set_ticks([])

    else:
        raise ParameterError(f"Unsupported axis type: {ax_type}")


def _coord_fft_hz(
    n: int, sr: float = 22050, n_fft: int | None = None, **_kwargs: Any
) -> _Array1D[np.float64]:
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
    return basis


def _coord_mel_hz(
    n: int,
    fmin: float | None = 0.0,
    fmax: float | None = None,
    sr: float = 22050,
    htk: bool = False,
    **_kwargs: Any,
) -> _Array1D[np.float64]:
    """Get the frequencies for Mel bins"""
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = 0.5 * sr

    basis = core.mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    return basis


def _coord_cqt_hz(
    n: int,
    fmin: _FloatLike_co | None = None,
    bins_per_octave: int = 12,
    sr: float = 22050,
    **_kwargs: Any,
) -> _Array1D[np.float64]:
    """Get CQT bin frequencies"""
    if fmin is None:
        fmin = core.note_to_hz("C1")

    # Apply tuning correction
    fmin = fmin * 2.0 ** (_kwargs.get("tuning", 0.0) / bins_per_octave)

    # we drop by half a bin so that CQT bins are centered vertically
    freqs = core.cqt_frequencies(
        n,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )

    if np.any(freqs > 0.5 * sr):
        warnings.warn(
            "Frequency axis exceeds Nyquist. "
            "Did you remember to set all spectrogram parameters in specshow?",
            stacklevel=4,
        )

    return freqs


def _coord_vqt_hz(
    n: int,
    fmin: _FloatLike_co | None = None,
    bins_per_octave: int = 12,
    sr: float = 22050,
    intervals: str | Collection[float] | None = None,
    unison: str | None = None,
    **_kwargs: Any,
) -> _Array1D[np.float64]:
    if fmin is None:
        fmin = core.note_to_hz("C1")

    if intervals is None:
        raise ParameterError("VQT axis coordinates cannot be defined without intervals")

    freqs = core.interval_frequencies(
        n, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave
    )

    if np.any(freqs > 0.5 * sr):
        warnings.warn(
            "Frequency axis exceeds Nyquist. "
            "Did you remember to set all spectrogram parameters in specshow?",
            stacklevel=4,
        )

    return freqs


def _coord_chroma(n: int, bins_per_octave: int = 12, **_kwargs: Any) -> np.ndarray:
    """Get chroma bin numbers"""
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n, endpoint=False)


def _coord_tempo(
    n: int, sr: float = 22050, hop_length: int = 512, **_kwargs: Any
) -> np.ndarray:
    """Tempo coordinates"""
    basis = core.tempo_frequencies(n + 1, sr=sr, hop_length=hop_length)[1:]
    return basis


def _coord_fourier_tempo(
    n: int,
    sr: float = 22050,
    hop_length: int = 512,
    win_length: int | None = None,
    **_kwargs: Any,
) -> np.ndarray:
    """Fourier tempogram coordinates"""
    if win_length is None:
        win_length = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fourier_tempo_frequencies(
        sr=sr, hop_length=hop_length, win_length=win_length
    )
    return basis


def _coord_n(n: int, **_kwargs: Any) -> np.ndarray:
    """Get bare positions"""
    return np.arange(n)


def _coord_time(
    n: int, sr: float = 22050, hop_length: int = 512, **_kwargs: Any
) -> np.ndarray:
    """Get time coordinates from frames"""
    times: np.ndarray = core.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return times


def _same_axes(x_axis, y_axis, xlim, ylim):
    """Check if two axes are similar, used to determine squared plots"""
    axes_compatible_and_not_none = (x_axis, y_axis) in _AXIS_COMPAT
    axes_same_lim = xlim == ylim
    return axes_compatible_and_not_none and axes_same_lim


def _radian_formatter(x, pos):
    """Format a tick value (in radians) as a rational multiple of pi"""
    m = x / np.pi
    # hard to imagine going finer than pi/16 (11°)
    frac = Fraction(m).limit_denominator(16)
    num, den = frac.numerator, frac.denominator

    if num == 0:
        return " 0"

    sign = "-" if num * den < 0 else " "
    num_abs = abs(num)

    # Build numerator string
    coeff = "" if num_abs == 1 else str(num_abs)

    if den == 1:
        return f"{sign}{coeff}π"
    else:
        return f"{sign}{coeff}π/{den}"


def colorbar_phase(
    im: matplotlib.cm.ScalarMappable,
    *,
    numticks: int = 9,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.FigureBase | None = None,
    **kwargs: Any,
) -> matplotlib.colorbar.Colorbar:
    """Attach a colorbar to an image representing phase data in radians.

    The colorbar will display ticks at rational multiples of π.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The image to which the colorbar will be attached.
        Generally this will be a `matplotlib.image.AxesImage` or `matplotlib.collections.QuadMesh`
        as returned by `specshow`.
    numticks : int > 0
        The number of ticks to display on the colorbar.
        Default is 9, corresponding to multiples of π/4.
    ax : matplotlib.axes.Axes or None
        The axes to which the colorbar will be attached.
        If None, the colorbar will be attached to the axes of `im`.
    fig : matplotlib.figure.Figure, SubFigure, or None
        The figure to which the colorbar will be attached.
        If None, the colorbar will be attached to the figure of `im`.
    **kwargs
        Additional keyword arguments to pass to `fig.colorbar`.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar object.

    See Also
    --------
    specshow
    colorbar_db
    matplotlib.colorbar.Colorbar

    Examples
    --------
    Attach a colorbar to a phase spectrogram

    >>> import matplotlib.pyplot as plt
    >>> import librosa
    >>> y, sr = librosa.loadx('trumpet')
    >>> S = librosa.stft(y)
    >>> fig, ax = plt.subplots()
    >>> im = librosa.display.specshow(S, ax=ax, y_axis='log', x_axis='time', vscale='phase')
    >>> librosa.display.colorbar_phase(im)
    >>> plt.show()

    Attach a colorbar to one subplot axes, and show as multiples of π/3.

    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> im_mag = librosa.display.specshow(S, ax=ax[0], y_axis='log', x_axis='time', vscale='dBFS')
    >>> cbar = librosa.display.colorbar_db(im_mag, ax=ax[0], label='dBFS')
    >>> im_ph = librosa.display.specshow(S, ax=ax[1], y_axis='log', x_axis='time', vscale='dphase')
    >>> cbar = librosa.display.colorbar_phase(im_ph, ax=ax[1], numticks=7)
    >>> ax[0].label_outer()
    >>> plt.show()
    """
    if fig is None:
        fig = im.figure

    if ax is None:
        ax = im.axes

    kwargs.setdefault("label", "radians")

    kwargs.setdefault("ticks", mplticker.LinearLocator(numticks=numticks))
    kwargs.setdefault("format", mplticker.FuncFormatter(_radian_formatter))

    cbar = fig.colorbar(
        im,
        ax=ax,
        **kwargs,
    )
    return cbar


def colorbar_db(
    im: matplotlib.cm.ScalarMappable,
    *,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.FigureBase | None = None,
    format: str | mplticker.Formatter = "% -3.f",
    **kwargs: Any,
) -> matplotlib.colorbar.Colorbar:
    """Attach a colorbar to an image representing decibel-scaled data.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        The image to which the colorbar will be attached.
        Generally this will be a `matplotlib.image.AxesImage` or `matplotlib.collections.QuadMesh`
        as returned by `specshow`.
    ax : matplotlib.axes.Axes or None
        The axes to which the colorbar will be attached.
        If None, the colorbar will be attached to the axes of `im`.
    fig : matplotlib.figure.Figure, SubFigure, or None
        The figure to which the colorbar will be attached.
        If None, the colorbar will be attached to the figure of `im`.
    format : str
        The format string for the colorbar ticks.
        Default is "% -3.f", which displays integer values.
        You can change this to a different format if needed.
    **kwargs
        Additional keyword arguments to pass to `fig.colorbar`.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar object.

    See Also
    --------
    specshow
    colorbar_phase
    matplotlib.colorbar.Colorbar

    Examples
    --------
    Attach a colorbar to a magnitude spectrogram

    >>> import matplotlib.pyplot as plt
    >>> import librosa
    >>> y, sr = librosa.loadx('trumpet')
    >>> S = librosa.stft(y)
    >>> fig, ax = plt.subplots()
    >>> im = librosa.display.specshow(S, ax=ax, y_axis='log', x_axis='time', vscale='dB')
    >>> librosa.display.colorbar_db(im)
    >>> plt.show()

    Attach a colorbar to one subplot axes.  We can also set a label for the colorbar.

    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> im_mag = librosa.display.specshow(S, ax=ax[0], y_axis='log', x_axis='time', vscale='dBFS')
    >>> cbar = librosa.display.colorbar_db(im_mag, ax=ax[0], label='dBFS')
    >>> im_ph = librosa.display.specshow(S, ax=ax[1], y_axis='log', x_axis='time', vscale='dphase')
    >>> cbar = librosa.display.colorbar_phase(im_ph, ax=ax[1])
    >>> ax[0].label_outer()
    >>> plt.show()
    """
    if fig is None:
        fig = im.figure

    if ax is None:
        ax = im.axes

    kwargs.setdefault("label", "dB")

    cbar = fig.colorbar(
        im,
        ax=ax,
        format=format,
        **kwargs,
    )

    return cbar
