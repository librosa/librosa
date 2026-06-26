#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Display
=======

Data visualization
------------------
.. autosummary::
    :toctree: generated/

    specshow
    waveshow
    wavebars
    wavef0
    multiplot

    colorbar_db
    colorbar_phase
    highlight
    legend_for_axes

Axis formatting
---------------
.. autosummary::
    :toctree: generated/

    TimeFormatter
    NoteFormatter
    SvaraFormatter
    FJSFormatter
    LogHzFormatter
    ChromaFormatter
    ChromaSvaraFormatter
    ChromaFJSFormatter
    TonnetzFormatter

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    infer_cmap
    AdaptiveWaveplot
    Transformf0

"""

from __future__ import annotations

import colorsys
import copy
import re
import warnings
import weakref
from fractions import Fraction
from itertools import cycle, product
from typing import TYPE_CHECKING, cast

import matplotlib.axes as mplaxes
import matplotlib.cm as cm
import matplotlib.collections as mcollections
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import colormaps as mcm
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerLine2D, HandlerPatch

from . import core, util
from .util.decorators import moved
from .util.exceptions import ParameterError

if TYPE_CHECKING:
    from typing import Any, Callable, Collection, Literal, Sequence

    import cycler
    import matplotlib
    import matplotlib.axes
    import matplotlib.figure
    import numpy.typing as npt
    import scipy.interpolate
    from matplotlib.artist import Artist
    from matplotlib.collections import PolyCollection, QuadMesh
    from matplotlib.colors import Colormap
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as MplPath
    from matplotlib.typing import ColorType

    from ._typing import ArrayLike, _Array1D, _FloatLike_co


__all__ = [
    "specshow",
    "waveshow",
    "wavebars",
    "wavef0",
    "multiplot",
    "highlight",
    "infer_cmap",
    "colorbar_db",
    "colorbar_phase",
    "legend_for_axes",
    "TimeFormatter",
    "NoteFormatter",
    "FJSFormatter",
    "LogHzFormatter",
    "ChromaFormatter",
    "ChromaSvaraFormatter",
    "ChromaFJSFormatter",
    "TonnetzFormatter",
    "AdaptiveWaveplot",
    "Transformf0",
]

# mypy: disable-error-code="attr-defined"

# Keeps adaptors alive as long as their Axes exists, preventing GC
_WAVESHOW_ADAPTORS: weakref.WeakKeyDictionary[mplaxes.Axes, set["AdaptiveWaveplot"]] = (
    weakref.WeakKeyDictionary()
)

# Nominal center frequencies for oct3 bands
__OCT3_FREQUENCIES = np.array(
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
    """

    major: bool

    def __init__(self, major: bool = True):
        super().__init__()
        self.major = major

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Apply the bounds check, then delegate to subclass formatting."""
        if x <= 0:
            return ""

        assert self.axis is not None
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
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
        assert self.axis is not None
        vmin, vmax = self.axis.get_view_interval()

        cents = vmax <= 2 * max(1, vmin)

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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.SvaraFormatter(261))
    >>> ax[1].set(ylabel='Note')
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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
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

    def __call__(self, x: float, pos: int | None = None) -> str:
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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = librosa.midi_to_hz(np.arange(48, 72))
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].bar(np.arange(len(values)), values)
    >>> ax[0].yaxis.set_major_formatter(librosa.display.LogHzFormatter())
    >>> ax[0].set(ylabel='Hz')
    >>> ax[1].bar(np.arange(len(values)), values)
    >>> ax[1].yaxis.set_major_formatter(librosa.display.NoteFormatter())
    >>> ax[1].set(ylabel='Note')
    """

    def __init__(self, major: bool = True):
        super().__init__(major=major)

    def _format_tick(self, x: float, pos: int | None = None) -> str:
        """Apply the formatter to position"""
        return f"{x:g}"


class AdaptiveEngFormatter(AdaptiveFormatterBase):
    """Engineering formatter wrapped with adaptive visibility."""

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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFormatter())
    >>> ax.set(ylabel='Pitch class')
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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(12)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.ChromaFJSFormatter(intervals="ji5", bins_per_octave=12))
    >>> ax.set(ylabel='Pitch class')
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

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.arange(6)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(values)
    >>> ax.yaxis.set_major_formatter(librosa.display.TonnetzFormatter())
    >>> ax.set(ylabel='Tonnetz')
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
            if start <= data[0] or end >= data[-1]:  # type: ignore[operator,index]
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


def __envelope(x, hop):
    """Compute the max-envelope of non-overlapping frames of x at length hop

    x is assumed to be multi-channel, of shape (n_channels, n_samples).
    """
    x_frame = np.abs(util.frame(x, frame_length=hop, hop_length=hop))
    return x_frame.max(axis=1)


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


def specshow(
    data: np.ndarray,
    *,
    x_coords: np.ndarray | None = None,
    y_coords: np.ndarray | None = None,
    x_axis: str | None = None,
    y_axis: str | None = None,
    vscale: str | None = None,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: int | None = None,
    win_length: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
    tempo_min: float | None = 16,
    tempo_max: float | None = 480,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
    key: str = "C:maj",
    Sa: float | None = None,
    mela: str | int | None = None,
    thaat: str | None = None,
    auto_aspect: bool = True,
    htk: bool = False,
    unicode: bool = True,
    intervals: str | np.ndarray | None = None,
    unison: str | None = None,
    top_db: float | None = 80.0,
    cmap_seq: str | colors.Colormap = "magma",
    cmap_bool: str | colors.Colormap = "gray_r",
    cmap_div: str | colors.Colormap = "coolwarm",
    cmap_cyclic: str | colors.Colormap = "twilight_shifted",
    div_thresh: float = 0.0,
    ax: mplaxes.Axes | None = None,
    **kwargs: Any,
) -> QuadMesh:
    """Display a spectrogram/chromagram/cqt/etc.

    For a detailed overview of this function, see :ref:`sphx_glr_auto_examples_plot_display.py`

    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

    x_coords, y_coords : np.ndarray [shape=data.shape[0 or 1]]
        Optional positioning coordinates of the input data.
        These can be use to explicitly set the location of each
        element ``data[i, j]``, e.g., for displaying beat-synchronous
        features in natural time coordinates.

        If not provided, they are inferred from ``x_axis`` and ``y_axis``.

    x_axis, y_axis : None or str
        Range for the x- and y-axes.

        Valid types are:

        - None, 'none', or 'off' : no axis decoration is displayed.

        Frequency types:

        - 'linear', 'fft', 'hz' : frequency range is determined by
          the FFT window and sampling rate.
        - 'log' : the spectrum is displayed on a log scale.
        - 'oct3' : the spectrum is displayed on a log scale with frequencies marked
          in scientific notation at 1/3-octave intervals
        - 'fft_note': the spectrum is displayed on a log scale with pitches marked.
        - 'fft_svara': the spectrum is displayed on a log scale with svara marked.
        - 'mel' : frequencies are determined by the mel scale.
        - 'mel_oct3' : like 'oct3' above, but using the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_oct3' : like 'oct3' above, but using the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.
        - 'cqt_svara' : like `cqt_note` but using Hindustani or Carnatic svara
        - 'vqt_hz' : like `cqt_hz` but using Variable-Q Transform (VQT) scale.
        - 'vqt_oct3' : like 'oct3' above, but using the VQT scale.
        - 'vqt_fjs' : like `cqt_note` but using Functional Just System (FJS)
          notation.  This requires a just intonation-based variable-Q
          transform representation.
        - 'vqt_note' : like 'cqt_note' but using the VQT scale.

        All frequency types are plotted in units of Hz.

        `oct3`-type use SI prefixes for frequencies, e.g., `1 kHz`, `2 MHz`, and are
        well adapted for scientific applications using high-frequency data.

        .. note::
            The 'log', 'fft_note', 'fft_svara', 'log_oct3', 'mel', and
            'mel_oct3' axes use symmetric-log scaling to retain frequency
            bins near 0 Hz.  CQT and VQT axes use logarithmic scaling.

        Any spectrogram parameters (hop_length, sr, bins_per_octave, etc.)
        used to generate the input data should also be provided when
        calling `specshow`.

        Categorical types:

        - 'chroma' : pitches are determined by the chroma filters.
          Pitch classes are arranged at integer locations (0-11) according to
          a given key.

        - `chroma_h`, `chroma_c`: pitches are determined by chroma filters,
          and labeled as svara in the Hindustani (`chroma_h`) or Carnatic (`chroma_c`)
          according to a given thaat (Hindustani) or melakarta raga (Carnatic).

        - 'chroma_fjs': pitches are determined by chroma filters using just
          intonation.  All pitch classes are annotated.

        - 'tonnetz' : axes are labeled by Tonnetz dimensions (0-5)
        - 'frames' : markers are shown as frame counts.

        Time types:

        - 'time' : markers are shown as milliseconds, seconds, minutes, or hours.
            Values are plotted in units of seconds.

        - 'h' : markers are shown as hours, minutes, and seconds.

        - 'm' : markers are shown as minutes and seconds.

        - 's' : markers are shown as seconds.

        - 'ms' : markers are shown as milliseconds.

        - 'lag' : like time, but past the halfway point counts as negative values.

        - 'lag_h' : same as lag, but in hours, minutes and seconds.

        - 'lag_m' : same as lag, but in minutes and seconds.

        - 'lag_s' : same as lag, but in seconds.

        - 'lag_ms' : same as lag, but in milliseconds.

        Rhythm:

        - 'tempo' : markers are shown as beats-per-minute (BPM)
            using a logarithmic scale.  This is useful for
            visualizing the outputs of `feature.tempogram`.

        - 'fourier_tempo' : same as `'tempo'`, but used when
            tempograms are calculated in the Frequency domain
            using `feature.fourier_tempogram`.

    vscale : str
        Optional value transformation for `data`.  The following are supported:

        - 'dB' : decibels with `1` as a reference amplitude

        - 'dB[<value>]' : decibels with the given value as a reference amplitude, e.g. 'dB[0.1]'.

        - 'dB[power]' : like above, but treating `data` as power rather than amplitude measurements.

        - 'dB[power,<value>]' : like above, but with an explicit reference power value, e.g. 'dB[power,0.1]'.

        - 'dBFS' : decibels relative to full scale, using `np.max(data)` as a reference amplitude

        - 'dBFS[power]' : like above, but treating `data` as power rather than amplitude measurements.

        - 'phase' : phase values in radians, with a range of `[-π, π]`.

        - 'dphase' : unwrapped phase differences in radians.  Each pixel corresponds to the residual between the
          observed phase and the expected phase if the frequency was stationary at the previous time step.
          Values are in the range of `[-π, π]`.

        - 'dphase_t' : as above, but differences are computed along the vertical axis instead of horizontal.
          This is intended for use with transposed spectrograms where the time axis is
          vertical and the frequency axis is horizontal.

        .. note::
            When using phase difference modes (`dphase` or `dphase_t`), the x and y coordinates must be provided
            via either the `x_axis` and `y_axis` parameters (e.g., `'time', 'fft'`), or explicitly by
            the `x_coords` and `y_coords` parameters.  All time-like and frequency-like axes are supported.

    sr : number > 0 [scalar]
        Sample rate used to determine time scale in x-axis.

    hop_length : int > 0 [scalar]
        Hop length, also used to determine time scale in x-axis

    n_fft : int > 0 or None
        Number of samples per frame in STFT/spectrogram displays.
        By default, this will be inferred from the shape of ``data``
        as ``2 * (d - 1)``.
        If ``data`` was generated using an odd frame length, the correct
        value can be specified here.

    win_length : int > 0 or None
        The number of samples per window.
        By default, this will be inferred to match ``n_fft``.
        This is primarily useful for specifying odd window lengths in
        Fourier tempogram displays.

    fmin : float > 0 [scalar] or None
        Frequency of the lowest spectrogram bin.  Used for Mel, CQT, and VQT
        scales.

        If ``y_axis`` is `cqt_hz` or `cqt_note` and ``fmin`` is not given,
        it is set by default to ``note_to_hz('C1')``.

    fmax : float > 0 [scalar] or None
        Used for setting the Mel frequency scales

    tempo_min : float > 0 [scalar]
        Lowest tempo (in beats per minute) for tempogram display.

    tempo_max : float > 0 [scalar]
        Highest tempo (in beats per minute) for tempogram display.

    tuning : float
        Tuning deviation from A440, in fractions of a bin.

        This is used for CQT frequency scales, so that ``fmin`` is adjusted
        to ``fmin * 2**(tuning / bins_per_octave)``.

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave.  Used for CQT frequency scale.

    key : str
        The reference key to use when using note axes (`cqt_note`, `chroma`).

    Sa : float or int
        If using Hindustani or Carnatic svara axis decorations, specify Sa.

        For `cqt_svara`, ``Sa`` should be specified as a frequency in Hz.

        For `chroma_c` or `chroma_h`, ``Sa`` should correspond to the position
        of Sa within the chromagram.
        If not provided, Sa will default to 0 (equivalent to `C`)

    mela : str or int, optional
        If using `chroma_c` or `cqt_svara` display mode, specify the melakarta raga.

    thaat : str, optional
        If using `chroma_h` display mode, specify the parent thaat.

    auto_aspect : bool
        Axes will have 'equal' aspect if the horizontal and vertical dimensions
        cover the same extent and their types match.

        To override, set to `False`.

    htk : bool
        If plotting on a mel frequency axis, specify which version of the mel
        scale to use.

            - `False`: use Slaney formula (default)
            - `True`: use HTK formula

        See `core.mel_frequencies` for more information.

    unicode : bool
        If using note or svara decorations, setting `unicode=True`
        will use unicode glyphs for accidentals and octave encoding.

        Setting `unicode=False` will use ASCII glyphs.  This can be helpful
        if your font does not support musical notation symbols.

    intervals : str or array of floats in [1, 2), optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the interval specification.

        See `core.interval_frequencies` for a description of supported values.

    unison : str, optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the pitch name of the unison
        interval.  If not provided, it will be inferred from `fmin` (for VQT display) or
        assumed as `'C'` (for chroma display).

    top_db : float
        If using a decibel scale, how many dB below the peak to allow
        before clipping.

    cmap_seq : str or matplotlib.colors.Colormap
        The name of the sequential colormap to use for decibel scales.
        Default is 'magma'.

    cmap_bool : str or matplotlib.colors.Colormap
        The name of the colormap to use for boolean data.
        Default is 'gray_r'.

    cmap_div : str or matplotlib.colors.Colormap
        The name of the diverging colormap to use for diverging data.
        Default is 'coolwarm'.

    cmap_cyclic : str or matplotlib.colors.Colormap
        The name of the cyclic colormap to use for phase data.
        Default is 'twilight_shifted'.

    div_thresh : float
        The threshold for determining whether to use a diverging colormap.
        If the data has values both above and below this threshold, then
        a diverging colormap is used.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    **kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.

        By default, the following options are set:

            - ``rasterized=True``
            - ``shading='auto'``
            - ``edgecolors='None'``

    Notes
    -----
    The ``cmap`` option if not provided via `kwargs`, is inferred from data automatically.
    If `vscale` is specified, the colormap will be sequential for decibels, and cyclic for phase
    and phase differences.

    If a diverging colormap is inferred, the color scale is normalized so that the center
    value (``div_thresh=0`` by default) is at the center of the colormap.

    To use matplotlib's default colormap, explicitly set ``cmap=None``.

    Returns
    -------
    colormesh : `matplotlib.collections.QuadMesh`
        The color mesh object produced by `matplotlib.pyplot.pcolormesh`

    See Also
    --------
    colorbar_db
    colorbar_phase
    infer_cmap : Automatic colormap detection
    matplotlib.pyplot.pcolormesh

    Examples
    --------
    Visualize an STFT magnitude spectrum using default parameters

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('choice', duration=15)
    >>> fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    >>> D = librosa.stft(y)
    >>> img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    ...                                vscale='dBFS', sr=sr, ax=ax[0])
    >>> ax[0].set(title='Linear-frequency magnitude spectrogram')
    >>> ax[0].label_outer()

    Or on a logarithmic scale, and using a larger hop

    >>> hop_length = 1024
    >>> D = librosa.stft(y, hop_length=hop_length)
    >>> librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
    ...                          vscale='dBFS', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-frequency magnitude spectrogram')
    >>> ax[1].label_outer()
    >>> librosa.display.colorbar_db(img, ax=ax)
    """
    all_params = dict(
        kwargs=kwargs,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        tuning=tuning,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=win_length,
        key=key,
        htk=htk,
        unicode=unicode,
        intervals=intervals,
        unison=unison,
    )

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    # Parse the value scale into a normalizer and possibly a colormap
    data, norm_cmap = __scale_data(
        data,
        vscale=vscale,
        top_db=top_db,
        x_coords=x_coords,
        y_coords=y_coords,
        cmap_seq=cmap_seq,
        cmap_cyclic=cmap_cyclic,
    )

    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn(
            "Trying to display complex-valued input. " "Showing magnitude instead.",
            stacklevel=2,
        )
        data = np.abs(data)

    if norm_cmap is not None:
        kwargs.setdefault("cmap", norm_cmap)
    elif "cmap" not in kwargs:
        # Neither vscale nor the user gave us a cmap, so we have to infer it
        kwargs["cmap"] = infer_cmap(
            data,
            cmap_seq=cmap_seq,
            cmap_bool=cmap_bool,
            cmap_div=cmap_div,
            div_thresh=div_thresh,
        )
        if isinstance(cmap_div, colors.Colormap):
            is_diverging_cmap = kwargs["cmap"] == cmap_div
        else:
            is_diverging_cmap = kwargs["cmap"] == mcm.get(cmap_div, None)

        if is_diverging_cmap:
            # If we have an inferred diverging colormap,
            # use a twoslope normalizer around the divergence threshold.
            # But only if the user didn't also set their own normalizer
            # If the user gave vmin/vmax values, move them from kwargs to the norm
            kwargs.setdefault(
                "norm",
                colors.TwoSlopeNorm(
                    vcenter=div_thresh,
                    vmin=kwargs.pop("vmin", None),
                    vmax=kwargs.pop("vmax", None),
                ),
            )

    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("edgecolors", "None")
    kwargs.setdefault("shading", "auto")
    if vscale is not None and "phase" in vscale:
        # If we're displaying phase, try to ensure that the color gamut
        # covers the full range.
        # A user can override this if they want to.
        kwargs.setdefault("vmin", -np.pi)
        kwargs.setdefault("vmax", np.pi)

    axes = __check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)

    __set_current_image(ax, out)

    # Set up axis scaling
    __scale_axes(axes, x_axis, "x", tempo_min=tempo_min, tempo_max=tempo_max)
    __scale_axes(axes, y_axis, "y", tempo_min=tempo_min, tempo_max=tempo_max)

    # Construct tickers and locators
    __decorate_axis(
        axes.xaxis,
        x_axis,
        key=key,
        Sa=Sa,
        mela=mela,
        thaat=thaat,
        unicode=unicode,
        fmin=fmin,
        unison=unison,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        n_bins=len(x_coords),
    )
    __decorate_axis(
        axes.yaxis,
        y_axis,
        key=key,
        Sa=Sa,
        mela=mela,
        thaat=thaat,
        unicode=unicode,
        fmin=fmin,
        unison=unison,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        n_bins=len(y_coords),
    )

    # If the plot is a self-similarity/covariance etc. plot, square it
    if __same_axes(x_axis, y_axis, axes.get_xlim(), axes.get_ylim()) and auto_aspect:
        axes.set_aspect("equal")

    return out


def __set_current_image(ax, img):
    """
    Set the current image when working in pyplot mode.

    If the provided ``ax`` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    """
    if ax is None:
        plt.sci(img)


def __mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""
    if coords is not None:
        if len(coords) not in (n, n + 1):
            raise ParameterError(
                f"Coordinate shape mismatch: {len(coords)}!={n} or {n}+1"
            )
        return coords

    coord_map: dict[str | None, Callable[..., np.ndarray]] = {
        "linear": __coord_fft_hz,
        "fft": __coord_fft_hz,
        "fft_note": __coord_fft_hz,
        "fft_svara": __coord_fft_hz,
        "hz": __coord_fft_hz,
        "oct3": __coord_fft_hz,
        "log_oct3": __coord_fft_hz,
        "log": __coord_fft_hz,
        "mel": __coord_mel_hz,
        "mel_oct3": __coord_mel_hz,
        "cqt": __coord_cqt_hz,
        "cqt_hz": __coord_cqt_hz,
        "cqt_note": __coord_cqt_hz,
        "cqt_svara": __coord_cqt_hz,
        "cqt_oct3": __coord_cqt_hz,
        "vqt_fjs": __coord_vqt_hz,
        "vqt_hz": __coord_vqt_hz,
        "vqt_note": __coord_vqt_hz,
        "vqt_oct3": __coord_vqt_hz,
        "chroma": __coord_chroma,
        "chroma_c": __coord_chroma,
        "chroma_h": __coord_chroma,
        "chroma_fjs": __coord_n,  # We can't use a 12-normalized tick locator here
        "time": __coord_time,
        "h": __coord_time,
        "m": __coord_time,
        "s": __coord_time,
        "ms": __coord_time,
        "lag": __coord_time,
        "lag_h": __coord_time,
        "lag_m": __coord_time,
        "lag_s": __coord_time,
        "lag_ms": __coord_time,
        "tonnetz": __coord_n,
        "off": __coord_n,
        "tempo": __coord_tempo,
        "fourier_tempo": __coord_fourier_tempo,
        "frames": __coord_n,
        None: __coord_n,
    }

    if ax_type not in coord_map:
        raise ParameterError(f"Unknown axis type: {ax_type}")
    return coord_map[ax_type](n, **kwargs)


def __check_axes(axes: mplaxes.Axes | None) -> mplaxes.Axes:
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        axes = plt.gca()
    elif not isinstance(axes, mplaxes.Axes):
        raise ParameterError(
            "`axes` must be an instance of matplotlib.axes.Axes. "
            "Found type(axes)={}".format(type(axes))
        )
    return axes


def __scale_axes(axes, ax_type, which, tempo_min, tempo_max):
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


def __decorate_axis(
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
            axis.set_major_locator(mplticker.FixedLocator(__OCT3_FREQUENCIES[5::3]))  # type: ignore[arg-type]
        else:
            axis.set_major_locator(mplticker.FixedLocator(__OCT3_FREQUENCIES[::3]))  # type: ignore[arg-type]
        axis.set_major_formatter(AdaptiveEngFormatter(major=True, unit="Hz"))
        axis.set_label_text("Frequency")
        # Minor ticks at the 1/3 octaves
        axis.set_minor_locator(mplticker.FixedLocator(__OCT3_FREQUENCIES, nbins=None))  # type: ignore[arg-type]
        axis.set_minor_formatter(AdaptiveEngFormatter(major=False, unit="Hz"))

    elif ax_type in ["frames"]:
        axis.set_label_text("Frames")

    elif ax_type in ["off", "none", None]:
        axis.set_label_text("")
        axis.set_ticks([])

    else:
        raise ParameterError(f"Unsupported axis type: {ax_type}")


def __coord_fft_hz(
    n: int, sr: float = 22050, n_fft: int | None = None, **_kwargs: Any
) -> _Array1D[np.float64]:
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
    return basis


def __coord_mel_hz(
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


def __coord_cqt_hz(
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


def __coord_vqt_hz(
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


def __coord_chroma(n: int, bins_per_octave: int = 12, **_kwargs: Any) -> np.ndarray:
    """Get chroma bin numbers"""
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n, endpoint=False)


def __coord_tempo(
    n: int, sr: float = 22050, hop_length: int = 512, **_kwargs: Any
) -> np.ndarray:
    """Tempo coordinates"""
    basis = core.tempo_frequencies(n + 1, sr=sr, hop_length=hop_length)[1:]
    return basis


def __coord_fourier_tempo(
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


def __coord_n(n: int, **_kwargs: Any) -> np.ndarray:
    """Get bare positions"""
    return np.arange(n)


def __coord_time(
    n: int, sr: float = 22050, hop_length: int = 512, **_kwargs: Any
) -> np.ndarray:
    """Get time coordinates from frames"""
    times: np.ndarray = core.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    return times


def __same_axes(x_axis, y_axis, xlim, ylim):
    """Check if two axes are similar, used to determine squared plots"""
    axes_compatible_and_not_none = (x_axis, y_axis) in _AXIS_COMPAT
    axes_same_lim = xlim == ylim
    return axes_compatible_and_not_none and axes_same_lim


def __scale_data(data, *, vscale, top_db, x_coords, y_coords, cmap_seq, cmap_cyclic):
    """Parse the vscale parameter and return the transformed data and colormap
    if necessary

    Parameters
    ----------
    data : np.ndarray
        The data to be scaled and visualized.
    vscale : str or None
        The value scale to apply to the data.
        If None, the data is returned as-is.
    top_db : float
        The maximum decibel level to display when using a dB scale.
        This is only used if `vscale` is set to a dB mode.
    x_coords, y_coords : np.ndarray
        Time and frequency coordinates for the data.
        These should be constructed using the `__mesh_coords` function.
    cmap_seq : str or matplotlib.colors.Colormap
        Default sequential colormap to use for dB scales.
    cmap_cyclic : str or matplotlib.colors.Colormap
        Default cyclic colormap to use for phase scales.

    Returns
    -------
    data : np.ndarray
        The scaled data, ready for visualization.
    cmap : matplotlib.colors.Colormap or None
        The colormap to use for visualization, or None if no scaling is applied.
    """
    # If vscale is None, we return the data as-is
    if vscale is None:
        return data, None

    # First check for the easy cases
    if vscale == "phase":
        # Phase should use a cyclic colormap
        return np.angle(data), cmap_cyclic

    elif vscale == "dphase":
        # Compute the difference of unwrapped phase
        diff = np.diff(np.unwrap(np.angle(data), axis=-1), axis=-1, prepend=0.0)
        # Correct it compared to the expected phase advance on this time-frequency grid
        #   - 2π*y counts radians per second
        #   - diff(x) counts seconds per frame
        #   - The product counts radians per frame
        diff -= np.multiply.outer(2 * np.pi * y_coords, np.diff(x_coords, prepend=0.0))
        # Wrap back to +-pi
        diff += np.pi
        np.mod(diff, 2 * np.pi, out=diff)
        diff -= np.pi
        # Use a cyclic colormap for the phase difference
        return diff, cmap_cyclic

    elif vscale == "dphase_t":
        # Same computation as above, but on the opposite axes
        diff = np.diff(np.unwrap(np.angle(data), axis=0), axis=0, prepend=0.0)
        diff -= np.multiply.outer(np.diff(y_coords, prepend=0.0), 2 * np.pi * x_coords)
        diff += np.pi
        np.mod(diff, 2 * np.pi, out=diff)
        diff -= np.pi
        return diff, cmap_cyclic

    else:
        # In some kind of dB mode
        _mode, scale_type, ref_ = __parse_vscale(vscale)
        if ref_ == "max":
            ref = np.max(np.abs(data))
        elif ref_ is None:
            ref = 1.0
        else:
            ref = float(ref_)

        if scale_type == "power":
            data = core.power_to_db(np.abs(data), top_db=top_db, ref=ref)
        else:
            data = core.amplitude_to_db(np.abs(data), top_db=top_db, ref=ref)

        # Use the default colormap for sequential data
        return data, cmap_seq


VSCALE_PATTERN = re.compile(
    r"^(?P<mode>dBFS|dB)"  # Match "dBFS" or "dB"
    r"(?:\[(?:(?P<type>power)"  # Optionally match [power
    r"(?:,(?P<ref_power>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))?"  # Optional ref_power
    r"|(?P<ref>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?))\])?$"  # Or ref alone
)


def __parse_vscale(vscale: str) -> tuple[str, str, float | str | None]:
    """Parse a vscale string into mode, scale_type, and reference value.

    Examples
    --------
    - 'dBFS' -> ('dBFS', 'amplitude', 'max')
    - 'dBFS[power]' -> ('dBFS', 'power', 'max')
    - 'dB[power,0.1]' -> ('dB', 'power', 0.1)
    - 'dB[0.1]' -> ('dB', 'amplitude', 0.1)
    - 'dB' -> ('dB', 'amplitude', None)

    Parameters
    ----------
    vscale : str

    Returns
    -------
    mode is one of 'dBFS' or 'dB'
    scale_type is one of 'power' or 'amplitude'
    ref is a float, None, or 'max'
    """
    match = VSCALE_PATTERN.fullmatch(vscale)
    if not match:
        raise ParameterError(f"Invalid vscale specification: {vscale}")

    mode = match.group("mode")

    scale_type = "power" if match.groupdict().get("type") else "amplitude"

    ref = match.groupdict().get("ref") or match.groupdict().get("ref_power")

    if mode == "dBFS":
        if ref is not None:
            raise ParameterError("dBFS vscale cannot have an explicit reference value")
        ref = "max"
    elif ref is not None:  # mode == 'dB'
        ref = float(ref)
    return mode, scale_type, ref


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
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
    >>> ax[2].set(title='Multiple waveforms')
    >>> ax[2].legend()
    >>> plt.show()

    Zooming in on a plot to show raw sample values

    >>> fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    >>> ax.set(xlim=[6.0, 6.01], title='Sample view', ylim=[-0.2, 0.2])
    >>> librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
    >>> ax.label_outer()
    >>> ax.legend()
    >>> ax2.legend()
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
    axes = __check_axes(ax)

    # Reduce by envelope calculation
    # this choice of hop ensures that the envelope has at most max_points values
    hop_length = max(1, y.shape[-1] // max_points)
    y_env = __envelope(y, hop_length)

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
    __decorate_axis(dec_axis, axis)

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

    axes = __check_axes(ax)

    hop = max(1, y.shape[-1] // n_bars)
    env = __envelope(y, hop)
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
        __decorate_axis(axes.yaxis, axis)
    else:
        __decorate_axis(axes.xaxis, axis)

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
    axes = __check_axes(ax)

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
        __decorate_axis(
            axes.xaxis,
            freq_axis,
            key=key,
            Sa=Sa,
            mela=mela,
            thaat=thaat,
            unicode=unicode,
        )
    else:
        __decorate_axis(
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


def __radian_formatter(x, pos):
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
    kwargs.setdefault("format", mplticker.FuncFormatter(__radian_formatter))

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


def _squeeze_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Check if two shape arrays are equivalent after squeezing out singleton dimensions."""
    return tuple(dim for dim in shape if dim > 1)


def _resolve_multiplot(
    func: Literal["waveshow", "wavebars", "specshow"],
) -> tuple[Callable[..., Any], int, list[str]]:
    """Resolve multiplot function names.

    Parameters
    ----------
    func : str
        The name of the display function to use for the multiplot.
        Accepted values are 'waveshow', 'wavebars', and 'specshow'.

    Returns
    -------
    function : callable
        The display function corresponding to the given name.
    dims : int
        The number of data dimensions that each call to the display function expects.
    badprops : list of str
        A list of property names that are not supported by the display function and should
        be removed from the style cycle when sharing properties.
    """
    display_map: dict[str, tuple[Callable[..., Any], int, list[str]]] = {
        "waveshow": (waveshow, 1, []),
        "wavebars": (wavebars, 1, []),
        "specshow": (specshow, 2, ["color"]),
    }

    try:
        return display_map[func]
    except KeyError as exc:
        raise ParameterError(f"Invalid display '{func}' for multiplot") from exc


def _mp_get_layout(
    data: tuple[np.ndarray, ...], dims: int, orient: Literal["h", "v"]
) -> tuple[tuple[int, ...], int, int, bool]:
    """Determine the layout of a multiplot grid based on the data shape and orientation.

    Parameters
    ----------
    data : tuple of ndarray
        The input data for the multiplot. The shape of this data will determine the layout of the grid.
    dims : int
        The number of data dimensions that each call to the display function expects.
    orient : str {'h', 'v'}
        The orientation of the multiplot grid. Accepted values are 'h' for horizontal and 'v' for vertical.

    Returns
    -------
    axshape : tuple of int
        The shape of the grid of axes, determined by the shape of the input data and the
        specified orientation.
    nrows : int
        The number of rows in the grid of axes.
    ncols : int
        The number of columns in the grid of axes.
    multi_input : bool
        If the input contains multiple separate arrays to plot,
        this flag is True.  Otherwise, False.
    """
    if orient not in ("h", "v"):
        raise ParameterError(f"Invalid value orient={orient}")

    multi_plot = False
    if len(data) == 1 and isinstance(data[0], np.ndarray) and data[0].ndim > dims:
        data_stack = np.asarray(data[0])
        axshape = data_stack.shape[:-dims]

    elif len(data) >= 1:
        multi_plot = True
        axshape = (len(data),)
    else:
        raise ParameterError("multiplot requires at least one data array to plot")

    if len(axshape) == 1:
        if orient == "v":
            nrows, ncols = axshape[0], 1
        else:
            nrows, ncols = 1, axshape[0]
    elif len(axshape) == 2:
        # Yes this is awkward, but it makes the type checker work.
        # In a sane world it would just be nrows, ncols = axshape
        nrows, ncols = axshape[0], axshape[-1]
    else:
        raise ParameterError(f"Invalid axes shape={axshape}")

    return axshape, nrows, ncols, multi_plot


def _mp_setup_axes(
    *,
    axes: matplotlib.axes.Axes | np.ndarray | None,
    fig: matplotlib.figure.FigureBase | None = None,
    fig_kw: dict | None = None,
    nrows: int,
    ncols: int,
    axshape: tuple[int, ...],
    orient: Literal["h", "v"],
    sharex: bool,
    sharey: bool,
) -> tuple[matplotlib.figure.FigureBase, npt.NDArray[np.object_], tuple[int, ...]]:
    """Set up the figure and axes for a multiplot grid.

    Parameters
    ----------
    axes : matplotlib.axes.Axes, np.ndarray, or None
        The axes to use for the multiplot. If None, a new figure and axes will be created.
        If a single Axes object is provided, it will be used for all subplots.
        If an array of Axes objects is provided, it must be compatible with the shape of the data.
    fig : matplotlib.figure.FigureBase or None
        The figure to use for the multiplot. If None, a new figure will be created if needed.
    fig_kw : dict or None
        Additional keyword arguments to pass to `plt.subplots` when creating a new figure.
    nrows : int
        The number of rows in the grid of axes.
    ncols : int
        The number of columns in the grid of axes.
    axshape : tuple of int
        The shape of the grid of axes, determined by the shape of the input data and the
        specified orientation.
    orient : str {'h', 'v'}
        The orientation of the multiplot grid. Accepted values are 'h' for horizontal and 'v' for vertical.
    sharex : bool
        Whether to share the x-axis among subplots when creating a new figure.
    sharey : bool
        Whether to share the y-axis among subplots when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.FigureBase
        The figure object for the multiplot.
    axes : np.ndarray
        An array of Axes objects for the multiplot, with shape compatible with the input data.
    output_shape : tuple of int
        The shape of the output array of display objects, determined by the shape of the axes.
    """
    output_shape = axshape

    if axes is None:
        if fig is None:
            if fig_kw is None:
                fig_kw = {}

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                squeeze=False,
                **fig_kw,
            )
        else:
            axes = fig.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                squeeze=False,
            )

    elif isinstance(axes, np.ndarray):
        output_shape = axes.shape

        if axes.ndim == 1:
            if orient == "v":
                axes = axes[:, np.newaxis]
            else:
                axes = axes[np.newaxis, :]

    else:
        if not isinstance(axes, np.ndarray):
            output_shape = tuple()

        axes = np.atleast_2d(np.asarray(axes))

    # Ensure that axes object is now encapsulated in numpy arrays
    axes = np.asarray(axes, dtype=object)

    # Populate fig with the figure from the axes object.
    fig = axes.flat[0].get_figure()

    if _squeeze_shape(axes.shape) != _squeeze_shape(axshape):
        raise ParameterError(f"axes shape={axes.shape} is incompatible with data shape")

    return fig, axes, output_shape


def _mp_setup_labels(
    labels: Sequence[str | None] | None, shape: tuple[int, ...]
) -> npt.NDArray[np.object_]:
    """Set up the labels for a multiplot grid.

    Parameters
    ----------
    labels : sequence of str or None
        The labels to apply to each subplot in the multiplot grid. If None, no labels
        will be applied. If a sequence is provided, it must be compatible with the shape of the axes.
    shape : tuple of int
        The shape of the grid of axes, determined by the shape of the input data and the
        specified orientation.

    Returns
    -------
    np.ndarray
        An array of labels for each subplot in the multiplot grid, with shape compatible with the
        axes.
    """
    if labels is None:
        return np.full(shape, None, dtype=object)

    return np.asarray(labels, dtype=object).reshape(shape)


def _mp_setup_prop_group(
    share_properties: bool | Literal["row", "col"] | ArrayLike | None,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Set up the property groups for a multiplot grid.

    This is used to determine how style properties (color, line style, etc.) are shared among
    different subplots in the grid.

    Parameters
    ----------
    share_properties : bool, str, sequence, or None
        The property sharing scheme for the multiplot grid. Accepted values are:
        - `None` or `False`: no properties are shared, and each subplot is treated as a unique group.
        - `True`: all subplots share the same properties and belong to a single group.
        - 'row': subplots in the same row share properties and belong to the same group.
        - 'col': subplots in the same column share properties and belong to the same group.
        - sequence: a sequence of group identifiers for each subplot. The length of the
          sequence must match the total number of subplots (i.e., the product of the shape
          of the axes).
    shape : tuple of int
        The shape of the grid of axes, determined by the shape of the input data and the
        specified orientation.

    Returns
    -------
    np.ndarray
        An array of group identifiers for each subplot in the multiplot grid, with shape compatible with the axes.
    """
    if share_properties is None or share_properties is False:
        return np.arange(np.prod(shape)).reshape(shape)

    if share_properties is True:
        return np.ones(shape, dtype=int)

    if isinstance(share_properties, str) and share_properties == "row":
        return np.asarray(np.indices(shape)[0])

    if isinstance(share_properties, str) and share_properties == "col":
        return np.asarray(np.indices(shape)[-1])

    prop_group = np.asarray(share_properties)

    if prop_group.size != np.prod(shape):
        raise ParameterError(
            f"Shape mismatch between axes={shape} "
            f"and share_properties={prop_group.shape}"
        )

    return prop_group.reshape(shape)


def _mp_setup_properties(
    prop_group: np.ndarray, badprops: list[str], prop_cycle: cycler.Cycler | None
) -> npt.NDArray[np.object_]:
    """Set up the properties for each subplot in a multiplot grid based on the property groups.

    Parameters
    ----------
    prop_group : np.ndarray
        An array of group identifiers for each subplot in the multiplot grid, with shape compatible with the axes.
    badprops : list of str
        A list of property names that are not supported by the display function
        and should be removed from the style cycle when sharing properties.
    prop_cycle : cycler.Cycler or None
        The property cycle to use for assigning properties to the subplots. If None, the
        default property cycle from `plt.rcParams["axes.prop_cycle"]` will be used.

    Returns
    -------
    np.ndarray
        An array of property dictionaries for each subplot in the multiplot grid, with shape compatible with the axes.
    """
    properties = np.empty(prop_group.shape, dtype=object)
    properties.fill(None)

    if prop_cycle is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]

    style_cycle = cycle(prop_cycle)
    style_map = {}

    for idx in np.ndindex(prop_group.shape):
        group = prop_group[idx]

        if group not in style_map:
            style = copy.deepcopy(next(style_cycle))
            for prop in badprops:
                style.pop(prop, None)
            style_map[group] = style

        properties[idx] = style_map[group]

    return properties


def multiplot(
    func: Literal["waveshow", "wavebars", "specshow"],
    *data: np.ndarray,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    fig: matplotlib.figure.FigureBase | None = None,
    orient: Literal["v", "h"] = "v",
    share_properties: bool | Literal["row", "col"] | np.ndarray | None = None,
    fig_kw: dict | None = None,
    sharex: bool = True,
    sharey: bool = True,
    label_outer: bool = True,
    labels: Sequence[str | None] | None = None,
    titles: Sequence[str | None] | None = None,
    prop_cycle: cycler.Cycler | None = None,
    **kwargs: Any,
) -> npt.NDArray[np.object_]:
    """Visualize multiple related waveforms or spectrograms on an array of subplots.

    Example use cases include:
        - Displaying multiple waveforms from a multi-channel audio file.
        - Displaying multiple spectrograms from a multi-channel audio file.

    Parameters
    ----------
    func : str
        The name of the display function to use for the multiplot. Accepted values are 'waveshow',
        'wavebars', and 'specshow'.

    *data : one or more `np.ndarray`s
        The input data for the multiplot.
        If one array is provided, it is interpreted as a multi-channel array, where the leading
        dimensions correspond to different channels or signals to plot.
        If multiple arrays are provided, each array is treated as a single channel or input
        signal, and visualized on its own subplot.

    axes : matplotlib.axes.Axes, np.ndarray, or None
        The axes to use for the multiplot. If None, a new axes array will be created on `fig`.
        If an array of Axes objects is provided, it must be compatible with the shape of the data.
        If a single axes object is provided, it will be interpreted as a 1x1 array (i.e. a single subplot).

    fig : matplotlib.figure.FigureBase or None
        The figure to use for the multiplot. If None, a new figure will be created if needed.
        If `axes` is provided, the figure will be inferred from `axes` and the `fig` parameter
        will be ignored.

    orient : str {'h', 'v'}
        The orientation of the multiplot grid. Accepted values are 'h' for horizontal
        and 'v' for vertical. This determines how the subplots are arranged when the
        input data has a single non-singleton dimension (e.g., shape (n, k) with k > 1).

    share_properties : bool, str, np.ndarray, or None
        The property sharing scheme for the multiplot grid. Accepted values are:

        - `None` or `False`: no properties are shared, and each subplot is treated as a unique group.
        - `True`: all subplots share the same properties and belong to a single group.
        - 'row': subplots in the same row share properties and belong to the same group
        - 'col': subplots in the same column share properties and belong to the same group.
        - np.ndarray: a custom array of group identifiers for each subplot. The shape of the
          array must match the shape of the axes grid.  Any two elements with the same value
          are considered to be in the same group and will share properties.

    fig_kw : dict or None
        Additional keyword arguments to pass to `plt.subplots` when creating a new figure.

    sharex : bool
        Whether to share the x-axis among subplots when creating a new figure.

    sharey : bool
        Whether to share the y-axis among subplots when creating a new figure.

    label_outer : bool
        Whether to only show labels on the outer axes when using shared axes.

    labels : sequence of str or None
        The labels to apply to each subplot in the multiplot grid. If None, no labels
        will be applied. If a sequence is provided, it must be compatible with the shape of the axes.

    titles : sequence of str or None
        The titles to apply to each subplot in the multiplot grid. If None, no titles
        will be applied. If a sequence is provided, it must be compatible with the shape of the axes.

    prop_cycle : cycler.Cycler or None
        The property cycle to use for assigning properties to the subplots. If None, the
        default property cycle from `plt.rcParams["axes.prop_cycle"]` will be used.

    **kwargs
        Additional keyword arguments to pass to the display function for each subplot.

    Returns
    -------
    np.ndarray
        An array of display objects returned by the display function for each subplot in the multiplot grid
        The shape of this array will be compatible with the shape of the axes grid.

    See Also
    --------
    waveshow
    wavebars
    specshow
    legend_for_axes

    Examples
    --------
    Display multiple synchronized signals stacked in an array.  We'll let multiplot create
    the figure and axes objects for us.

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('choice', duration=10)
    >>> yh, yp = librosa.effects.hpss(y)
    >>> librosa.display.multiplot('waveshow', y, yh, yp,
    ...                           labels=['Original', 'Harmonic', 'Percussive'],
    ...                           # The remaining parameters are passed through to waveshow
    ...                           sr=sr,
    ...                           invert=True)
    >>> librosa.display.legend_for_axes()  # Helper to create a single legend across subplots
    >>> plt.show()

    Multiplot can also accept preconstructed axes as input, provided that they
    are compatible with the shape of the data.  The below example does this
    with a spectrogram display.

    >>> y_stack = librosa.to_multi(y, yh, yp)
    >>> stft = librosa.stft(y=y_stack)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(8, 8))
    >>> img = librosa.display.multiplot('specshow', stft, axes=ax,
    ...                                 titles=['Original', 'Harmonic', 'Percussive'],
    ...                                 x_axis='time', y_axis='log', vscale='dBFS')
    >>> librosa.display.colorbar_db(img[0], ax=ax, label='dBFS')
    >>> plt.show()
    """
    # Identify the display function and the expected data dimensions for each subplot
    function, dims, badprops = _resolve_multiplot(func)

    # Determine the layout of the multiplot grid based on the data shape and orientation
    axshape, nrows, ncols, multi_input = _mp_get_layout(data, dims, orient)

    # Set up the figure and axes for the multiplot grid
    fig, axes, output_shape = _mp_setup_axes(
        axes=axes,
        fig=fig,
        fig_kw=fig_kw,
        nrows=nrows,
        ncols=ncols,
        axshape=axshape,
        orient=orient,
        sharex=sharex,
        sharey=sharey,
    )

    # Set up the labels and properties for each subplot in the multiplot grid
    labels = _mp_setup_labels(labels, axes.shape)
    titles = _mp_setup_labels(titles, axes.shape)
    prop_group = _mp_setup_prop_group(share_properties, axes.shape)
    properties: np.ndarray = _mp_setup_properties(prop_group, badprops, prop_cycle)

    # Allocate the output array
    output = np.empty_like(axes, dtype=object)

    # Iterate over each subplot and call the display function with the appropriate data, axes, labels, and properties
    for idx in np.ndindex(axshape):
        flat_idx = np.ravel_multi_index(idx, axshape)
        if multi_input:
            # User provided variadic inputs, so use flat indexing
            datum = data[flat_idx]
        else:
            # User already stacked the inputs into one array.
            datum = data[0][idx]
        output.flat[flat_idx] = function(
            datum,
            ax=axes.flat[flat_idx],
            label=labels.flat[flat_idx],
            **properties.flat[flat_idx],
            **kwargs,
        )
        if titles.flat[flat_idx] is not None:
            axes.flat[flat_idx].set_title(titles.flat[flat_idx])
        if label_outer:
            axes.flat[flat_idx].label_outer()

    # Reshape the output array to match the shape of the axes grid
    return output.reshape(output_shape)


def legend_for_axes(
    axes: matplotlib.axes.Axes | np.ndarray | list[matplotlib.axes.Axes] | None = None,
    *,
    fig: matplotlib.figure.Figure | None = None,
    **kwargs: Any,
) -> matplotlib.legend.Legend:
    """Create a figure-level legend for a collection of axes.

    This is similar to `matplotlib.figure.Figure.legend`, but it limits
    the handle collection to only those belonging to the specified axes.
    This makes it easier to create different legends for subsets of a subplot array.

    Parameters
    ----------
    axes : matplotlib.axes.Axes or array-like of Axes, optional
        Axes to include in the legend aggregation.
        If not provided, axes are taken from `fig.axes`, or from the
        current figure if `fig` is not provided.

    fig : matplotlib.figure.Figure, optional
        Figure on which to create the legend.
        If not provided, it is inferred from `axes`, or from `plt.gcf()`
        if `axes` is also not provided.

    **kwargs
        Additional keyword arguments passed to `matplotlib.figure.Figure.legend`.

    Returns
    -------
    legend : matplotlib.legend.Legend
        The created legend.

    Examples
    --------
    If no axes are provided, we aggregate legends across all subplots on the current figure:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10, 10, 100)
    >>> fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    >>> ax[0].plot(x, label='Line', color='C0')
    >>> ax[1].plot(x**2, label='Parabola', color='C1')
    >>> librosa.display.legend_for_axes()
    >>> plt.show()

    You can also specify a subset of axes to aggregate, and control the legend placement:

    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    >>> ax[0, 0].plot(x, label='Line', color='C0')
    >>> ax[0, 1].plot(x**2, label='Parabola', color='C1')
    >>> ax[1, 0].plot(x**3, label='Cubic', color='C2')
    >>> ax[1, 1].plot(x**4, label='Quartic', color='C3')
    >>> librosa.display.legend_for_axes(axes=ax[0], loc='outside upper center')
    >>> librosa.display.legend_for_axes(axes=ax[1], loc='outside lower center')
    >>> plt.show()
    """
    if axes is None:
        if fig is None:
            fig = plt.gcf()
        axes = fig.axes

    axes_array = np.atleast_1d(np.asarray(axes, dtype=object))

    if len(axes_array.flat) == 0:
        raise ParameterError("No axes provided for legend aggregation")

    if fig is None:
        fig = axes_array.flat[0].figure

    for ax in axes_array.flat:
        if ax.figure is not fig:
            raise ParameterError("All axes must belong to the same figure")

    handles: list[Artist] = []
    labels: list[str] = []

    for ax in axes_array.flat:
        hlist, llist = ax.get_legend_handles_labels()
        handles.extend(hlist)
        labels.extend(llist)

    return fig.legend(handles, labels, **kwargs)


def _get_ax_bright_highlight(
    ax: mplaxes.Axes,
    luminance_threshold: float = 0.5,
) -> bool:
    """Determine whether the axes should produce a bright or dark
    highlight.

    This is based on a few things:
    - If the axes has mappable data, we take the median color of that
      data.
    - If the axes has no mappable data, we take the facecolor of the
      axes.
    - If the axes is transparent, we take the facecolor of the figure.

    From the resulting color, we calculate the luminance by RGB->YIQ
    conversion.  Luminance above threshold is considered light, and should
    therefore produce a dark highlight.  Luminance below threshold is
    considered dark, and should produce a bright highlight.
    """
    mappable = None

    for child in ax.get_children():
        if isinstance(child, cm.ScalarMappable) and child.get_array() is not None:
            mappable = child
            break

    if mappable is not None:
        data = mappable.get_array()
        # Calculate median, ignoring NaNs
        median_val = np.nanmedian(np.asarray(data))
        # Map through the normalization and colormap
        normed_val = mappable.norm(median_val)
        rgba = mappable.get_cmap()(normed_val)
    else:
        # If there's no mappable data, get the axes facecolor
        rgba = ax.get_facecolor()
        # And if the axes is transparent, pull from the figure
        if len(rgba) == 4 and rgba[3] == 0.0:
            rgba = ax.figure.get_facecolor()

    # Calculate relative luminance
    luminance = colorsys.rgb_to_yiq(*rgba[:3])[0]

    return luminance <= luminance_threshold


def highlight(
    *,
    artist: Artist | None = None,
    ax: mplaxes.Axes | None = None,
    color: ColorType | None = None,
    bright_color: ColorType = "white",
    dark_color: ColorType = "black",
    luminance_threshold: float = 0.5,
    **kwargs: Any,
) -> list[mpe.AbstractPathEffect]:
    """Apply a contrasting highlight effect to a matplotlib artist.

    This is primarily useful for providing contrast between an artist
    (e.g., a line plot) and an underlying image (e.g., a spectrogram or scatter plot).
    For example, if the underlying image is predominantly dark (under the choice of colormap),
    then a bright highlight (default "white") should be used.
    If the underlying image is predominantly bright, then a dark highlight (default "black")
    should be used.

    This function is designed to automatically infer which kind of highlight should be applied
    based on the contents of the `ax` axes object, if any.  If no color-mapped data can be
    identified on `ax`, then the axes facecolor or figure facecolor will be used as fallbacks.

    If an `artist` is provided, the highlight effect will be applied in-place, but this is
    optional. (See examples below.)

    The choices for bright and dark highlight colors, as well as the luminance threshold for
    determining which to use, can be customized via the `bright_color`, `dark_color`, and
    `luminance_threshold` parameters.

    Alternatively, the user can bypass the automatic color inference and directly specify a
    highlight color via the `color` parameter.

    Parameters
    ----------
    artist : matplotlib.artist.Artist, optional
        The artist to which the highlight effect should be applied.  If not provided, the
        function will still return the appropriate path effect object(s) based on the contents
        of `ax`, but will not apply them to any artist.

    ax : matplotlib.axes.Axes, optional
        The axes to inspect for color-mapped data to determine the appropriate highlight color.
        If not provided, the function will attempt to infer an appropriate axes object from
        `artist`, and if that fails, will default to the current axes (`plt.gca()`).

    color : color specifier, optional
        A color specification to use directly for the highlight, bypassing the automatic color
        inference.  If not provided, the function will determine whether to use `bright_color`
        or `dark_color` based on the contents of `ax` and the `luminance_threshold`.

    bright_color : color specifier, default 'white'
        The color to use for the highlight if the underlying axes is determined to be dark.

    dark_color : color specifier, default 'black'
        The color to use for the highlight if the underlying axes is determined to be bright.

    luminance_threshold : float, default 0.5
        The luminance threshold for determining whether the underlying axes is considered bright or dark.
        Luminance is calculated by converting the relevant color to YIQ color space and taking
        the Y (luminance) component.  If the luminance is above this threshold, the axes is
        considered bright and `dark_color` will be used for the highlight.  If the luminance is
        below this threshold, the axes is considered dark and `bright_color` will be used for
        the highlight.

    **kwargs : dict
        Additional keyword arguments to pass to `matplotlib.patheffects.withStroke` when
        creating the highlight effect.  Common options include `linewidth` (default to 2) and
        `alpha` (default to 1.0).

        .. note:: `foreground`, if provided, will override the `color` parameter and the
          automatic color inference.  To avoid confusion, it's recommended to specify highlight
          color via the `color` parameter and not to provide `foreground` in `kwargs`.

    Returns
    -------
    effects : list of matplotlib.patheffects.AbstractPathEffect
        A list of path effect objects that implement the highlight.  If `artist` was provided,
        these effects will have been applied to the artist in-place.  If `artist` was not
        provided, these effects can be applied to any artist via `artist.set_path_effects(effects)`.

    Examples
    --------
    Plotting an f₀ contour with and without highlighting, in bright or dark colormaps

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.loadx('trumpet')
    >>> f0, _, _ = librosa.pyin(y, fmin=100, fmax=1000)
    >>> times = librosa.times_like(f0)
    >>> D = librosa.stft(y)
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log_oct3', ax=ax[0, 0],
    ...                          vscale='dBFS')
    >>> ax[0, 0].plot(times, f0)
    >>> ax[0, 0].set_title('Dark image, no highlight')
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log_oct3', ax=ax[0, 1],
    ...                          vscale='dBFS')
    >>> line = ax[0, 1].plot(times, f0)[0]  # 'plot' returns a list of artists
    >>> librosa.display.highlight(artist=line)
    >>> ax[0, 1].set_title('Dark image, highlighted')
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log_oct3', ax=ax[1, 0],
    ...                          vscale='dBFS', cmap='gray_r')
    >>> ax[1, 0].plot(times, f0)
    >>> ax[1, 0].set_title('Bright image, no highlight')
    >>> librosa.display.specshow(D, x_axis='time', y_axis='log_oct3', ax=ax[1, 1],
    ...                          vscale='dBFS', cmap='gray_r')
    >>> # We can also construct the highlight first and then supply it to the plot command
    >>> hl = librosa.display.highlight(ax=ax[1, 1])
    >>> ax[1, 1].plot(times, f0, path_effects=hl)
    >>> ax[1, 1].set_title('Bright image, highlighted')
    >>> for a in ax.flat:
    ...     a.label_outer()
    >>> plt.show()
    """
    # 1. Resolve Axes
    if ax is None:
        if artist is not None and hasattr(artist, "axes") and artist.axes is not None:
            ax = cast("mplaxes.Axes", artist.axes)
        else:
            ax = plt.gca()

    # 2. Infer highlight color
    color = kwargs.pop("foreground", color)
    if color is None:
        if _get_ax_bright_highlight(ax, luminance_threshold):
            # Axes is dark, so we want a bright highlight
            stroke_color = bright_color
        else:
            # Axes is bright, so we want a dark highlight
            stroke_color = dark_color

    else:
        # Use the user-specified highlight color
        stroke_color = color

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1.0)

    # 3. Create and apply the effect
    effects: list[mpe.AbstractPathEffect] = [mpe.withStroke(foreground=stroke_color, **kwargs)]

    if artist is not None:
        artist.set_path_effects(effects)

    return effects
