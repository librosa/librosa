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

    cmap
    AdaptiveWaveplot

"""
from __future__ import annotations
from itertools import product
import warnings

import numpy as np
from matplotlib import colormaps as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt

from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co

if TYPE_CHECKING:
    import matplotlib
    from matplotlib.collections import QuadMesh, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MplPath
    from matplotlib.markers import MarkerStyle
    from matplotlib.colors import Colormap


__all__ = [
    "specshow",
    "waveshow",
    "cmap",
    "TimeFormatter",
    "NoteFormatter",
    "FJSFormatter",
    "LogHzFormatter",
    "ChromaFormatter",
    "ChromaSvaraFormatter",
    "ChromaFJSFormatter",
    "TonnetzFormatter",
    "AdaptiveWaveplot",
]

# mypy: disable-error-code="attr-defined"


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

    Manually set the physical time unit of the x-axis to milliseconds

    >>> times = np.arange(100)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(unit='ms'))
    >>> ax.set(xlabel='Time (ms)')

    For lag plots

    >>> times = np.arange(60)
    >>> values = np.random.randn(len(times))
    >>> fig, ax = plt.subplots()
    >>> ax.plot(times, values)
    >>> ax.xaxis.set_major_formatter(librosa.display.TimeFormatter(lag=True))
    >>> ax.set(xlabel='Lag')
    """

    def __init__(self, lag: bool = False, unit: Optional[str] = None):
        if unit not in ["h", "m", "s", "ms", None]:
            raise ParameterError(f"Unknown time unit: {unit}")

        self.unit = unit
        self.lag = lag

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Return the time format as pos"""
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
        elif self.unit == None and (vmax - vmin >= 1):
            s = f"{value:.2g}"
        elif self.unit == "ms":
            s = "{:.3g}".format(value * 1000)
        elif self.unit == None and (vmax - vmin < 1):
            s = f"{value:.3f}"

        return f"{sign:s}{s:s}"


class NoteFormatter(mplticker.Formatter):
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

    def __init__(
        self,
        octave: bool = True,
        major: bool = True,
        key: str = "C:maj",
        unicode: bool = True,
    ):
        self.octave = octave
        self.major = major
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        cents = vmax <= 2 * max(1, vmin)

        return core.hz_to_note(
            x, octave=self.octave, cents=cents, key=self.key, unicode=self.unicode
        )


class SvaraFormatter(mplticker.Formatter):
    """Ticker formatter for Svara

    Parameters
    ----------
    octave : bool
        If ``True``, display the octave number along with the note name.

        Otherwise, only show the note name (and cent deviation)

    major : bool
        If ``True``, ticks are always labeled.

        If ``False``, ticks are only labeled if the span is less than 2 octaves

    Sa : number > 0
        Frequency (in Hz) of Sa

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
        mela: Optional[Union[str, int]] = None,
        unicode: bool = True,
    ):
        if Sa is None:
            raise ParameterError(
                "Sa frequency is required for svara display formatting"
            )

        self.Sa = Sa
        self.octave = octave
        self.major = major
        self.abbr = abbr
        self.mela = mela
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

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


class FJSFormatter(mplticker.Formatter):
    """Ticker formatter for Functional Just System (FJS) notation

    Parameters
    ----------
    fmin : float
        The unison frequency for this axis

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

    def __init__(
        self,
        *,
        fmin: int,
        n_bins: int,
        bins_per_octave: int,
        intervals: Union[str, Collection[float]],
        major: bool = True,
        unison: Optional[str] = None,
        unicode: bool = True,
    ):
        self.fmin = fmin
        self.major = major
        self.unison = unison
        self.unicode = unicode
        self.intervals = intervals
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.frequencies_ = core.interval_frequencies(
            n_bins, fmin=fmin, intervals=intervals, bins_per_octave=bins_per_octave
        )

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ""

        # Only use cent precision if our vspan is less than an octave
        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        # Map the given frequency to the nearest JI interval
        idx = util.match_events(np.atleast_1d(x), self.frequencies_)[0]

        label: str = core.hz_to_fjs(
            self.frequencies_[idx],
            fmin=self.fmin,
            unison=self.unison,
            unicode=self.unicode,
        )
        return label


class LogHzFormatter(mplticker.Formatter):
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
        self.major = major

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Apply the formatter to position"""
        if x <= 0:
            return ""

        vmin, vmax = self.axis.get_view_interval()

        if not self.major and vmax > 4 * max(1, vmin):
            return ""

        return f"{x:g}"


class ChromaFormatter(mplticker.Formatter):
    """A formatter for chroma axes

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

    def __init__(self, key: str = "C:maj", unicode: bool = True):
        self.key = key
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Format for chroma positions"""
        return core.midi_to_note(
            int(x), octave=False, cents=False, key=self.key, unicode=self.unicode
        )


class ChromaSvaraFormatter(mplticker.Formatter):
    """A formatter for chroma axes with svara instead of notes.

    If mela is given, Carnatic svara names will be used.

    Otherwise, Hindustani svara names will be used.

    If `Sa` is not given, it will default to 0 (equivalent to `C`).

    See Also
    --------
    ChromaFormatter

    """

    def __init__(
        self,
        Sa: Optional[float] = None,
        mela: Optional[Union[int, str]] = None,
        abbr: bool = True,
        unicode: bool = True,
    ):
        if Sa is None:
            Sa = 0
        self.Sa = Sa
        self.mela = mela
        self.abbr = abbr
        self.unicode = unicode

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
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

    def __init__(
        self,
        *,
        intervals: Union[str, Collection[float]],
        unison: str = "C",
        unicode: bool = True,
        bins_per_octave: Optional[int] = None,
    ):
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
            self.bins_per_octave: int = bins_per_octave
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

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
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

    def __call__(self, x: float, pos: Optional[int] = None) -> str:
        """Format for tonnetz positions"""
        return [r"5$_x$", r"5$_y$", r"m3$_x$", r"m3$_y$", r"M3$_x$", r"M3$_y$"][int(x)]


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

    See Also
    --------
    waveshow
    """

    def __init__(
        self,
        times: np.ndarray,
        y: np.ndarray,
        steps: Line2D,
        envelope: PolyCollection,
        sr: float = 22050,
        max_samples: int = 11025,
        transpose: bool = False,
    ):
        self.times = times
        self.samples = y
        self.steps = steps
        self.envelope = envelope
        self.sr = sr
        self.max_samples = max_samples
        self.transpose = transpose
        self.cid: Optional[int] = None
        self.ax: Optional[mplaxes.Axes] = None

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
        self.ax = ax
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
        if self.ax:
            self.ax.callbacks.disconnect(self.cid)
            self.cid = None
            if strict:
                self.ax = None

    def update(self, ax: mplaxes.Axes) -> None:
        """Update the matplotlib display according to the current viewport limits.

        This is a callback function, and should not be used directly.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to update
        """
        lims = ax.viewLim

        if self.transpose:
            dim = lims.height * self.sr
            start, end = lims.y0, lims.y1
            xdata, ydata = self.samples, self.times
            data = self.steps.get_ydata()
        else:
            dim = lims.width * self.sr
            start, end = lims.x0, lims.x1
            xdata, ydata = self.times, self.samples
            data = self.steps.get_xdata()
        # Does our width cover fewer than max_samples?
        # If so, then use the sample-based plot
        if dim <= self.max_samples:
            self.envelope.set_visible(False)
            self.steps.set_visible(True)

            # Now check our viewport
            if start <= data[0] or end >= data[-1]:
                # Viewport expands beyond current data in steps; update
                # we want to cover a window of self.max_samples centered on the current viewport
                midpoint_time = (start + end) / 2
                idx_start = np.searchsorted(
                    self.times, midpoint_time - 0.5 * self.max_samples / self.sr
                )
                self.steps.set_data(
                    xdata[idx_start : idx_start + self.max_samples],
                    ydata[idx_start : idx_start + self.max_samples],
                )
        else:
            # Otherwise, use the envelope plot
            self.envelope.set_visible(True)
            self.steps.set_visible(False)

        ax.figure.canvas.draw_idle()


def cmap(
    data: np.ndarray,
    *,
    robust: bool = True,
    cmap_seq: str = "magma",
    cmap_bool: str = "gray_r",
    cmap_div: str = "coolwarm",
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
    cmap_seq : str
        The sequential colormap name
    cmap_bool : str
        The boolean colormap name
    cmap_div : str
        The diverging colormap name

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap to use for ``data``

    See Also
    --------
    matplotlib.pyplot.colormaps
    """
    data = np.atleast_1d(data)

    if data.dtype == "bool":
        return mcm[cmap_bool]

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    min_val, max_val = np.percentile(data, [min_p, max_p])

    if min_val >= 0 or max_val <= 0:
        return mcm[cmap_seq]

    return mcm[cmap_div]


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
)
_freq_ax_types = (
    "linear",
    "fft",
    "hz",
    "fft_note",
    "fft_svara",
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
    "log",
    "tonnetz",
    "frames",
)

_AXIS_COMPAT = set(
    [(t, t) for t in _misc_ax_types]
    + [t for t in product(_chroma_ax_types, _chroma_ax_types)]
    + [t for t in product(_cqt_ax_types, _cqt_ax_types)]
    + [t for t in product(_freq_ax_types, _freq_ax_types)]
    + [t for t in product(_time_ax_types, _time_ax_types)]
    + [t for t in product(_lag_ax_types, _lag_ax_types)]
)


def specshow(
    data: np.ndarray,
    *,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    x_axis: Optional[str] = None,
    y_axis: Optional[str] = None,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    tempo_min: Optional[float] = 16,
    tempo_max: Optional[float] = 480,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
    key: str = "C:maj",
    Sa: Optional[Union[float, int]] = None,
    mela: Optional[Union[str, int]] = None,
    thaat: Optional[str] = None,
    auto_aspect: bool = True,
    htk: bool = False,
    unicode: bool = True,
    intervals: Optional[Union[str, np.ndarray]] = None,
    unison: Optional[str] = None,
    ax: Optional[mplaxes.Axes] = None,
    **kwargs: Any,
) -> QuadMesh:
    """Display a spectrogram/chromagram/cqt/etc.

    For a detailed overview of this function, see :ref:`sphx_glr_auto_examples_plot_display.py`

    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Matrix to display (e.g., spectrogram)

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

    x_axis, y_axis : None or str
        Range for the x- and y-axes.

        Valid types are:

        - None, 'none', or 'off' : no axis decoration is displayed.

        Frequency types:

        - 'linear', 'fft', 'hz' : frequency range is determined by
          the FFT window and sampling rate.
        - 'log' : the spectrum is displayed on a log scale.
        - 'fft_note': the spectrum is displayed on a log scale with pitches marked.
        - 'fft_svara': the spectrum is displayed on a log scale with svara marked.
        - 'mel' : frequencies are determined by the mel scale.
        - 'cqt_hz' : frequencies are determined by the CQT scale.
        - 'cqt_note' : pitches are determined by the CQT scale.
        - 'cqt_svara' : like `cqt_note` but using Hindustani or Carnatic svara
        - 'vqt_fjs' : like `cqt_note` but using Functional Just System (FJS)
          notation.  This requires a just intonation-based variable-Q
          transform representation.

        All frequency types are plotted in units of Hz.

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

    x_coords, y_coords : np.ndarray [shape=data.shape[0 or 1]]
        Optional positioning coordinates of the input data.
        These can be use to explicitly set the location of each
        element ``data[i, j]``, e.g., for displaying beat-synchronous
        features in natural time coordinates.

        If not provided, they are inferred from ``x_axis`` and ``y_axis``.

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

    intervals : str or array of floats in [1, 2), optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the interval specification.

        See `core.interval_frequencies` for a description of supported values.

    unison : str, optional
        If using an FJS notation (`chroma_fjs`, `vqt_fjs`), the pitch name of the unison
        interval.  If not provided, it will be inferred from `fmin` (for VQT display) or
        assumed as `'C'` (for chroma display).

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

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    **kwargs : additional keyword arguments
        Arguments passed through to `matplotlib.pyplot.pcolormesh`.

        By default, the following options are set:

            - ``rasterized=True``
            - ``shading='auto'``
            - ``edgecolors='None'``

        The ``cmap`` option if not provided, is inferred from data automatically.
        Set ``cmap=None`` to use matplotlib's default colormap.

    Returns
    -------
    colormesh : `matplotlib.collections.QuadMesh`
        The color mesh object produced by `matplotlib.pyplot.pcolormesh`

    See Also
    --------
    cmap : Automatic colormap detection
    matplotlib.pyplot.pcolormesh

    Examples
    --------
    Visualize an STFT power spectrum using default parameters

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=15)
    >>> fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    >>> img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
    ...                                sr=sr, ax=ax[0])
    >>> ax[0].set(title='Linear-frequency power spectrogram')
    >>> ax[0].label_outer()

    Or on a logarithmic scale, and using a larger hop

    >>> hop_length = 1024
    >>> D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)),
    ...                             ref=np.max)
    >>> librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length,
    ...                          x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-frequency power spectrogram')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn(
            "Trying to display complex-valued input. " "Showing magnitude instead.",
            stacklevel=2,
        )
        data = np.abs(data)

    kwargs.setdefault("cmap", cmap(data))
    kwargs.setdefault("rasterized", True)
    kwargs.setdefault("edgecolors", "None")
    kwargs.setdefault("shading", "auto")

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

    coord_map: Dict[Optional[str], Callable[..., np.ndarray]] = {
        "linear": __coord_fft_hz,
        "fft": __coord_fft_hz,
        "fft_note": __coord_fft_hz,
        "fft_svara": __coord_fft_hz,
        "hz": __coord_fft_hz,
        "log": __coord_fft_hz,
        "mel": __coord_mel_hz,
        "cqt": __coord_cqt_hz,
        "cqt_hz": __coord_cqt_hz,
        "cqt_note": __coord_cqt_hz,
        "cqt_svara": __coord_cqt_hz,
        "vqt_fjs": __coord_vqt_hz,
        "vqt_hz": __coord_vqt_hz,
        "vqt_note": __coord_vqt_hz,
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


def __check_axes(axes: Optional[mplaxes.Axes]) -> mplaxes.Axes:
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
    if ax_type == "mel":
        mode = "symlog"
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type in [
        "cqt",
        "cqt_hz",
        "cqt_note",
        "cqt_svara",
        "vqt_hz",
        "vqt_note",
        "vqt_fjs",
    ]:
        mode = "log"
        kwargs[base] = 2

    elif ax_type in ["log", "fft_note", "fft_svara"]:
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
        axis.set_major_locator(mplticker.FixedLocator(np.arange(6)))
        axis.set_label_text("Tonnetz")

    elif ax_type == "chroma":
        axis.set_major_formatter(ChromaFormatter(key=key, unicode=unicode))
        degrees = core.key_to_degrees(key)
        axis.set_major_locator(
            mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
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
            mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
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
            mplticker.FixedLocator(np.add.outer(12 * np.arange(10), degrees).ravel())
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

        axis.set_major_locator(mplticker.FixedLocator(degrees))
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
            fmin = core.note_to_hz("C1")
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
                )
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
                ),
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
                ),
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
            mplticker.LogLocator(base=2.0, subs=2.0 ** (np.arange(1, 12) / 12.0))
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

    elif ax_type in ["frames"]:
        axis.set_label_text("Frames")

    elif ax_type in ["off", "none", None]:
        axis.set_label_text("")
        axis.set_ticks([])

    else:
        raise ParameterError(f"Unsupported axis type: {ax_type}")


def __coord_fft_hz(
    n: int, sr: float = 22050, n_fft: Optional[int] = None, **_kwargs: Any
) -> np.ndarray:
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
    return basis


def __coord_mel_hz(
    n: int,
    fmin: Optional[float] = 0.0,
    fmax: Optional[float] = None,
    sr: float = 22050,
    htk: bool = False,
    **_kwargs: Any,
) -> np.ndarray:
    """Get the frequencies for Mel bins"""
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = 0.5 * sr

    basis = core.mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    return basis


def __coord_cqt_hz(
    n: int,
    fmin: Optional[_FloatLike_co] = None,
    bins_per_octave: int = 12,
    sr: float = 22050,
    **_kwargs: Any,
) -> np.ndarray:
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
    fmin: Optional[_FloatLike_co] = None,
    bins_per_octave: int = 12,
    sr: float = 22050,
    intervals: Optional[Union[str, Collection[float]]] = None,
    unison: Optional[str] = None,
    **_kwargs: Any,
) -> np.ndarray:
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
    win_length: Optional[int] = None,
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


def waveshow(
    y: np.ndarray,
    *,
    sr: float = 22050,
    max_points: int = 11025,
    axis: Optional[str] = "time",
    offset: float = 0.0,
    marker: Union[str, MplPath, MarkerStyle] = "",
    where: str = "post",
    label: Optional[str] = None,
    transpose: bool = False,
    ax: Optional[mplaxes.Axes] = None,
    x_axis: Optional[Union[str, Deprecated]] = Deprecated(),
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

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : number > 0 [scalar]
        sampling rate of ``y`` (samples per second)

    max_points : positive integer
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

    x_axis : Deprecated
        Equivalent to `axis` parameter, included for backward compatibility.

        .. warning:: This parameter is deprecated as of 0.10.0 and
            will be removed in 1.0.  Use `axis=` instead going forward.

    ax : matplotlib.axes.Axes or None
        Axes to plot on instead of the default `plt.gca()`.

    offset : float
        Horizontal offset (in seconds) to start the waveform plot

    marker : string
        Marker symbol to use for sample values. (default: no markers)

        See Also: `matplotlib.markers`.

    where : string, {'pre', 'mid', 'post'}
        This setting determines how both waveform and envelope plots interpolate
        between observations.

        See `matplotlib.pyplot.step` for details.

        Default: 'post'

    label : string [optional]
        The label string applied to this plot.
        Note that the label

    transpose : bool
        If `True`, display the wave vertically instead of horizontally.

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
    AdaptiveWaveplot
    matplotlib.pyplot.step
    matplotlib.pyplot.fill_between
    matplotlib.pyplot.fill_betweenx
    matplotlib.markers

    Examples
    --------
    Plot a monophonic waveform with an envelope view

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[0])
    >>> ax[0].set(title='Envelope view, mono')
    >>> ax[0].label_outer()

    Or a stereo waveform

    >>> y, sr = librosa.load(librosa.ex('choice', hq=True), mono=False, duration=10)
    >>> librosa.display.waveshow(y, sr=sr, ax=ax[1])
    >>> ax[1].set(title='Envelope view, stereo')
    >>> ax[1].label_outer()

    Or harmonic and percussive components with transparency

    >>> y, sr = librosa.load(librosa.ex('choice'), duration=10)
    >>> y_harm, y_perc = librosa.effects.hpss(y)
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax[2], label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax[2], label='Percussive')
    >>> ax[2].set(title='Multiple waveforms')
    >>> ax[2].legend()

    Zooming in on a plot to show raw sample values

    >>> fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    >>> ax.set(xlim=[6.0, 6.01], title='Sample view', ylim=[-0.2, 0.2])
    >>> librosa.display.waveshow(y, sr=sr, ax=ax, marker='.', label='Full signal')
    >>> librosa.display.waveshow(y_harm, sr=sr, alpha=0.5, ax=ax2, label='Harmonic')
    >>> librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
    >>> ax.label_outer()
    >>> ax.legend()
    >>> ax2.legend()

    Plotting a transposed wave along with a self-similarity matrix

    >>> fig, ax = plt.subplot_mosaic("hSSS;hSSS;hSSS;.vvv")
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
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
    """
    util.valid_audio(y, mono=False)

    # Pad an extra channel dimension, if necessary
    if y.ndim == 1:
        y = y[np.newaxis, :]

    if max_points <= 0:
        raise ParameterError(f"max_points={max_points} must be strictly positive")

    # Create the adaptive drawing object
    axes = __check_axes(ax)

    # Handle the x_axis->axis rename deprecation
    axis = rename_kw(
        old_name="x_axis",
        old_value=x_axis,
        new_name="axis",
        new_value=axis,
        version_deprecated="0.10.0",
        version_removed="1.0",
    )

    # Reduce by envelope calculation
    # this choice of hop ensures that the envelope has at most max_points values
    hop_length = max(1, y.shape[-1] // max_points)
    y_env = __envelope(y, hop_length)

    # Split the envelope into top and bottom
    y_bottom, y_top = -y_env[-1], y_env[0]

    times = offset + core.times_like(y, sr=sr, hop_length=1)

    # Only plot up to max_points worth of data here
    xdata, ydata = times[:max_points], y[0, :max_points]
    filler = axes.fill_between
    signal = "xlim_changed"
    dec_axis = axes.xaxis
    if transpose:
        ydata, xdata = xdata, ydata
        filler = axes.fill_betweenx
        signal = "ylim_changed"
        dec_axis = axes.yaxis

    (steps,) = axes.step(xdata, ydata, marker=marker, where=where, **kwargs)

    # Pull color property from the steps object, if we don't already have it
    if "color" not in kwargs:
        kwargs.setdefault("color", steps.get_color())

    envelope = filler(
        times[: len(y_top) * hop_length : hop_length],
        y_bottom,
        y_top,
        step=where,
        label=label,
        **kwargs,
    )
    adaptor = AdaptiveWaveplot(
        times, y[0], steps, envelope, sr=sr, max_samples=max_points, transpose=transpose
    )

    adaptor.connect(axes, signal=signal)

    # Force an initial update to ensure the state is consistent
    adaptor.update(axes)

    # Construct tickers and locators
    __decorate_axis(dec_axis, axis)

    return adaptor
