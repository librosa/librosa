#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image and spectral visualization
================================

This module provides functionalities for visualizing 2D spectral data
and matrices as images. It forms the core of `specshow`, handling the
coordinate grids, color mapping, and rendering of spectrograms and
chromagrams.
"""

# Standard library imports for type checking and future compatibility
from __future__ import annotations
from typing import TYPE_CHECKING
import re
import warnings

# Third-party imports for plotting and numerical operations
import matplotlib.axes as mplaxes
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import colormaps as mcm
import numpy as np

# Core and utility imports for audio processing
from .. import core
from ..util.exceptions import ParameterError
 
# Image visualization utilities for spectral and 2D matrix displays
if TYPE_CHECKING:
    from typing import Any, Callable
    from matplotlib.collections import QuadMesh

# Module imports for image formatting and coordinate handling
from .formatting import (
    _check_axes,
    _coord_chroma,
    _coord_cqt_hz,
    _coord_fft_hz,
    _coord_fourier_tempo,
    _coord_mel_hz,
    _coord_n,
    _coord_tempo,
    _coord_time,
    _coord_vqt_hz,
    _decorate_axis,
    _same_axes,
    _scale_axes,
    infer_cmap,
    cmap
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

    For a detailed overview of display functionality, see the
    :ref:`Display and visualization tutorial <tutorial-display-index>`.

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
    y_coords = _mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = _mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    # Parse the value scale into a normalizer and possibly a colormap
    data, norm_cmap = _scale_data(
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
            "Trying to display complex-valued input. Showing magnitude instead.",
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

        if isinstance(cmap_bool, colors.Colormap):
            is_boolean_cmap = kwargs["cmap"] == cmap_bool
        else:
            is_boolean_cmap = kwargs["cmap"] == mcm.get(cmap_bool, None)
        # Harden this check to ensure that it only hits when
        # data is really boolean
        is_boolean_cmap &= (data.dtype.kind == "b")

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
        elif is_boolean_cmap:
            # If we have an inferred boolean colormap, use a boundary norm
            # But only if the user didn't also set their own normalizer
            kwargs.setdefault(
                "norm",
                colors.BoundaryNorm(
                    boundaries=[0, 0.5, 1], ncolors=kwargs["cmap"].N
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

    axes = _check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)

    _set_current_image(ax, out)

    # Set up axis scaling
    _scale_axes(axes, x_axis, "x", tempo_min=tempo_min, tempo_max=tempo_max)
    _scale_axes(axes, y_axis, "y", tempo_min=tempo_min, tempo_max=tempo_max)

    # Construct tickers and locators
    _decorate_axis(
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
    _decorate_axis(
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
    if _same_axes(x_axis, y_axis, axes.get_xlim(), axes.get_ylim()) and auto_aspect:
        axes.set_aspect("equal")

    return out


def _set_current_image(ax, img):
    """
    Set the current image when working in pyplot mode.

    If the provided ``ax`` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    """
    if ax is None:
        plt.sci(img)


def _mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""
    if coords is not None:
        if len(coords) not in (n, n + 1):
            raise ParameterError(
                f"Coordinate shape mismatch: {len(coords)}!={n} or {n}+1"
            )
        return coords

    coord_map: dict[str | None, Callable[..., np.ndarray]] = {
        "linear": _coord_fft_hz,
        "fft": _coord_fft_hz,
        "fft_note": _coord_fft_hz,
        "fft_svara": _coord_fft_hz,
        "hz": _coord_fft_hz,
        "oct3": _coord_fft_hz,
        "log_oct3": _coord_fft_hz,
        "log": _coord_fft_hz,
        "mel": _coord_mel_hz,
        "mel_oct3": _coord_mel_hz,
        "cqt": _coord_cqt_hz,
        "cqt_hz": _coord_cqt_hz,
        "cqt_note": _coord_cqt_hz,
        "cqt_svara": _coord_cqt_hz,
        "cqt_oct3": _coord_cqt_hz,
        "vqt_fjs": _coord_vqt_hz,
        "vqt_hz": _coord_vqt_hz,
        "vqt_note": _coord_vqt_hz,
        "vqt_oct3": _coord_vqt_hz,
        "chroma": _coord_chroma,
        "chroma_c": _coord_chroma,
        "chroma_h": _coord_chroma,
        "chroma_fjs": _coord_n,  # We can't use a 12-normalized tick locator here
        "time": _coord_time,
        "h": _coord_time,
        "m": _coord_time,
        "s": _coord_time,
        "ms": _coord_time,
        "lag": _coord_time,
        "lag_h": _coord_time,
        "lag_m": _coord_time,
        "lag_s": _coord_time,
        "lag_ms": _coord_time,
        "tonnetz": _coord_n,
        "off": _coord_n,
        "tempo": _coord_tempo,
        "fourier_tempo": _coord_fourier_tempo,
        "frames": _coord_n,
        None: _coord_n,
    }

    if ax_type not in coord_map:
        raise ParameterError(f"Unknown axis type: {ax_type}")
    return coord_map[ax_type](n, **kwargs)


def _scale_data(data, *, vscale, top_db, x_coords, y_coords, cmap_seq, cmap_cyclic):
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
        These should be constructed using the `_mesh_coords` function.
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
        _mode, scale_type, ref_ = _parse_vscale(vscale)
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


def _parse_vscale(vscale: str) -> tuple[str, str, float | str | None]:
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
