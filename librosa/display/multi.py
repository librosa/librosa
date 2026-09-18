#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-plot orchestration
========================

This module offers utilities for orchestrating multiple synchronized
plots and highlighting specific data regions. It supports creating
complex visualizations with shared axes and unified legends, integrating
with tools like `multiplot` and `highlight`.
"""
# mypy: disable-error-code="attr-defined"

# Standard library imports for type checking and future compatibility
from __future__ import annotations
from typing import TYPE_CHECKING, cast

# Third-party imports for plotting and numerical operations
import colorsys
import copy
from itertools import cycle

import matplotlib.axes as mplaxes
import matplotlib.cm as cm
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np

# Core and utility imports for exception handling
from ..util.exceptions import ParameterError

# Type checking imports for development and static analysis
if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Sequence
    
    import cycler
    import matplotlib
    import matplotlib.figure
    import numpy.typing as npt
    from matplotlib.artist import Artist
    from matplotlib.typing import ColorType

    from .._typing import ArrayLike

# Module imports for image and signal display functions
from .image import specshow
from .signal import waveshow, wavebars


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

    *data : one or more `np.ndarray`
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
    >>> plt.gcf().legend()
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
