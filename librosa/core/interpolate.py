#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interpolation along time and frequency axes."""

import warnings
from typing import Literal

import numpy as np

from .._typing import _InterpKind
from ..util.exceptions import ParameterError
from ..util.utils import is_unique
from .convert import times_like

__all__ = ["interp_time", "interp_frequency"]


def _validate_coords(
    coords: np.ndarray,
    n_in: int,
    axis: int,
    *,
    name: str = "Coordinates",
) -> None:
    """Validate coordinate array shape and uniqueness along the target axis.

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array to validate.
    n_in : int
        Expected length along the target axis.
    axis : int
        The target axis.
    name : str
        Name of the coordinate variable for error/warning messages.

    Raises
    ------
    ParameterError
        If ``coords`` shape along ``axis`` does not match ``n_in``.
    """
    if coords.ndim == 1:
        if len(coords) != n_in:
            raise ParameterError(
                f"{name} length={len(coords)} does not match input shape={n_in} along axis={axis}"
            )
        if not is_unique(coords, axis=0):
            warnings.warn(
                f"{name} are not unique. This may produce incorrect interpolations.",
                stacklevel=3,
            )
    else:
        c_ax = axis if axis >= 0 else axis + coords.ndim
        if c_ax < 0 or c_ax >= coords.ndim or coords.shape[c_ax] != n_in:
            raise ParameterError(
                f"{name} shape={coords.shape} is incompatible with input shape length={n_in} along axis={axis}"
            )
        if not np.all(is_unique(coords, axis=c_ax)):
            warnings.warn(
                f"{name} are not unique. This may produce incorrect interpolations.",
                stacklevel=3,
            )


def _interp_slice(
    c: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    *,
    kind: _InterpKind,
    fill_value: float | tuple[float, float] | Literal["extrapolate"],
    assume_sorted: bool,
) -> np.ndarray:
    """Interpolate a single 1D slice using scipy.interpolate.interp1d.

    Parameters
    ----------
    c : np.ndarray
        Source coordinates (1D).
    v : np.ndarray
        Source values (1D).
    t : np.ndarray
        Target coordinates (1D).
    kind : str
        Interpolation kind.
    fill_value : float, tuple of float, or 'extrapolate'
        Fill value for out-of-bounds evaluation.
    assume_sorted : bool
        Whether coordinates are assumed sorted.

    Returns
    -------
    res : np.ndarray
        Interpolated values at target coordinates.
    """
    import scipy.interpolate

    idx = np.isfinite(c) & np.isfinite(v)
    if np.count_nonzero(idx) < 1:
        fill = fill_value if isinstance(fill_value, (int, float)) else 0.0
        return np.full(len(t), fill, dtype=v.dtype)

    c_clean = c[idx]
    v_clean = v[idx]

    # If coordinates are not asserted to be sorted, check for non-increasing or duplicate values
    if not assume_sorted and len(c_clean) > 1 and np.any(np.diff(c_clean) <= 0):
        c_clean, inv, counts = np.unique(c_clean, return_inverse=True, return_counts=True)
        if len(c_clean) < 2:
            fill = (
                v_clean[0]
                if len(c_clean) == 1
                else (fill_value if isinstance(fill_value, (int, float)) else 0.0)
            )
            return np.full(len(t), fill, dtype=v.dtype)
        v_clean = np.bincount(inv, weights=v_clean) / counts
        assume_sorted = True

    fill = fill_value if isinstance(fill_value, (int, float)) else 0.0
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            interp = scipy.interpolate.interp1d(
                c_clean,
                v_clean,
                bounds_error=False,
                fill_value=fill_value,
                assume_sorted=assume_sorted,
                kind=kind,
                copy=False,
            )
            res = interp(t)
    except (ValueError, FloatingPointError):
        return np.full(len(t), fill, dtype=v.dtype)

    return np.nan_to_num(res, copy=False, nan=fill)


def _interp_axis(
    x: np.ndarray,
    *,
    coords: np.ndarray,
    target_coords: np.ndarray,
    axis: int,
    kind: _InterpKind = "linear",
    fill_value: float | tuple[float, float] | Literal["extrapolate"] = 0.0,
    assume_sorted: bool = False,
    coord_name: str = "Coordinates",
) -> np.ndarray:
    """Interpolate an array along a specified axis.

    Parameters
    ----------
    x : np.ndarray
        The input array to interpolate.
    coords : np.ndarray
        Source coordinates corresponding to elements of ``x`` along ``axis``.
    target_coords : np.ndarray
        Target coordinates at which to evaluate the interpolator.
    axis : int
        The axis along which to interpolate.
    kind : str
        Interpolation type. See `scipy.interpolate.interp1d`.
    fill_value : float, tuple of float, or 'extrapolate'
        Fill value for out-of-bounds evaluation.
    assume_sorted : bool
        Whether coordinates are assumed sorted.
    coord_name : str
        Variable name for error messages.

    Returns
    -------
    x_out : np.ndarray
        The interpolated array.
    """
    if axis < -x.ndim or axis >= x.ndim:
        raise ParameterError(
            f"axis={axis} is out of bounds for input array with ndim={x.ndim}"
        )
    ax = axis if axis >= 0 else axis + x.ndim
    n_in = x.shape[ax]

    _validate_coords(coords, n_in, ax, name=coord_name)

    # Fast path: 1D source and 1D target coordinates with fully finite coordinates
    if (
        coords.ndim == 1
        and target_coords.ndim == 1
        and np.all(np.isfinite(coords))
    ):
        import scipy.interpolate

        interp = scipy.interpolate.interp1d(
            coords,
            x,
            axis=ax,
            kind=kind,
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=assume_sorted,
            copy=False,
        )
        res = interp(target_coords)
        fill = fill_value if isinstance(fill_value, (int, float)) else 0.0
        return np.nan_to_num(res, copy=False, nan=fill)

    # General path: multidimensional coordinates, multidimensional targets, or NaNs
    c_ndim = coords.ndim if coords.ndim > 1 else 0
    t_ndim = target_coords.ndim if target_coords.ndim > 1 else 0
    max_ndim = max(x.ndim, c_ndim, t_ndim)

    neg_axis = axis if axis < 0 else axis - x.ndim
    norm_ax = neg_axis + max_ndim

    x_aligned = x.reshape([1] * (max_ndim - x.ndim) + list(x.shape))
    if coords.ndim > 1:
        coords_aligned = coords.reshape([1] * (max_ndim - coords.ndim) + list(coords.shape))
    else:
        coords_aligned = coords

    if target_coords.ndim > 1:
        target_aligned = target_coords.reshape([1] * (max_ndim - target_coords.ndim) + list(target_coords.shape))
        n_out = target_aligned.shape[norm_ax]
    else:
        target_aligned = target_coords
        n_out = len(target_aligned)

    bc_x = list(x_aligned.shape)
    bc_x[norm_ax] = 1

    if coords_aligned.ndim == 1:
        c_shape = [n_in if i == norm_ax else 1 for i in range(max_ndim)]
        bc_c = list(c_shape)
        bc_c[norm_ax] = 1
    else:
        bc_c = list(coords_aligned.shape)
        bc_c[norm_ax] = 1

    shapes = [bc_x, bc_c]
    if target_aligned.ndim > 1:
        bc_t = list(target_aligned.shape)
        bc_t[norm_ax] = 1
        shapes.append(bc_t)

    try:
        lead_shape = np.broadcast_shapes(*shapes)
    except ValueError as exc:
        raise ParameterError(
            f"Cannot broadcast array shapes along non-interpolation axes: "
            f"x={x.shape}, coords={coords.shape}, target={target_coords.shape}"
        ) from exc

    full_in = list(lead_shape)
    full_in[norm_ax] = n_in
    full_out = list(lead_shape)
    full_out[norm_ax] = n_out

    x_b = np.broadcast_to(x_aligned, full_in)
    if coords_aligned.ndim == 1:
        c_shape = [n_in if i == norm_ax else 1 for i in range(len(full_in))]
        c_b = np.broadcast_to(coords_aligned.reshape(c_shape), full_in)
    else:
        c_b = np.broadcast_to(coords_aligned, full_in)

    if target_aligned.ndim == 1:
        t_shape = [n_out if i == norm_ax else 1 for i in range(len(full_out))]
        t_b = np.broadcast_to(target_aligned.reshape(t_shape), full_out)
    else:
        t_b = np.broadcast_to(target_aligned, full_out)

    x_s = np.swapaxes(x_b, norm_ax, -1)
    c_s = np.swapaxes(c_b, norm_ax, -1)
    t_s = np.swapaxes(t_b, norm_ax, -1)

    def _curried_slice(_c: np.ndarray, _v: np.ndarray, _t: np.ndarray) -> np.ndarray:
        return _interp_slice(
            _c,
            _v,
            _t,
            kind=kind,
            fill_value=fill_value,
            assume_sorted=assume_sorted,
        )

    vfunc = np.vectorize(_curried_slice, signature="(n),(n),(m)->(m)")
    res_s = vfunc(c_s, x_s, t_s)
    return np.swapaxes(res_s, norm_ax, -1)


def interp_time(
    x: np.ndarray,
    *,
    target_times: np.ndarray,
    times: np.ndarray | None = None,
    sr: float = 22050,
    hop_length: int = 512,
    kind: _InterpKind = "linear",
    fill_value: float | tuple[float, float] | Literal["extrapolate"] = 0.0,
    axis: int = -1,
    assume_sorted: bool = False,
) -> np.ndarray:
    """Interpolate an array along a time axis.

    Given an input array representing time-series features (such as an STFT,
    spectrogram, novelty curve, or chromagram), this function interpolates the
    values along the specified time axis onto a new grid of target times.

    Times can be provided explicitly via ``times``, or inferred automatically
    from ``sr`` and ``hop_length`` (using `times_like`).

    Parameters
    ----------
    x : np.ndarray
        The input array to interpolate.
    target_times : np.ndarray
        The target time coordinates (in seconds) at which to evaluate the
        interpolator.
        If 1-dimensional, all slices along other dimensions are evaluated
        at these times.
        If multi-dimensional, the shape along ``axis`` must match the target
        length, and other dimensions must be broadcast-compatible with ``x``.
    times : np.ndarray, optional
        The time coordinates corresponding to elements of ``x`` along ``axis``.
        If not provided, times are inferred using `times_like(x, sr=sr, hop_length=hop_length, axis=axis)`.
        Can be 1-dimensional with length ``x.shape[axis]``, or multi-dimensional
        with shape matching or broadcastable to ``x``.
    sr : number > 0 [scalar]
        Audio sampling rate. Only used if ``times is None``.
    hop_length : int > 0 [scalar]
        Number of samples between successive frames. Only used if ``times is None``.
    kind : str
        Interpolation type. See `scipy.interpolate.interp1d`.
        Common choices: ``'linear'``, ``'nearest'``, ``'cubic'``.
    fill_value : float, tuple of float, or 'extrapolate'
        Value to fill when evaluating outside the observed time bounds.
        Defaults to 0.0.
    axis : int
        The axis along which to interpolate time. Defaults to -1.
    assume_sorted : bool
        If True, the source time coordinates are assumed to be monotonically
        increasing. Defaults to False. If ``times is None``, this is automatically
        treated as True.

    Returns
    -------
    x_interp : np.ndarray
        The interpolated array. Dimension ``axis`` will have length matching
        the target time dimension.

    See Also
    --------
    interp_frequency
    times_like
    scipy.interpolate.interp1d

    Examples
    --------
    Resample a spectrogram in time to a fixed 100 Hz frame rate:

    >>> y, sr = librosa.loadx("trumpet", duration=1.0)
    >>> S = np.abs(librosa.stft(y))
    >>> times = librosa.times_like(S, sr=sr)
    >>> target_times = np.arange(0, times[-1], 0.01)
    >>> S_resampled = librosa.interp_time(S, target_times=target_times, sr=sr)
    >>> S_resampled.shape
    (1025, 96)

    Resample with reassigned times onto a regular time grid:

    >>> freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr)
    >>> t_grid = np.linspace(0, 1.0, 100)
    >>> S_locked = librosa.interp_time(mags, target_times=t_grid, times=times)
    >>> S_locked.shape
    (1025, 100)
    """
    if times is None:
        times = times_like(x, sr=sr, hop_length=hop_length, axis=axis)
        assume_sorted = True

    return _interp_axis(
        x,
        coords=times,
        target_coords=target_times,
        axis=axis,
        kind=kind,
        fill_value=fill_value,
        assume_sorted=assume_sorted,
        coord_name="Times",
    )


def interp_frequency(
    x: np.ndarray,
    *,
    target_freqs: np.ndarray,
    freqs: np.ndarray,
    kind: _InterpKind = "linear",
    fill_value: float | tuple[float, float] | Literal["extrapolate"] = 0.0,
    axis: int = -2,
    assume_sorted: bool = False,
) -> np.ndarray:
    """Interpolate an array along a frequency axis.

    Given a frequency-based energy representation (such as an STFT spectrogram,
    tempogram, or reassigned spectrogram), this function interpolates the values
    along the specified frequency axis onto a new grid of target frequencies.

    This function supports both uniform frequency representations and
    time-varying frequency grids (e.g. from `reassigned_spectrogram`).

    Parameters
    ----------
    x : np.ndarray
        The input array to interpolate.
    target_freqs : np.ndarray
        The target frequency coordinates at which to evaluate the interpolator.
        If 1-dimensional, all slices along other dimensions are evaluated
        at these frequencies.
        If multi-dimensional, the shape along ``axis`` must match the target
        length, and other dimensions must be broadcast-compatible with ``x``.
    freqs : np.ndarray
        The frequency values corresponding to ``x``'s elements along ``axis``.
        Can be 1-dimensional with length ``x.shape[axis]`` (e.g., from
        `fft_frequencies`, `cqt_frequencies`, or `tempo_frequencies`), or
        multi-dimensional with shape matching or broadcastable to ``x``
        (e.g., instantaneous frequencies from `reassigned_spectrogram`).
    kind : str
        Interpolation type. See `scipy.interpolate.interp1d`.
        Common choices: ``'linear'``, ``'nearest'``, ``'cubic'``.
    fill_value : float, tuple of float, or 'extrapolate'
        Value to fill when evaluating outside the observed frequency bounds.
        Defaults to 0.0.
    axis : int
        The axis along which to interpolate frequency. Defaults to -2.
    assume_sorted : bool
        If True, the source frequency coordinates are assumed to be monotonically
        increasing. Defaults to False.

    Returns
    -------
    x_interp : np.ndarray
        The interpolated array. Dimension ``axis`` will have length matching
        the target frequency dimension.

    See Also
    --------
    interp_time
    interp_harmonics
    f0_harmonics
    scipy.interpolate.interp1d

    Examples
    --------
    Interpolate an STFT spectrogram onto a logarithmic CQT frequency grid:

    >>> y, sr = librosa.loadx("trumpet", duration=1.0)
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> cqt_freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz("C1"))
    >>> S_cqt_grid = librosa.interp_frequency(S, target_freqs=cqt_freqs, freqs=fft_freqs)
    >>> S_cqt_grid.shape
    (84, 44)

    Lock a reassigned spectrogram to a uniform frequency grid:

    >>> freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr)
    >>> f_grid = np.linspace(0, sr / 2, 512)
    >>> S_locked = librosa.interp_frequency(mags, target_freqs=f_grid, freqs=freqs)
    >>> S_locked.shape
    (512, 44)
    """
    return _interp_axis(
        x,
        coords=freqs,
        target_coords=target_freqs,
        axis=axis,
        kind=kind,
        fill_value=fill_value,
        assume_sorted=assume_sorted,
        coord_name="Frequencies",
    )
