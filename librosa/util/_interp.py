#!/usr/bin/env python
"""Internal 1D interpolation utilities built on NumPy searchsorted and SciPy splines."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..exceptions import ParameterShapeError, ParameterValueError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from numpy.typing import ArrayLike, DTypeLike

    from .._typing import _InterpKind

_STEP_AND_LINEAR_KINDS = frozenset({"linear", "nearest", "nearest-up", "zero", "previous", "next"})
_SPLINE_ORDERS: dict[str, int] = {"slinear": 1, "quadratic": 2, "cubic": 3}


def _resolve_fill_value(
    fill_value: float | tuple[float, float] | Literal["extrapolate"] | ArrayLike,
    bounds_error: bool | None,
) -> tuple[bool, bool, Any, Any]:
    """Resolve out-of-bounds behavior into (extrapolate, bounds_error, fill_below, fill_above)."""
    if bounds_error is None:
        bounds_error = fill_value != "extrapolate"

    if isinstance(fill_value, str):
        if fill_value != "extrapolate":
            raise ParameterValueError(f"Unsupported string fill_value={fill_value!r}")
        if bounds_error:
            raise ParameterValueError("Cannot extrapolate and raise at the same time.")
        return True, False, np.nan, np.nan

    if isinstance(fill_value, (tuple, list)):
        if len(fill_value) != 2:
            raise ParameterValueError(
                f"Tuple fill_value must have length 2 (below, above), got length={len(fill_value)}"
            )
        return False, bounds_error, fill_value[0], fill_value[1]

    return False, bounds_error, fill_value, fill_value


def _output_dtype(
    x_dtype: np.dtype[Any],
    y_dtype: np.dtype[Any],
    x_new_dtype: np.dtype[Any],
    kind: str,
) -> np.dtype[Any]:
    """Match scipy.interpolate.interp1d output dtype promotion rules."""
    if kind in _SPLINE_ORDERS:
        return np.dtype(np.complex128 if np.issubdtype(y_dtype, np.complexfloating) else np.float64)

    y_inexact = y_dtype if np.issubdtype(y_dtype, np.inexact) else np.dtype(np.float64)
    if kind == "linear":
        if np.issubdtype(x_dtype, np.floating) and np.issubdtype(x_new_dtype, np.floating):
            x_float = np.result_type(x_dtype, x_new_dtype, np.float32)
        else:
            x_float = np.dtype(np.float64)
        return np.result_type(y_inexact, x_float)

    return np.dtype(y_inexact)


def _sort_1d_along_axis(x: np.ndarray, y: np.ndarray, ax: int) -> tuple[np.ndarray, np.ndarray]:
    """Sort 1D coordinate array ``x`` ascending and permute ``y`` along axis ``ax`` to match."""
    if x.shape[0] <= 1 or np.all(x[1:] >= x[:-1]):
        return x, y
    if np.all(x[1:] < x[:-1]):
        # Strictly decreasing (e.g., tempo_frequencies): O(1) slice reversal
        rev_slice: list[slice] = [slice(None)] * y.ndim
        rev_slice[ax] = slice(None, None, -1)
        return x[::-1], y[tuple(rev_slice)]
    ind = np.argsort(x, kind="mergesort")
    return x[ind], np.take(y, ind, axis=ax)


def _gather_axis(
    y: np.ndarray,
    idx: np.ndarray,
    *,
    ax: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
    """Gather elements along axis ``ax`` with guaranteed newly allocated writable buffer."""
    gathered = y[(slice(None),) * ax + (idx,)]
    # 0-d array indexing can yield a NumPy scalar or a view of y; always copy when ndim == 0
    return np.asarray(gathered, dtype=dtype).copy() if idx.ndim == 0 else np.asarray(gathered, dtype=dtype)


def _set_mask_inplace(
    out: np.ndarray,
    mask: np.ndarray,
    val: Any,
    *,
    ax: int | None = None,
) -> None:
    """Assign ``val`` in-place on freshly allocated ``out`` where ``mask`` is True."""
    if not np.any(mask):
        return
    fill = np.asarray(val, dtype=out.dtype)
    if mask.ndim == 0:
        out[...] = fill
    elif ax is None:
        out[mask] = fill
    else:
        out[(slice(None),) * ax + (mask,)] = fill


def _apply_oob_and_nan_inplace(
    out: np.ndarray,
    *,
    x_new: np.ndarray,
    below: np.ndarray,
    above: np.ndarray,
    extrapolate: bool,
    fill_below: Any,
    fill_above: Any,
    kind: str,
    ax: int | None,
) -> np.ndarray:
    """Apply out-of-bounds fill values or extrapolation NaN edge masks in-place on ``out``."""
    if extrapolate:
        if kind == "previous":
            _set_mask_inplace(out, below, np.nan, ax=ax)
        elif kind == "next":
            _set_mask_inplace(out, above, np.nan, ax=ax)
    else:
        _set_mask_inplace(out, below, fill_below, ax=ax)
        _set_mask_inplace(out, above, fill_above, ax=ax)

    if kind in ("linear", "zero"):
        _set_mask_inplace(out, np.isnan(x_new), np.nan, ax=ax)
    return out


def _linear_blend(
    xq: np.ndarray,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    out: np.ndarray,
    y_hi: np.ndarray,
    *,
    out_dtype: np.dtype[Any],
    check_nonfinite_y: bool,
    bc_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Compute vectorized linear interpolation from bracket bounds ``(x_lo, x_hi, out=y_lo, y_hi)``."""
    dx = x_hi - x_lo
    hit_lo = xq == x_lo
    hit_hi = xq == x_hi
    safe_dx = np.where(dx == 0, 1.0, dx)
    safe_num = np.where(hit_lo | hit_hi, 0.0, xq - x_lo)
    slope_dtype = np.finfo(cast("DTypeLike", out_dtype)).dtype
    slope = np.where(hit_hi, 1.0, np.where(hit_lo, 0.0, safe_num / safe_dx)).astype(slope_dtype, copy=False)
    slope_b = slope.reshape(bc_shape) if bc_shape is not None else slope

    if not check_nonfinite_y:
        # Both `out` (y_lo) and `y_hi` are freshly allocated buffers: blend in-place with zero temporaries
        y_hi -= out
        y_hi *= slope_b
        out += y_hi
        return out

    # Preserve exact grid-point values even when the adjacent grid value is non-finite
    hit_lo_b = hit_lo.reshape(bc_shape) if bc_shape is not None else hit_lo
    hit_hi_b = hit_hi.reshape(bc_shape) if bc_shape is not None else hit_hi
    blended = out + slope_b * (y_hi - out)
    res = np.where(hit_lo_b, out, blended)
    return np.where(hit_hi_b, y_hi, res)


def _eval_from_pos(
    xq: np.ndarray,
    pos: np.ndarray,
    *,
    n: int,
    kind: str,
    gather_x: Callable[[np.ndarray], np.ndarray],
    gather_y: Callable[[np.ndarray], np.ndarray],
    out_dtype: np.dtype[Any],
    check_nonfinite_y: bool,
    bc_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Evaluate linear or step interpolation from ``searchsorted`` positions."""
    if kind == "linear":
        hi = np.clip(pos, 1, n - 1)
        lo = hi - 1
        return _linear_blend(
            xq,
            gather_x(lo),
            gather_x(hi),
            gather_y(lo),
            gather_y(hi),
            out_dtype=out_dtype,
            check_nonfinite_y=check_nonfinite_y,
            bc_shape=bc_shape,
        )
    if kind in ("nearest", "nearest-up"):
        hi = np.clip(pos, 1, n - 1)
        lo = hi - 1
        d_lo = xq - gather_x(lo)
        d_hi = gather_x(hi) - xq
        idx = np.where((d_lo <= d_hi) if kind == "nearest" else (d_lo < d_hi), lo, hi)
        return gather_y(idx)
    if kind in ("zero", "previous"):
        return gather_y(np.clip(pos - 1, 0, n - 1))
    return gather_y(np.clip(pos, 0, n - 1))


def _interp1d(
    x: ArrayLike,
    y: ArrayLike,
    x_new: ArrayLike,
    *,
    axis: int = -1,
    kind: _InterpKind = "linear",
    bounds_error: bool | None = None,
    fill_value: float | tuple[float, float] | Literal["extrapolate"] | ArrayLike = np.nan,
    assume_sorted: bool = False,
    copy: bool = False,  # noqa: ARG001 - retained for hybrid_tempogram(**kwargs) compatibility
) -> np.ndarray:
    """Interpolate a 1-D function across an N-D array along ``axis``.

    Matches the behavior of ``scipy.interpolate.interp1d(x, y, ...)(x_new)`` while
    implementing linear and step modes directly on NumPy primitives and delegating
    spline modes (``'quadratic'``, ``'cubic'``, deprecated ``'slinear'``) to
    ``scipy.interpolate.make_interp_spline``.
    """
    if kind not in _STEP_AND_LINEAR_KINDS and kind not in _SPLINE_ORDERS:
        raise ParameterValueError(f"Unsupported interpolation kind={kind!r}")

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    x_new_arr = np.asarray(x_new)

    if x_arr.ndim != 1:
        raise ParameterShapeError(f"x must be 1-dimensional, got shape={x_arr.shape}")
    n = x_arr.shape[0]
    if n < 1:
        raise ParameterValueError("x must contain at least 1 element")
    if y_arr.ndim < 1:
        raise ParameterShapeError(f"y must have at least 1 dimension, got ndim={y_arr.ndim}")
    if axis < -y_arr.ndim or axis >= y_arr.ndim:
        raise ParameterShapeError(f"axis={axis} is out of bounds for y.ndim={y_arr.ndim}")
    ax = axis % y_arr.ndim
    if y_arr.shape[ax] != n:
        raise ParameterShapeError(f"len(x)={n} does not match y.shape[{ax}]={y_arr.shape[ax]}")

    if not assume_sorted:
        x_arr, y_arr = _sort_1d_along_axis(x_arr, y_arr, ax)

    extrapolate, check_bounds, fill_below, fill_above = _resolve_fill_value(fill_value, bounds_error)
    below = x_new_arr < x_arr[0]
    above = x_new_arr > x_arr[-1]
    if check_bounds and (np.any(below) or np.any(above)):
        raise ParameterValueError("A value in x_new is outside the interpolation range.")

    bc_shape = (1,) * ax + x_new_arr.shape + (1,) * (y_arr.ndim - 1 - ax)

    if kind in _SPLINE_ORDERS:
        import scipy.interpolate

        spline = scipy.interpolate.make_interp_spline(x_arr, y_arr, k=_SPLINE_ORDERS[kind], axis=ax)
        out = np.asarray(spline(x_new_arr))
        if x_new_arr.ndim == 0 and out.ndim == 0:
            out = out.copy()
        if not extrapolate:
            _set_mask_inplace(out, below, fill_below, ax=ax)
            _set_mask_inplace(out, above, fill_above, ax=ax)
        return out

    out_dtype = _output_dtype(x_arr.dtype, y_arr.dtype, x_new_arr.dtype, kind)
    if n == 1:
        out = _gather_axis(y_arr, np.zeros(x_new_arr.shape, dtype=np.intp), ax=ax, dtype=out_dtype)
    else:
        side: Literal["left", "right"] = "right" if kind in ("linear", "zero", "previous") else "left"
        pos = np.searchsorted(x_arr, x_new_arr, side=side)
        out = _eval_from_pos(
            x_new_arr,
            pos,
            n=n,
            kind=kind,
            gather_x=lambda idx: x_arr[idx],
            gather_y=lambda idx: _gather_axis(y_arr, idx, ax=ax, dtype=out_dtype),
            out_dtype=out_dtype,
            check_nonfinite_y=not np.all(np.isfinite(y_arr)),
            bc_shape=bc_shape,
        )

    return _apply_oob_and_nan_inplace(
        out,
        x_new=x_new_arr,
        below=below,
        above=above,
        extrapolate=extrapolate,
        fill_below=fill_below,
        fill_above=fill_above,
        kind=kind,
        ax=ax,
    )


def _interp1d_batched(
    x: ArrayLike,
    y: ArrayLike,
    x_new: ArrayLike,
    *,
    kind: _InterpKind = "linear",
    bounds_error: bool | None = None,
    fill_value: float | tuple[float, float] | Literal["extrapolate"] | ArrayLike = np.nan,
    assume_sorted: bool = False,
) -> np.ndarray:
    """Evaluate batched 1-D interpolation along the trailing axis ``-1``.

    Parameters
    ----------
    x : np.ndarray [shape=(..., n) or (n,)]
        Source coordinates. Either shared 1-D coordinates or per-slice coordinates
        matching the leading batch shape of ``y`` and ``x_new``.
    y : np.ndarray [shape=(..., n)]
        Source values along ``axis=-1``.
    x_new : np.ndarray [shape=(..., m)]
        Target query coordinates along ``axis=-1``. Must share the leading batch
        dimensions of ``y`` (and of ``x`` when ``x`` is multi-dimensional).
    kind : str
        Interpolation kind.
    bounds_error : bool or None
        Whether to raise on out-of-bounds queries.
    fill_value : float, tuple, 'extrapolate', or ArrayLike
        Out-of-bounds fill value or extrapolation behavior.
    assume_sorted : bool
        If True, ``x`` is assumed sorted ascending along ``axis=-1``.
    """
    if kind not in _STEP_AND_LINEAR_KINDS and kind not in _SPLINE_ORDERS:
        raise ParameterValueError(f"Unsupported interpolation kind={kind!r}")

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    xq_arr = np.asarray(x_new)

    if y_arr.ndim < 1 or xq_arr.ndim < 1:
        raise ParameterShapeError("y and x_new must have at least 1 dimension")
    if x_arr.ndim == 1:
        if x_arr.shape[0] != y_arr.shape[-1]:
            raise ParameterShapeError(f"len(x)={x_arr.shape[0]} does not match y.shape[-1]={y_arr.shape[-1]}")
    elif x_arr.shape != y_arr.shape:
        raise ParameterShapeError(f"x.shape={x_arr.shape} must match y.shape={y_arr.shape}")
    if xq_arr.shape[:-1] != y_arr.shape[:-1]:
        raise ParameterShapeError(f"Batch shape mismatch between y={y_arr.shape} and x_new={xq_arr.shape}")

    batch_shape = y_arr.shape[:-1]
    n = y_arr.shape[-1]
    m = xq_arr.shape[-1]
    if n < 1:
        raise ParameterValueError("x must contain at least 1 element")

    out_dtype = _output_dtype(x_arr.dtype, y_arr.dtype, xq_arr.dtype, kind)
    b_size = int(np.prod(batch_shape)) if batch_shape else 1
    if b_size == 0:
        return np.empty((*batch_shape, m), dtype=out_dtype)

    if kind in _SPLINE_ORDERS or not batch_shape:
        if not batch_shape:
            return _interp1d(
                x_arr,
                y_arr,
                xq_arr,
                axis=-1,
                kind=kind,
                bounds_error=bounds_error,
                fill_value=fill_value,
                assume_sorted=assume_sorted,
            )
        y_2d = y_arr.reshape(b_size, n)
        xq_2d = xq_arr.reshape(b_size, m)
        if x_arr.ndim == 1:
            if not assume_sorted:
                x_arr, y_2d = _sort_1d_along_axis(x_arr, y_2d, 1)
            rows = [
                _interp1d(
                    x_arr,
                    y_2d[i],
                    xq_2d[i],
                    axis=0,
                    kind=kind,
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    assume_sorted=True,
                )
                for i in range(b_size)
            ]
        else:
            x_2d = x_arr.reshape(b_size, n)
            rows = [
                _interp1d(
                    x_2d[i],
                    y_2d[i],
                    xq_2d[i],
                    axis=0,
                    kind=kind,
                    bounds_error=bounds_error,
                    fill_value=fill_value,
                    assume_sorted=assume_sorted,
                )
                for i in range(b_size)
            ]
        return np.stack(rows, axis=0).reshape(*batch_shape, m)

    extrapolate, check_bounds, fill_below, fill_above = _resolve_fill_value(fill_value, bounds_error)
    xq_c = np.ascontiguousarray(xq_arr)
    check_nonfinite_y = not np.all(np.isfinite(y_arr))
    side: Literal["left", "right"] = "right" if kind in ("linear", "zero", "previous") else "left"

    if x_arr.ndim == 1:
        # Shared 1D x with N-D y and N-D xq: single vectorized searchsorted + take_along_axis
        y_c = y_arr
        if not assume_sorted:
            x_arr, y_c = _sort_1d_along_axis(x_arr, y_c, y_c.ndim - 1)

        below = xq_c < x_arr[0]
        above = xq_c > x_arr[-1]
        if check_bounds and (np.any(below) or np.any(above)):
            raise ParameterValueError("A value in x_new is outside the interpolation range.")

        if n == 1:
            out = np.broadcast_to(y_c[..., :1], xq_c.shape).astype(out_dtype, copy=True)
        else:
            pos = np.searchsorted(x_arr, xq_c, side=side)
            out = _eval_from_pos(
                xq_c,
                pos,
                n=n,
                kind=kind,
                gather_x=lambda idx: x_arr[idx],
                gather_y=lambda idx: np.take_along_axis(y_c, idx, axis=-1).astype(out_dtype, copy=False),
                out_dtype=out_dtype,
                check_nonfinite_y=check_nonfinite_y,
            )

        return _apply_oob_and_nan_inplace(
            out,
            x_new=xq_c,
            below=below,
            above=above,
            extrapolate=extrapolate,
            fill_below=fill_below,
            fill_above=fill_above,
            kind=kind,
            ax=None,
        )

    # Multi-dimensional x: ensure contiguous and sort along -1 if needed
    x_c = np.ascontiguousarray(x_arr)
    y_c = np.ascontiguousarray(y_arr)
    if not assume_sorted and n > 1 and not np.all(x_c[..., 1:] >= x_c[..., :-1]):
        idx_sort = np.argsort(x_c, axis=-1, kind="mergesort")
        x_c = np.take_along_axis(x_c, idx_sort, axis=-1)
        y_c = np.take_along_axis(y_c, idx_sort, axis=-1)

    x_2d = x_c.reshape(b_size, n)
    y_2d = y_c.reshape(b_size, n)
    xq_2d = xq_c.reshape(b_size, m)

    below_2d = xq_2d < x_2d[:, :1]
    above_2d = xq_2d > x_2d[:, -1:]
    if check_bounds and (np.any(below_2d) or np.any(above_2d)):
        raise ParameterValueError("A value in x_new is outside the interpolation range.")

    x_flat = x_2d.ravel()
    y_flat = y_2d.ravel()
    row_off = (np.arange(b_size, dtype=np.intp) * n)[:, None]

    if n == 1:
        out_2d = np.broadcast_to(y_2d[:, :1], (b_size, m)).astype(out_dtype, copy=True)
    else:
        pos = np.empty((b_size, m), dtype=np.intp)
        for b in range(b_size):
            pos[b] = np.searchsorted(x_2d[b], xq_2d[b], side=side)

        out_2d = _eval_from_pos(
            xq_2d,
            pos,
            n=n,
            kind=kind,
            gather_x=lambda idx: x_flat[row_off + idx],
            gather_y=lambda idx: y_flat[row_off + idx].astype(out_dtype, copy=False),
            out_dtype=out_dtype,
            check_nonfinite_y=check_nonfinite_y,
        )

    out_2d = _apply_oob_and_nan_inplace(
        out_2d,
        x_new=xq_2d,
        below=below_2d,
        above=above_2d,
        extrapolate=extrapolate,
        fill_below=fill_below,
        fill_above=fill_above,
        kind=kind,
        ax=None,
    )
    return out_2d.reshape(*batch_shape, m)
