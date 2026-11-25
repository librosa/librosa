#!/usr/bin/env python
"""Unit tests for librosa.interp_time and librosa.interp_frequency."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.interpolate

import librosa
from librosa.display.formatting import Transformf0
from librosa.util._interp import _interp1d, _interp1d_batched
from librosa.util.exceptions import ParameterShapeError, ParameterValueError

if TYPE_CHECKING:
    from librosa._typing import _InterpKind


def test_interp_time_1d():
    times = np.linspace(0, 1, 100)
    x = np.sin(2 * np.pi * 5 * times)
    target_times = np.linspace(0, 1, 150)

    res = librosa.interp_time(x, target_times=target_times, times=times, axis=0)
    assert res.shape == (150,)

    ref = scipy.interpolate.interp1d(times, x, axis=0, bounds_error=False, fill_value=0.0)(target_times)
    assert np.allclose(res, ref)


def test_interp_time_implicit():
    sr = 22050
    hop_length = 512
    n_frames = 100
    x = np.ones((10, n_frames))

    target_times = np.linspace(0, 1.0, 50)
    res = librosa.interp_time(x, target_times=target_times, sr=sr, hop_length=hop_length, axis=-1)
    assert res.shape == (10, 50)

    inferred_times = librosa.times_like(x, sr=sr, hop_length=hop_length, axis=-1)
    ref = librosa.interp_time(x, target_times=target_times, times=inferred_times, axis=-1)
    assert np.allclose(res, ref)


def test_interp_time_2d(rng):
    n_freqs = 32
    n_frames = 64
    x = rng.standard_normal((n_freqs, n_frames))
    times = np.linspace(0, 2.0, n_frames)
    target_times = np.linspace(0, 2.0, 48)

    res = librosa.interp_time(x, target_times=target_times, times=times, axis=-1)
    assert res.shape == (n_freqs, 48)

    for i in range(n_freqs):
        ref_slice = scipy.interpolate.interp1d(times, x[i], bounds_error=False, fill_value=0.0)(target_times)
        assert np.allclose(res[i], ref_slice)


def test_interp_time_axes(rng):
    x = rng.standard_normal((10, 20, 30))
    t = np.linspace(0, 1, 20)
    target_t = np.linspace(0, 1, 15)

    res_pos = librosa.interp_time(x, target_times=target_t, times=t, axis=1)
    res_neg = librosa.interp_time(x, target_times=target_t, times=t, axis=-2)
    assert res_pos.shape == (10, 15, 30)
    assert np.allclose(res_pos, res_neg)


def test_interp_frequency_1d():
    freqs = np.linspace(0, 1000, 11)
    x = np.arange(11, dtype=float)
    target_freqs = np.array([250.0, 500.0, 750.0])

    res = librosa.interp_frequency(x, target_freqs=target_freqs, freqs=freqs, axis=0)
    assert res.shape == (3,)
    assert np.allclose(res, [2.5, 5.0, 7.5])


def test_interp_frequency_2d(rng):
    sr = 22050
    n_fft = 2048
    n_frames = 20
    x = rng.standard_normal((n_fft // 2 + 1, n_frames))
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    cqt_freqs = librosa.cqt_frequencies(n_bins=84, fmin=32.7)

    res = librosa.interp_frequency(x, target_freqs=cqt_freqs, freqs=fft_freqs, axis=-2)
    assert res.shape == (84, n_frames)

    for j in range(n_frames):
        ref_col = scipy.interpolate.interp1d(fft_freqs, x[:, j], bounds_error=False, fill_value=0.0)(cqt_freqs)
        assert np.allclose(res[:, j], ref_col)


def test_interp_fill_value():
    coords = np.array([1.0, 2.0, 3.0])
    x = np.array([10.0, 20.0, 30.0])
    target = np.array([0.0, 2.0, 4.0])

    # Default fill_value = 0.0
    res_default = librosa.interp_time(x, target_times=target, times=coords, axis=0)
    assert np.allclose(res_default, [0.0, 20.0, 0.0])

    # Custom fill_value
    res_custom = librosa.interp_time(x, target_times=target, times=coords, fill_value=-999.0, axis=0)
    assert np.allclose(res_custom, [-999.0, 20.0, -999.0])

    # Extrapolate
    res_extrap = librosa.interp_time(x, target_times=target, times=coords, fill_value="extrapolate", axis=0)
    assert np.allclose(res_extrap, [0.0, 20.0, 40.0])


def test_interp_kinds():
    coords = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 8.0, 27.0])
    target = np.array([0.5, 1.5, 2.5])

    res_linear = librosa.interp_frequency(x, target_freqs=target, freqs=coords, kind="linear", axis=0)
    assert np.allclose(res_linear, [0.5, 4.5, 17.5])

    res_nearest = librosa.interp_frequency(x, target_freqs=target, freqs=coords, kind="nearest", axis=0)
    ref_nearest = scipy.interpolate.interp1d(coords, x, kind="nearest")(target)
    assert np.allclose(res_nearest, ref_nearest)


def test_interp_complex():
    times = np.arange(10, dtype=float)
    x = np.arange(10, dtype=float) + 1j * np.arange(10, 20, dtype=float)
    target = np.array([0.5, 1.5])

    res = librosa.interp_time(x, target_times=target, times=times, axis=0)
    assert np.iscomplexobj(res)
    assert np.allclose(res.real, [0.5, 1.5])
    assert np.allclose(res.imag, [10.5, 11.5])


def test_interp_variable_grids():
    y, sr = librosa.loadx("trumpet", duration=0.5)
    freqs, times, mags = librosa.reassigned_spectrogram(y, sr=sr, fill_nan=False)

    # 1. Variable frequencies locked to regular target frequency grid
    target_freqs = np.linspace(100, 2000, 64)
    with pytest.warns(UserWarning, match="not unique"):
        mags_f_locked = librosa.interp_frequency(mags, target_freqs=target_freqs, freqs=freqs, axis=-2)
    assert mags_f_locked.shape == (64, mags.shape[-1])
    assert not np.isnan(mags_f_locked).any()

    # 2. Variable times locked to regular target time grid
    target_times = np.linspace(0, 0.5, 50)
    with pytest.warns(UserWarning, match="not unique"):
        mags_t_locked = librosa.interp_time(mags, target_times=target_times, times=times, axis=-1)
    assert mags_t_locked.shape == (mags.shape[-2], 50)
    assert not np.isnan(mags_t_locked).any()


def test_interp_multidim_target():
    n_freqs = 10
    n_frames = 20
    x = np.arange(n_freqs * n_frames, dtype=float).reshape(n_freqs, n_frames)
    freqs = np.linspace(0, 100, n_freqs)

    # Multi-dimensional target grid: time-varying target frequencies of shape (5, n_frames)
    target_freqs = np.linspace(10, 90, 5)[:, None] + np.linspace(0, 5, n_frames)[None, :]
    assert target_freqs.shape == (5, n_frames)

    res = librosa.interp_frequency(x, target_freqs=target_freqs, freqs=freqs, axis=-2)
    assert res.shape == (5, n_frames)

    for t in range(n_frames):
        ref_t = scipy.interpolate.interp1d(freqs, x[:, t], bounds_error=False, fill_value=0.0)(target_freqs[:, t])
        assert np.allclose(res[:, t], ref_t)


def test_interp_validation_and_warnings():
    x = np.ones((10, 20))

    # Axis out of bounds
    with pytest.raises(ParameterShapeError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(10), axis=5)

    with pytest.raises(ParameterShapeError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(10), axis=-5)

    # Incompatible coordinate length (1D)
    with pytest.raises(ParameterShapeError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(15), axis=-1)

    # Incompatible coordinate shape (multidimensional)
    with pytest.raises(ParameterShapeError):
        librosa.interp_frequency(x, target_freqs=np.ones(5), freqs=np.ones((8, 20)), axis=-2)

    # Incompatible target coordinates broadcast
    with pytest.raises(ParameterShapeError):
        librosa.interp_frequency(x, target_freqs=np.ones((5, 12)), freqs=np.arange(10, dtype=float), axis=-2)

    # Non-unique coordinates warning (1D)
    non_unique_1d = np.array([0, 1, 1, 3, 4, 5, 6, 7, 8, 9])
    with pytest.warns(UserWarning, match="not unique"):
        librosa.interp_frequency(x, target_freqs=np.ones(5), freqs=non_unique_1d, axis=0)

    # Non-unique coordinates warning (2D)
    non_unique_2d = np.tile(non_unique_1d[:, None], (1, 20))
    with pytest.warns(UserWarning, match="not unique"):
        librosa.interp_frequency(x, target_freqs=np.ones(5), freqs=non_unique_2d, axis=-2)


def test_interp_nan_slice_and_spline_fallback():
    # All NaN coordinates for a slice in 2D
    x = np.ones((5, 2))
    freqs = np.array(
        [
            [0.0, np.nan],
            [1.0, np.nan],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, np.nan],
        ]
    )
    target = np.array([0.5, 1.5])
    with pytest.warns(UserWarning, match="not unique"):
        res = librosa.interp_frequency(x, target_freqs=target, freqs=freqs, axis=0, fill_value=-1.0)
    assert res.shape == (2, 2)
    # Column 0 is valid
    assert np.allclose(res[:, 0], [1.0, 1.0])
    # Column 1 was all-NaN, should be filled with fill_value
    assert np.allclose(res[:, 1], [-1.0, -1.0])

    # Insufficient points for cubic spline (cubic requires >= 4 points, but we only have 2 finite)
    freqs_sparse = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, np.nan],
            [3.0, np.nan],
            [4.0, np.nan],
        ]
    )
    with pytest.warns(UserWarning, match="not unique"):
        res_cubic = librosa.interp_frequency(
            x, target_freqs=target, freqs=freqs_sparse, axis=0, kind="cubic", fill_value=-99.0
        )
    assert res_cubic.shape == (2, 2)
    # Column 0 has 5 points -> valid cubic
    assert np.allclose(res_cubic[:, 0], [1.0, 1.0])
    # Column 1 has only 2 points -> ValueError caught, fallback to fill_value
    assert np.allclose(res_cubic[:, 1], [-99.0, -99.0])


def test_interp_slinear_deprecated_and_valid_kinds():
    coords = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 8.0, 27.0])
    target = np.array([0.5, 1.5, 2.5])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        for kind in ("linear", "nearest", "nearest-up", "previous", "next", "zero", "quadratic", "cubic"):
            librosa.interp_time(x, target_times=target, times=coords, kind=kind, axis=0)
            librosa.interp_frequency(x, target_freqs=target, freqs=coords, kind=kind, axis=0)

    with pytest.warns(FutureWarning, match="slinear") as record_t:
        librosa.interp_time(x, target_times=target, times=coords, kind="slinear", axis=0)
    assert len(record_t) == 1

    with pytest.warns(FutureWarning, match="slinear") as record_f:
        librosa.interp_frequency(x, target_freqs=target, freqs=coords, kind="slinear", axis=0)
    assert len(record_f) == 1


@pytest.mark.parametrize(
    "kind",
    ["linear", "slinear", "nearest", "nearest-up", "zero", "previous", "next", "quadratic", "cubic"],
)
@pytest.mark.parametrize("fill_value", [0.0, (-1.0, 2.0), np.nan, "extrapolate"])
def test_internal_interp1d_kinds_and_fill(kind, fill_value):
    x = np.linspace(0.0, 4.0, 9)
    rng = np.random.default_rng(123)
    y = rng.standard_normal((2, 9, 3))
    x_new = np.array([-0.5, 0.0, 0.25, 0.5, 1.25, 2.0, 3.5, 4.0, 4.75])

    ref = scipy.interpolate.interp1d(x, y, axis=1, kind=kind, bounds_error=False, fill_value=fill_value)(x_new)
    got = _interp1d(x, y, x_new, axis=1, kind=kind, bounds_error=False, fill_value=fill_value)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128, np.int32])
@pytest.mark.parametrize("kind", ["linear", "nearest", "previous", "cubic"])
def test_internal_interp1d_dtypes_and_edge_cases(dtype, kind):
    x = np.array([10.0, 5.0, 2.0, 0.0, -3.0], dtype=np.float32)  # strictly decreasing
    if np.issubdtype(dtype, np.complexfloating):
        y = (np.arange(5) + 2j * np.arange(5, 10)).astype(dtype)
    else:
        y = np.arange(5, dtype=dtype)
    x_new = np.array([-4.0, -3.0, 1.0, 5.0, 11.0], dtype=np.float32)

    ref = scipy.interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value=0.0)(x_new)
    got = _interp1d(x, y, x_new, axis=0, kind=kind, bounds_error=False, fill_value=0.0)
    assert got.dtype == ref.dtype
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_internal_interp1d_special_cases():
    # Unsorted with duplicate x, n==1, NaN in x_new, NaN in y, inf endpoints (tempo_frequencies style)
    x_dup = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 0.0])
    y_dup = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    xq_dup = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ref_dup = scipy.interpolate.interp1d(x_dup, y_dup, kind="linear")(xq_dup)
    got_dup = _interp1d(x_dup, y_dup, xq_dup, kind="linear")
    assert np.allclose(got_dup, ref_dup)

    # n == 1
    for k in ("linear", "nearest", "previous", "next", "zero"):
        ref1 = scipy.interpolate.interp1d([2.0], [7.0], kind=k, bounds_error=False, fill_value=-1.0)([1.0, 2.0, 3.0])
        got1 = _interp1d([2.0], [7.0], [1.0, 2.0, 3.0], kind=k, bounds_error=False, fill_value=-1.0)
        assert np.allclose(got1, ref1)

    # Descending tempo_frequencies with np.inf at index 0
    f_tempo = np.array([np.inf, 600.0, 300.0, 200.0, 150.0, 120.0])
    y_tempo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    f_out = np.multiply.outer([0.5, 1.0, 2.0], f_tempo)
    ref_inf = scipy.interpolate.interp1d(f_tempo, y_tempo, bounds_error=False, fill_value=0.0)(f_out)
    got_inf = _interp1d(f_tempo, y_tempo, f_out, axis=0, bounds_error=False, fill_value=0.0)
    assert np.allclose(got_inf, ref_inf)

    # NaN in x_new and adjacent NaN in y
    xq_nan = np.array([0.0, np.nan, 1.5, 2.0])
    y_nan = np.array([10.0, np.nan, 30.0])
    x_grid = np.array([0.0, 1.0, 2.0])
    step_kinds: tuple[_InterpKind, ...] = ("linear", "nearest", "nearest-up", "zero", "previous", "next")
    for k in step_kinds:
        ref_n = scipy.interpolate.interp1d(x_grid, y_nan, kind=k, bounds_error=False, fill_value=0.0)(xq_nan)
        got_n = _interp1d(x_grid, y_nan, xq_nan, kind=k, bounds_error=False, fill_value=0.0)
        assert np.allclose(got_n, ref_n, equal_nan=True)

    # Guard errors
    with pytest.raises(ParameterValueError, match="Unsupported interpolation kind"):
        _interp1d(x_grid, y_nan, xq_nan, kind="bad")  # type: ignore[arg-type]
    with pytest.raises(ParameterValueError, match="Cannot extrapolate"):
        _interp1d(x_grid, y_nan, xq_nan, bounds_error=True, fill_value="extrapolate")
    with pytest.raises(ParameterValueError, match="outside the interpolation range"):
        _interp1d(x_grid, y_nan, np.array([-1.0]), bounds_error=True)


def test_internal_interp1d_batched(rng):
    x = np.linspace(0, 10, 12)[None, None, :] + rng.uniform(-0.2, 0.2, (2, 4, 12))
    x = np.sort(x, axis=-1).astype(np.float32)
    y = rng.standard_normal((2, 4, 12)).astype(np.float32)
    xq = np.linspace(-1, 11, 15)[None, None, :] + rng.uniform(-0.1, 0.1, (2, 4, 15))
    xq = xq.astype(np.float32)

    all_kinds: tuple[_InterpKind, ...] = ("linear", "nearest", "nearest-up", "zero", "previous", "next", "cubic")
    for k in all_kinds:
        got = _interp1d_batched(x, y, xq, kind=k, bounds_error=False, fill_value=-5.0)
        ref = np.stack(
            [
                scipy.interpolate.interp1d(x[i, j], y[i, j], kind=k, bounds_error=False, fill_value=-5.0)(xq[i, j])
                for i in range(2)
                for j in range(4)
            ],
            axis=0,
        ).reshape(2, 4, 15)
        assert got.shape == ref.shape
        expected_dtype = np.float64 if k == "cubic" else y.dtype
        assert got.dtype == expected_dtype
        assert np.allclose(got, ref, rtol=1e-5, atol=1e-5)

    # Descending tempo frequencies in interp_harmonics
    f_tempo = librosa.tempo_frequencies(32, sr=22050)
    y_tempo = np.abs(rng.standard_normal(len(f_tempo)))
    harms = [0.5, 1.0, 2.0]
    ref_h = scipy.interpolate.interp1d(f_tempo, y_tempo, bounds_error=False, fill_value=0.0)(
        np.multiply.outer(harms, f_tempo)
    )
    got_h = librosa.interp_harmonics(y_tempo, freqs=f_tempo, harmonics=harms, axis=0)
    assert np.allclose(got_h, ref_h)


def test_internal_interp1d_adversarial(rng):
    # 1. interp_harmonics with axis=0 on varying 2D freqs -> (H, F, T)
    f_2d = np.linspace(100, 1000, 16)[:, None] + rng.uniform(-5.0, 5.0, (16, 4))
    y_2d = np.abs(rng.standard_normal((16, 4))).astype(np.float32)
    h_vals = [0.5, 1.0, 2.0]
    got_h2d = librosa.interp_harmonics(y_2d, freqs=f_2d.astype(np.float32), harmonics=h_vals, axis=0)
    assert got_h2d.shape == (3, 16, 4)
    for t in range(4):
        ref_t = scipy.interpolate.interp1d(f_2d[:, t], y_2d[:, t], bounds_error=False, fill_value=0.0)(
            np.multiply.outer(h_vals, f_2d[:, t])
        )
        assert np.allclose(got_h2d[:, :, t], ref_t, rtol=1e-5, atol=1e-5)

    # 2. High dynamic-range geometric coordinates
    x_geo = np.geomspace(1e-6, 1e6, 200)
    y_geo = np.log10(x_geo)
    xq_geo = np.geomspace(2e-6, 9e5, 300)
    ref_geo = scipy.interpolate.interp1d(x_geo, y_geo, kind="linear", bounds_error=False, fill_value=0.0)(xq_geo)
    got_geo = _interp1d(x_geo, y_geo, xq_geo, kind="linear", bounds_error=False, fill_value=0.0)
    assert np.allclose(got_geo, ref_geo, rtol=1e-12, atol=1e-12)

    # 3. Exact midpoint ties: nearest vs nearest-up
    x_mid = np.array([0.0, 1.0, 2.0, 3.0])
    y_mid = np.array([10.0, 20.0, 30.0, 40.0])
    xq_mid = np.array([0.5, 1.5, 2.5])
    assert np.allclose(_interp1d(x_mid, y_mid, xq_mid, kind="nearest"), [10.0, 20.0, 30.0])
    assert np.allclose(_interp1d(x_mid, y_mid, xq_mid, kind="nearest-up"), [20.0, 30.0, 40.0])

    # 4. Non-contiguous transposed y
    S_t = np.ascontiguousarray(rng.standard_normal((16, 32))).T  # shape (32, 16), F-contiguous
    f_nc = np.linspace(0.0, 1.0, 16)
    xq_nc = np.linspace(0.0, 1.0, 11)
    ref_nc = scipy.interpolate.interp1d(f_nc, S_t, axis=-1, kind="linear", bounds_error=False, fill_value=0.0)(xq_nc)
    got_nc = _interp1d(f_nc, S_t, xq_nc, axis=-1, kind="linear", bounds_error=False, fill_value=0.0)
    assert np.allclose(got_nc, ref_nc)

    # 5. Transformf0 with NaN gaps in f0
    f0 = np.array([220.0, np.nan, 440.0, 330.0, np.nan])
    trans = Transformf0(f0, sr=22050, hop_length=512)
    pts = np.array([[0.0, 1.0], [0.025, 1.0], [0.05, 1.0], [10.0, 1.0]])
    out_pts = trans.transform_non_affine(pts)
    assert np.isclose(out_pts[0, 1], 2.0 ** (1.0 / 12) * 220.0)
    assert np.isnan(out_pts[1, 1])
    assert np.isnan(out_pts[3, 1])

    # 6. Scalar (0-D) x_new on 1D and 2D y (both in-bounds and OOB)
    x_s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_1d_s = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    y_2d_s = np.stack([y_1d_s, y_1d_s + 5.0])
    scalar_kinds: tuple[_InterpKind, ...] = ("linear", "nearest", "previous", "cubic")
    for k in scalar_kinds:
        for q in (1.5, -1.0, 5.0):
            r1 = _interp1d(x_s, y_1d_s, q, kind=k, bounds_error=False, fill_value=-3.0)
            ref1 = scipy.interpolate.interp1d(x_s, y_1d_s, kind=k, bounds_error=False, fill_value=-3.0)(q)
            assert np.allclose(r1, ref1)

            r2 = _interp1d(x_s, y_2d_s, q, axis=-1, kind=k, bounds_error=False, fill_value=-3.0)
            ref2 = scipy.interpolate.interp1d(x_s, y_2d_s, axis=-1, kind=k, bounds_error=False, fill_value=-3.0)(q)
            assert r2.shape == ref2.shape
            assert np.allclose(r2, ref2)

    # 7. 2D tempo_frequencies with inf at index 0 produces finite results matching 1D (no NaN/warnings)
    f_t2d = librosa.tempo_frequencies(64, sr=22050)
    S_t2d = np.abs(rng.standard_normal((len(f_t2d), 8)))
    harms_t = [0.5, 1.0, 2.0, 3.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got_t2d = librosa.interp_harmonics(S_t2d, freqs=f_t2d, harmonics=harms_t, axis=0)
    assert not np.isnan(got_t2d).any()
    for col in range(S_t2d.shape[1]):
        got_col = librosa.interp_harmonics(S_t2d[:, col], freqs=f_t2d, harmonics=harms_t, axis=0)
        assert np.allclose(got_t2d[:, :, col], got_col)


def test_internal_interp1d_batched_coverage(rng):
    # Shared 1D x across all 9 kinds with ascending, strictly descending, and unsorted coordinates
    y = rng.standard_normal((2, 3, 10)).astype(np.float32)
    xq = np.linspace(-1.0, 11.0, 14)[None, None, :] + rng.uniform(-0.2, 0.2, (2, 3, 14))
    xq = xq.astype(np.float32)

    all_9_kinds: tuple[_InterpKind, ...] = (
        "linear",
        "slinear",
        "nearest",
        "nearest-up",
        "zero",
        "previous",
        "next",
        "quadratic",
        "cubic",
    )
    x_variants = [
        np.linspace(0.0, 10.0, 10, dtype=np.float32),
        np.linspace(10.0, 0.0, 10, dtype=np.float32),
        np.array([0.0, 5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 10.0], dtype=np.float32),
    ]
    for x_1d in x_variants:
        for k in all_9_kinds:
            for fv in ((-1.5, 2.5), "extrapolate"):
                got = _interp1d_batched(x_1d, y, xq, kind=k, bounds_error=False, fill_value=fv)
                ref = np.stack(
                    [
                        scipy.interpolate.interp1d(x_1d, y[i, j], kind=k, bounds_error=False, fill_value=fv)(xq[i, j])
                        for i in range(2)
                        for j in range(3)
                    ],
                    axis=0,
                ).reshape(2, 3, 14)
                assert got.shape == ref.shape
                assert np.allclose(got, ref, rtol=1e-4, atol=1e-4, equal_nan=True)

    # Multi-D unsorted and strictly descending x, complex64, NaNs, n==1, 1D batch_shape==(), empty batch (0, 10)
    x_unsorted = rng.uniform(0.0, 10.0, (2, 8)).astype(np.float64)
    y_c64 = (rng.standard_normal((2, 8)) + 1j * rng.standard_normal((2, 8))).astype(np.complex64)
    xq_2d = np.linspace(-1.0, 11.0, 10)[None, :] + np.zeros((2, 10))
    multidim_kinds: tuple[_InterpKind, ...] = ("linear", "nearest", "previous", "next", "quadratic")
    for k in multidim_kinds:
        got_u = _interp1d_batched(x_unsorted, y_c64, xq_2d, kind=k, bounds_error=False, fill_value=(-1.0, 1.0))
        ref_u = np.stack(
            [
                scipy.interpolate.interp1d(
                    x_unsorted[i], y_c64[i], kind=k, bounds_error=False, fill_value=(-1.0, 1.0)
                )(xq_2d[i])
                for i in range(2)
            ]
        )
        assert np.allclose(got_u, ref_u, rtol=1e-5, atol=1e-5)

    # Descending multi-D x with n==1 and n==5, bounds_error=True, NaN in x_new/y
    x_desc = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0]), (2, 1))
    y_desc = rng.standard_normal((2, 5))
    y_desc[0, 2] = np.nan
    xq_nan_2d = np.array([[1.0, np.nan, 3.5, 5.0], [0.5, 2.5, 4.5, 5.5]])
    got_dn = _interp1d_batched(x_desc, y_desc, xq_nan_2d, kind="linear", bounds_error=False, fill_value=0.0)
    ref_dn = np.stack(
        [
            scipy.interpolate.interp1d(x_desc[i], y_desc[i], kind="linear", bounds_error=False, fill_value=0.0)(
                xq_nan_2d[i]
            )
            for i in range(2)
        ]
    )
    assert np.allclose(got_dn, ref_dn, equal_nan=True)

    with pytest.raises(ParameterValueError, match="outside the interpolation range"):
        _interp1d_batched(x_desc, y_desc, xq_nan_2d, kind="linear", bounds_error=True)
    with pytest.raises(ParameterValueError, match="outside the interpolation range"):
        _interp1d_batched(x_desc[0], y_desc, xq_nan_2d, kind="linear", bounds_error=True)

    # n==1 batched (both 1D x and multi-D x), 1D batch_shape==(), empty batch (0, 10)
    x_n1 = np.array([[2.0], [3.0]])
    y_n1 = np.array([[10.0], [20.0]])
    xq_n1 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    assert np.allclose(
        _interp1d_batched(x_n1, y_n1, xq_n1, bounds_error=False, fill_value=-1.0),
        [[-1.0, 10.0, -1.0], [-1.0, 20.0, -1.0]],
    )
    assert np.allclose(
        _interp1d_batched(np.array([2.0]), y_n1, xq_n1, bounds_error=False, fill_value=-1.0),
        [[-1.0, 10.0, -1.0], [20.0, -1.0, -1.0]],
    )
    assert np.allclose(
        _interp1d_batched(x_desc[0], y_desc[0], np.array([1.5, 3.5])),
        [0.5 * (y_desc[0, 3] + y_desc[0, 4]), 0.5 * (y_desc[0, 1] + y_desc[0, 2])],
        equal_nan=True,
    )

    empty_kinds: tuple[_InterpKind, ...] = ("linear", "cubic")
    for k in empty_kinds:
        empty_out = _interp1d_batched(np.linspace(0, 1, 10), np.empty((0, 10)), np.empty((0, 5)), kind=k)
        assert empty_out.shape == (0, 5)


def test_internal_interp1d_validation_and_caller_parity(rng):
    x = np.linspace(0.0, 4.0, 5)
    y = np.arange(10.0).reshape(2, 5)

    # Empty x_new
    empty_res = _interp1d(x, y, np.array([]), axis=-1, kind="linear", fill_value=0.0)
    assert empty_res.shape == (2, 0)

    # n == 1 with extrapolate across linear/step kinds (previous=NaN below, next=NaN above)
    extrap_step_kinds: tuple[_InterpKind, ...] = ("linear", "nearest", "nearest-up", "zero", "previous", "next")
    for k in extrap_step_kinds:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            r1 = _interp1d([2.0], [7.0], [1.0, 2.0, 3.0], kind=k, fill_value="extrapolate")
        if k == "previous":
            assert np.allclose(r1, [np.nan, 7.0, 7.0], equal_nan=True)
        elif k == "next":
            assert np.allclose(r1, [7.0, 7.0, np.nan], equal_nan=True)
        else:
            assert np.allclose(r1, [7.0, 7.0, 7.0])
        if k != "linear":
            ref1 = scipy.interpolate.interp1d([2.0], [7.0], kind=k, fill_value="extrapolate")([1.0, 2.0, 3.0])
            assert np.allclose(r1, ref1, equal_nan=True)

    # Validation guards
    with pytest.raises(ParameterShapeError, match="1-dimensional"):
        _interp1d(np.ones((2, 2)), y, x)
    with pytest.raises(ParameterValueError, match="at least 1 element"):
        _interp1d(np.array([]), np.empty((2, 0)), x)
    with pytest.raises(ParameterShapeError, match="out of bounds"):
        _interp1d(x, y, x, axis=5)
    with pytest.raises(ParameterShapeError, match="does not match"):
        _interp1d(x, y, x, axis=0)
    with pytest.raises(ParameterValueError, match="Unsupported string fill_value"):
        _interp1d(x, y, x, bounds_error=False, fill_value="invalid")
    with pytest.raises(ParameterValueError, match="length 2"):
        _interp1d(x, y, x, bounds_error=False, fill_value=(1.0, 2.0, 3.0))

    # Direct caller parity against scipy.interpolate.interp1d
    # 1. path_to_steps on descending DTW path with duplicate column indices
    path_desc = np.array([[5, 4], [4, 4], [3, 3], [2, 2], [2, 1], [1, 1], [0, 1], [0, 0]])
    frames_in = np.arange(path_desc[:, 1].min(), path_desc[:, 1].max() + 1)
    ref_steps = scipy.interpolate.interp1d(path_desc[:, 1], path_desc[:, 0], kind="linear")(frames_in)
    assert np.allclose(librosa.sequence.path_to_steps(path_desc), ref_steps)

    # 2. interp_broadcast multi-channel with fill_value & nearest/cubic
    x1 = rng.standard_normal((2, 16)).astype(np.float32)
    x2 = rng.standard_normal((2, 12)).astype(np.float32)
    p1 = np.linspace(0.0, 1.0, 16)
    p2 = np.linspace(0.2, 0.9, 12)
    broadcast_kinds: tuple[_InterpKind, ...] = ("nearest", "cubic")
    for k in broadcast_kinds:
        b1, b2 = librosa.util.interp_broadcast(
            x1=x1, x1_pos=p1, x2=x2, x2_pos=p2, axis=-1, kind=k, fill_value=-2.0, op=None
        )
        r_b1 = scipy.interpolate.interp1d(p1, x1, axis=-1, kind=k, bounds_error=False, fill_value=-2.0)(p1)
        r_b2 = scipy.interpolate.interp1d(p2, x2, axis=-1, kind=k, bounds_error=False, fill_value=-2.0)(p1)
        assert np.allclose(b1, r_b1, rtol=1e-5, atol=1e-5)
        assert np.allclose(b2, r_b2, rtol=1e-5, atol=1e-5)

    # 3. Transformf0 round-trip with inverted() and NaN gaps
    f0 = np.array([220.0, np.nan, 440.0, 330.0])
    tf = Transformf0(f0, sr=22050, hop_length=512)
    pts = np.array([[0.0, 2.0], [0.025, 2.0], [0.05, -1.5], [10.0, 1.0]])
    rt = tf.inverted().transform_non_affine(tf.transform_non_affine(pts))
    assert np.allclose(rt[[0, 2]], pts[[0, 2]])
    assert np.isnan(rt[1, 1]) and np.isnan(rt[3, 1])
