#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for librosa.interp_time and librosa.interp_frequency."""

import numpy as np
import pytest
import scipy.interpolate

import librosa
from librosa.util.exceptions import ParameterError


def test_interp_time_1d():
    times = np.linspace(0, 1, 100)
    x = np.sin(2 * np.pi * 5 * times)
    target_times = np.linspace(0, 1, 150)

    res = librosa.interp_time(x, target_times=target_times, times=times, axis=0)
    assert res.shape == (150,)

    ref = scipy.interpolate.interp1d(times, x, axis=0, bounds_error=False, fill_value=0.0)(
        target_times
    )
    assert np.allclose(res, ref)


def test_interp_time_implicit():
    sr = 22050
    hop_length = 512
    n_frames = 100
    x = np.ones((10, n_frames))

    target_times = np.linspace(0, 1.0, 50)
    res = librosa.interp_time(
        x, target_times=target_times, sr=sr, hop_length=hop_length, axis=-1
    )
    assert res.shape == (10, 50)

    inferred_times = librosa.times_like(x, sr=sr, hop_length=hop_length, axis=-1)
    ref = librosa.interp_time(
        x, target_times=target_times, times=inferred_times, axis=-1
    )
    assert np.allclose(res, ref)


def test_interp_time_2d():
    n_freqs = 32
    n_frames = 64
    x = np.random.randn(n_freqs, n_frames)
    times = np.linspace(0, 2.0, n_frames)
    target_times = np.linspace(0, 2.0, 48)

    res = librosa.interp_time(x, target_times=target_times, times=times, axis=-1)
    assert res.shape == (n_freqs, 48)

    for i in range(n_freqs):
        ref_slice = scipy.interpolate.interp1d(
            times, x[i], bounds_error=False, fill_value=0.0
        )(target_times)
        assert np.allclose(res[i], ref_slice)


def test_interp_time_axes():
    x = np.random.randn(10, 20, 30)
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

    res = librosa.interp_frequency(
        x, target_freqs=target_freqs, freqs=freqs, axis=0
    )
    assert res.shape == (3,)
    assert np.allclose(res, [2.5, 5.0, 7.5])


def test_interp_frequency_2d():
    sr = 22050
    n_fft = 2048
    n_frames = 20
    x = np.random.randn(n_fft // 2 + 1, n_frames)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    cqt_freqs = librosa.cqt_frequencies(n_bins=84, fmin=32.7)

    res = librosa.interp_frequency(
        x, target_freqs=cqt_freqs, freqs=fft_freqs, axis=-2
    )
    assert res.shape == (84, n_frames)

    for j in range(n_frames):
        ref_col = scipy.interpolate.interp1d(
            fft_freqs, x[:, j], bounds_error=False, fill_value=0.0
        )(cqt_freqs)
        assert np.allclose(res[:, j], ref_col)


def test_interp_fill_value():
    coords = np.array([1.0, 2.0, 3.0])
    x = np.array([10.0, 20.0, 30.0])
    target = np.array([0.0, 2.0, 4.0])

    # Default fill_value = 0.0
    res_default = librosa.interp_time(x, target_times=target, times=coords, axis=0)
    assert np.allclose(res_default, [0.0, 20.0, 0.0])

    # Custom fill_value
    res_custom = librosa.interp_time(
        x, target_times=target, times=coords, fill_value=-999.0, axis=0
    )
    assert np.allclose(res_custom, [-999.0, 20.0, -999.0])

    # Extrapolate
    res_extrap = librosa.interp_time(
        x, target_times=target, times=coords, fill_value="extrapolate", axis=0
    )
    assert np.allclose(res_extrap, [0.0, 20.0, 40.0])


def test_interp_kinds():
    coords = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 8.0, 27.0])
    target = np.array([0.5, 1.5, 2.5])

    res_linear = librosa.interp_frequency(
        x, target_freqs=target, freqs=coords, kind="linear", axis=0
    )
    assert np.allclose(res_linear, [0.5, 4.5, 17.5])

    res_nearest = librosa.interp_frequency(
        x, target_freqs=target, freqs=coords, kind="nearest", axis=0
    )
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
    mags_f_locked = librosa.interp_frequency(
        mags, target_freqs=target_freqs, freqs=freqs, axis=-2
    )
    assert mags_f_locked.shape == (64, mags.shape[-1])
    assert not np.isnan(mags_f_locked).any()

    # 2. Variable times locked to regular target time grid
    target_times = np.linspace(0, 0.5, 50)
    mags_t_locked = librosa.interp_time(
        mags, target_times=target_times, times=times, axis=-1
    )
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

    res = librosa.interp_frequency(
        x, target_freqs=target_freqs, freqs=freqs, axis=-2
    )
    assert res.shape == (5, n_frames)

    for t in range(n_frames):
        ref_t = scipy.interpolate.interp1d(
            freqs, x[:, t], bounds_error=False, fill_value=0.0
        )(target_freqs[:, t])
        assert np.allclose(res[:, t], ref_t)


def test_interp_validation_and_warnings():
    x = np.ones((10, 20))

    # Axis out of bounds
    with pytest.raises(ParameterError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(10), axis=5)

    with pytest.raises(ParameterError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(10), axis=-5)

    # Incompatible coordinate length (1D)
    with pytest.raises(ParameterError):
        librosa.interp_time(x, target_times=np.ones(5), times=np.ones(15), axis=-1)

    # Incompatible coordinate shape (multidimensional)
    with pytest.raises(ParameterError):
        librosa.interp_frequency(
            x, target_freqs=np.ones(5), freqs=np.ones((8, 20)), axis=-2
        )

    # Incompatible target coordinates broadcast
    with pytest.raises(ParameterError):
        librosa.interp_frequency(
            x, target_freqs=np.ones((5, 12)), freqs=np.ones(10), axis=-2
        )

    # Non-unique coordinates warning (1D)
    non_unique_1d = np.array([0, 1, 1, 3, 4, 5, 6, 7, 8, 9])
    with pytest.warns(UserWarning, match="not unique"):
        librosa.interp_frequency(
            x, target_freqs=np.ones(5), freqs=non_unique_1d, axis=0
        )

    # Non-unique coordinates warning (2D)
    non_unique_2d = np.tile(non_unique_1d[:, None], (1, 20))
    with pytest.warns(UserWarning, match="not unique"):
        librosa.interp_frequency(
            x, target_freqs=np.ones(5), freqs=non_unique_2d, axis=-2
        )


def test_interp_nan_slice_and_spline_fallback():
    # All NaN coordinates for a slice in 2D
    x = np.ones((5, 2))
    freqs = np.array([
        [0.0, np.nan],
        [1.0, np.nan],
        [2.0, np.nan],
        [3.0, np.nan],
        [4.0, np.nan],
    ])
    target = np.array([0.5, 1.5])
    res = librosa.interp_frequency(x, target_freqs=target, freqs=freqs, axis=0, fill_value=-1.0)
    assert res.shape == (2, 2)
    # Column 0 is valid
    assert np.allclose(res[:, 0], [1.0, 1.0])
    # Column 1 was all-NaN, should be filled with fill_value
    assert np.allclose(res[:, 1], [-1.0, -1.0])

    # Insufficient points for cubic spline (cubic requires >= 4 points, but we only have 2 finite)
    freqs_sparse = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, np.nan],
        [3.0, np.nan],
        [4.0, np.nan],
    ])
    res_cubic = librosa.interp_frequency(
        x, target_freqs=target, freqs=freqs_sparse, axis=0, kind="cubic", fill_value=-99.0
    )
    assert res_cubic.shape == (2, 2)
    # Column 0 has 5 points -> valid cubic
    assert np.allclose(res_cubic[:, 0], [1.0, 1.0])
    # Column 1 has only 2 points -> ValueError caught, fallback to fill_value
    assert np.allclose(res_cubic[:, 1], [-99.0, -99.0])

