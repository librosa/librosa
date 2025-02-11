#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Unit tests for the effects module"""
import warnings

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

from contextlib import nullcontext as dnr
import numpy as np
import pytest

import librosa

__EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")


@pytest.fixture(scope="module", params=["test1_44100.wav"])
def y_multi(request):
    infile = request.param
    return librosa.load(os.path.join("tests", "data", infile), sr=None, mono=False)


@pytest.fixture(scope="module", params=[22050, 44100])
def ysr(request):
    return librosa.load(__EXAMPLE_FILE, sr=request.param)


@pytest.mark.parametrize(
    "rate,ctx",
    [
        (0.25, dnr()),
        (0.25, dnr()),
        (1.0, dnr()),
        (2.0, dnr()),
        (4.0, dnr()),
        (-1, pytest.raises(librosa.ParameterError)),
        (0, pytest.raises(librosa.ParameterError)),
    ],
)
@pytest.mark.parametrize("n_fft", [2048, 2049])
def test_time_stretch(ysr, rate, ctx, n_fft):

    with ctx:
        y, sr = ysr
        ys = librosa.effects.time_stretch(y, rate=rate, n_fft=n_fft)

        orig_duration = librosa.get_duration(y=y, sr=sr)
        new_duration = librosa.get_duration(y=ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert np.allclose(orig_duration, rate * new_duration, rtol=1e-2, atol=1e-3)


def test_time_stretch_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.effects.time_stretch(y[0], rate=1.1)
    C1 = librosa.effects.time_stretch(y[1], rate=1.1)
    Call = librosa.effects.time_stretch(y, rate=1.1)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("n_steps", [-1.5, 1.5, 5])
@pytest.mark.parametrize(
    "bins_per_octave,ctx",
    [
        (12, dnr()),
        (24, dnr()),
        (-1, pytest.raises(librosa.ParameterError)),
        (0, pytest.raises(librosa.ParameterError)),
    ],
)
@pytest.mark.parametrize("n_fft", [2048, 2049])
def test_pitch_shift(ysr, n_steps, bins_per_octave, ctx, n_fft):

    with ctx:
        y, sr = ysr
        ys = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave, n_fft=n_fft
        )

        orig_duration = librosa.get_duration(y=y, sr=sr)
        new_duration = librosa.get_duration(y=ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert orig_duration == new_duration


def test_pitch_shift_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.effects.pitch_shift(y[0], sr=sr, n_steps=1)
    C1 = librosa.effects.pitch_shift(y[1], sr=sr, n_steps=1)
    Call = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_mono(align_zeros):

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1], dtype=float)
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2], dtype=float)
    intervals = np.asarray([[2, 4], [6, 8], [0, 2], [8, 10], [4, 6]])

    y_out = librosa.effects.remix(y, intervals, align_zeros=align_zeros)
    assert np.allclose(y_out, y_t)


@pytest.mark.parametrize("align_zeros", [False, True])
def test_remix_stereo(align_zeros):

    # without zc alignment
    y = np.asarray([1, 1, -1, -1, 2, 2, -1, -1, 1, 1], dtype=float)
    y_t = np.asarray([-1, -1, -1, -1, 1, 1, 1, 1, 2, 2], dtype=float)
    y = np.vstack([y, y])
    y_t = np.vstack([y_t, y_t])

    intervals = np.asarray([[2, 4], [6, 8], [0, 2], [8, 10], [4, 6]])

    y_out = librosa.effects.remix(y, intervals, align_zeros=align_zeros)
    assert np.allclose(y_out, y_t)


def test_hpss(ysr):

    y, sr = ysr

    y_harm, y_perc = librosa.effects.hpss(y)

    # Make sure that the residual energy is generally small
    y_residual = y - y_harm - y_perc

    rms_orig = librosa.feature.rms(y=y)
    rms_res = librosa.feature.rms(y=y_residual)

    assert np.percentile(rms_orig, 0.01) > np.percentile(rms_res, 0.99)


def test_hpss_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    CH0, CP0 = librosa.effects.hpss(y[0])
    CH1, CP1 = librosa.effects.hpss(y[1])
    CHall, CPall = librosa.effects.hpss(y)

    # Check each channel
    assert np.allclose(CH0, CHall[0])
    assert np.allclose(CP0, CPall[0])
    assert np.allclose(CH1, CHall[1])
    assert np.allclose(CP1, CPall[1])

    # Verify that they're not all the same
    assert not np.allclose(CHall[0], CHall[1])
    assert not np.allclose(CPall[0], CPall[1])


def test_percussive(ysr):

    y, sr = ysr

    yh1, yp1 = librosa.effects.hpss(y)

    yp2 = librosa.effects.percussive(y)

    assert np.allclose(yp1, yp2)


def test_harmonic(ysr):

    y, sr = ysr

    yh1, yp1 = librosa.effects.hpss(y)

    yh2 = librosa.effects.harmonic(y)

    assert np.allclose(yh1, yh2)


@pytest.fixture(scope="module", params=[False, True], ids=["mono", "stereo"])
def y_trim(request):
    # construct 5 seconds of stereo silence
    # Stick a sine wave in the middle three seconds

    sr = 22050
    trim_duration = 3.0
    y = np.sin(2 * np.pi * 440.0 * np.arange(0, trim_duration * sr) / sr)
    y = librosa.util.pad_center(y, size=5 * sr)

    if request.param:
        y = np.vstack([y, np.zeros_like(y)])
    return y


@pytest.mark.parametrize("top_db", [60, 40, 20])
@pytest.mark.parametrize("ref", [1, np.max])
@pytest.mark.parametrize("trim_duration", [3.0])
def test_trim(y_trim, top_db, ref, trim_duration):

    yt, idx = librosa.effects.trim(y_trim, top_db=top_db, ref=ref)

    # Test for index position
    fidx = [slice(None)] * y_trim.ndim
    fidx[-1] = slice(*idx.tolist())
    assert np.allclose(yt, y_trim[tuple(fidx)])

    # Verify logamp
    rms = librosa.feature.rms(y=librosa.to_mono(yt), center=False)
    logamp = librosa.power_to_db(rms**2, ref=ref, top_db=None)
    assert np.all(logamp > -top_db)

    # Verify logamp
    rms_all = librosa.feature.rms(y=librosa.to_mono(y_trim)).squeeze()
    logamp_all = librosa.power_to_db(rms_all**2, ref=ref, top_db=None)

    start = int(librosa.samples_to_frames(idx[0]))
    stop = int(librosa.samples_to_frames(idx[1]))
    assert np.all(logamp_all[:start] <= -top_db)
    assert np.all(logamp_all[stop:] <= -top_db)

    # Verify duration
    duration = librosa.get_duration(y=yt)
    assert np.allclose(duration, trim_duration, atol=1e-1), duration


def test_trim_empty():

    y = np.zeros(1)

    yt, idx = librosa.effects.trim(y, ref=1)

    assert yt.size == 0
    assert idx[0] == 0
    assert idx[1] == 0


def test_trim_multi(y_multi):
    # Test for https://github.com/librosa/librosa/issues/1489
    y, sr = y_multi
    librosa.effects.trim(y=y)


def test_split_multi(y_multi):
    # Test for https://github.com/librosa/librosa/issues/1489
    y, sr = y_multi

    librosa.effects.split(y=y)


@pytest.fixture(
    scope="module",
    params=[0, 1, 2, 3],
    ids=["constant", "end-silent", "full-signal", "gaps"],
)
def y_split_idx(request):

    sr = 8192
    y = np.ones(5 * sr)

    if request.param == 0:
        # Constant
        idx_true = np.asarray([[0, 5 * sr]])

    elif request.param == 1:
        # end-silent
        y[::2] *= -1
        y[4 * sr :] = 0
        idx_true = np.asarray([[0, 4 * sr]])

    elif request.param == 2:
        # begin-silent
        y[::2] *= -1
        idx_true = np.asarray([[0, 5 * sr]])
    else:
        # begin and end are silent
        y[::2] *= -1

        # Zero out all but two intervals
        y[:sr] = 0
        y[2 * sr : 3 * sr] = 0
        y[4 * sr :] = 0

        # The true non-silent intervals
        idx_true = np.asarray([[sr, 2 * sr], [3 * sr, 4 * sr]])

    return y, idx_true


@pytest.mark.parametrize("frame_length", [1024, 2048, 4096])
@pytest.mark.parametrize("hop_length", [256, 512, 1024])
@pytest.mark.parametrize("top_db", [20, 60, 80])
def test_split(y_split_idx, frame_length, hop_length, top_db):

    y, idx_true = y_split_idx

    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    assert np.all(intervals <= y.shape[-1])

    int_match = librosa.util.match_intervals(intervals, idx_true)

    for i in range(len(intervals)):
        i_true = idx_true[int_match[i]]

        assert np.all(np.abs(i_true - intervals[i]) <= frame_length), intervals[i]


@pytest.mark.parametrize("coef", [0.5, 0.99])
@pytest.mark.parametrize("zi", [None, 0, [0]])
@pytest.mark.parametrize("return_zf", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preemphasis(coef, zi, return_zf: bool, dtype):
    x = np.arange(10, dtype=dtype)

    if return_zf:
        y, zf = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)
    else:
        y = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)

    assert np.allclose(y[1:], x[1:] - coef * x[:-1])
    assert x.dtype == y.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preemphasis_continue(dtype):

    # Compare pre-emphasis computed in parts to that of the whole sequence in one go
    x = np.arange(64, dtype=dtype)

    y1, zf1 = librosa.effects.preemphasis(x[:32], return_zf=True)
    y2, zf2 = librosa.effects.preemphasis(x[32:], return_zf=True, zi=zf1)

    y_all, zf_all = librosa.effects.preemphasis(x, return_zf=True)

    assert np.allclose(y_all, np.concatenate([y1, y2]))
    assert np.allclose(zf2, zf_all)
    assert x.dtype == y_all.dtype


def test_preemphasis_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0, zf0 = librosa.effects.preemphasis(y[0], return_zf=True)
    C1, zf1 = librosa.effects.preemphasis(y[1], return_zf=True)
    Call, zf = librosa.effects.preemphasis(y, return_zf=True)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert np.allclose(zf0, zf[0])
    assert np.allclose(zf1, zf[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])
    assert not np.allclose(zf[0], zf[1])


def test_deemphasis_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0, zf0 = librosa.effects.deemphasis(y[0], return_zf=True)
    C1, zf1 = librosa.effects.deemphasis(y[1], return_zf=True)
    Call, zf = librosa.effects.deemphasis(y, return_zf=True)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])
    assert np.allclose(zf0, zf[0])
    assert np.allclose(zf1, zf[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])
    assert not np.allclose(zf[0], zf[1])


@pytest.mark.parametrize("coef", [0.5, 0.99])
@pytest.mark.parametrize("zi", [None, 0, [0]])
@pytest.mark.parametrize("return_zf", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_deemphasis(coef, zi, return_zf, dtype):
    x = np.arange(10, dtype=dtype)

    y = librosa.effects.preemphasis(x, coef=coef, zi=zi, return_zf=return_zf)

    if return_zf:
        y, zf = y

    y_deemph = librosa.effects.deemphasis(y, coef=coef, zi=zi)

    assert np.allclose(x, y_deemph)
    assert x.dtype == y_deemph.dtype
