#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq
"""
from __future__ import division

import warnings

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import librosa
import numpy as np

import pytest

from test_core import srand


def __test_cqt_size(y, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, sparsity, res_type):

    cqt_output = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            filter_scale=filter_scale,
            norm=norm,
            sparsity=sparsity,
            res_type=res_type,
        )
    )

    assert cqt_output.shape[0] == n_bins

    return cqt_output


def __test_vqt_size(y, sr, hop_length, fmin, n_bins, gamma, bins_per_octave,
                    tuning, filter_scale, norm, sparsity, res_type):

    vqt_output = np.abs(
        librosa.vqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            gamma=gamma,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            filter_scale=filter_scale,
            norm=norm,
            sparsity=sparsity,
            res_type=res_type))

    assert vqt_output.shape[0] == n_bins

    return vqt_output


def make_signal(sr, duration, fmin="C1", fmax="C8"):
    """ Generates a linear sine sweep """

    if fmin is None:
        fmin = 0.01
    else:
        fmin = librosa.note_to_hz(fmin) / sr

    if fmax is None:
        fmax = 0.5
    else:
        fmax = librosa.note_to_hz(fmax) / sr

    return np.sin(np.cumsum(2 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax), num=int(duration * sr))))


@pytest.fixture(scope="module")
def sr_cqt():
    return 11025


@pytest.fixture(scope="module")
def y_cqt(sr_cqt):
    return make_signal(sr_cqt, 5.0)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("hop_length", [-1, 0, 16, 63, 65])
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_bad_hop(y_cqt, sr_cqt, hop_length, bpo):
    # incorrect hop lengths for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 15
    librosa.cqt(y=y_cqt, sr=sr_cqt, hop_length=hop_length, n_bins=bpo * 6, bins_per_octave=bpo)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_exceed_passband(y_cqt, sr_cqt, bpo):
    # Filters going beyond nyquist: 500 Hz -> 4 octaves = 8000 > 11025/2
    librosa.cqt(y=y_cqt, sr=sr_cqt, fmin=500, n_bins=4 * bpo, bins_per_octave=bpo)


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("filter_scale", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("hop_length", [256, 512])
def test_cqt(y_cqt, sr_cqt, hop_length, fmin, n_bins, bins_per_octave, tuning, filter_scale, norm, res_type, sparsity):

    C = librosa.cqt(
        y=y_cqt,
        sr=sr_cqt,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
    )

    # type is complex
    assert np.iscomplexobj(C)

    # number of bins is correct
    assert C.shape[0] == n_bins


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 76])
@pytest.mark.parametrize("gamma", [None, 0, 2.5])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("filter_scale", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("hop_length", [256, 512])
def test_vqt(y_cqt, sr_cqt, hop_length, fmin, n_bins, gamma,
             bins_per_octave, tuning, filter_scale, norm, res_type, sparsity):

    C = librosa.vqt(
        y=y_cqt,
        sr=sr_cqt,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=gamma,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
    )

    # type is complex
    assert np.iscomplexobj(C)

    # number of bins is correct
    assert C.shape[0] == n_bins


@pytest.fixture(scope="module")
def y_hybrid():
    return make_signal(11025, 5.0, None)


@pytest.mark.parametrize("sr", [11025])
@pytest.mark.parametrize("hop_length", [512])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 48, 72, 74, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("resolution", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
def test_hybrid_cqt(
    y_hybrid, sr, hop_length, fmin, n_bins, bins_per_octave, tuning, resolution, norm, sparsity, res_type
):
    # This test verifies that hybrid and full cqt agree down to 1e-4
    # on 99% of bins which are nonzero (> 1e-8) in either representation.

    C2 = librosa.hybrid_cqt(
        y_hybrid,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=resolution,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
    )

    C1 = np.abs(
        librosa.cqt(
            y_hybrid,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            filter_scale=resolution,
            norm=norm,
            sparsity=sparsity,
            res_type=res_type,
        )
    )

    assert C1.shape == C2.shape

    # Check for numerical comparability
    idx1 = C1 > 1e-4 * C1.max()
    idx2 = C2 > 1e-4 * C2.max()

    perc = 0.99

    thresh = 1e-3

    idx = idx1 | idx2

    assert np.percentile(np.abs(C1[idx] - C2[idx]), perc) < thresh * max(C1.max(), C2.max())


@pytest.mark.parametrize("note_min", [12, 18, 24, 30, 36])
@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize("y", [np.sin(2 * np.pi * librosa.midi_to_hz(60) * np.arange(2 * 22050) / 22050.0)])
def test_cqt_position(y, sr, note_min):

    C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min))) ** 2

    # Average over time
    Cbar = np.median(C, axis=1)

    # Find the peak
    idx = np.argmax(Cbar)

    assert idx == 60 - note_min

    # Make sure that the max outside the peak is sufficiently small
    Cscale = Cbar / Cbar[idx]
    Cscale[idx] = np.nan
    assert np.nanmax(Cscale) < 6e-1, Cscale

    Cscale[idx - 1 : idx + 2] = np.nan
    assert np.nanmax(Cscale) < 5e-2, Cscale


@pytest.mark.parametrize("note_min", [12, 18, 24, 30, 36])
@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize("y", [np.sin(2 * np.pi * librosa.midi_to_hz(60) * np.arange(2 * 22050) / 22050.0)])
def test_vqt_position(y, sr, note_min):

    C = np.abs(librosa.vqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min)))**2

    # Average over time
    Cbar = np.median(C, axis=1)

    # Find the peak
    idx = np.argmax(Cbar)

    assert idx == 60 - note_min

    # Make sure that the max outside the peak is sufficiently small
    Cscale = Cbar / Cbar[idx]
    Cscale[idx] = np.nan
    assert np.nanmax(Cscale) < 7.5e-1, Cscale

    Cscale[idx-1:idx+2] = np.nan
    assert np.nanmax(Cscale) < 2.5e-1, Cscale


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_vqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.vqt(y, sr=44100, n_bins=36)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_late():

    y = np.zeros(16)
    librosa.cqt(y, sr=22050)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_vqt_fail_short_late():

    y = np.zeros(16)
    librosa.vqt(y, sr=22050)


@pytest.fixture(scope="module", params=[11025, 16384, 22050, 32000, 44100])
def sr_impulse(request):
    return request.param


@pytest.fixture(scope="module", params=range(1, 9))
def hop_impulse(request):
    return 64 * request.param


@pytest.fixture(scope="module")
def y_impulse(sr_impulse, hop_impulse):
    x = np.zeros(sr_impulse)
    center = int((len(x) / (2.0 * float(hop_impulse))) * hop_impulse)
    x[center] = 1
    return x


def test_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #348
    # Updated in #417 to use integrated energy, rather than frame-wise max

    C = np.abs(librosa.cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse))

    response = np.mean(C ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    # Test that integrated energy is approximately constant
    assert np.max(continuity) < 5e-4, continuity


def test_vqt_impulse(y_impulse, sr_impulse, hop_impulse):

    C = np.abs(librosa.vqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse))

    response = np.mean(C ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    # Test that integrated energy is approximately constant
    assert np.max(continuity) < 5e-4, continuity

def test_hybrid_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #341
    # Updated in #417 to use integrated energy instead of pointwise max

    hcqt = librosa.hybrid_cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse, tuning=0)

    response = np.mean(np.abs(hcqt) ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    assert np.max(continuity) < 5e-4, continuity


@pytest.fixture(scope="module")
def sr_white():
    return 22050


@pytest.fixture(scope="module")
def y_white(sr_white):
    srand()
    return np.random.randn(30 * sr_white)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [24, 36])
def test_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):

    C = np.abs(librosa.cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale))

    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr_white, fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])

    # Only compare statistics across the time dimension
    # we want ~ constant mean and variance across frequencies
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [24, 36])
@pytest.mark.parametrize("gamma", [2.5])
def test_vqt_white_noise(y_white, sr_white, fmin, n_bins, gamma, scale):

    C = np.abs(librosa.vqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, gamma=gamma, scale=scale))

    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr_white, fmin, n_bins=n_bins, gamma=gamma)
        C /= np.sqrt(lengths[:, np.newaxis])

    # Only compare statistics across the time dimension
    # we want ~ constant mean and variance across frequencies
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [72, 84])
def test_hybrid_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):
    C = librosa.hybrid_cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale)

    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr_white, fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])

    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.fixture(scope="module", params=[22050, 44100])
def sr_icqt(request):
    return request.param


@pytest.fixture(scope="module")
def y_icqt(sr_icqt):
    return make_signal(sr_icqt, 1.5, fmin="C2", fmax="C4")


@pytest.mark.parametrize("over_sample", [1, 3])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("hop_length", [384, 512])
@pytest.mark.parametrize("length", [None, True])
@pytest.mark.parametrize("res_type", ["scipy", "kaiser_fast", "polyphase"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_icqt(y_icqt, sr_icqt, scale, hop_length, over_sample, length, res_type, dtype):

    bins_per_octave = over_sample * 12
    n_bins = 7 * bins_per_octave

    C = librosa.cqt(
        y_icqt, sr=sr_icqt, n_bins=n_bins, bins_per_octave=bins_per_octave, scale=scale, hop_length=hop_length
    )

    if length:
        _len = len(y_icqt)
    else:
        _len = None
    yinv = librosa.icqt(
        C,
        sr=sr_icqt,
        scale=scale,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        length=_len,
        res_type=res_type,
        dtype=dtype,
    )

    assert yinv.dtype == dtype

    # Only test on the middle section
    if length:
        assert len(y_icqt) == len(yinv)
    else:
        yinv = librosa.util.fix_length(yinv, len(y_icqt))

    y_icqt = y_icqt[sr_icqt // 2 : -sr_icqt // 2]
    yinv = yinv[sr_icqt // 2 : -sr_icqt // 2]

    residual = np.abs(y_icqt - yinv)
    # We'll tolerate 10% RMSE
    # error is lower on more recent numpy/scipy builds

    resnorm = np.sqrt(np.mean(residual ** 2))
    assert resnorm <= 0.1, resnorm


@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(55, 55 * 2 ** 3, length=sr // 8, sr=sr)
    return y


@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("use_length", [False, True])
@pytest.mark.parametrize("over_sample", [1, 3])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("pad_mode", ["reflect"])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("momentum", [0, 0.99])
@pytest.mark.parametrize("random_state", [None, 0, np.random.RandomState()])
@pytest.mark.parametrize("fmin", [40.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", [None, "random"])
def test_griffinlim_cqt(
    y_chirp,
    hop_length,
    window,
    use_length,
    over_sample,
    fmin,
    res_type,
    pad_mode,
    scale,
    momentum,
    init,
    random_state,
    dtype,
):

    if use_length:
        length = len(y_chirp)
    else:
        length = None

    sr = 22050
    bins_per_octave = 12 * over_sample
    n_bins = 6 * bins_per_octave
    C = librosa.cqt(
        y_chirp,
        sr=sr,
        hop_length=hop_length,
        window=window,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
        scale=scale,
        pad_mode=pad_mode,
        res_type=res_type,
    )

    Cmag = np.abs(C)

    y_rec = librosa.griffinlim_cqt(
        Cmag,
        hop_length=hop_length,
        window=window,
        sr=sr,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        scale=scale,
        pad_mode=pad_mode,
        n_iter=3,
        momentum=momentum,
        random_state=random_state,
        length=length,
        res_type=res_type,
        init=init,
        dtype=dtype,
    )

    y_inv = librosa.icqt(
        Cmag,
        sr=sr,
        fmin=fmin,
        hop_length=hop_length,
        window=window,
        bins_per_octave=bins_per_octave,
        scale=scale,
        length=length,
        res_type=res_type,
    )

    # First check for length
    if use_length:
        assert len(y_rec) == length

    assert y_rec.dtype == dtype

    # Check that the data is okay
    assert np.all(np.isfinite(y_rec))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_badinit():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, init="garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_momentum():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, momentum=-1)


def test_griffinlim_cqt_momentum_warn():
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim_cqt(x, momentum=2)
