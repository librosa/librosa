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

from typing import Optional
import librosa
import numpy as np
import scipy.stats

import pytest

from test_core import srand


def __test_cqt_size(
    y,
    sr,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    sparsity,
    res_type,
):

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


def make_signal(sr, duration, fmin: Optional[str] = "C1", fmax: Optional[str] = "C8"):
    """Generates a linear sine sweep"""

    if fmin is None:
        fmin_normfreq = 0.01
    else:
        fmin_normfreq = librosa.note_to_hz(fmin) / sr

    if fmax is None:
        fmax_normfreq = 0.5
    else:
        fmax_normfreq = librosa.note_to_hz(fmax) / sr

    return np.sin(
        np.cumsum(
            2
            * np.pi
            * np.logspace(
                np.log10(fmin_normfreq), np.log10(fmax_normfreq), num=int(duration * sr)
            )
        )
    )


@pytest.fixture(scope="module")
def sr_cqt():
    return 11025


@pytest.fixture(scope="module")
def y_cqt(sr_cqt):
    return make_signal(sr_cqt, 2.0)


@pytest.fixture(scope="module")
def y_cqt_110(sr_cqt):
    return librosa.tone(110.0, sr=sr_cqt, duration=0.75)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("hop_length", [-1, 0])
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_bad_hop(y_cqt, sr_cqt, hop_length, bpo):
    # incorrect hop lengths for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 15
    librosa.cqt(
        y=y_cqt, sr=sr_cqt, hop_length=hop_length, n_bins=bpo * 6, bins_per_octave=bpo
    )


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_exceed_passband(y_cqt, sr_cqt, bpo):
    # Filters going beyond nyquist: 500 Hz -> 4 octaves = 8000 > 11025/2
    librosa.cqt(y=y_cqt, sr=sr_cqt, fmin=500, n_bins=4 * bpo, bins_per_octave=bpo)


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("filter_scale", [1])
@pytest.mark.parametrize("norm", [1])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("hop_length", [512, 2000])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")  # this is fine here
@pytest.mark.filterwarnings(
    "ignore:Trying to estimate tuning"
)  # we can ignore this too
def test_cqt(
    y_cqt_110,
    sr_cqt,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    res_type,
    sparsity,
):

    C = librosa.cqt(
        y=y_cqt_110,
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

    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    # check for peaks if 110 is within range
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)

        # This is our most common peak index in the CQT spectrum
        # we use the mode here over frames to sidestep transient effects
        # at the beginning and end of the CQT
        # common_peak = scipy.stats.mode(peaks, keepdims=True)[0][0]
        common_peak = np.argmax(np.bincount(peaks))

        # Convert peak index to frequency
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)

        # Check that it matches 110, which is an analysis frequency
        assert np.isclose(peak_frequency, 110)


@pytest.mark.parametrize("fmin", [librosa.note_to_hz("C1")])
@pytest.mark.parametrize("bins_per_octave", [12])
@pytest.mark.parametrize("n_bins", [88])
def test_cqt_early_downsample(y_cqt_110, sr_cqt, n_bins, fmin, bins_per_octave):
    with pytest.warns(FutureWarning, match="Support for VQT with res_type=None"):
        C = librosa.cqt(
            y=y_cqt_110,
            sr=sr_cqt,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            res_type=None,
        )

    # type is complex
    assert np.iscomplexobj(C)

    # number of bins is correct
    assert C.shape[0] == n_bins

    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    # check for peaks if 110 is within range
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)

        # This is our most common peak index in the CQT spectrum
        # we use the mode here over frames to sidestep transient effects
        # at the beginning and end of the CQT
        # common_peak = scipy.stats.mode(peaks, keepdims=True)[0][0]
        common_peak = np.argmax(np.bincount(peaks))

        # Convert peak index to frequency
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)

        # Check that it matches 110, which is an analysis frequency
        assert np.isclose(peak_frequency, 110)


@pytest.mark.parametrize("hop_length", [256, 512])
def test_cqt_frame_rate(y_cqt_110, sr_cqt, hop_length):

    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=hop_length, res_type="polyphase")

    # At sr=11025, hop of 256 gives 17 frames for default
    # analysis
    # hop of 512 gives 33 frames

    if hop_length == 256:
        assert C.shape[1] == 33
    elif hop_length == 512:
        assert C.shape[1] == 17
    else:
        # Unsupported test case
        assert False


def test_cqt_odd_hop(y_cqt_110, sr_cqt):
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=1001, res_type="polyphase")


def test_icqt_odd_hop(y_cqt_110, sr_cqt):
    C = librosa.cqt(y=y_cqt_110, sr=sr_cqt, hop_length=1001, res_type="polyphase")
    yi = librosa.icqt(
        C, sr=sr_cqt, hop_length=1001, res_type="polyphase", length=len(y_cqt_110)
    )


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24])
@pytest.mark.parametrize("gamma", [None, 0, 2.5])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [0])
@pytest.mark.parametrize("filter_scale", [1])
@pytest.mark.parametrize("norm", [1])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("hop_length", [512])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_vqt(
    y_cqt_110,
    sr_cqt,
    hop_length,
    fmin,
    n_bins,
    gamma,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    res_type,
    sparsity,
):

    C = librosa.vqt(
        y=y_cqt_110,
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

    if fmin is None:
        fmin = librosa.note_to_hz("C1")

    # check for peaks if 110 is within range
    if 110 <= fmin * 2 ** (n_bins / bins_per_octave):
        peaks = np.argmax(np.abs(C), axis=0)

        # This is our most common peak index in the CQT spectrum
        # we use the mode here over frames to sidestep transient effects
        # at the beginning and end of the CQT
        # common_peak = scipy.stats.mode(peaks, keepdims=True)[0][0]
        common_peak = np.argmax(np.bincount(peaks))

        # Convert peak index to frequency
        peak_frequency = fmin * 2 ** (common_peak / bins_per_octave)

        # Check that it matches 110, which is an analysis frequency
        assert np.isclose(peak_frequency, 110)


@pytest.fixture(scope="module")
def y_hybrid():
    return make_signal(11025, 5.0, None)


@pytest.mark.parametrize("sr", [11025])
@pytest.mark.parametrize("hop_length", [512, 2000])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 48, 72, 74, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("resolution", [1])
@pytest.mark.parametrize("norm", [1])
@pytest.mark.parametrize("res_type", ["polyphase"])
def test_hybrid_cqt(
    y_hybrid,
    sr,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    resolution,
    norm,
    sparsity,
    res_type,
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

    assert np.percentile(np.abs(C1[idx] - C2[idx]), perc) < thresh * max(
        C1.max(), C2.max()
    )


@pytest.mark.parametrize("note_min", [12, 18, 24, 30, 36])
@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize(
    "y", [np.sin(2 * np.pi * librosa.midi_to_hz(60) * np.arange(2 * 22050) / 22050.0)]
)
def test_cqt_position(y, sr, note_min: int):

    C = np.abs(librosa.cqt(y, sr=sr, fmin=float(librosa.midi_to_hz(note_min)))) ** 2

    # Average over time
    Cbar = np.median(C, axis=1)

    # Find the peak
    idx = int(np.argmax(Cbar))

    assert idx == 60 - note_min

    # Make sure that the max outside the peak is sufficiently small
    Cscale = Cbar / Cbar[idx]
    Cscale[idx] = np.nan
    assert np.nanmax(Cscale) < 6e-1, Cscale

    Cscale[idx - 1 : idx + 2] = np.nan
    assert np.nanmax(Cscale) < 5e-2, Cscale


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36)


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


@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #348
    # Updated in #417 to use integrated energy, rather than frame-wise max

    C = np.abs(librosa.cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse))

    response = np.mean(C**2, axis=1)

    continuity = np.abs(np.diff(response))

    # Test that integrated energy is approximately constant
    assert np.max(continuity) < 5e-4, continuity


@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_hybrid_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #341
    # Updated in #417 to use integrated energy instead of pointwise max

    hcqt = librosa.hybrid_cqt(
        y=y_impulse, sr=sr_impulse, hop_length=hop_impulse, tuning=0
    )

    response = np.mean(np.abs(hcqt) ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    assert np.max(continuity) < 5e-4, continuity


@pytest.fixture(scope="module")
def sr_white():
    return 22050


@pytest.fixture(scope="module")
def y_white(sr_white):
    srand()
    return np.random.randn(10 * sr_white)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [24, 36])
def test_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):

    C = np.abs(
        librosa.cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale)
    )

    if not scale:
        freqs = librosa.cqt_frequencies(fmin=fmin, n_bins=n_bins)
        lengths, _ = librosa.filters.wavelet_lengths(sr=sr_white, freqs=freqs)
        C /= np.sqrt(lengths[:, np.newaxis])

    # Only compare statistics across the time dimension
    # we want ~ constant mean and variance across frequencies
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [72, 84])
def test_hybrid_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):
    C = librosa.hybrid_cqt(
        y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale
    )

    if not scale:
        freqs = fmin * 2.0 ** (np.arange(n_bins) / 12)
        lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr_white)
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
@pytest.mark.parametrize("res_type", ["soxr_hq", "polyphase"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.filterwarnings(
    "ignore:n_fft=.*is too large"
)  # our test signal is short; this is fine
def test_icqt(y_icqt, sr_icqt, scale, hop_length, over_sample, length, res_type, dtype):

    bins_per_octave = over_sample * 12
    n_bins = 7 * bins_per_octave

    C = librosa.cqt(
        y_icqt,
        sr=sr_icqt,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        scale=scale,
        hop_length=hop_length,
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
        yinv = librosa.util.fix_length(yinv, size=len(y_icqt))

    y_icqt = y_icqt[sr_icqt // 2 : -sr_icqt // 2]
    yinv = yinv[sr_icqt // 2 : -sr_icqt // 2]

    residual = np.abs(y_icqt - yinv)
    # We'll tolerate 10% RMSE
    # error is lower on more recent numpy/scipy builds

    resnorm = np.sqrt(np.mean(residual**2))
    assert resnorm <= 0.1, resnorm


@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(fmin=55, fmax=55 * 2**3, length=sr // 8, sr=sr)
    return y


@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("use_length", [False, True])
@pytest.mark.parametrize("over_sample", [1, 3])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("pad_mode", ["reflect"])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("momentum", [0.99])
@pytest.mark.parametrize("random_state", [0])
@pytest.mark.parametrize("fmin", [40.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", [None])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
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
        n_iter=2,
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


@pytest.mark.parametrize("momentum", [0, 0.95])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_griffinlim_cqt_momentum(y_chirp, momentum):

    C = librosa.cqt(y=y_chirp, sr=22050, res_type="polyphase")
    y_rec = librosa.griffinlim_cqt(
        np.abs(C), sr=22050, n_iter=2, momentum=momentum, res_type="polyphase"
    )

    assert np.all(np.isfinite(y_rec))


@pytest.mark.parametrize("random_state", [None, 0, np.random.RandomState()])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_griffinlim_cqt_rng(y_chirp, random_state):

    C = librosa.cqt(y=y_chirp, sr=22050, res_type="polyphase")
    y_rec = librosa.griffinlim_cqt(
        np.abs(C), sr=22050, n_iter=2, random_state=random_state, res_type="polyphase"
    )

    assert np.all(np.isfinite(y_rec))


@pytest.mark.parametrize("init", [None, "random"])
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_griffinlim_cqt_init(y_chirp, init):
    C = librosa.cqt(y=y_chirp, sr=22050, res_type="polyphase")
    y_rec = librosa.griffinlim_cqt(
        np.abs(C), sr=22050, n_iter=2, init=init, res_type="polyphase"
    )

    assert np.all(np.isfinite(y_rec))


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_griffinlim_cqt_badinit():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, init="garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.filterwarnings("ignore:n_fft=.*is too large")
def test_griffinlim_cqt_badrng():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, random_state="garbage")  # type: ignore


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_bad_momentum():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, momentum=-1)


def test_griffinlim_cqt_momentum_warn():
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim_cqt(x, momentum=2)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cqt_precision(y_cqt, sr_cqt, dtype):
    C = librosa.cqt(y=y_cqt, sr=sr_cqt, dtype=dtype)
    assert np.dtype(C.dtype) == np.dtype(dtype)


@pytest.mark.parametrize("n_bins_missing", range(-11, 11))
def test_cqt_partial_octave(y_cqt, sr_cqt, n_bins_missing):
    # Test what happens when n_bins is +- 1 bin from complete octaves
    librosa.cqt(y=y_cqt, sr=sr_cqt, n_bins=72 - n_bins_missing, bins_per_octave=12)


def test_vqt_provided_intervals(y_cqt, sr_cqt):

    # Generate a 20-ET vqt
    V1 = librosa.vqt(
        y=y_cqt, sr=sr_cqt, bins_per_octave=20, n_bins=60, intervals="equal"
    )

    # Generate the same thing with a pre-set list of intervals
    intervals = 2.0 ** (np.arange(20) / 20.0)

    V2 = librosa.vqt(y=y_cqt, sr=sr_cqt, n_bins=60, intervals=intervals)

    assert np.allclose(V1, V2)
