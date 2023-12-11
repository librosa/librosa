#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.filters
#
# This test suite verifies that librosa core routines match (numerically) the output
# of various DPWE matlab implementations on a broad range of input parameters.
#
# All test data is generated by the Matlab script "makeTestData.m".
# Each test loads in a .mat file which contains the input and desired output for a given
# function.  The test then runs the librosa implementation and verifies the results
# against the desired output, typically via numpy.allclose().
#

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

from contextlib import nullcontext as dnr
import warnings
import glob
import numpy as np
import scipy.io
import scipy.signal
from typing import Any, ContextManager

import pytest

import librosa


# -- utilities --#
def files(pattern):
    test_files = glob.glob(pattern)
    test_files.sort()
    return test_files


def load(infile):
    DATA = scipy.io.loadmat(infile, chars_as_strings=True)
    return DATA


# --           --#


# -- Tests     --#
@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-hz_to_mel-*.mat"))
)
def test_hz_to_mel(infile):
    DATA = load(infile)
    z = librosa.hz_to_mel(DATA["f"], htk=DATA["htk"])

    assert np.allclose(z, DATA["result"])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-mel_to_hz-*.mat"))
)
def test_mel_to_hz(infile):

    DATA = load(infile)
    z = librosa.mel_to_hz(DATA["f"], htk=DATA["htk"])
    assert np.allclose(z, DATA["result"])

    # Test for scalar conversion too
    z0 = librosa.mel_to_hz(DATA["f"][0], htk=DATA["htk"])
    assert np.allclose(z0, DATA["result"][0])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-hz_to_octs-*.mat"))
)
def test_hz_to_octs(infile):
    DATA = load(infile)
    z = librosa.hz_to_octs(DATA["f"])

    assert np.allclose(z, DATA["result"])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-melfb-*.mat"))
)
@pytest.mark.filterwarnings("ignore:Empty filters detected")
def test_melfb(infile):

    DATA = load(infile)

    wts = librosa.filters.mel(
        sr=DATA["sr"][0, 0],
        n_fft=DATA["nfft"][0, 0],
        n_mels=DATA["nfilts"][0, 0],
        fmin=DATA["fmin"][0, 0],
        fmax=DATA["fmax"][0, 0],
        htk=DATA["htk"][0, 0],
    )

    # Our version only returns the real-valued part.
    # Pad out.
    wts = np.pad(wts, [(0, 0), (0, DATA["nfft"][0, 0] // 2 - 1)], mode="constant")

    assert wts.shape == DATA["wts"].shape
    assert np.allclose(wts, DATA["wts"])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-melfbnorm-*.mat"))
)
def test_melfbnorm(infile):
    DATA = load(infile)
    # if DATA['norm'] is empty, pass None.
    if DATA["norm"].shape[-1] == 0:
        norm = None
    else:
        norm = DATA["norm"][0, 0]
    wts = librosa.filters.mel(
        sr=DATA["sr"][0, 0],
        n_fft=DATA["nfft"][0, 0],
        n_mels=DATA["nfilts"][0, 0],
        fmin=DATA["fmin"][0, 0],
        fmax=DATA["fmax"][0, 0],
        htk=DATA["htk"][0, 0],
        norm=norm,
    )
    # Pad out.
    wts = np.pad(wts, [(0, 0), (0, DATA["nfft"][0, 0] // 2 - 1)], mode="constant")

    assert wts.shape == DATA["wts"].shape
    assert np.allclose(wts, DATA["wts"])


@pytest.mark.parametrize("norm", [1, 2, np.inf])
def test_mel_norm(norm):

    M = librosa.filters.mel(sr=22050, n_fft=2048, norm=norm)
    if norm == 1:
        assert np.allclose(np.sum(np.abs(M), axis=1), 1)
    elif norm == 2:
        assert np.allclose(np.sum(np.abs(M ** 2), axis=1), 1)
    elif norm == np.inf:
        assert np.allclose(np.max(np.abs(M), axis=1), 1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_mel_badnorm():
    librosa.filters.mel(sr=22050, n_fft=2048, norm="garbage")  # type: ignore


def test_mel_gap():

    # This configuration should trigger some empty filters
    sr = 44100
    n_fft = 1024
    fmin = 0
    fmax = 2000
    n_mels = 128
    htk = True

    with pytest.warns(UserWarning, match="Empty filters"):
        librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk)


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "feature-chromafb-*.mat"))
)
def test_chromafb(infile):

    DATA = load(infile)

    octwidth = DATA["octwidth"][0, 0]
    if octwidth == 0:
        octwidth = None

    # Convert A440 parameter to tuning parameter
    A440 = DATA["a440"][0, 0]

    tuning = DATA["nchroma"][0, 0] * (np.log2(A440) - np.log2(440.0))

    wts = librosa.filters.chroma(
        sr=DATA["sr"][0, 0],
        n_fft=DATA["nfft"][0, 0],
        n_chroma=DATA["nchroma"][0, 0],
        tuning=tuning,
        ctroct=DATA["ctroct"][0, 0],
        octwidth=octwidth,
        norm=2,
        base_c=False,
    )

    # Our version only returns the real-valued part.
    # Pad out.
    wts = np.pad(wts, [(0, 0), (0, DATA["nfft"][0, 0] // 2 - 1)], mode="constant")

    assert wts.shape == DATA["wts"].shape
    assert np.allclose(wts, DATA["wts"])


# Testing two tones, 261.63 Hz and 440 Hz
@pytest.mark.parametrize("freq", [261.63, 440])
def test_chroma_issue1295(freq):

    tone_1 = librosa.tone(frequency=freq, sr=22050, duration=1)
    chroma_1 = librosa.feature.chroma_stft(
        y=tone_1, sr=22050, n_chroma=120, base_c=True
    )

    actual_argmax = np.unravel_index(chroma_1.argmax(), chroma_1.shape)

    if freq == 261.63:
        assert actual_argmax == (118, 0) # type: ignore[comparison-overlap]
    elif freq == 440:
        assert actual_argmax == (90, 0) # type: ignore[comparison-overlap]


@pytest.mark.parametrize("n", [16, 16.0, 16.25, 16.75])
@pytest.mark.parametrize(
    "window_name",
    [
        "barthann",
        "bartlett",
        "blackman",
        "blackmanharris",
        "bohman",
        "boxcar",
        "cosine",
        "flattop",
        "hamming",
        "hann",
        "nuttall",
        "parzen",
        "triang",
    ],
)
def test__window(n, window_name):

    window = getattr(scipy.signal.windows, window_name)

    wdec = librosa.filters.__float_window(window)

    if n == int(n):
        n = int(n)
        assert np.allclose(wdec(n), window(n))
    else:
        wf = wdec(n)
        fn = int(np.floor(n))
        assert not np.any(wf[fn:])


@pytest.mark.parametrize("sr", [11025])
@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C3")])
@pytest.mark.parametrize("n_bins", [12, 24])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("filter_scale", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("pad_fft", [False, True])
def test_constant_q(sr, fmin, n_bins, bins_per_octave, filter_scale, pad_fft, norm):

    with pytest.warns(FutureWarning, match="Deprecated"):
        F, lengths = librosa.filters.constant_q(
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
            pad_fft=pad_fft,
            norm=norm,
        )

    assert np.all(lengths <= F.shape[1])

    assert len(F) == n_bins

    if not pad_fft:
        return

    assert np.mod(np.log2(F.shape[1]), 1.0) == 0.0

    # Check for vanishing negative frequencies
    F_fft = np.abs(np.fft.fft(F, axis=1))
    # Normalize by row-wise peak
    F_fft = F_fft / np.max(F_fft, axis=1, keepdims=True)
    assert not np.any(F_fft[:, -F_fft.shape[1] // 2 :] > 1e-4)


@pytest.mark.parametrize("sr", [11025])
@pytest.mark.parametrize("fmin", [librosa.note_to_hz("C3")])
@pytest.mark.parametrize("n_bins", [12, 24])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("filter_scale", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("pad_fft", [False, True])
@pytest.mark.parametrize("gamma", [0, 10, None])
def test_wavelet(sr, fmin, n_bins, bins_per_octave, filter_scale, pad_fft, norm, gamma):

    freqs = librosa.cqt_frequencies(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    F, lengths = librosa.filters.wavelet(
        freqs=freqs,
        sr=sr,
        filter_scale=filter_scale,
        pad_fft=pad_fft,
        norm=norm,
        gamma=gamma
    )

    assert np.all(lengths <= F.shape[1])

    assert len(F) == n_bins

    if not pad_fft:
        return

    assert np.mod(np.log2(F.shape[1]), 1.0) == 0.0

    # Check for vanishing negative frequencies
    F_fft = np.abs(np.fft.fft(F, axis=1))
    # Normalize by row-wise peak
    F_fft = F_fft / np.max(F_fft, axis=1, keepdims=True)
    assert np.max(F_fft[:, -F_fft.shape[1] // 2 :]) < 1e-3


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavelet_lengths_badscale():
    librosa.filters.wavelet_lengths(freqs=2**np.arange(3), filter_scale=-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavelet_lengths_badgamma():
    librosa.filters.wavelet_lengths(freqs=2**np.arange(3), gamma=-1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavelet_lengths_badfreqs():
    librosa.filters.wavelet_lengths(freqs=2**np.arange(3) -20)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavelet_lengths_badfreqsorder():
    librosa.filters.wavelet_lengths(freqs=2**np.arange(3)[::-1])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_wavelet_lengths_noalpha():
    librosa.filters.wavelet_lengths(freqs=[64], alpha=None)



@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "sr,fmin,n_bins,bins_per_octave,filter_scale,norm",
    [
        (11025, 11025 / 2.0, 1, 12, 1, 1),
        (11025, -60, 1, 12, 1, 1),
        (11025, 60, 1, -12, 1, 1),
        (11025, 60, -1, 12, 1, 1),
        (11025, 60, 1, 12, -1, 1),
        (11025, 60, 1, 12, 1, -1),
    ],
)
def test_constant_q_badparams(sr, fmin, n_bins, bins_per_octave, filter_scale, norm):
    with pytest.warns(FutureWarning, match="Deprecated"):
        librosa.filters.constant_q(
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=filter_scale,
            pad_fft=True,
            norm=norm,
        )


def test_window_bandwidth():

    hann_bw = librosa.filters.window_bandwidth("hann")
    hann_scipy_bw = librosa.filters.window_bandwidth(scipy.signal.windows.hann)
    assert hann_bw == hann_scipy_bw


def test_window_bandwidth_dynamic():

    # Test with a window constructor guaranteed to not exist in
    # the dictionary.
    # should behave like a box filter, which has enbw == 1
    assert librosa.filters.window_bandwidth(lambda n: np.ones(n)) == 1


@pytest.mark.xfail(raises=ValueError)
def test_window_bandwidth_missing():
    librosa.filters.window_bandwidth("made up window name")


def binstr(m):

    out = []
    for row in m:
        line = [" "] * len(row)
        for i in np.flatnonzero(row):
            line[i] = "."
        out.append("".join(line))
    return "\n".join(out)


@pytest.mark.parametrize("n_octaves", [2, 3, 4])
@pytest.mark.parametrize("semitones", [1, 3])
@pytest.mark.parametrize("n_chroma", [12, 24, 36])
@pytest.mark.parametrize("fmin", [None] + list(librosa.midi_to_hz(range(48, 61))))
@pytest.mark.parametrize("base_c", [False, True])
@pytest.mark.parametrize("window", [None, [1]])
def test_cq_to_chroma(n_octaves, semitones, n_chroma, fmin, base_c, window):

    bins_per_octave = 12 * semitones
    n_bins = n_octaves * bins_per_octave

    ctx: ContextManager[Any]
    if np.mod(bins_per_octave, n_chroma) != 0:
        ctx = pytest.raises(librosa.ParameterError)
    else:
        ctx = dnr()

    with ctx:
        # Fake up a cqt matrix with the corresponding midi notes

        if fmin is None:
            midi_base = 24  # C2
        else:
            midi_base = librosa.hz_to_midi(fmin)

        midi_notes = np.linspace(
            midi_base,
            midi_base + n_bins * 12.0 / bins_per_octave,
            endpoint=False,
            num=n_bins,
        )
        #  We don't care past 2 decimals here.
        # the log2 inside hz_to_midi can cause problems though.
        midi_notes = np.around(midi_notes, decimals=2)
        C = np.diag(midi_notes)

        cq2chr = librosa.filters.cq_to_chroma(
            n_input=C.shape[0],
            bins_per_octave=bins_per_octave,
            n_chroma=n_chroma,
            fmin=fmin,
            base_c=base_c,
            window=window,
        )

        chroma = cq2chr.dot(C)
        for i in range(n_chroma):
            v = chroma[i][chroma[i] != 0]
            v = np.around(v, decimals=2)

            if base_c:
                resid = np.mod(v, 12)
            else:
                resid = np.mod(v - 9, 12)

            resid = np.round(resid * n_chroma / 12.0)
            assert np.allclose(np.mod(i - resid, 12), 0.0), i - resid


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_get_window_fail():

    librosa.filters.get_window(None, 32) # type: ignore


@pytest.mark.parametrize("window", ["hann", "hann", 4.0, ("kaiser", 4.0)])
def test_get_window(window):

    w1 = librosa.filters.get_window(window, 32)
    w2 = scipy.signal.get_window(window, 32)

    assert np.allclose(w1, w2)


def test_get_window_func():

    w1 = librosa.filters.get_window(scipy.signal.windows.boxcar, 32)
    w2 = scipy.signal.get_window("boxcar", 32)
    assert np.allclose(w1, w2)


@pytest.mark.parametrize(
    "pre_win", [scipy.signal.windows.hann(16), list(scipy.signal.windows.hann(16)), [1, 1, 1]]
)
def test_get_window_pre(pre_win):
    win = librosa.filters.get_window(pre_win, len(pre_win))
    assert np.allclose(win, pre_win)


def test_semitone_filterbank():
    # We test against Chroma Toolbox' elliptical semitone filterbank
    # load data from chroma toolbox
    gt_fb = scipy.io.loadmat(
        os.path.join(
            "tests", "data", "filter-muliratefb-MIDI_FB_ellip_pitch_60_96_22050_Q25"
        ),
        squeeze_me=True,
    )["h"]

    # standard parameters reproduce settings from chroma toolbox
    mut_ft_ba, mut_srs_ba = librosa.filters.semitone_filterbank(flayout="ba")
    mut_ft_sos, mut_srs_sos = librosa.filters.semitone_filterbank(flayout="sos")

    for cur_filter_id in range(len(mut_ft_ba)):
        cur_filter_gt = gt_fb[cur_filter_id + 23]
        cur_filter_mut = mut_ft_ba[cur_filter_id]
        cur_filter_mut_sos = scipy.signal.sos2tf(mut_ft_sos[cur_filter_id])

        cur_a_gt = cur_filter_gt[0]
        cur_b_gt = cur_filter_gt[1]
        cur_a_mut = cur_filter_mut[1]
        cur_b_mut = cur_filter_mut[0]
        cur_a_mut_sos = cur_filter_mut_sos[1]
        cur_b_mut_sos = cur_filter_mut_sos[0]

        # we deviate from the chroma toolboxes for pitches 94 and 95
        # (filters 70 and 71) by processing them with a higher samplerate
        if (cur_filter_id != 70) and (cur_filter_id != 71):
            assert np.allclose(cur_a_gt, cur_a_mut)
            assert np.allclose(cur_b_gt, cur_b_mut, atol=1e-4)

            assert np.allclose(cur_a_gt, cur_a_mut_sos)
            assert np.allclose(cur_b_gt, cur_b_mut_sos, atol=1e-4)


@pytest.mark.parametrize("n", [9, 17])
@pytest.mark.parametrize("window", ["hann", "rect"])
@pytest.mark.parametrize("angle", [None, np.pi / 4, np.pi / 6])
@pytest.mark.parametrize("slope", [1, 2, 0.5])
@pytest.mark.parametrize("zero_mean", [False, True])
def test_diagonal_filter(n, window, angle, slope, zero_mean):

    kernel = librosa.filters.diagonal_filter(
        window, n, slope=slope, angle=angle, zero_mean=zero_mean
    )

    # In the no-rotation case, check that the filter is shaped correctly
    if angle == np.pi / 4 and not zero_mean:
        win_unnorm = librosa.filters.get_window(window, n, fftbins=False)
        win_unnorm /= win_unnorm.sum()
        assert np.allclose(np.diag(kernel), win_unnorm)

    # First check: zero-mean
    if zero_mean:
        assert np.isclose(kernel.sum(), 0)
    else:
        assert np.isclose(kernel.sum(), 1) and np.all(kernel >= 0)

    # Now check if the angle transposes correctly
    if angle is None:
        # If we're using the slope API, then the transposed kernel
        # will have slope 1/slope
        k2 = librosa.filters.diagonal_filter(
            window, n, slope=1.0 / slope, angle=angle, zero_mean=zero_mean
        )
    else:
        # If we're using the angle API, then the transposed kernel
        # will have angle pi/2 - angle
        k2 = librosa.filters.diagonal_filter(
            window, n, slope=slope, angle=np.pi / 2 - angle, zero_mean=zero_mean
        )

    assert np.allclose(k2, kernel.T)
