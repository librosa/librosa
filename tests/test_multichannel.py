#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for multi-channel functionality
#

from __future__ import print_function

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import librosa
import glob
import numpy as np
import scipy.io
import pytest
import warnings
from unittest import mock


@pytest.fixture(scope="module", params=["test1_44100.wav"])
def y_multi(request):
    infile = request.param
    return librosa.load(os.path.join("tests", "data", infile), sr=None, mono=False)


@pytest.fixture(scope="module")
def s_multi(y_multi):
    y, sr = y_multi
    return np.abs(librosa.stft(y)), sr


@pytest.fixture(scope="module")
def tfr_multi(y_multi):
    y, sr = y_multi
    return librosa.reassigned_spectrogram(y, fill_nan=True)


def test_stft_multi(y_multi):

    # Verify that a stereo STFT matches on
    # each channel individually
    y, sr = y_multi

    D = librosa.stft(y)

    D0 = librosa.stft(y[0])
    D1 = librosa.stft(y[1])

    # Check each channel
    assert np.allclose(D[0], D0)
    assert np.allclose(D[1], D1)

    # Check that they're not both the same
    assert not np.allclose(D0, D1)


def test_istft_multi(y_multi):

    # Verify that a stereo ISTFT matches on each channel
    y, sr = y_multi

    # Assume the forward transform works properly in stereo
    D = librosa.stft(y)

    # Invert per channel
    y0m = librosa.istft(D[0])
    y1m = librosa.istft(D[1])

    # Invert both channels at once
    ys = librosa.istft(D)

    # Check each channel
    assert np.allclose(y0m, ys[0])
    assert np.allclose(y1m, ys[1])

    # Check that they're not both the same
    assert not np.allclose(ys[0], ys[1])


def test_griffinlim_multi(y_multi):
    y, sr = y_multi

    # Compute the stft
    D = librosa.stft(y)

    # Run a couple of iterations of griffin-lim
    yout = librosa.griffinlim(np.abs(D), n_iter=2, length=y.shape[-1])

    # Check the lengths
    assert np.allclose(y.shape, yout.shape)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
def test_cqt_multi(y_multi, scale, res_type):

    y, sr = y_multi

    # Assuming single-channel CQT is well behaved
    C0 = librosa.cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.cqt(y=y, sr=sr, scale=scale, res_type=res_type)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
def test_hybrid_cqt_multi(y_multi, scale, res_type):

    y, sr = y_multi

    # Assuming single-channel CQT is well behaved
    C0 = librosa.hybrid_cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.hybrid_cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.hybrid_cqt(y=y, sr=sr, scale=scale, res_type=res_type)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("length", [None, 22050])
def test_icqt_multi(y_multi, scale, length):

    y, sr = y_multi

    # Assuming the forward transform is well-behaved
    C = librosa.cqt(y=y, sr=sr, scale=scale)

    yboth = librosa.icqt(C, sr=sr, scale=scale, length=length)
    y0 = librosa.icqt(C[0], sr=sr, scale=scale, length=length)
    y1 = librosa.icqt(C[1], sr=sr, scale=scale, length=length)

    if length is not None:
        assert yboth.shape[-1] == length

    # Check each channel
    assert np.allclose(yboth[0], y0)
    assert np.allclose(yboth[1], y1)

    # Check that they're not the same
    assert not np.allclose(yboth[0], yboth[1])


def test_griffinlim_cqt_multi(y_multi):
    y, sr = y_multi

    # Compute the stft
    C = librosa.cqt(y, sr=sr)

    # Run a couple of iterations of griffin-lim
    yout = librosa.griffinlim_cqt(np.abs(C), n_iter=2, length=y.shape[-1])

    # Check the lengths
    assert np.allclose(y.shape, yout.shape)


def test_spectral_centroid_multi(s_multi):

    S, sr = s_multi

    freq = None

    # Assuming single-channel CQT is well behaved
    C0 = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_centroid_multi_variable(s_multi):

    S, sr = s_multi

    freq = np.random.randn(*S.shape)

    # compare each channel
    C0 = librosa.feature.spectral_centroid(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_centroid(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_centroid(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_bandwidth_multi(s_multi):
    S, sr = s_multi

    freq = None

    # compare each channel
    C0 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_bandwidth_multi_variable(s_multi):
    S, sr = s_multi

    freq = np.random.randn(*S.shape)

    # compare each channel
    C0 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_bandwidth(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_bandwidth(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_contrast_multi(s_multi):
    S, sr = s_multi

    freq = None

    # compare each channel
    C0 = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_contrast(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_rolloff_multi(s_multi):
    S, sr = s_multi

    freq = None

    # compare each channel
    C0 = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S[0])
    C1 = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S[1])
    Call = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_rolloff_multi_variable(s_multi):
    S, sr = s_multi

    freq = np.random.randn(*S.shape)

    # compare each channel
    C0 = librosa.feature.spectral_rolloff(sr=sr, freq=freq[0], S=S[0])
    C1 = librosa.feature.spectral_rolloff(sr=sr, freq=freq[1], S=S[1])
    Call = librosa.feature.spectral_rolloff(sr=sr, freq=freq, S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_spectral_flatness_multi(s_multi):
    S, sr = s_multi

    # compare each channel
    C0 = librosa.feature.spectral_flatness(S=S[0])
    C1 = librosa.feature.spectral_flatness(S=S[1])
    Call = librosa.feature.spectral_flatness(S=S)

    # Check each channel
    assert np.allclose(C0, Call[0], atol=1e-5)
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_poly_multi_static(s_multi):
    mags, sr = s_multi

    Pall = librosa.feature.poly_features(S=mags, order=5)

    # Compute per channel
    P0 = librosa.feature.poly_features(S=mags[0], order=5)
    P1 = librosa.feature.poly_features(S=mags[1], order=5)

    # Check results
    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)
    assert not np.allclose(P0, P1)


def test_poly_multi_varying(tfr_multi):

    # Get some time-varying frequencies
    times, freqs, mags = tfr_multi
    Pall = librosa.feature.poly_features(S=mags, freq=freqs, order=5)

    # Compute per channel
    P0 = librosa.feature.poly_features(S=mags[0], freq=freqs[0], order=5)
    P1 = librosa.feature.poly_features(S=mags[1], freq=freqs[1], order=5)

    # Check results
    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)
    assert not np.allclose(P0, P1)


def test_rms_multi(s_multi):
    S, sr = s_multi

    # compare each channel
    C0 = librosa.feature.rms(S=S[0])
    C1 = librosa.feature.rms(S=S[1])
    Call = librosa.feature.rms(S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_zcr_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.zero_crossing_rate(y=y[0])
    C1 = librosa.feature.zero_crossing_rate(y=y[1])
    Call = librosa.feature.zero_crossing_rate(y=y)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_chroma_stft_multi(s_multi):
    S, sr = s_multi

    # compare each channel
    C0 = librosa.feature.chroma_stft(S=S[0], tuning=0)
    C1 = librosa.feature.chroma_stft(S=S[1], tuning=0)
    Call = librosa.feature.chroma_stft(S=S, tuning=0)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_chroma_cqt_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.chroma_cqt(y=y[0], tuning=0)
    C1 = librosa.feature.chroma_cqt(y=y[1], tuning=0)
    Call = librosa.feature.chroma_cqt(y=y, tuning=0)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_chroma_cens_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.chroma_cens(y=y[0], tuning=0)
    C1 = librosa.feature.chroma_cens(y=y[1], tuning=0)
    Call = librosa.feature.chroma_cens(y=y, tuning=0)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_tonnetz_multi(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.tonnetz(y=y[0], tuning=0)
    C1 = librosa.feature.tonnetz(y=y[1], tuning=0)
    Call = librosa.feature.tonnetz(y=y, tuning=0)

    # Check each channel
    assert np.allclose(C0, Call[0], atol=1e-7)
    assert np.allclose(C1, Call[1], atol=1e-7)

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_mfcc_multi(s_multi):
    S, sr = s_multi

    # compare each channel
    C0 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[0], top_db=None))
    C1 = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S[1], top_db=None))
    Call = librosa.feature.mfcc(S=librosa.core.amplitude_to_db(S=S, top_db=None))

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.skip(reason="power_to_db leaks information across channels")
def test_mfcc_multi_time(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.mfcc(y=y[0])
    C1 = librosa.feature.mfcc(y=y[1])
    Call = librosa.feature.mfcc(y=y)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_melspectrogram_multi(s_multi):
    S, sr = s_multi

    # compare each channel
    C0 = librosa.feature.melspectrogram(S=S[0])
    C1 = librosa.feature.melspectrogram(S=S[1])
    Call = librosa.feature.melspectrogram(S=S)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_melspectrogram_multi_time(y_multi):
    y, sr = y_multi

    # compare each channel
    C0 = librosa.feature.melspectrogram(y=y[0])
    C1 = librosa.feature.melspectrogram(y=y[1])
    Call = librosa.feature.melspectrogram(y=y)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize("rate", [0.5, 2])
def test_phase_vocoder(y_multi, rate):
    y, sr = y_multi
    D = librosa.stft(y)

    D0 = librosa.phase_vocoder(D[0], rate)
    D1 = librosa.phase_vocoder(D[1], rate)
    D2 = librosa.phase_vocoder(D, rate)

    assert np.allclose(D2[0], D0)
    assert np.allclose(D2[1], D1)
    assert not np.allclose(D2[0], D2[1])


@pytest.mark.parametrize("delay", [1, -1])
def test_stack_memory_multi(delay):
    data = np.random.randn(2, 5, 200)

    # compare each channel
    C0 = librosa.feature.stack_memory(data[0], delay=delay)
    C1 = librosa.feature.stack_memory(data[1], delay=delay)
    Call = librosa.feature.stack_memory(data, delay=delay)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


def test_interp_harmonics_multi_static(s_multi):
    S, sr = s_multi

    freqs = librosa.fft_frequencies(sr=sr)
    Hall = librosa.interp_harmonics(S, freqs, [0.5, 1, 2])
    H0 = librosa.interp_harmonics(S[0], freqs, [0.5, 1, 2])
    H1 = librosa.interp_harmonics(S[1], freqs, [0.5, 1, 2])

    assert np.allclose(Hall[0], H0)
    assert np.allclose(Hall[1], H1)

    assert not np.allclose(H0, H1)


def test_interp_harmonics_multi_vary(tfr_multi):
    times, freqs, mags = tfr_multi

    # Force slinear mode here to deal with non-unique frequencies
    Hall = librosa.interp_harmonics(mags, freqs, [0.5, 1, 2], kind="slinear")
    H0 = librosa.interp_harmonics(mags[0], freqs[0], [0.5, 1, 2], kind="slinear")
    H1 = librosa.interp_harmonics(mags[1], freqs[1], [0.5, 1, 2], kind="slinear")

    assert np.allclose(Hall[0], H0)
    assert np.allclose(Hall[1], H1)

    assert not np.allclose(H0, H1)


@pytest.mark.parametrize("filter_peaks", [False, True])
def test_salience_multi_static(s_multi, filter_peaks):
    S, sr = s_multi

    freqs = librosa.fft_frequencies(sr=sr)

    sal_all = librosa.salience(
        S,
        freqs,
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )
    sal_0 = librosa.salience(
        S[0],
        freqs,
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )
    sal_1 = librosa.salience(
        S[1],
        freqs,
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )

    assert np.allclose(sal_all[0], sal_0)
    assert np.allclose(sal_all[1], sal_1)
    assert not np.allclose(sal_0, sal_1)


@pytest.mark.parametrize("filter_peaks", [False, True])
def test_salience_multi_dynamic(tfr_multi, filter_peaks):
    times, freqs, S = tfr_multi

    sal_all = librosa.salience(
        S,
        freqs,
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )
    sal_0 = librosa.salience(
        S[0],
        freqs[0],
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )
    sal_1 = librosa.salience(
        S[1],
        freqs[1],
        [0.5, 1, 2, 3],
        kind="slinear",
        filter_peaks=filter_peaks,
        fill_value=0,
    )

    assert np.allclose(sal_all[0], sal_0)
    assert np.allclose(sal_all[1], sal_1)
    assert not np.allclose(sal_0, sal_1)


@pytest.mark.parametrize("center", [False, True])
def test_iirt_multi(y_multi, center):
    y, sr = y_multi
    Call = librosa.iirt(y=y, sr=sr, center=center)
    C0 = librosa.iirt(y=y[0], sr=sr, center=center)
    C1 = librosa.iirt(y=y[1], sr=sr, center=center)

    assert np.allclose(Call[0], C0)
    assert np.allclose(Call[1], C1)

    assert not np.allclose(C0, C1)


def test_lpc_multi(y_multi):
    y, sr = y_multi

    Lall = librosa.lpc(y, 6)
    L0 = librosa.lpc(y[0], 6)
    L1 = librosa.lpc(y[1], 6)

    assert np.allclose(Lall[0], L0)
    assert np.allclose(Lall[1], L1)
    assert not np.allclose(L0, L1)


def test_yin_multi(y_multi):
    y, sr = y_multi

    Pall = librosa.yin(y, 30, 300)
    P0 = librosa.yin(y[0], 30, 300)
    P1 = librosa.yin(y[1], 30, 300)

    assert np.allclose(Pall[0], P0)
    assert np.allclose(Pall[1], P1)

    assert not np.allclose(P0, P1)


@pytest.mark.parametrize('ref', [None, 1.0])
def test_piptrack_multi(s_multi, ref):
    S, sr = s_multi

    pall, mall = librosa.piptrack(S=S, sr=sr, ref=ref)
    p0, m0 = librosa.piptrack(S=S[0], sr=sr, ref=ref)
    p1, m1 = librosa.piptrack(S=S[1], sr=sr, ref=ref)

    assert np.allclose(pall[0], p0)
    assert np.allclose(pall[1], p1)
    assert np.allclose(mall[0], m0)
    assert np.allclose(mall[1], m1)
    assert not np.allclose(p0, p1)
    assert not np.allclose(m0, m1)
