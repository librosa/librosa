#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.onset

import pytest
from contextlib import nullcontext as dnr

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass


import warnings

import numpy as np
import librosa


@pytest.fixture(scope="module")
def ysr():
    # Generate a pulse train at 120BPM
    sr = 22050
    y = np.zeros(5 * sr)
    y[:: sr // 2] = 1
    return y, sr


@pytest.mark.parametrize(
    "feature", [None, librosa.feature.melspectrogram, librosa.feature.chroma_stft]
)
@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("lag", [1])
@pytest.mark.parametrize("max_size", [1])
@pytest.mark.parametrize("aggregate", [np.mean])
@pytest.mark.parametrize("detrend", [False, True])
def test_onset_strength_audio(
    ysr,
    feature,
    n_fft,
    hop_length,
    center,
    lag,
    max_size,
    detrend,
    aggregate,
):

    # This test only really checks output shape and dimension
    # frame length, hop length, and centering matter
    # feature kind of matters
    # lag, max_size, detrending, and aggregation do not matter.

    y, sr = ysr
    oenv = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        S=None,
        detrend=detrend,
        center=center,
        aggregate=aggregate,
        feature=feature,
        n_fft=n_fft,
        hop_length=hop_length,
        lag=lag,
        max_size=max_size,
    )

    assert oenv.ndim == 1

    # Calculate the number of frames that we should have for a centered analysis
    # Note that centering is enforced at the feature level, so the output shape
    # does not actually depend on the center parameter.
    # This may be something to change in the future.
    target_shape = 1 + len(y) // hop_length

    if not detrend:
        assert np.all(oenv >= 0)

    assert oenv.shape[-1] == target_shape


def test_onset_strength_lag(melspec_sr):

    S, sr = melspec_sr
    oenv_skip = librosa.onset.onset_strength(S=S[..., ::2], sr=sr, lag=1)
    oenv_lag = librosa.onset.onset_strength(S=S, sr=sr, lag=2)

    # Lag of 2 should do the same thing at every other frame as
    # doing a decimated onset strength with lag=1.  We do need
    # to be careful of boundary effects though, hence the edge slicing here.
    assert np.allclose(oenv_lag[:-2:2], oenv_skip[1:])


def test_onset_strength_max(melspec_sr):

    S, sr = melspec_sr

    oenv1 = librosa.onset.onset_strength(S=S, sr=sr, max_size=1)
    oenv2 = librosa.onset.onset_strength(S=S, sr=sr, max_size=3)

    # When we compare to the max over a larger set from the previous
    # frame, the novelty can only decrease.
    assert np.all(oenv1 >= oenv2)


def test_onset_strength_aggregate(melspec_sr):
    S, sr = melspec_sr
    oenv_mean = librosa.onset.onset_strength(S=S, sr=sr, aggregate=np.mean)
    oenv_max = librosa.onset.onset_strength(S=S, sr=sr, aggregate=np.max)
    oenv_min = librosa.onset.onset_strength(S=S, sr=sr, aggregate=np.min)

    assert np.all(oenv_mean >= oenv_min)
    assert np.all(oenv_mean <= oenv_max)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badlag(ysr):
    y, sr = ysr
    librosa.onset.onset_strength(y=y, sr=sr, lag=0)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badmax(ysr):
    y, sr = ysr
    librosa.onset.onset_strength(y=y, sr=sr, max_size=0)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_noinput():
    librosa.onset.onset_strength(y=None, S=None)


@pytest.fixture(scope="module")
def melspec_sr(ysr):
    y, sr = ysr
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    return S, sr


# TODO: reduce this parameter grid
@pytest.mark.parametrize(
    "feature", [None, librosa.feature.melspectrogram, librosa.feature.chroma_stft]
)
@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("detrend", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("aggregate", [None, np.mean, np.max])
def test_onset_strength_spectrogram(
    melspec_sr, feature, n_fft, hop_length, detrend, center, aggregate
):
    S, sr = melspec_sr
    oenv = librosa.onset.onset_strength(
        y=None,
        sr=sr,
        S=S,
        detrend=detrend,
        center=center,
        aggregate=aggregate,
        feature=feature,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    assert oenv.ndim == 1

    target_shape = S.shape[-1]

    if not detrend:
        assert np.all(oenv >= 0)

    assert oenv.shape[-1] == target_shape


@pytest.mark.parametrize("lag", [1, 2, 3])
@pytest.mark.parametrize("aggregate", [np.mean, np.max])
def test_onset_strength_multi_noagg(melspec_sr, lag, aggregate):

    S, sr = melspec_sr
    # We only test with max_size=1 here to make the sub-band slicing test simple
    odf_multi = librosa.onset.onset_strength_multi(
        S=S, lag=lag, max_size=1, aggregate=False
    )
    odf_mean = librosa.onset.onset_strength_multi(
        S=S, lag=lag, max_size=1, aggregate=aggregate
    )

    # With no aggregation, output shape should = input shape
    assert odf_multi.shape == S.shape

    # Result should average out to the same as mean aggregation
    assert np.allclose(odf_mean, aggregate(odf_multi, axis=0))


@pytest.fixture(scope="module")
def channels(melspec_sr):
    S, _ = melspec_sr
    return np.linspace(0, S.shape[0], num=5, dtype=int)


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_onset_strength_multi(melspec_sr, lag, channels):

    S, sr = melspec_sr
    # We only test with max_size=1 here to make the sub-band slicing test simple
    odf_multi = librosa.onset.onset_strength_multi(
        S=S, lag=lag, max_size=1, channels=channels
    )

    assert len(odf_multi) == len(channels) - 1

    for i, (s, t) in enumerate(zip(channels, channels[1:])):
        odf_single = librosa.onset.onset_strength(S=S[s:t], lag=lag, max_size=1)
        assert np.allclose(odf_single, odf_multi[i])


@pytest.fixture(scope="module", params=[64, 512, 2048])
def hop(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True], ids=["audio", "oenv"])
def oenv(ysr, hop, request):

    if request.param:
        y, sr = ysr
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    else:
        return None


@pytest.mark.parametrize("bt", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_onset_detect_real(ysr, oenv, hop, bt, normalize):

    y, sr = ysr
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=oenv,
        hop_length=hop,
        backtrack=bt,
        normalize=normalize,
    )
    if bt:
        assert np.all(onsets >= 0)
    else:
        assert np.all(onsets > 0)

    assert np.all(onsets < len(y) * sr // hop)
    if oenv is not None:
        assert np.all(onsets < len(oenv))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_detect_nosignal():
    librosa.onset.onset_detect(y=None, onset_envelope=None)


@pytest.mark.parametrize("sr", [4000])
@pytest.mark.parametrize("y", [np.zeros(4000), np.ones(4000), -np.ones(4000)])
@pytest.mark.parametrize("hop_length", [64, 512, 2048])
def test_onset_detect_const(y, sr, hop_length):

    # Disable padding here
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        onset_envelope=None,
        hop_length=hop_length,
    )

    # We'll allow one onset at the start of the signal for these examples
    # when y is all-ones, zero-padding induces an onset at the beginning of the
    # signal
    assert len(onsets) == 0 or (y[0] != 0 and len(onsets) == 1)


def test_onset_detect_dense_const():

    # Make an empty onset strength envelope
    z = np.zeros(512)
    onsets = librosa.onset.onset_detect(onset_envelope=z, sparse=False)
    assert z.shape == onsets.shape
    assert onsets.dtype is np.dtype(bool)
    assert not np.any(onsets)


@pytest.mark.parametrize(
    "units, ctx",
    [
        ("frames", dnr()),
        ("time", dnr()),
        ("samples", dnr()),
        ("bad units", pytest.raises(librosa.ParameterError)),
    ],
)
@pytest.mark.parametrize("hop_length", [512, 1024])
def test_onset_units(ysr, hop_length, units, ctx):

    y, sr = ysr

    with ctx:
        b1 = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        b2 = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units=units)

        t1 = librosa.frames_to_time(b1, sr=sr, hop_length=hop_length)

        if units == "time":
            t2 = b2

        elif units == "samples":
            t2 = librosa.samples_to_time(b2, sr=sr)

        elif units == "frames":
            t2 = librosa.frames_to_time(b2, sr=sr, hop_length=hop_length)

        else:
            assert False

        assert np.allclose(t1, t2)


@pytest.fixture(scope="module", params=[False, True], ids=["oenv", "rms"])
def energy(ysr, hop, request):
    y, sr = ysr
    if request.param:
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    else:
        return librosa.feature.rms(y=y, hop_length=hop)


def test_onset_backtrack(ysr, oenv, hop, energy):
    y, sr = ysr

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=oenv, hop_length=hop, backtrack=False
    )

    # Test backtracking
    onsets_bt = librosa.onset.onset_backtrack(onsets, energy)

    # Make sure there are no negatives
    assert np.all(onsets_bt >= 0)

    # And that we never roll forward
    assert np.all(onsets_bt <= onsets)

    # And that the detected peaks are actually minima
    assert np.all(energy[onsets_bt] <= energy[np.maximum(0, onsets_bt - 1)])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_backtrack_fail(ysr):
    y, sr = ysr
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, sparse=False)


def test_onset_sparse(ysr, oenv, hop, energy):
    y, sr = ysr

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=oenv, hop_length=hop, sparse=True
    )
    onsetsd = librosa.onset.onset_detect(
        y=y, sr=sr, onset_envelope=oenv, hop_length=hop, sparse=False
    )

    assert np.allclose(onsets, np.flatnonzero(onsetsd))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_noagg():
    S = np.zeros((3, 3))
    librosa.onset.onset_strength(S=S, aggregate=False)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_onset_strength_badref():
    S = np.zeros((3, 3))
    librosa.onset.onset_strength(S=S, ref=S[:, :2])


def test_onset_strength_multi_ref(rng):
    # Make a random positive spectrum
    S = 1 + np.abs(rng.standard_normal(size=(1025, 10)))

    # Test with a null reference
    null_ref = np.zeros_like(S)

    onsets = librosa.onset.onset_strength_multi(
        S=S, ref=null_ref, aggregate=False, center=False
    )

    # since the reference is zero everywhere, S - ref = S
    # past the setup phase (first frame)
    assert np.allclose(onsets[:, 1:], S[:, 1:])


def test_onset_detect_inplace_normalize():

    # This test will fail if the in-place normalization modifies
    # the input onset envelope
    oenv_in = np.ones(50)
    oenv_in[10] = 2
    oenv_orig = oenv_in.copy()

    librosa.onset.onset_detect(onset_envelope=oenv_in, normalize=True)

    assert np.allclose(oenv_in, oenv_orig) and oenv_in is not oenv_orig
