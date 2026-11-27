#!/usr/bin/env python
# CREATED:2013-03-11 18:14:30 by Brian McFee <brm2132@columbia.edu>
#  unit tests for librosa.onset

# Disable cache
import os
from contextlib import nullcontext as dnr
from contextlib import suppress

import pytest

with suppress(KeyError):
    os.environ.pop("LIBROSA_CACHE_DIR")


import itertools

import numpy as np

import librosa


@pytest.fixture(scope="module")
def ysr():
    # Generate a pulse train at 120BPM
    sr = 22050
    y = np.zeros(5 * sr)
    y[:: sr // 2] = 1
    return y, sr


@pytest.mark.parametrize("feature", [None, librosa.feature.melspectrogram, librosa.feature.chroma_stft])
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


def test_onset_strength_badlag(ysr):
    y, sr = ysr
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(y=y, sr=sr, lag=0)


def test_onset_strength_badmax(ysr):
    y, sr = ysr
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(y=y, sr=sr, max_size=0)


def test_onset_strength_noinput():
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(y=None, S=None)


@pytest.fixture(scope="module")
def melspec_sr(ysr):
    y, sr = ysr
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    return S, sr


@pytest.mark.parametrize("feature", [None, librosa.feature.chroma_stft])
@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("hop_length", [256, 512])
@pytest.mark.parametrize("detrend", [False, True])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("aggregate", [None, np.max])
def test_onset_strength_spectrogram(melspec_sr, feature, n_fft, hop_length, detrend, center, aggregate):
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

    S, _sr = melspec_sr
    # We only test with max_size=1 here to make the sub-band slicing test simple
    odf_multi = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, aggregate=False)
    odf_mean = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, aggregate=aggregate)

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

    S, _sr = melspec_sr
    # We only test with max_size=1 here to make the sub-band slicing test simple
    odf_multi = librosa.onset.onset_strength_multi(S=S, lag=lag, max_size=1, channels=channels)

    assert len(odf_multi) == len(channels) - 1

    for i, (s, t) in enumerate(itertools.pairwise(channels)):
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


def test_onset_detect_nosignal():
    with pytest.raises(librosa.ParameterValueError):
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
        ("bad units", pytest.raises(librosa.ParameterValueError)),
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
            raise AssertionError

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

    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, hop_length=hop, backtrack=False)

    # Test backtracking
    onsets_bt = librosa.onset.onset_backtrack(onsets, energy)

    # Make sure there are no negatives
    assert np.all(onsets_bt >= 0)

    # And that we never roll forward
    assert np.all(onsets_bt <= onsets)

    # And that the detected peaks are actually minima
    assert np.all(energy[onsets_bt] <= energy[np.maximum(0, onsets_bt - 1)])


def test_onset_backtrack_dense(ysr):
    y, sr = ysr
    oenv = librosa.onset.onset_strength(y=y, sr=sr)

    onsets_sparse = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=True, sparse=True)
    onsets_dense = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=True, sparse=False)

    assert onsets_dense.dtype == np.dtype(bool)
    assert onsets_dense.shape == oenv.shape
    assert np.array_equal(np.flatnonzero(onsets_dense), np.unique(onsets_sparse))

    # Direct dense onset_backtrack on 1D boolean events, including empty detections
    raw_dense = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False, sparse=False)
    bt_dense = librosa.onset.onset_backtrack(raw_dense, oenv)
    assert np.array_equal(bt_dense, onsets_dense)

    all_false = np.zeros_like(raw_dense, dtype=bool)
    assert np.array_equal(librosa.onset.onset_backtrack(all_false, oenv), all_false)

    with pytest.raises(librosa.ParameterShapeError):
        librosa.onset.onset_backtrack(raw_dense, oenv[:-1])


def test_onset_sparse(ysr, oenv, hop):
    y, sr = ysr

    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, hop_length=hop, sparse=True)
    onsetsd = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=oenv, hop_length=hop, sparse=False)

    assert np.allclose(onsets, np.flatnonzero(onsetsd))


def test_onset_strength_noagg():
    S = np.zeros((3, 3))
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(S=S, aggregate=False)


def test_onset_strength_badref():
    S = np.zeros((3, 3))
    with pytest.raises(librosa.ParameterShapeError):
        librosa.onset.onset_strength(S=S, S_ref=S[:, :2])


def test_onset_strength_multi_ref(rng):
    # Make a random positive spectrum
    S = 1 + np.abs(rng.standard_normal(size=(1025, 10)))

    # Test with a null reference
    null_ref = np.zeros_like(S)

    onsets = librosa.onset.onset_strength_multi(S=S, S_ref=null_ref, aggregate=False, center=False)

    # since the reference is zero everywhere, S - S_ref = S
    # past the setup phase (first frame)
    assert np.allclose(onsets[:, 1:], S[:, 1:])


def test_onset_strength_deprecated_ref(rng):
    S = 1 + np.abs(rng.standard_normal(size=(32, 10)))
    null_ref = np.zeros_like(S)

    with pytest.warns(
        FutureWarning,
        match=r"(?s)onset_strength\(\) keyword argument 'ref' has been renamed to 'S_ref' in version 1\.1\.0.*1\.2\.0",
    ) as record:
        onsets_old = librosa.onset.onset_strength(S=S, ref=null_ref, center=False)
    assert len(record) == 1

    onsets_new = librosa.onset.onset_strength(S=S, S_ref=null_ref, center=False)
    assert np.allclose(onsets_old, onsets_new)

    with pytest.warns(
        FutureWarning,
        match=r"(?s)onset_strength\(\) keyword argument 'ref' has been renamed to 'S_ref' in version 1\.1\.0.*1\.2\.0",
    ) as record_none:
        onsets_old_none = librosa.onset.onset_strength(S=S, ref=None, center=False)
    assert len(record_none) == 1

    onsets_default = librosa.onset.onset_strength(S=S, center=False)
    assert np.allclose(onsets_old_none, onsets_default)


def test_onset_strength_multi_deprecated_ref(rng):
    S = 1 + np.abs(rng.standard_normal(size=(32, 10)))
    null_ref = np.zeros_like(S)
    pattern = (
        r"(?s)onset_strength_multi\(\) keyword argument 'ref' "
        r"has been renamed to 'S_ref' in version 1\.1\.0.*1\.2\.0"
    )

    with pytest.warns(FutureWarning, match=pattern) as record:
        onsets_old = librosa.onset.onset_strength_multi(S=S, ref=null_ref, aggregate=False, center=False)
    assert len(record) == 1

    onsets_new = librosa.onset.onset_strength_multi(S=S, S_ref=null_ref, aggregate=False, center=False)
    assert np.allclose(onsets_old, onsets_new)

    with pytest.warns(FutureWarning, match=pattern) as record_none:
        onsets_old_none = librosa.onset.onset_strength_multi(S=S, ref=None, aggregate=False, center=False)
    assert len(record_none) == 1

    onsets_default = librosa.onset.onset_strength_multi(S=S, aggregate=False, center=False)
    assert np.allclose(onsets_old_none, onsets_default)


@pytest.mark.parametrize("func", [librosa.onset.onset_strength, librosa.onset.onset_strength_multi])
@pytest.mark.parametrize("use_none_ref", [False, True])
def test_onset_strength_ref_conflict(rng, func, use_none_ref):
    S = 1 + np.abs(rng.standard_normal(size=(32, 10)))
    null_ref = np.zeros_like(S)
    old_ref = None if use_none_ref else null_ref

    with pytest.raises(librosa.ParameterValueError):
        func(S=S, S_ref=null_ref, ref=old_ref)


def test_onset_strength_deprecated_ref_complex(ysr):
    y, _ = ysr
    S_complex = librosa.stft(y[:2048])

    with pytest.warns(FutureWarning, match="renamed to 'S_ref'"), pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(S=S_complex, ref=S_complex, mode="complex")


def test_onset_detect_inplace_normalize():

    # This test will fail if the in-place normalization modifies
    # the input onset envelope
    oenv_in = np.ones(50)
    oenv_in[10] = 2
    oenv_orig = oenv_in.copy()

    librosa.onset.onset_detect(onset_envelope=oenv_in, normalize=True)

    assert np.allclose(oenv_in, oenv_orig) and oenv_in is not oenv_orig


def test_onset_strength_mode_validation(ysr):
    y, sr = ysr
    S_complex = librosa.stft(y[:2048])
    S_real = np.abs(S_complex)

    # Invalid mode string
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(y=y, sr=sr, mode="invalid")  # type: ignore[arg-type]

    # max_size > 1 with complex mode
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(S=S_complex, max_size=3, mode="complex")

    # S_ref is not None with complex mode
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(S=S_complex, S_ref=S_complex, mode="complex")

    # Real S with mode="complex"
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(S=S_real, mode="complex")

    # Real feature with mode="complex"
    with pytest.raises(librosa.ParameterValueError):
        librosa.onset.onset_strength(y=y, sr=sr, mode="complex", feature=librosa.feature.melspectrogram)


def test_onset_strength_complex_linear_phase(rng):
    F, T = 16, 60
    omega = np.linspace(0.1, 1.5, F)[:, None]
    n = np.arange(T)[None, :]
    phase = omega * n
    mag = rng.uniform(0.5, 2.0, size=(F, T))

    S = mag * np.exp(1j * phase)

    for lag in [1, 2, 3]:
        # Without centering: compare frames n >= lag + 1
        oenv_complex = librosa.onset.onset_strength(S=S, lag=lag, center=False)
        oenv_mag = librosa.onset.onset_strength(S=mag, lag=lag, center=False)

        assert oenv_complex.shape == (T,)
        assert oenv_mag.shape == (T,)
        assert np.allclose(oenv_complex[lag + 1 :], oenv_mag[lag + 1 :])

        # With centering
        oenv_complex_c = librosa.onset.onset_strength(S=S, lag=lag, center=True)
        oenv_mag_c = librosa.onset.onset_strength(S=mag, lag=lag, center=True)
        assert oenv_complex_c.shape == (T,)
        assert oenv_mag_c.shape == (T,)
        shift = (lag + 1) + 2048 // (2 * 512)
        assert np.allclose(oenv_complex_c[shift:], oenv_mag_c[shift:])


def test_onset_strength_complex_phase_reset():
    F, T = 8, 50
    omega = 0.5
    phase = omega * np.arange(T)

    # Constant magnitude everywhere
    mag = np.ones((F, T), dtype=np.float64)

    # Phase reset at frame 25
    phase[25:] += np.pi
    S = mag * np.exp(1j * phase[None, :])

    oenv_mag = librosa.onset.onset_strength(S=np.abs(S), center=False)
    assert np.allclose(oenv_mag, 0.0)

    oenv_complex = librosa.onset.onset_strength(S=S, center=False)
    # Peak must occur at frame 25
    assert np.argmax(oenv_complex) == 25
    assert oenv_complex[25] > 0.5


def test_onset_strength_audio_modes(ysr):
    y, sr = ysr

    oenv_flux = librosa.onset.onset_strength(y=y, sr=sr, mode="flux")
    oenv_default = librosa.onset.onset_strength(y=y, sr=sr)
    assert np.allclose(oenv_flux, oenv_default)

    oenv_complex = librosa.onset.onset_strength(y=y, sr=sr, mode="complex")
    assert oenv_complex.shape == oenv_flux.shape
    assert np.all(oenv_complex >= 0)
    assert np.any(oenv_complex > 0)


def test_onset_strength_custom_features(ysr):
    y, sr = ysr

    # complex mode with cqt
    oenv_cqt_complex = librosa.onset.onset_strength(y=y, sr=sr, mode="complex", feature=librosa.cqt)
    assert oenv_cqt_complex.ndim == 1
    assert len(oenv_cqt_complex) == 1 + len(y) // 512

    # flux mode with stft (stft has no sr kwarg, handled by _call_with_kwargs)
    oenv_stft_flux = librosa.onset.onset_strength(y=y, sr=sr, mode="flux", feature=librosa.stft)
    assert oenv_stft_flux.ndim == 1
    assert len(oenv_stft_flux) == 1 + len(y) // 512

    # flux mode with cqt (cqt has no n_fft kwarg, handled by _call_with_kwargs)
    oenv_cqt_flux = librosa.onset.onset_strength(y=y, sr=sr, mode="flux", feature=librosa.cqt)
    assert oenv_cqt_flux.ndim == 1

    # custom lambda
    oenv_lambda = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        mode="complex",
        feature=lambda y, hop_length: librosa.stft(y, hop_length=hop_length),
    )
    assert len(oenv_lambda) == len(oenv_cqt_complex)


def test_onset_strength_complex_spec_input(ysr):
    y, _ = ysr
    D = librosa.stft(y)

    # auto-inference of complex mode
    oenv_auto = librosa.onset.onset_strength(S=D)
    oenv_explicit = librosa.onset.onset_strength(S=D, mode="complex")
    assert np.allclose(oenv_auto, oenv_explicit)

    # explicit flux mode on complex S converts to magnitude
    oenv_flux_from_complex = librosa.onset.onset_strength(S=D, mode="flux")
    oenv_flux_from_mag = librosa.onset.onset_strength(S=np.abs(D), mode="flux")
    assert np.allclose(oenv_flux_from_complex, oenv_flux_from_mag)


def test_onset_strength_complex_multichannel(rng):
    y = rng.standard_normal(size=(2, 22050))
    sr = 22050

    # multi-channel audio input
    oenv = librosa.onset.onset_strength(y=y, sr=sr, mode="complex")
    assert oenv.shape == (2, 44)

    # onset_strength_multi with sub-band channels
    channels = [0, 128, 256, 512, 1025]
    odf = librosa.onset.onset_strength_multi(y=y, sr=sr, mode="complex", channels=channels)
    assert odf.shape == (2, len(channels) - 1, 44)


def test_onset_strength_complex_dtypes_and_edge_cases(rng):
    # complex64 -> float32
    S32 = (rng.standard_normal(size=(10, 20)) + 1j * rng.standard_normal(size=(10, 20))).astype(np.complex64)
    res32 = librosa.onset.onset_strength(S=S32)
    assert res32.dtype == np.float32

    # complex128 -> float64
    S64 = (rng.standard_normal(size=(10, 20)) + 1j * rng.standard_normal(size=(10, 20))).astype(np.complex128)
    res64 = librosa.onset.onset_strength(S=S64)
    assert res64.dtype == np.float64

    # All zeros
    S_zeros = np.zeros((10, 20), dtype=np.complex64)
    res_zeros = librosa.onset.onset_strength(S=S_zeros)
    assert not np.isnan(res_zeros).any()
    assert not np.isinf(res_zeros).any()
    assert np.all(res_zeros == 0.0)

    # Short input T <= lag + 1
    S_short = np.ones((10, 2), dtype=np.complex64)
    res_center = librosa.onset.onset_strength(S=S_short, lag=2, center=True)
    assert res_center.shape == (2,)
    assert np.all(res_center == 0.0)

    res_nocenter = librosa.onset.onset_strength(S=S_short, lag=2, center=False)
    assert res_nocenter.shape == (3,)
    assert np.all(res_nocenter == 0.0)
