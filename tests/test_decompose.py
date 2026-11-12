#!/usr/bin/env python
# CREATED: 2013-10-06 22:31:29 by Dawen Liang <dl2771@columbia.edu>
# unit tests for librosa.decompose

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import numpy as np
import scipy.sparse

import librosa
import sklearn.decomposition

import pytest
from typing import Union


def test_default_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_given_decompose():

    D = sklearn.decomposition.NMF(random_state=0)

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, transformer=D)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_decompose_fit(rng):

    D = sklearn.decomposition.NMF(random_state=0)

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    # Do a first fit
    (W, H) = librosa.decompose.decompose(X, transformer=D, fit=True)

    # Make random data and decompose with the same basis
    X = np.asarray(rng.standard_normal(size=X.shape) ** 2)
    (W2, H2) = librosa.decompose.decompose(X, transformer=D, fit=False)

    # Make sure the basis hasn't changed
    assert np.allclose(W, W2)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_decompose_multi_sort():
    librosa.decompose.decompose(np.zeros((3, 3, 3)), sort=True)


@pytest.mark.filterwarnings("ignore:Maximum number of iterations")
def test_decompose_multi(rng):
    X = rng.random(size=(2, 20, 100))

    # Fit with multichannel data
    components, activations = librosa.decompose.decompose(
        X, n_components=20, random_state=0
    )

    # Reshape the data
    Xflat = np.vstack([X[0], X[1]])
    c_flat, a_flat = librosa.decompose.decompose(Xflat, n_components=20, random_state=0)

    assert np.allclose(c_flat[: X.shape[1]], components[0])
    assert np.allclose(c_flat[X.shape[1] :], components[1])
    assert np.allclose(activations, a_flat)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_decompose_fit_false():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, fit=False)


def test_sorted_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, sort=True, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_decompose_custom_components_fit_true(rng):
    X = np.abs(rng.standard_normal(size=(20, 50)))
    W_init = np.abs(rng.standard_normal(size=(20, 5))) + 1e-4

    # solver='cd'
    comps_cd, acts_cd = librosa.decompose.decompose(
        X, components=W_init, fit=True, random_state=0
    )
    assert comps_cd.shape == (20, 5)
    assert acts_cd.shape == (5, 50)

    # solver='mu'
    comps_mu, acts_mu = librosa.decompose.decompose(
        X, components=W_init, fit=True, solver="mu", random_state=0
    )
    assert comps_mu.shape == (20, 5)
    assert acts_mu.shape == (5, 50)


def test_decompose_custom_components_fit_false(rng):
    X = np.abs(rng.standard_normal(size=(20, 50)))
    W_fixed = np.abs(rng.standard_normal(size=(20, 5))) + 1e-4

    comps, acts = librosa.decompose.decompose(X, components=W_fixed, fit=False)
    assert comps.shape == (20, 5)
    assert acts.shape == (5, 50)
    assert np.allclose(comps, W_fixed)


def test_decompose_components_and_activations(rng):
    X = np.abs(rng.standard_normal(size=(20, 50)))
    W_init = np.abs(rng.standard_normal(size=(20, 5))) + 1e-4
    H_init = np.abs(rng.standard_normal(size=(5, 50))) + 1e-4

    comps, acts = librosa.decompose.decompose(
        X, components=W_init, activations=H_init, fit=True, solver="mu", random_state=0
    )
    assert comps.shape == (20, 5)
    assert acts.shape == (5, 50)


def test_decompose_custom_components_errors(rng):
    X = np.abs(rng.standard_normal(size=(20, 50)))
    W_bad = np.abs(rng.standard_normal(size=(15, 5)))

    # Feature dimension mismatch
    with pytest.raises(librosa.ParameterError):
        librosa.decompose.decompose(X, components=W_bad)

    # n_components mismatch
    with pytest.raises(librosa.ParameterError):
        librosa.decompose.decompose(
            X, components=np.ones((20, 5)), n_components=10
        )

    # activations shape mismatch
    with pytest.raises(librosa.ParameterError):
        librosa.decompose.decompose(
            X, components=np.ones((20, 5)), activations=np.ones((5, 40))
        )


def test_decompose_components_multi(rng):
    X = np.abs(rng.standard_normal(size=(2, 20, 50)))
    W_2d = np.abs(rng.standard_normal(size=(20, 6))) + 1e-4

    comps, acts = librosa.decompose.decompose(X, components=W_2d, fit=False)
    assert comps.shape == (2, 20, 6)
    assert acts.shape == (6, 50)


@pytest.mark.filterwarnings("ignore:Maximum number of iterations")
def test_pitch_nmf():
    sr = 22050
    n_fft = 2048
    y = librosa.chirp(fmin=220, fmax=880, sr=sr, duration=1.0)
    S = np.abs(librosa.stft(y, n_fft=n_fft)).astype(np.float32)

    # Default pitch_nmf (n_pitches=84, n_residual=5 => 89 components)
    comps, acts = librosa.decompose.pitch_nmf(S, sr=sr, max_iter=20)
    assert comps.shape == (1025, 89)
    assert acts.shape == (89, S.shape[1])
    assert not np.isnan(comps).any()
    assert not np.isnan(acts).any()

    # Inferred n_fft
    comps_inf, acts_inf = librosa.decompose.pitch_nmf(
        S, n_pitches=36, fmin="C3", n_residual=1, max_iter=10
    )
    assert comps_inf.shape == (1025, 37)
    assert acts_inf.shape == (37, S.shape[1])

    # fit=False
    comps_fix, acts_fix = librosa.decompose.pitch_nmf(
        S, n_pitches=24, fmin="A2", fit=False
    )
    assert comps_fix.shape == (1025, 29)
    assert acts_fix.shape == (29, S.shape[1])

    # Multichannel spectrogram
    S_multi = np.stack([S, S[::-1]])
    comps_mc, acts_mc = librosa.decompose.pitch_nmf(
        S_multi, n_pitches=12, fmin="C3", n_residual=3, max_iter=10
    )
    assert comps_mc.shape == (2, 1025, 15)
    assert acts_mc.shape == (15, S.shape[1])


@pytest.fixture(scope="module")
def y22050():
    sr = 22050
    y = librosa.chirp(fmin=55, fmax=880, sr=sr, duration=5.0)
    y += librosa.clicks(
        times=np.arange(0, 5, 0.5), sr=sr, length=len(y), click_duration=0.25
    )
    return y


@pytest.fixture(scope="module")
def D22050(y22050):
    return librosa.stft(y22050)


@pytest.fixture(scope="module")
def S22050(D22050):
    return np.abs(D22050)


@pytest.mark.parametrize("window", [31, (5, 5)])
@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("margin", [1.0, 3.0, (1.0, 1.0), (9.0, 10.0)])
def test_real_hpss(S22050, window, power, mask, margin):
    H, P = librosa.decompose.hpss(
        S22050, kernel_size=window, power=power, mask=mask, margin=margin
    )

    if margin == 1.0 or margin == (1.0, 1.0):
        if mask:
            assert np.allclose(H + P, np.ones_like(S22050))
        else:
            assert np.allclose(H + P, S22050)
    else:
        if mask:
            assert np.all(H + P <= np.ones_like(S22050))
        else:
            assert np.all(H + P <= S22050)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_hpss_margin_error(S22050):
    H, P = librosa.decompose.hpss(S22050, margin=0.9)


def test_complex_hpss(D22050):
    H, P = librosa.decompose.hpss(D22050)
    assert np.allclose(H + P, D22050)


def test_nn_filter_mean(rng):

    X = rng.standard_normal(size=(10, 100))

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X)

    X_filtered = librosa.decompose.nn_filter(X)

    # Normalize the recurrence matrix so dotting computes an average
    rec = librosa.util.normalize(rec.astype(float), axis=0, norm=1)

    assert np.allclose(X_filtered, X.dot(rec))


def test_nn_filter_mean_rec(rng):

    X = rng.standard_normal(size=(10, 100))

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X)

    # Knock out the first three rows of links
    rec[:, :3] = False

    X_filtered = librosa.decompose.nn_filter(X, rec=rec)

    for i in range(3):
        assert np.allclose(X_filtered[:, i], X[:, i])

    # Normalize the recurrence matrix
    rec = librosa.util.normalize(rec.astype(float), axis=0, norm=1)
    assert np.allclose(X_filtered[:, 3:], (X.dot(rec))[:, 3:])


def test_nn_filter_mean_rec_sparse(rng):

    X = rng.standard_normal(size=(10, 100))

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X, sparse=True)

    X_filtered = librosa.decompose.nn_filter(X, rec=rec)

    # Normalize the recurrence matrix
    rec = librosa.util.normalize(rec.toarray().astype(float), axis=0, norm=1)
    assert np.allclose(X_filtered, (X.dot(rec)))


@pytest.fixture(scope="module")
def s_multi():
    y = librosa.chirp(fmin=55, fmax=880, duration=5.0)
    y = np.vstack([y, y[::-1]])
    return np.abs(librosa.stft(y))


@pytest.mark.parametrize("useR,sparse", [(False, False), (True, False), (True, True)])
def test_nn_filter_multi(s_multi, useR, sparse):

    R = librosa.segment.recurrence_matrix(s_multi, mode="affinity", sparse=sparse)
    if useR:
        R_multi = R
    else:
        R_multi = None

    s_filt = librosa.decompose.nn_filter(
        s_multi, rec=R_multi, mode="affinity", sparse=sparse
    )
    # Always use the same recurrence matrix for comparison
    s_filt0 = librosa.decompose.nn_filter(s_multi[0], rec=R)
    s_filt1 = librosa.decompose.nn_filter(s_multi[1], rec=R)

    assert np.allclose(s_filt[0], s_filt0)
    assert np.allclose(s_filt[1], s_filt1)
    assert not np.allclose(s_filt0, s_filt1)


def test_nn_filter_avg(rng):

    X = rng.standard_normal(size=(10, 100))

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X, mode="affinity")

    X_filtered = librosa.decompose.nn_filter(X, rec=rec, aggregate=np.average)

    # Normalize the recurrence matrix so dotting computes an average
    rec = librosa.util.normalize(rec, axis=0, norm=1)

    assert np.allclose(X_filtered, X.dot(rec))


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "x,y", [(10, 10), (100, 20), (20, 100), (100, 101), (101, 101)]
)
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("data", [np.zeros((10, 100))])
def test_nn_filter_badselfsim(data, x, y, sparse, rng):

    # Build a recurrence matrix, just for testing purposes
    rec: Union[np.ndarray, scipy.sparse.csr_array] = rng.standard_normal(size=(x, y))
    if sparse:
        rec = scipy.sparse.csr_array(rec)

    librosa.decompose.nn_filter(data, rec=rec)
