#!/usr/bin/env python
# CREATED: 2013-10-06 22:31:29 by Dawen Liang <dl2771@columbia.edu>
# unit tests for librosa.decompose

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import numpy as np
import scipy.sparse

import librosa
import sklearn.decomposition

import pytest

from test_core import srand


def test_default_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_given_decompose():

    D = sklearn.decomposition.NMF(random_state=0)

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, transformer=D)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_decompose_fit():

    srand()

    D = sklearn.decomposition.NMF(random_state=0)

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    # Do a first fit
    (W, H) = librosa.decompose.decompose(X, transformer=D, fit=True)

    # Make random data and decompose with the same basis
    X = np.random.randn(*X.shape) ** 2
    (W2, H2) = librosa.decompose.decompose(X, transformer=D, fit=False)

    # Make sure the basis hasn't changed
    assert np.allclose(W, W2)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_decompose_fit_false():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])
    (W, H) = librosa.decompose.decompose(X, fit=False)


def test_sorted_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, sort=True, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


@pytest.fixture
def y22050():
    y, _ = librosa.load(os.path.join("tests", "data", "test1_22050.wav"))
    return y


@pytest.fixture
def D22050(y22050):
    return librosa.stft(y22050)


@pytest.fixture
def S22050(D22050):
    return np.abs(D22050)


@pytest.mark.parametrize("window", [31, (5, 5)])
@pytest.mark.parametrize("power", [1, 2, 10])
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


def test_nn_filter_mean():

    srand()
    X = np.random.randn(10, 100)

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X)

    X_filtered = librosa.decompose.nn_filter(X)

    # Normalize the recurrence matrix so dotting computes an average
    rec = librosa.util.normalize(rec.astype(np.float), axis=0, norm=1)

    assert np.allclose(X_filtered, X.dot(rec))


def test_nn_filter_mean_rec():

    srand()
    X = np.random.randn(10, 100)

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X)

    # Knock out the first three rows of links
    rec[:, :3] = False

    X_filtered = librosa.decompose.nn_filter(X, rec=rec)

    for i in range(3):
        assert np.allclose(X_filtered[:, i], X[:, i])

    # Normalize the recurrence matrix
    rec = librosa.util.normalize(rec.astype(np.float), axis=0, norm=1)
    assert np.allclose(X_filtered[:, 3:], (X.dot(rec))[:, 3:])


def test_nn_filter_mean_rec_sparse():

    srand()
    X = np.random.randn(10, 100)

    # Build a recurrence matrix, just for testing purposes
    rec = librosa.segment.recurrence_matrix(X, sparse=True)

    X_filtered = librosa.decompose.nn_filter(X, rec=rec)

    # Normalize the recurrence matrix
    rec = librosa.util.normalize(rec.toarray().astype(np.float), axis=0, norm=1)
    assert np.allclose(X_filtered, (X.dot(rec)))


def test_nn_filter_avg():

    srand()
    X = np.random.randn(10, 100)

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
def test_nn_filter_badselfsim(data, x, y, sparse):

    srand()
    # Build a recurrence matrix, just for testing purposes
    rec = np.random.randn(x, y)
    if sparse:
        rec = scipy.sparse.csr_matrix(rec)

    librosa.decompose.nn_filter(data, rec=rec)
