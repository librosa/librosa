#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Tests for segmentation functions"""
import warnings

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import numpy as np
import scipy
from scipy.spatial.distance import cdist, pdist, squareform
import pytest

from test_core import srand

import librosa

__EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")


@pytest.mark.parametrize("n", [20, 250])
@pytest.mark.parametrize("k", [None, 5])
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_cross_similarity(n, k, metric):

    srand()
    # Make a data matrix
    data_ref = np.random.randn(3, n)
    data = np.random.randn(3, n + 7)

    D = librosa.segment.cross_similarity(data, data_ref, k=k, metric=metric)

    assert D.shape == (data_ref.shape[1], data.shape[1])

    if k is not None:
        real_k = min(k, n)
        assert not np.any(D.sum(axis=0) != real_k)


def test_cross_similarity_sparse():

    srand()
    data_ref = np.random.randn(3, 50)
    data = np.random.randn(3, 100)

    D_sparse = librosa.segment.cross_similarity(data, data_ref, sparse=True)
    D_dense = librosa.segment.cross_similarity(data, data_ref, sparse=False)

    assert scipy.sparse.isspmatrix(D_sparse)
    assert np.allclose(D_sparse.todense(), D_dense)


def test_cross_similarity_distance():

    srand()
    data_ref = np.random.randn(3, 50)
    data = np.random.randn(3, 70)
    distance = cdist(data.T, data_ref.T, metric="sqeuclidean").T
    rec = librosa.segment.cross_similarity(
        data, data_ref, mode="distance", metric="sqeuclidean", sparse=True
    )

    i, j, vals = scipy.sparse.find(rec)
    assert np.allclose(vals, distance[i, j])


@pytest.mark.parametrize("metric", ["sqeuclidean", "cityblock"])
@pytest.mark.parametrize("bandwidth", [None, 1])
def test_cross_similarity_affinity(metric, bandwidth):

    srand()
    data_ref = np.random.randn(3, 70)
    data = np.random.randn(3, 50)
    distance = cdist(data_ref.T, data.T, metric=metric)
    rec = librosa.segment.cross_similarity(
        data, data_ref, mode="affinity", metric=metric, sparse=True, bandwidth=bandwidth
    )

    i, j, vals = scipy.sparse.find(rec)
    logvals = np.log(vals)

    ratio = -logvals / distance[i, j]

    if bandwidth is None:
        assert np.allclose(-logvals, distance[i, j] * np.nanmax(ratio))
    else:
        assert np.allclose(-logvals, distance[i, j] * bandwidth)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cross_similarity_badmode():

    srand()
    data_ref = np.random.randn(3, 70)
    data = np.random.randn(3, 50)

    rec = librosa.segment.cross_similarity(
        data, data_ref, mode="NOT A MODE", metric="sqeuclidean", sparse=True
    )


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cross_similarity_bad_bandwidth():

    srand()
    data_ref = np.random.randn(3, 50)
    data = np.random.randn(3, 70)
    rec = librosa.segment.cross_similarity(data, data_ref, bandwidth=-2)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cross_similarity_fail_mismatch():
    D1 = np.zeros((3, 3))
    D2 = np.zeros((2, 3))
    librosa.segment.cross_similarity(D1, D2)


@pytest.mark.parametrize("n", [20, 250])
@pytest.mark.parametrize("k", [None, 5])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("self", [False, True])
def test_recurrence_matrix(n, k, width, sym, metric, self):

    srand()
    # Make a data matrix
    data = np.random.randn(3, n)

    D = librosa.segment.recurrence_matrix(
        data, k=k, width=width, sym=sym, axis=-1, metric=metric, self=self
    )

    # First test for symmetry
    if sym:
        assert np.allclose(D, D.T)

    # Test for target-axis invariance
    D_trans = librosa.segment.recurrence_matrix(
        data.T, k=k, width=width, sym=sym, axis=0, metric=metric, self=self
    )
    assert np.allclose(D, D_trans)

    # If not symmetric, test for correct number of links
    if not sym and k is not None:
        real_k = min(k, n - width)
        if self:
            real_k += 1
        assert not np.any(D.sum(axis=0) != real_k)

    if self:
        assert np.allclose(np.diag(D), True)

    # Make sure the +- width diagonal is hollow
    # It's easier to test if zeroing out the triangles leaves nothing
    idx = np.tril_indices(n, k=width)

    D[idx] = False
    D.T[idx] = False
    assert not np.any(D)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("data", [np.ones((3, 10))])
@pytest.mark.parametrize("width", [-1, 0, 11])
def test_recurrence_badwidth(data, width):
    librosa.segment.recurrence_matrix(data, width=width)


@pytest.mark.parametrize("self", [False, True])
def test_recurrence_sparse(self):

    srand()
    data = np.random.randn(3, 100)
    D_sparse = librosa.segment.recurrence_matrix(data, sparse=True, self=self)
    D_dense = librosa.segment.recurrence_matrix(data, sparse=False, self=self)

    assert scipy.sparse.isspmatrix(D_sparse)
    assert np.allclose(D_sparse.todense(), D_dense)

    if self:
        assert np.allclose(D_sparse.diagonal(), True)
    else:
        assert np.allclose(D_sparse.diagonal(), False)


@pytest.mark.parametrize("self", [False, True])
def test_recurrence_distance(self):

    srand()
    data = np.random.randn(3, 100)
    distance = squareform(pdist(data.T, metric="sqeuclidean"))
    rec = librosa.segment.recurrence_matrix(
        data, mode="distance", metric="sqeuclidean", sparse=True, self=self
    )

    i, j, vals = scipy.sparse.find(rec)
    assert np.allclose(vals, distance[i, j])
    assert np.allclose(rec.diagonal(), 0.0)


@pytest.mark.parametrize("metric", ["sqeuclidean", "cityblock"])
@pytest.mark.parametrize("bandwidth", [None, 1])
@pytest.mark.parametrize("self", [False, True])
def test_recurrence_affinity(metric, bandwidth, self):

    srand()
    data = np.random.randn(3, 100)
    distance = squareform(pdist(data.T, metric=metric))
    rec = librosa.segment.recurrence_matrix(
        data,
        mode="affinity",
        metric=metric,
        sparse=True,
        bandwidth=bandwidth,
        self=self,
    )

    if self:
        assert np.allclose(rec.diagonal(), 1.0)
    else:
        assert np.allclose(rec.diagonal(), 0.0)

    i, j, vals = scipy.sparse.find(rec)
    logvals = np.log(vals)

    # After log-scaling, affinity will match distance up to a constant factor
    ratio = -logvals / distance[i, j]
    if bandwidth is None:
        # Estimate the global bandwidth using non-zero distances
        assert np.allclose(-logvals, distance[i, j] * np.nanmax(ratio))
    else:
        assert np.allclose(-logvals, distance[i, j] * bandwidth)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_recurrence_badmode():

    srand()
    data = np.random.randn(3, 100)

    rec = librosa.segment.recurrence_matrix(
        data, mode="NOT A MODE", metric="sqeuclidean", sparse=True
    )


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_recurrence_bad_bandwidth():

    srand()
    data = np.random.randn(3, 100)
    rec = librosa.segment.recurrence_matrix(data, bandwidth=-2)


@pytest.mark.parametrize("n", [10, 100, 500])
@pytest.mark.parametrize("pad", [False, True])
def test_recurrence_to_lag(n, pad):

    srand()
    data = np.random.randn(17, n)

    rec = librosa.segment.recurrence_matrix(data)

    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
    lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0)

    assert np.allclose(lag, lag2.T)

    x = Ellipsis
    if pad:
        x = slice(n)

    for i in range(n):
        assert np.allclose(rec[:, i], np.roll(lag[:, i], i)[x])


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("size", [(17,), (17, 34), (17, 17, 17)])
def test_recurrence_to_lag_fail(size):
    librosa.segment.recurrence_to_lag(np.zeros(size))


@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize(
    "rec", [librosa.segment.recurrence_matrix(np.random.randn(3, 100), sparse=True)]
)
@pytest.mark.parametrize("fmt", ["csc", "csr", "lil", "bsr", "dia"])
def test_recurrence_to_lag_sparse(pad, axis, rec, fmt):

    rec_dense = rec.toarray()

    rec = rec.asformat(fmt)

    lag_sparse = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)
    lag_dense = librosa.segment.recurrence_to_lag(rec_dense, pad=pad, axis=axis)

    assert scipy.sparse.issparse(lag_sparse)
    assert rec.format == lag_sparse.format
    assert rec.dtype == lag_sparse.dtype
    assert np.allclose(lag_sparse.toarray(), lag_dense)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("pad", [False, True])
def test_lag_to_recurrence(n, pad):

    srand()
    data = np.random.randn(17, n)

    rec = librosa.segment.recurrence_matrix(data)
    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
    lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0).T

    rec2 = librosa.segment.lag_to_recurrence(lag)

    assert np.allclose(rec, rec2)
    assert np.allclose(lag, lag2)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("size", [(17,), (17, 35), (17, 17, 17)])
def test_lag_to_recurrence_badsize(size):
    librosa.segment.lag_to_recurrence(np.zeros(size))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("pad", [False, True])
def test_lag_to_recurrence_sparse(axis, pad):

    srand()
    data = np.random.randn(3, 10)
    rec = librosa.segment.recurrence_matrix(data, sparse=True)
    lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)

    lag_dense = lag.toarray()

    rec_sparse = librosa.segment.lag_to_recurrence(lag, axis=axis)
    rec_dense = librosa.segment.lag_to_recurrence(lag_dense, axis=axis)

    assert scipy.sparse.issparse(rec_sparse)
    assert rec_sparse.format == lag.format
    assert rec_sparse.dtype == lag.dtype
    assert np.allclose(rec_sparse.toarray(), rec_dense)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_lag_to_recurrence_sparse_badaxis():

    srand()

    data = np.random.randn(3, 100)
    R = librosa.segment.recurrence_matrix(data, sparse=True)
    L = librosa.segment.recurrence_to_lag(R)
    librosa.segment.lag_to_recurrence(L, axis=2)


def test_timelag_filter():
    srand()
    X = np.random.randn(15, 15)
    d_pos0 = librosa.segment.timelag_filter(lambda X: X)
    assert np.allclose(X, d_pos0(X))


def test_timelag_filter_pos1():
    srand()
    X = np.random.randn(15, 15)
    d_pos1 = librosa.segment.timelag_filter(lambda _, X: X, index=1)
    assert np.allclose(X, d_pos1(None, X))


@pytest.fixture(scope="module")
def ysr():
    return librosa.load(__EXAMPLE_FILE)


@pytest.fixture(scope="module")
def mfcc(ysr):
    y, sr = ysr
    return librosa.feature.mfcc(y=y, sr=sr)


@pytest.fixture(scope="module")
def beats(ysr):
    y, sr = ysr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return beats


@pytest.mark.parametrize("n_segments", [1, 2, 3, 4, 100])
def test_subsegment(mfcc, beats, n_segments):

    subseg = librosa.segment.subsegment(mfcc, beats, n_segments=n_segments, axis=-1)

    # Make sure that the boundaries are within range
    assert subseg.min() >= 0
    assert subseg.max() <= mfcc.shape[-1]

    # Make sure that all input beats are retained
    for b in beats:
        assert b in subseg

    # Do we have a 0 marker?
    assert 0 in subseg

    # Did we over-segment?  +2 here for 0- and end-padding
    assert len(subseg) <= n_segments * (len(beats) + 2)

    # Verify that running on the transpose gives the same answer
    ss2 = librosa.segment.subsegment(mfcc.T, beats, n_segments=n_segments, axis=0)
    assert np.allclose(subseg, ss2)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("n_segments", [-1, 0])
def test_subsegment_badn(mfcc, beats, n_segments):
    librosa.segment.subsegment(mfcc, beats, n_segments=n_segments, axis=-1)


@pytest.fixture
def R_input():
    X = np.random.randn(30, 5)

    return X.dot(X.T)


@pytest.mark.parametrize("window", ["rect", "hann"])
@pytest.mark.parametrize("n", [5, 9])
@pytest.mark.parametrize("max_ratio", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("min_ratio", [None, 1.0])
@pytest.mark.parametrize("n_filters", [1, 2, 5])
@pytest.mark.parametrize("zero_mean", [False, True])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("kwargs", [dict(), dict(mode="reflect")])
def test_path_enhance(
    R_input, window, n, max_ratio, min_ratio, n_filters, zero_mean, clip, kwargs
):

    R_smooth = librosa.segment.path_enhance(
        R_input,
        window=window,
        n=n,
        max_ratio=max_ratio,
        min_ratio=min_ratio,
        n_filters=n_filters,
        zero_mean=zero_mean,
        clip=clip,
        **kwargs,
    )

    assert R_smooth.shape == R_input.shape
    assert np.all(np.isfinite(R_smooth))
    assert R_smooth.dtype == R_input.dtype

    if clip:
        assert np.min(R_smooth) >= 0


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_path_enhance_badratio(R_input):
    # We can't have min_ratio > max_ratio
    librosa.segment.path_enhance(R_input, n=5, min_ratio=3, max_ratio=2)
