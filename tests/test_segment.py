#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for segmentation functions'''
import warnings

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from nose.tools import raises

from test_core import srand

import librosa
__EXAMPLE_FILE = 'data/test1_22050.wav'

warnings.resetwarnings()
warnings.simplefilter('always')


def test_recurrence_matrix():

    def __test(n, k, width, sym, metric):
        srand()
        # Make a data matrix
        data = np.random.randn(3, n)

        D = librosa.segment.recurrence_matrix(data, k=k, width=width, sym=sym, axis=-1, metric=metric)

        # First test for symmetry
        if sym:
            assert np.allclose(D, D.T)

        # Test for target-axis invariance
        D_trans = librosa.segment.recurrence_matrix(data.T, k=k, width=width, sym=sym, axis=0, metric=metric)
        assert np.allclose(D, D_trans)

        # If not symmetric, test for correct number of links
        if not sym and k is not None:
            real_k = min(k, n - width)
            assert not np.any(D.sum(axis=1) != real_k)

        # Make sure the +- width diagonal is hollow
        # It's easier to test if zeroing out the triangles leaves nothing
        idx = np.tril_indices(n, k=width)

        D[idx] = False
        D.T[idx] = False
        assert not np.any(D)

    for n in [20, 250]:
        for k in [None, n//4]:
            for sym in [False, True]:
                for width in [-1, 0, 1, 3, 5]:
                    for metric in ['l2', 'cosine']:
                        tester = __test
                        if width < 1:
                            tester = raises(librosa.ParameterError)(__test)

                        yield tester, n, k, width, sym, metric


def test_recurrence_sparse():

    srand()
    data = np.random.randn(3, 100)
    D_sparse = librosa.segment.recurrence_matrix(data, sparse=True)
    D_dense = librosa.segment.recurrence_matrix(data, sparse=False)

    assert scipy.sparse.isspmatrix(D_sparse)
    assert np.allclose(D_sparse.todense(), D_dense)


def test_recurrence_distance():

    srand()
    data = np.random.randn(3, 100)
    distance = squareform(pdist(data.T, metric='sqeuclidean'))
    rec = librosa.segment.recurrence_matrix(data, mode='distance',
                                            metric='sqeuclidean',
                                            sparse=True)

    i, j, vals = scipy.sparse.find(rec)
    assert np.allclose(vals, distance[i, j])


def test_recurrence_affinity():

    def __test(metric, bandwidth):
        srand()
        data = np.random.randn(3, 100)
        distance = squareform(pdist(data.T, metric=metric))
        rec = librosa.segment.recurrence_matrix(data, mode='affinity',
                                                metric=metric,
                                                sparse=True,
                                                bandwidth=bandwidth)

        i, j, vals = scipy.sparse.find(rec)
        logvals = np.log(vals)

        # After log-scaling, affinity will match distance up to a constant factor
        ratio = -logvals / distance[i, j]
        if bandwidth is None:
            assert np.allclose(ratio, ratio[0])
        else:
            assert np.allclose(ratio, bandwidth)

    for metric in ['sqeuclidean', 'cityblock']:
        for bandwidth in [None, 1]:
            yield __test, metric, bandwidth


@raises(librosa.ParameterError)
def test_recurrence_badmode():

    srand()
    data = np.random.randn(3, 100)

    rec = librosa.segment.recurrence_matrix(data, mode='NOT A MODE',
                                            metric='sqeuclidean',
                                            sparse=True)


@raises(librosa.ParameterError)
def test_recurrence_bad_bandwidth():

    srand()
    data = np.random.randn(3, 100)
    rec = librosa.segment.recurrence_matrix(data, bandwidth=-2)


def test_recurrence_to_lag():

    def __test(n, pad):
        srand()
        data = np.random.randn(17, n)

        rec = librosa.segment.recurrence_matrix(data)

        lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
        lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0).T

        assert np.allclose(lag, lag2)

        x = Ellipsis
        if pad:
            x = slice(n)

        for i in range(n):
            assert np.allclose(rec[:, i], np.roll(lag[:, i], i)[x])

    @raises(librosa.ParameterError)
    def __test_fail(size):
        librosa.segment.recurrence_to_lag(np.zeros(size))

    for n in [10, 100, 1000]:
        for pad in [False, True]:
            yield __test, n, pad

    yield __test_fail, (17,)
    yield __test_fail, (17, 34)
    yield __test_fail, (17, 17, 17)


def test_recurrence_to_lag_sparse():

    srand()

    def __test(pad, axis, rec):

        rec_dense = rec.toarray()

        lag_sparse = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=axis)
        lag_dense = librosa.segment.recurrence_to_lag(rec_dense, pad=pad, axis=axis)

        assert scipy.sparse.issparse(lag_sparse)
        assert rec.format == lag_sparse.format
        assert rec.dtype == lag_sparse.dtype
        assert np.allclose(lag_sparse.toarray(), lag_dense)

    data = np.random.randn(3, 100)
    R_sparse = librosa.segment.recurrence_matrix(data, sparse=True)

    for pad in [False, True]:
        for axis in [0, 1, -1]:
            yield __test, pad, axis, R_sparse


def test_lag_to_recurrence():

    def __test(n, pad):
        srand()
        data = np.random.randn(17, n)

        rec = librosa.segment.recurrence_matrix(data)
        lag = librosa.segment.recurrence_to_lag(rec, pad=pad, axis=-1)
        lag2 = librosa.segment.recurrence_to_lag(rec.T, pad=pad, axis=0).T

        rec2 = librosa.segment.lag_to_recurrence(lag)

        assert np.allclose(rec, rec2)
        assert np.allclose(lag, lag2)

    @raises(librosa.ParameterError)
    def __test_fail(size):
        librosa.segment.lag_to_recurrence(np.zeros(size))

    for n in [10, 100, 1000]:
        for pad in [False, True]:
            yield __test, n, pad

    yield __test_fail, (17,)
    yield __test_fail, (17, 35)
    yield __test_fail, (17, 17, 17)


def test_lag_to_recurrence_sparse():

    srand()

    def __test(axis, lag):

        lag_dense = lag.toarray()

        rec_sparse = librosa.segment.lag_to_recurrence(lag, axis=axis)
        rec_dense = librosa.segment.lag_to_recurrence(lag_dense, axis=axis)

        assert scipy.sparse.issparse(rec_sparse)
        assert rec_sparse.format == lag.format
        assert rec_sparse.dtype == lag.dtype
        assert np.allclose(rec_sparse.toarray(), rec_dense)

    data = np.random.randn(3, 100)
    R = librosa.segment.recurrence_matrix(data, sparse=True)

    for pad in [False, True]:
        for axis in [0, 1, -1]:
            L = librosa.segment.recurrence_to_lag(R, pad=pad, axis=axis)
            yield __test, axis, L


@raises(librosa.ParameterError)
def test_lag_to_recurrence_sparse_badaxis():

    srand()

    data = np.random.randn(3, 100)
    R = librosa.segment.recurrence_matrix(data, sparse=True)
    L = librosa.segment.recurrence_to_lag(R)
    librosa.segment.lag_to_recurrence(L, axis=2)


def test_timelag_filter():

    def pos0_filter(X):
        return X

    def pos1_filter(_, X):
        return X

    def __test_positional(n):
        srand()
        dpos0 = librosa.segment.timelag_filter(pos0_filter)
        dpos1 = librosa.segment.timelag_filter(pos1_filter, index=1)

        X = np.random.randn(n, n)

        assert np.allclose(X, dpos0(X))
        assert np.allclose(X, dpos1(None, X))

    yield __test_positional, 25


def test_subsegment():

    y, sr = librosa.load(__EXAMPLE_FILE)

    X = librosa.feature.mfcc(y=y, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)

    def __test(n_segments):

        subseg = librosa.segment.subsegment(X, beats, n_segments=n_segments, axis=-1)

        # Make sure that the boundaries are within range
        assert subseg.min() >= 0
        assert subseg.max() <= X.shape[-1]

        # Make sure that all input beats are retained
        for b in beats:
            assert b in subseg

        # Do we have a 0 marker?
        assert 0 in subseg

        # Did we over-segment?  +2 here for 0- and end-padding
        assert len(subseg) <= n_segments * (len(beats) + 2)

        # Verify that running on the transpose gives the same answer
        ss2 = librosa.segment.subsegment(X.T, beats, n_segments=n_segments, axis=0)
        assert np.allclose(subseg, ss2)

    for n_segments in [0, 1, 2, 3, 4, 100]:
        if n_segments < 1:
            tf = raises(librosa.ParameterError, IndexError)(__test)
        else:
            tf = __test
        yield tf, n_segments
