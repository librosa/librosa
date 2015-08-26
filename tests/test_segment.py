#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for segmentation functions'''

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import matplotlib
matplotlib.use('Agg')
import numpy as np
from nose.tools import raises

import librosa
__EXAMPLE_FILE = 'data/test1_22050.wav'


def test_band_infinite():

    def __test(width, n, v_in, v_out):
        B = librosa.segment.__band_infinite(n, width, v_in=v_in, v_out=v_out)

        idx = np.tril_indices(n, k=width-1)

        # First check the shape
        assert np.allclose(B.shape, [n, n])

        # Check symmetry
        assert np.allclose(B, B.T)

        # Check in-band first
        Bincheck = B.copy()
        Bincheck[idx] = v_in
        Bincheck.T[idx] = v_in

        assert np.all(Bincheck == v_in)

        # If we're too small to have an out-of-band, then we're done
        if not len(idx[0]):
            return

        Boutcheck = B.copy()
        Boutcheck[idx] = v_out
        Boutcheck.T[idx] = v_out

        assert np.all(Boutcheck == v_out)

    @raises(librosa.ParameterError)
    def __test_fail(width, n, v_in, v_out):
        librosa.segment.__band_infinite(n, width, v_in=v_in, v_out=v_out)

    for width in [0, 1, 2, 3, 5, 9]:
        for n in [width//2, width, width+1, width * 2, width**2]:
            if width > n:
                yield __test_fail, width, n, -1, +1
            else:
                yield __test, width, n, -1, +1


def test_recurrence_matrix():

    def __test(n, k, width, sym):
        # Make a data matrix
        data = np.random.randn(3, n)

        D = librosa.segment.recurrence_matrix(data, k=k, width=width, sym=sym, axis=-1)


        # First test for symmetry
        if sym:
            assert np.allclose(D, D.T)

        # Test for target-axis invariance
        D_trans = librosa.segment.recurrence_matrix(data.T, k=k, width=width, sym=sym, axis=0)
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


    for n in [10, 100, 1000]:
        for k in [None, int(n/4)]:
            for sym in [False, True]:
                for width in [-1, 0, 1, 3, 5]:
                    tester = __test
                    if width < 1:
                        tester = raises(librosa.ParameterError)(__test)

                    yield tester, n, k, width, sym


def test_recurrence_to_lag():

    def __test(n, pad):
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


def test_lag_to_recurrence():

    def __test(n, pad):
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


def test_structure_feature():

    def __test(n, pad):
        # Make a data matrix
        data = np.random.randn(17, n)

        # Make a recurrence plot
        rec = librosa.segment.recurrence_matrix(data)

        # Make a structure feature
        st = librosa.segment.structure_feature(rec, pad=pad, inverse=False)

        # Test for each column
        if pad:
            x = slice(n)
        else:
            x = Ellipsis

        for i in range(n):
            assert np.allclose(rec[:, i], np.roll(st[:, i], i)[x])

        # Invert it
        rec2 = librosa.segment.structure_feature(st, pad=pad, inverse=True)

        assert np.allclose(rec, rec2)

    for n in [10, 100, 1000]:
        for pad in [False, True]:
            yield __test, n, pad


def test_timelag_filter():

    def pos0_filter(X):
        return X

    def pos1_filter(_, X):
        return X

    def __test_positional(n):
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
