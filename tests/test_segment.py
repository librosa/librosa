#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Tests for segmentation functions'''

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np


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

    for width in [0, 1, 2, 3, 5, 9]:
        for n in [width, width+1, width * 2, width*width]:
            
            yield __test, width, n, -1, +1


def test_recurrence_matrix():

    def __test(n, k, width, sym):
        # Make a data matrix
        data = np.random.randn(3, n)

        D = librosa.segment.recurrence_matrix(data, k=k, width=width, sym=sym)

        # First test for symmetry
        if sym:
            assert np.allclose(D, D.T)

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
            for width in [1, 3, 5]:
                for sym in [False, True]:
                    yield __test, n, k, width, sym


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
