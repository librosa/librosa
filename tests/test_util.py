#!/usr/bin/env python
# CREATED:2014-01-18 14:09:05 by Brian McFee <brm2132@columbia.edu>
# unit tests for util routines 

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import numpy as np
import librosa
from nose.tools import raises


def test_example_audio_file():

    assert os.path.exists(librosa.util.example_audio_file())


def test_frame():

    # Generate a random time series
    def __test(P):
        frame, hop = P

        y = np.random.randn(8000)
        y_frame = librosa.util.frame(y, frame_length=frame, hop_length=hop)

        for i in range(y_frame.shape[1]):
            assert np.allclose(y_frame[:, i], y[ i * hop : (i * hop + frame)])

    for frame in [256, 1024, 2048]:
        for hop_length in [64, 256, 512]:
            yield (__test, [frame, hop_length])


def test_pad_center():
    
    def __test(y, n, axis, mode):

        y_out = librosa.util.pad_center(y, n, axis=axis, mode=mode)

        n_len = y.shape[axis]
        n_pad = int((n - n_len) / 2)

        eq_slice = [Ellipsis] * y.ndim
        eq_slice[axis] = slice(n_pad, n_pad + n_len)

        assert np.allclose(y, y_out[eq_slice])

    @raises(ValueError)
    def __test_fail(y, n, axis, mode):
        librosa.util.pad_center(y, n, axis=axis, mode=mode)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for mode in ['constant', 'edge', 'reflect']:
                for n in [0, 10]:
                    yield __test, y, n + y.shape[axis], axis, mode

                for n in [0, 10]:
                    yield __test_fail, y, n, axis, mode


def test_fix_length():

    def __test(y, n, axis):

        y_out = librosa.util.fix_length(y, n, axis=axis)

        eq_slice = [Ellipsis] * y.ndim
        eq_slice[axis] = slice(y.shape[axis])

        if n > y.shape[axis]:
            assert np.allclose(y, y_out[eq_slice])
        else:
            assert np.allclose(y[eq_slice], y)

    for shape in [(16,), (16, 16)]:
        y = np.ones(shape)

        for axis in [0, -1]:
            for n in [-5, 0, 5]:
                yield __test, y, n + y.shape[axis], axis


def test_fix_frames():

    @raises(ValueError)
    def __test_fail(frames, x_min, x_max, pad):
        librosa.util.fix_frames(frames, x_min, x_max, pad)

    def __test_pass(frames, x_min, x_max, pad):

        f_fix = librosa.util.fix_frames(frames,
                                        x_min=x_min,
                                        x_max=x_max,
                                        pad=pad)

        if x_min is not None:
            if pad:
                assert f_fix[0] == x_min
            assert np.all(f_fix >= x_min)

        if x_max is not None:
            if pad:
                assert f_fix[-1] == x_max
            assert np.all(f_fix <= x_max)

    for low in [-20, 0, 20]:
        for high in [low + 20, low + 50, low + 100]:

            frames = np.random.randint(low, high=high, size=15)

            for x_min in [None, 0, 20]:
                for x_max in [None, 20, 100]:
                    for pad in [False, True]:
                        if np.any(frames < 0):
                            yield __test_fail, frames, x_min, x_max, pad
                        else:
                            yield __test_pass, frames, x_min, x_max, pad


def test_normalize():

    def __test_pass(X, norm, axis):
        X_norm = librosa.util.normalize(X, norm=norm, axis=axis)

        X_norm = np.abs(X_norm)

        if norm == np.inf:
            values = np.max(X_norm, axis=axis)
        elif norm == -np.inf:
            values = np.min(X_norm, axis=axis)
        elif norm == 0:
            # XXX: normalization here isn't quite right
            values = np.ones(1)
        else:
            values = np.sum(X_norm**norm, axis=axis)**(1./norm)

        assert np.allclose(values, np.ones_like(values))

    @raises(ValueError)
    def __test_fail(X, norm, axis):
        librosa.util.normalize(X, norm=norm, axis=axis)

    for ndims in [1, 2, 3]:
        X = np.random.randn(* ([16] * ndims))

        for axis in range(X.ndim):
            for norm in [np.inf, -np.inf, 0, 0.5, 1.0, 2.0]:
                yield __test_pass, X, norm, axis

            for norm in ['inf', -0.5, -2]:
                yield __test_fail, X, norm, axis


def test_axis_sort():

    def __test_pass(data, axis, index, value):

        if index:
            Xsorted, idx = librosa.util.axis_sort(data,
                                                  axis=axis,
                                                  index=index,
                                                  value=value)

            cmp_slice = [Ellipsis] * X.ndim
            cmp_slice[axis] = idx

            assert np.allclose(X[cmp_slice], Xsorted)

        else:
            Xsorted = librosa.util.axis_sort(data,
                                             axis=axis,
                                             index=index,
                                             value=value)

        compare_axis = np.mod(1 - axis, 2)

        sort_values = value(Xsorted, axis=compare_axis)

        assert np.allclose(sort_values, np.sort(sort_values))

    @raises(ValueError)
    def __test_fail(data, axis, index, value):
        librosa.util.axis_sort(data, axis=axis, index=index, value=value)

    for ndim in [1, 2, 3]:
        X = np.random.randn(*([10] * ndim))

        for axis in [0, 1, -1]:
            for index in [False, True]:
                for value in [np.min, np.mean, np.max]:

                    if ndim == 2:
                        yield __test_pass, X, axis, index, value
                    else:
                        yield __test_fail, X, axis, index, value


def test_match_intervals():

    def __make_intervals(n):
        
        return np.cumsum(np.abs(np.random.randn(n, 2)), axis=1)

    def __compare_intervals(i1, i2):

        return np.maximum(0, np.minimum(i1[-1], i2[-1])
                          - np.maximum(i1[0], i2[0]))

    def __make_truth(ints1, ints2):

        n1 = ints1.shape[0]

        y = np.empty(n1)

        for i in range(n1):
            y[i] = np.argmax([__compare_intervals(ints1[i], i2) for i2 in ints2])

        return y

    def __test(n, m):
        ints1 = __make_intervals(n)
        ints2 = __make_intervals(m)

        y_true = __make_truth(ints1, ints2)

        y_pred = librosa.util.match_intervals(ints1, ints2)

        assert np.allclose(y_pred, y_true)

    @raises(ValueError)
    def __test_fail(n, m):
        ints1 = __make_intervals(n)
        ints2 = __make_intervals(m)
        librosa.util.match_intervals(ints1, ints2)

    for n in [0, 1, 5, 20, 500]:
        for m in [0, 1, 5, 20, 500, 5000]:
            if n == 0 or m == 0:
                yield __test_fail, n, m
            else:
                yield __test, n, m
    
    # TODO:   2015-01-20 17:04:55 by Brian McFee <brian.mcfee@nyu.edu>
    # add coverage for shape errors


def test_match_events():

    def __make_events(n):
        return np.abs(np.random.randn(n))

    def __make_truth(ev1, ev2):
        n1 = len(ev1)

        y = np.empty(n1)
        for i in range(n1):
            y[i] = np.argmin([np.abs(ev1[i] - ev) for ev in ev2])

        return y

    def __test(n, m):
        ev1 = __make_events(n)
        ev2 = __make_events(m)

        y_true = __make_truth(ev1, ev2)

        y_pred = librosa.util.match_events(ev1, ev2)

        assert np.allclose(y_pred, y_true)

    @raises(ValueError)
    def __test_fail(n, m):
        ev1 = __make_events(n)
        ev2 = __make_events(m)
        librosa.util.match_events(ev1, ev2)

    for n in [0, 1, 5, 20, 500]:
        for m in [0, 1, 5, 20, 500, 5000]:
            if n == 0 or m == 0:
                yield __test_fail, n, m
            else:
                yield __test, n, m


def test_localmax():

    def __test_axis(ndim, axis):

        data = np.random.randn(*([50] * ndim))

        lm = librosa.util.localmax(data, axis=axis)

        for hits in np.argwhere(lm):
            for offset in [-1, 1]:
                compare_idx = hits.copy()
                compare_idx[axis] += offset

                if compare_idx[axis] < 0:
                    continue

                if compare_idx[axis] >= data.shape[axis]:
                    continue

                if offset < 0:
                    assert data[tuple(hits)] > data[tuple(compare_idx)]
                else:
                    assert data[tuple(hits)] >= data[tuple(compare_idx)]

    for ndim in [1, 2, 3]:
        for axis in range(ndim):
            yield __test_axis, ndim, axis
