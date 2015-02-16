#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import print_function

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np

from nose.tools import raises, eq_


def test_delta():
    # Note: this test currently only checks first-order differences
    #       and width=3 filters

    def __test(width, order, axis, x):
        # Compare trimmed and untrimmed versions
        delta = librosa.feature.delta(x,
                                      width=width,
                                      order=order,
                                      axis=axis,
                                      trim=False)
        delta_t = librosa.feature.delta(x,
                                        width=width,
                                        order=order,
                                        axis=axis,
                                        trim=True)

        # Check that trimming matches the expected shape
        eq_(x.shape, delta_t.shape)

        # Check that trimming gives the right values in the right places
        _s = [Ellipsis] * delta.ndim
        _s[axis] = slice(1 + width//2, -(1 + width//2))
        delta_retrim = delta[_s]
        assert np.allclose(delta_t, delta_retrim)

        # Check that the delta values line up with the data
        # for a width=3 filter, delta[i] = x[i+1] - x[i-1]
        _s_front = [Ellipsis] * delta.ndim
        _s_back = [Ellipsis] * delta.ndim
        _s_front[axis] = slice(1 + width//2, None)
        _s_back[axis] = slice(None, -(1 + width//2))

        assert np.allclose(x, (delta[_s_front] + delta[_s_back])[_s_back])

    x = np.vstack([np.arange(5.0)**2] * 2)

    for width in [-1, 0, 1, 2, 3, 4]:
        for order in [0, 1]:
            for axis in range(x.ndim):
                tf = __test
                if width != 3:
                    tf = raises(ValueError)(__test)
                if order != 1:
                    tf = raises(ValueError)(__test)

                yield tf, width, order, axis, x


def test_stack_memory():

    def __test(data, n_steps, delay):
        data_stack = librosa.feature.stack_memory(data,
                                                  n_steps=n_steps,
                                                  delay=delay)

        # If we're one-dimensional, reshape for testing
        if data.ndim == 1:
            data = data.reshape((1, -1))

        d, t = data.shape

        eq_(data_stack.shape[0], n_steps * d)
        eq_(data_stack.shape[1], t)

        for i in range(d):
            for step in range(1, n_steps):
                assert np.allclose(data[i, :- step * delay],
                                   data_stack[step * d + i, step * delay:])

    for ndim in [1, 2]:
        data = np.random.randn(* ([5] * ndim))

        for n_steps in [1, 2, 3, 4]:
            for delay in [1, 2, 4]:
                yield __test, data, n_steps, delay


def test_sync():

    @raises(ValueError)
    def __test_fail(data, frames):
        librosa.feature.sync(data, frames)

    def __test_pass(data, frames):
        # By default, mean aggregation
        dsync = librosa.feature.sync(data, frames)
        assert np.allclose(dsync, 2 * np.ones_like(dsync))

        # Explicit mean aggregation
        dsync = librosa.feature.sync(data, frames, aggregate=np.mean)
        assert np.allclose(dsync, 2 * np.ones_like(dsync))

        # Max aggregation
        dsync = librosa.feature.sync(data, frames, aggregate=np.max)
        assert np.allclose(dsync, 4 * np.ones_like(dsync))

        # Min aggregation
        dsync = librosa.feature.sync(data, frames, aggregate=np.min)
        assert np.allclose(dsync, np.zeros_like(dsync))

    for ndim in [1, 2, 3]:
        shaper = [1] * ndim
        shaper[-1] = -1

        data = np.mod(np.arange(135), 5)
        frames = np.flatnonzero(data[0] == 0)

        data = np.reshape(data, shaper)

        if ndim > 2:
            yield __test_fail, data, frames

        else:
            yield __test_pass, data, frames
