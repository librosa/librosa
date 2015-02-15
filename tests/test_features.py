#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np

from nose.tools import raises, eq_


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
