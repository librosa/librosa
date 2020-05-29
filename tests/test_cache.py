#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 16:20:25 by Brian McFee <brian.mcfee@nyu.edu>
#   unit tests for librosa cache
"""Tests for librosa.cache"""

import os
import sys
import tempfile
import shutil
import numpy as np

import pytest
import librosa._cache


# Disable any initial cache settings
for key in [
        "DIR", "MMAP", "COMPRESS", "VERBOSE", "LEVEL",
        "BYTES_LIMIT", "RESIZE_INTERVAL"]:
    os.environ.pop("LIBROSA_CACHE_{:s}".format(key), None)


@pytest.fixture
def local_cache():
    cache_dir = tempfile.mkdtemp()
    cache = librosa._cache.CacheManager(
        cache_dir, verbose=0, level=10, bytes_limit=96, cache_resize_interval=0.01)
    yield cache
    shutil.rmtree(cache_dir)


def func(x):
    return np.random.randn(x)


def test_cache_disabled():
    # When there's no cache directory in the environment,
    # librosa.cache is a no-op.
    cache = librosa._cache.CacheManager(None, verbose=0, level=10)
    func_cache = cache(level=-10)(func)
    assert func == func_cache


def test_cache_enabled(local_cache):
    local_cache.clear()
    func_cache = local_cache(level=-10)(func)

    # The cache should be active now, so func_cache should be a different object from func
    assert func_cache != func

    # check the function output looks right
    y = func_cache(5)
    assert y.shape == (5,)

    # issue three calls to func - they should all be the same
    for i in range(3):
        assert np.allclose(func_cache(5), y)
        assert len(local_cache.memory.store_backend.get_items()) == 1

    # overflow the cache so that the value for func_cache(5) will change
    for i in range(6, 500):
        x = func_cache(i)

    # see that it's a different random array because the cache has been cleared
    assert not np.allclose(func_cache(5), y)
