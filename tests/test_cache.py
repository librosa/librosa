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
for key in ["DIR", "MMAP", "COMPRESS", "VERBOSE", "LEVEL"]:
    try:
        os.environ.pop("LIBROSA_CACHE_{:s}".format(key))
    except KeyError:
        pass


@pytest.fixture
def local_cache():
    cache_dir = tempfile.mkdtemp()
    cache = librosa._cache.CacheManager(cache_dir, verbose=0, level=10)
    yield cache
    shutil.rmtree(cache_dir)


def func(x):

    return np.arange(x)


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

    # issue three calls to func
    y = func(5)

    for i in range(3):
        assert np.allclose(func_cache(5), y)
