#!/usr/bin/env python
# CREATED:2015-02-14 16:20:25 by Brian McFee <brian.mcfee@nyu.edu>
#   unit tests for librosa cache
"""Tests for librosa.cache"""

import contextlib
import os
import shutil
import tempfile

import numpy as np
import pytest

import librosa._cache

# Disable any initial cache settings
for key in ["DIR", "MMAP", "COMPRESS", "VERBOSE", "LEVEL"]:
    with contextlib.suppress(KeyError):
        os.environ.pop(f"LIBROSA_CACHE_{key:s}")


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

    for _i in range(3):
        assert np.allclose(func_cache(5), y)


def test_cache_real_function(local_cache):
    # Test wrapping and calling an actual librosa function through the cache
    local_cache.clear()
    cached_semitone = local_cache(level=10)(librosa.filters.semitone_filterbank)

    fb1, sr1 = cached_semitone()
    fb2, sr2 = cached_semitone()

    assert np.allclose(sr1, sr2)
    assert len(fb1) == len(fb2)


def test_cache_levels():
    cache_dir = tempfile.mkdtemp()
    try:
        # Manager with level=20
        cm20 = librosa._cache.CacheManager(cache_dir, verbose=0, level=20)
        # Function with level=10 should be wrapped
        f10 = cm20(level=10)(func)
        assert f10 != func

        # Function with level=20 should be wrapped
        f20 = cm20(level=20)(func)
        assert f20 != func

        # Function with level=30 should NOT be wrapped
        f30 = cm20(level=30)(func)
        assert f30 == func
    finally:
        shutil.rmtree(cache_dir)


def test_cache_docstring_consistency():
    import importlib
    import inspect
    import pkgutil
    import re

    submodules = [
        m
        for _, m, p in pkgutil.walk_packages(librosa.__path__, prefix="librosa.")
        if not p and not m.endswith(".version") and not m.endswith("._typing") and not m.endswith("._cache")
    ]
    cache_re = re.compile(r"caches at level\s+(\d+)", re.IGNORECASE)

    mismatches = []
    for modname in sorted(submodules):
        mod = importlib.import_module(modname)
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and obj.__module__ == modname:
                doc = inspect.getdoc(obj) or ""
                m_doc = cache_re.search(doc)
                doc_level = int(m_doc.group(1)) if m_doc else None
                try:
                    src = inspect.getsource(obj)
                except Exception:
                    src = ""
                m = re.search(r"@(?:cache|librosa\._cache\.cache)\(level=(\d+)\)", src)
                dec_level = int(m.group(1)) if m else None
                if dec_level != doc_level:
                    mismatches.append((f"{modname}.{name}", dec_level, doc_level))

    assert not mismatches, f"Docstring vs decorator cache level mismatches found: {mismatches}"
