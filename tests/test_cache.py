#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 16:20:25 by Brian McFee <brian.mcfee@nyu.edu>
#   unit tests for librosa cache
'''Tests for librosa.cache'''

import os
import sys
import tempfile
import shutil
import numpy as np

from nose.tools import with_setup, eq_

import warnings
warnings.resetwarnings()
warnings.simplefilter('always')

# Disable any initial cache settings
for key in ['DIR', 'MMAP', 'COMPRESS', 'VERBOSE', 'LEVEL']:
    try:
        os.environ.pop('LIBROSA_CACHE_{:s}'.format(key))
    except KeyError:
        pass


def cache_construct():
    '''Make a temp directory for the librosa cache'''
    cache_dir = tempfile.mkdtemp()
    os.environ['LIBROSA_CACHE_DIR'] = cache_dir


def cache_teardown():
    '''Blow away the temp directory'''

    cache_dir = os.environ.pop('LIBROSA_CACHE_DIR')
    shutil.rmtree(cache_dir)


def func(x):

    return np.arange(x)


def test_cache_disabled():

    os.environ.pop('LIBROSA_CACHE_DIR', None)
    sys.modules.pop('librosa.cache', None)
    import librosa.cache

    func_cache = librosa.cache(level=-10)(func)

    # When there's no cache directory in the environment,
    # librosa.cache is a no-op.
    eq_(func, func_cache)


@with_setup(cache_construct, cache_teardown)
def test_cache_enabled():

    sys.modules.pop('librosa.cache', None)
    import librosa.cache
    librosa.cache.clear()

    func_cache = librosa.cache(level=-10)(func)

    # The cache should be active now
    assert func_cache != func

    # issue three calls to func
    y = func(5)

    for i in range(3):
        assert np.allclose(func_cache(5), y)
