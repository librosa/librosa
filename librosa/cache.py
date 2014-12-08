#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function caching"""

import os
import sys
from joblib import Memory


class CacheManager(Memory):
    '''The librosa cache manager class extends joblib.Memory
    with a __call__ attribute, so that it may act as a function.

    This allows us to override the librosa.cache module's __call__
    field, thereby allowing librosa.cache to act as a decorator function.
    '''

    def __call__(self, function):
        '''Decorator function.  Adds an input/output cache to
        the specified function.'''

        if self.cachedir is not None:
            return self.cache(function)
        else:
            return function

# Instantiate the cache from the environment
CACHE = CacheManager(os.environ.get('LIBROSA_CACHE_DIR', None),
                     mmap_mode=os.environ.get('LIBROSA_CACHE_MMAP', None),
                     compress=os.environ.get('LIBROSA_CACHE_COMPRESS', False),
                     verbose=int(os.environ.get('LIBROSA_CACHE_VERBOSE', 0)))

# Override the module's __call__ attribute
sys.modules[__name__] = CACHE
