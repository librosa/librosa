#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function caching"""

import os
import sys
from joblib import Memory
from decorator import decorator


class CacheManager(Memory):
    '''The librosa cache manager class extends joblib.Memory
    with a __call__ attribute, so that it may act as a function.

    This allows us to override the librosa.cache module's __call__
    field, thereby allowing librosa.cache to act as a decorator function.
    '''

    def __call__(self, function):
        '''Decorator function.  Adds an input/output cache to
        the specified function.'''
        return self.cache(function)

# Instantiate the cache from the environment
CACHE = CacheManager(os.environ.get('LIBROSA_CACHEDIR', None))

# Override the module's __call__ attribute
sys.modules[__name__] = CACHE
