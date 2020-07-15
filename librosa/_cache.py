#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function caching"""

import os
import time
import functools
from joblib import Memory


class CacheManager(object):
    '''The librosa cache manager class wraps joblib.Memory
    with a __call__ attribute, so that it may act as a function.

    Additionally, it provides a caching level filter, so that
    different functions can be cached or not depending on the user's
    preference for speed vs. storage usage.
    '''

    def __init__(self, *args, cache_resize_interval=20, level=10, **kwargs):
        # Initialize the memory object
        self.memory = Memory(*args, **kwargs)
        # The level parameter controls which data we cache
        # smaller numbers mean less caching
        self.level = level
        # call reduce_size no more frequently than cache_resize_interval
        self._throttled_reduce_size = throttle(
            self.reduce_size, cache_resize_interval)
        self._throttled_reduce_size()

    def __call__(self, level):
        '''Example usage:

        @cache(level=2)
        def semi_important_function(some_arguments):
            ...
        '''
        def wrapper(function):
            '''Decorator function.  Adds an input/output cache to
            the specified function.'''

            if self.memory.location is not None and self.level >= level:
                cached_function = self.memory.cache(function)

                def inner(*a, **kw):
                    '''Call function and maybe resize cache.'''
                    _ = cached_function(*a, **kw)
                    self._throttled_reduce_size()
                    return _

                return functools.wraps(function)(inner)
            return function
        return wrapper

    def clear(self, *args, **kwargs):
        return self.memory.clear(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.memory.eval(*args, **kwargs)

    def format(self, *args, **kwargs):
        return self.memory.format(*args, **kwargs)

    def reduce_size(self, *args, **kwargs):
        return self.memory.reduce_size(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self.memory.warn(*args, **kwargs)


def throttle(func, interval):
    '''Don't call a function if it has been called in the last `interval` seconds.'''
    if not interval:
        return func
    @functools.wraps(func)
    def inner(*a, _ignore_cache=False, **kw):
        # rerun function
        tnow = time.time()
        if _ignore_cache or (tnow - inner.last_ran) > interval:
            inner.last_result = func(*a, **kw)
            inner.last_ran = tnow

        return inner.last_result
    inner.last_ran = 0
    inner.last_result = None
    return inner

# Instantiate the cache from the environment
cache = CacheManager(
    os.environ.get('LIBROSA_CACHE_DIR', None),
    bytes_limit=float(os.environ.get('LIBROSA_CACHE_BYTES_LIMIT', 0)) or None,
    cache_resize_interval=float(os.environ.get('LIBROSA_CACHE_RESIZE_INTERVAL', 20)),
    mmap_mode=os.environ.get('LIBROSA_CACHE_MMAP', None),
    compress=os.environ.get('LIBROSA_CACHE_COMPRESS', False),
    verbose=int(os.environ.get('LIBROSA_CACHE_VERBOSE', 0)),
    level=int(os.environ.get('LIBROSA_CACHE_LEVEL', 10)))
