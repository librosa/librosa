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

    def __init__(self, location, level=10, **kwargs):
        super(CacheManager, self).__init__(location, **kwargs)
        # The level parameter controls which data we cache
        # smaller numbers mean less caching
        self.level = level

    def __call__(self, level):
        '''Example usage:

        @cache(level=2)
        def semi_important_function(some_arguments):
            ...
        '''
        def wrapper(function):
            '''Decorator function.  Adds an input/output cache to
            the specified function.'''

            from decorator import FunctionMaker

            def decorator_apply(dec, func):
                """Decorate a function by preserving the signature even if dec
                is not a signature-preserving decorator.

                This recipe is derived from
                http://micheles.googlecode.com/hg/decorator/documentation.html#id14
                """

                return FunctionMaker.create(
                    func, 'return decorated(%(signature)s)',
                    dict(decorated=dec(func)), __wrapped__=func)

            if self.location is not None and self.level >= level:
                return decorator_apply(self.cache, function)

            else:
                return function
        return wrapper


# Instantiate the cache from the environment
CACHE = CacheManager(os.environ.get('LIBROSA_CACHE_DIR', None),
                     mmap_mode=os.environ.get('LIBROSA_CACHE_MMAP', None),
                     compress=os.environ.get('LIBROSA_CACHE_COMPRESS', False),
                     verbose=int(os.environ.get('LIBROSA_CACHE_VERBOSE', 0)),
                     level=int(os.environ.get('LIBROSA_CACHE_LEVEL', 10)))

# Override the module's __call__ attribute
sys.modules[__name__] = CACHE
