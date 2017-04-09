#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-15 10:06:03 by Brian McFee <brian.mcfee@nyu.edu>
'''Helpful tools for deprecation'''

import warnings
from decorator import decorator
from functools import wraps
import six

__all__ = ['moved', 'deprecated', 'optional_jit']

def moved(moved_from, version, version_removed):
    '''This is a decorator which can be used to mark functions
    as moved/renamed.

    It will result in a warning being emitted when the function is used.
    '''

    def __wrapper(func, *args, **kwargs):
        '''Warn the user, and then proceed.'''
        code = six.get_function_code(func)
        warnings.warn_explicit(
            "{:s}\n\tThis function was moved to '{:s}.{:s}' in "
            "librosa version {:s}."
            "\n\tThis alias will be removed in librosa version "
            "{:s}.".format(moved_from, func.__module__,
                           func.__name__, version, version_removed),

            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


def deprecated(version, version_removed):
    '''This is a decorator which can be used to mark functions
    as deprecated.

    It will result in a warning being emitted when the function is used.'''

    def __wrapper(func, *args, **kwargs):
        '''Warn the user, and then proceed.'''
        code = six.get_function_code(func)
        warnings.warn_explicit(
            "{:s}.{:s}\n\tDeprecated as of librosa version {:s}."
            "\n\tIt will be removed in librosa version {:s}."
            .format(func.__module__, func.__name__,
                    version, version_removed),
            category=DeprecationWarning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


'''Define the optional_jit decorator
   If numba is importable, use numba.jit.
   Else create a no-op decorator.
'''
try:
    from numba.decorators import jit as optional_jit
except ImportError:
    # Decorator with optional arguments borrowed from
    # http://stackoverflow.com/a/10288927
    def magical_decorator(decorator):
        @wraps(decorator)
        def inner(*args, **kw):
            if len(args) == 1 and not kw and callable(args[0]):
                return decorator()(args[0])
            else:
                return decorator(*args, **kw)
        return inner

    @magical_decorator
    def optional_jit(*_, **__):
        def __wrapper(func, *args, **kwargs):
            return func(*args, **kwargs)
        return decorator(__wrapper)
