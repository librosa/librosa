#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-15 10:06:03 by Brian McFee <brian.mcfee@nyu.edu>
'''Helpful decorators'''

import warnings
from decorator import decorator


def moved(moved_from, version, version_removed):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    def __wrapper(func, *args, **kwargs):
        warnings.warn_explicit(
            ("\n\tFunction '{:s}' was moved to '{:s}.{:s}' in "
             "librosa version {:s}."
             "\n\tThis alias will be removed in librosa "
             "version {:s}.").format(moved_from, func.__module__,
                                     func.__name__, version, version_removed),

            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


def deprecated(version, version_removed):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    def __wrapper(func, *args, **kwargs):
        warnings.warn_explicit(
            ("\n\tFunction '{:s}.{:s}' is deprecated as of "
             "librosa version {:s}."
             "\n\tIt will be removed in librosa "
             "version {:s}.").format(func.__module__, func.__name__,
                                     version, version_removed),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)
