#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-15 10:06:03 by Brian McFee <brian.mcfee@nyu.edu>
"""Helpful tools for deprecation"""

from typing import Any, Callable, Iterable, Optional, Set, TypeVar, Union
import warnings
import functools
from decorator import decorator
import numpy as np
from numpy.typing import DTypeLike

__all__ = ["moved", "deprecated", "vectorize"]


def moved(
    *, moved_from: str, version: str, version_removed: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """This is a decorator which can be used to mark functions
    as moved/renamed.

    It will result in a warning being emitted when the function is used.
    """

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        warnings.warn(
            "{:s}\n\tThis function was moved to '{:s}.{:s}' in "
            "librosa version {:s}."
            "\n\tThis alias will be removed in librosa version "
            "{:s}.".format(
                moved_from, func.__module__, func.__name__, version, version_removed
            ),
            category=FutureWarning,
            stacklevel=3,  # Would be 2, but the decorator adds a level
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


def deprecated(
    *, version: str, version_removed: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """This is a decorator which can be used to mark functions
    as deprecated.

    It will result in a warning being emitted when the function is used."""

    def __wrapper(func, *args, **kwargs):
        """Warn the user, and then proceed."""
        warnings.warn(
            "{:s}.{:s}\n\tDeprecated as of librosa version {:s}."
            "\n\tIt will be removed in librosa version {:s}.".format(
                func.__module__, func.__name__, version, version_removed
            ),
            category=FutureWarning,
            stacklevel=3,  # Would be 2, but the decorator adds a level
        )
        return func(*args, **kwargs)

    return decorator(__wrapper)


_F = TypeVar("_F", bound=Callable[..., Any])


def vectorize(
    *,
    otypes: Optional[Union[str, Iterable[DTypeLike]]] = None,
    doc: Optional[str] = None,
    excluded: Optional[Iterable[Union[int, str]]] = None,
    cache: bool = False,
    signature: Optional[str] = None
) -> Callable[[_F], _F]:
    """This function is not quite a decorator, but is used as a wrapper
    to np.vectorize that preserves scalar behavior.
    """

    def __wrapper(function):
        vecfunc = np.vectorize(
            function,
            otypes=otypes,
            doc=doc,
            excluded=excluded,
            cache=cache,
            signature=signature,
        )

        @functools.wraps(function)
        def _vec(*args, **kwargs):
            y = vecfunc(*args, **kwargs)
            if np.isscalar(args[0]):
                return y.item()
            else:
                return y

        return _vec

    return __wrapper
