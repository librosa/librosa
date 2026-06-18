#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-15 10:06:03 by Brian McFee <brian.mcfee@nyu.edu>
"""Helpful tools for deprecation"""
from __future__ import annotations

import functools
import inspect
import warnings
from typing import TYPE_CHECKING

import numpy as np
from decorator import decorator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any, Protocol, type_check_only

    from numpy.typing import DTypeLike

    @type_check_only
    class _Decorator(Protocol):
        def __call__[**P, R](self, fn: Callable[P, R], /) -> Callable[P, R]: ...

__all__ = ["moved", "deprecated", "vectorize"]


def moved(*, moved_from: str, version: str, version_removed: str) -> _Decorator:
    """Mark functions as moved/renamed.

    Using the decorated (old) function will result in a warning.

    Parameters
    ----------
    moved_from : str
        The old location of the function
    version : str
        The version in which the function was moved
    version_removed : str
        The version in which the old alias will be removed

    Returns
    -------
    decorator : Callable
        A decorator that can be applied to the old function name
    """

    def __wrapper[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
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


def deprecated(*, version: str, version_removed: str) -> _Decorator:
    """Mark a function as deprecated.

    Using the decorated (old) function will result in a warning.

    Parameters
    ----------
    version : str
        The version in which the function was deprecated
    version_removed : str
        The version in which the function will be removed

    Returns
    -------
    decorator : Callable
        A decorator that can be applied to the deprecated function
    """

    def __wrapper[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
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


def vectorize(
    *,
    otypes: str | Iterable[DTypeLike] | None = None,
    doc: str | None = None,
    excluded: Iterable[int | str] | None = None,
    cache: bool = False,
    signature: str | None = None,
) -> _Decorator:
    """Wrap a function for use with np.vectorize.

    This function is not quite a decorator, but is used as a wrapper
    to np.vectorize that preserves scalar behavior.

    Parameters
    ----------
    otypes : str or list of dtype, optional
        The output data type(s).  This is required if the function returns
        a scalar.  If the function returns an array, this is optional.
    doc : str, optional
        The docstring for the vectorized function.  If None, the original
        function's docstring will be used.
    excluded : list of int or str, optional
        List of argument indices or names to exclude from vectorization.
    cache : bool, optional
        If True, cache the results of the function calls.  This can speed up
        repeated calls with the same arguments, but may consume more memory.
    signature : str, optional
        The signature of the function for generalized ufunc behavior.

    Returns
    -------
    decorator : Callable
        A decorator that can be applied to the function to vectorize it.

    See Also
    --------
    np.vectorize : The underlying vectorization function from NumPy.
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


def future_default(*, param_name: str, old_default: Any, new_default: Any, version: str) -> _Decorator:
    """Mark a function parameter as having a future change to a default parameter value.

    Parameters
    ----------
    param_name : str
        The name of the parameter whose default value is changing.
    old_default : Any
        The current default value of the parameter.
    new_default : Any
        The future default value of the parameter.
    version : str
        The version in which the default value will change.

    Returns
    -------
    decorator : Callable
        A decorator that can be applied to the function to warn about the future change in default value
    """
    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def __wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # functools.wraps does not inject defaults into args/kwargs
            bound = sig.bind(*args, **kwargs)

            if param_name not in bound.arguments:
                warnings.warn(
                    f"The default value of '{param_name}' will change from "
                    f"{old_default!r} to {new_default!r} in librosa version {version}. "
                    f"To suppress this warning, explicitly pass '{param_name}={old_default!r}'.",
                    FutureWarning,
                    stacklevel=2
                )

            return func(*args, **kwargs)
        return __wrapper
    return decorator
