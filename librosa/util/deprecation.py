#!/usr/bin/env python
"""Deprecation utilities"""

import inspect
import warnings
from typing import Any


class Deprecated:
    """A placeholder class to catch usage of deprecated variable names"""

    def __repr__(self) -> str:
        """Pretty-print display for deprecated objects"""
        return "<DEPRECATED parameter>"


def rename_kw(
    *, old_name: str, old_value: Any, new_name: str, new_value: Any, version_deprecated: str, version_removed: str
) -> Any:
    """Handle renamed arguments.

    Parameters
    ----------
    old_name : str
        The name of the old argument
    old_value : Any
        The value of the old argument
    new_name : str
        The name of the new argument
    new_value : Any
        The value of the new argument
    version_deprecated : str
        The version at which the old name became deprecated
    version_removed : str
        The version at which the old name will be removed

    Returns
    -------
    value : Any
        - ``new_value`` if ``old_value`` of type `Deprecated`
        - ``old_value`` otherwise

    Warnings
    --------
    if ``old_value`` is not of type `Deprecated`
    """
    if isinstance(old_value, Deprecated):
        return new_value
    else:
        stack = inspect.stack()
        dep_func = stack[1]

        warnings.warn(
            f"{dep_func[3]:s}() keyword argument '{old_name:s}' has been "
            f"renamed to '{new_name:s}' in version {version_deprecated}."
            "\n\tThis alias will be removed in version "
            f"{version_removed}.",
            category=FutureWarning,
            stacklevel=3,
        )

        return old_value


def _warn_spmatrix(x: Any, *, stacklevel: int = 3) -> None:
    """Emit a FutureWarning if ``x`` is a legacy ``scipy.sparse.spmatrix``."""
    import scipy.sparse

    if isinstance(x, scipy.sparse.spmatrix):
        warnings.warn(
            "Support for scipy.sparse.spmatrix inputs is deprecated as of librosa version 1.1.0 "
            "and will be removed in version 2.0.0. Convert to a scipy.sparse array "
            "(e.g., scipy.sparse.csc_array or scipy.sparse.csr_array) instead.",
            category=FutureWarning,
            stacklevel=stacklevel,
        )


def _warn_interp_kind(kind: Any, *, stacklevel: int = 3) -> None:
    """Emit a FutureWarning if ``kind`` is a deprecated interpolation mode."""
    if kind == "slinear":
        warnings.warn(
            "Interpolation mode kind='slinear' is deprecated as of librosa version 1.1.0 "
            "and will be removed in version 2.0.0. Use kind='linear' instead.",
            category=FutureWarning,
            stacklevel=stacklevel,
        )
