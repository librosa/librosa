#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast Fourier Transform (FFT) library container"""
import scipy.fft

from types import ModuleType
from typing import Optional
from ..util.decorators import deprecated


__all__ = ["get_fftlib", "set_fftlib"]

# Object to hold FFT interfaces
__FFTLIB: Optional[ModuleType] = scipy.fft


@deprecated(version="0.11.0", version_removed="1.0")
def set_fftlib(lib: Optional[ModuleType] = None) -> None:
    """Set the FFT library used by librosa.

    .. warning:: This functionality is deprecated in librosa 0.11 and will be
        removed in 1.0.  To achieve the same effect, use either the
        `scipy.fft.set_backend` context manager or
        `scipy.fft.set_global_backend` function.

    Parameters
    ----------
    lib : None or module
        Must implement an interface compatible with `scipy.fft`.
        If ``None``, reverts to `scipy.fft`.

    Examples
    --------
    Use `pyfftw`:

    >>> import pyfftw
    >>> librosa.set_fftlib(pyfftw.interfaces.numpy_fft)

    Reset to default `scipy` implementation

    >>> librosa.set_fftlib()
    """
    global __FFTLIB
    if lib is None:
        lib = scipy.fft

    __FFTLIB = lib


def get_fftlib() -> ModuleType:
    """Get the FFT library currently used by librosa

    Returns
    -------
    fft : module
        The FFT library currently used by librosa.
        Must API-compatible with `numpy.fft`.
    """
    if __FFTLIB is None:
        # This path should never occur because importing
        # this module will call set_fftlib
        assert False  # pragma: no cover

    return __FFTLIB
