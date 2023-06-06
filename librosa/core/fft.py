#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast Fourier Transform (FFT) library container"""
from types import ModuleType
from typing import Optional


__all__ = ["get_fftlib", "set_fftlib"]

# Object to hold FFT interfaces
__FFTLIB: Optional[ModuleType] = None


def set_fftlib(lib: Optional[ModuleType] = None) -> None:
    """Set the FFT library used by librosa.

    Parameters
    ----------
    lib : None or module
        Must implement an interface compatible with `numpy.fft`.
        If ``None``, reverts to `numpy.fft`.

    Examples
    --------
    Use `pyfftw`:

    >>> import pyfftw
    >>> librosa.set_fftlib(pyfftw.interfaces.numpy_fft)

    Reset to default `numpy` implementation

    >>> librosa.set_fftlib()
    """
    global __FFTLIB
    if lib is None:
        from numpy import fft

        lib = fft

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


# Set the FFT library to numpy's, by default
set_fftlib(None)
