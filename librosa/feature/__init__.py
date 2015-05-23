#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature extraction
==================

Spectral features
-----------------

.. autosummary::
    :toctree: generated/

    chroma_stft
    chroma_cqt
    melspectrogram
    mfcc
    rmse
    spectral_centroid
    spectral_bandwidth
    spectral_contrast
    spectral_rolloff
    poly_features
    tonnetz
    zero_crossing_rate


Feature manipulation
--------------------

.. autosummary::
    :toctree: generated/

    delta
    stack_memory
    sync


Deprecated
----------

.. autosummary::
    :toctree: generated/

    logfsgram
"""
from .utils import *  # pylint: disable=wildcard-import
from .spectral import *  # pylint: disable=wildcard-import

__all__ = [_ for _ in dir() if not _.startswith('_')]
