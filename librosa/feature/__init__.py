#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature extraction routines

Spectral features
=================
.. autosummary::
    :toctree: generated/

    chromagram
    melspectrogram
    mfcc
    rmse
    spectral_centroid
    spectral_bandwidth
    spectral_contrast
    spectral_rolloff
    poly_features
    zero_crossing_rate

Feature manipulation
====================
.. autosummary::
    :toctree: generated/

    delta
    stack_memory
    sync

Deprecated
==========
.. autosummary::
    :toctree: generated/

    logfsgram
"""

from .utils import *  # pylint: disable=wildcard-import
from .spectral import *  # pylint: disable=wildcard-import
