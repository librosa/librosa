#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities
=========

Array operations
----------------
.. autosummary::
    :toctree: generated/

    frame
    pad_center
    fix_length
    fix_frames

    axis_sort
    normalize
    sparsify_rows

    buf_to_float


Matching
--------
.. autosummary::
    :toctree: generated/

    match_intervals
    match_events

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    localmax
    peak_pick


Input validation
----------------
.. autosummary::
    :toctree: generated/

    valid_audio
    valid_int
    valid_intervals


File operations
---------------

.. autosummary::
    :toctree: generated/

    example_audio_file
    find_files


sklearn integration
-------------------

.. autosummary::
    :toctree: generated/

    FeatureExtractor


Deprecated
----------
.. autosummary::
    :toctree: generated/

    buf_to_int

"""

from .utils import *  # pylint: disable=wildcard-import
from .files import *  # pylint: disable=wildcard-import
from .feature_extractor import *  # pylint: disable=wildcard-import
from . import decorators
from . import exceptions

__all__ = [_ for _ in dir() if not _.startswith('_')]

