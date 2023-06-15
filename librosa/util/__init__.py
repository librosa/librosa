#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Skip pydocstyle checks that erroneously trigger on "example"
# noqa: D405,D214,D407
"""
Utilities
=========

Array operations
----------------
.. autosummary::
    :toctree: generated/

    frame
    pad_center
    expand_to
    fix_length
    fix_frames
    index_to_slice
    softmask
    stack
    sync

    axis_sort
    normalize
    shear
    sparsify_rows

    buf_to_float
    tiny

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
    localmin
    peak_pick
    nnls
    cyclic_gradient
    dtype_c2r
    dtype_r2c
    count_unique
    is_unique
    abs2
    phasor


Input validation
----------------
.. autosummary::
    :toctree: generated/

    valid_audio
    valid_int
    valid_intervals
    is_positive_int


File operations
---------------
.. autosummary::
    :toctree: generated/

    example
    example_info
    list_examples
    find_files
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
