#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core IO and DSP
===============
Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    samples_like
    times_like

    get_fftlib
    set_fftlib
"""

import lazy_loader as lazy
from .version import version as __version__

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
