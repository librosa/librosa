#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""librosa.feature contains routines for computing spectral, tonal,
rhythmic features from audio signals.

Utility functions for manipulating and inverting features are also provided.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
