#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualizing signals and spectral data.

Librosa display functionality is built on top of matplotlib.
The display module provides a collection of convenience functions
that make construction of common visualizations easier, and also provides a set of
custom tick formatters for musical axes (time, frequency, pitch, etc.).
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
