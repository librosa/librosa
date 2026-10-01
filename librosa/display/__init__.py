#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualizing signals and spectral data.

Librosa display functionality is built on top of matplotlib.
The display module provides a collection of convenience functions
that make construction of common visualizations easier, and also provides a set of
custom tick formatters for musical axes (time, frequency, pitch, etc.).
"""

import lazy_loader as lazy

__getattr__, __dir__, _ = lazy.attach_stub(__name__, __file__)

__all__ = [
    "AdaptiveWaveplot",
    "ChromaFJSFormatter",
    "ChromaFormatter",
    "ChromaSvaraFormatter",
    "FJSFormatter",
    "LogHzFormatter",
    "NoteFormatter",
    "SvaraFormatter",
    "TimeFormatter",
    "TonnetzFormatter",
    "Transformf0",
    "cmap",
    "colorbar_db",
    "colorbar_phase",
    "highlight",
    "infer_cmap",
    "legend_for_axes",
    "multiplot",
    "specshow",
    "wavebars",
    "wavef0",
    "waveshow",
]
