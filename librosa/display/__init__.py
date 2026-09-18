#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualizing signals and spectral data.

Librosa display functionality is built on top of matplotlib.
The display module provides a collection of convenience functions
that make construction of common visualizations easier, and also provides a set of
custom tick formatters for musical axes (time, frequency, pitch, etc.).
"""

# Standard library imports for future compatibility
from __future__ import annotations

# Module imports for formatting and utility functions
from .formatting import (
    AdaptiveWaveplot,
    ChromaFJSFormatter,
    ChromaFormatter,
    ChromaSvaraFormatter,
    FJSFormatter,
    LogHzFormatter,
    NoteFormatter,
    TimeFormatter,
    TonnetzFormatter,
    Transformf0,
    cmap,
    infer_cmap,
    colorbar_db,
    colorbar_phase,
    _same_axes,
    _WAVESHOW_ADAPTORS
)

# Module imports for image and signal display functions
from .image import specshow, _parse_vscale
from .signal import waveshow, wavebars, wavef0
from .multi import (
    multiplot,
    highlight,
    legend_for_axes,
    _squeeze_shape,
    _resolve_multiplot,
    _mp_get_layout,
    _mp_setup_axes,
    _mp_setup_labels,
    _mp_setup_prop_group,
    _mp_setup_properties,
)

# Public API for the display module
__all__ = [
    "specshow",
    "waveshow",
    "wavebars",
    "wavef0",
    "multiplot",
    "highlight",
    "infer_cmap",
    "colorbar_db",
    "colorbar_phase",
    "legend_for_axes",
    "TimeFormatter",
    "NoteFormatter",
    "FJSFormatter",
    "LogHzFormatter",
    "ChromaFormatter",
    "ChromaSvaraFormatter",
    "ChromaFJSFormatter",
    "TonnetzFormatter",
    "AdaptiveWaveplot",
    "Transformf0",
]
