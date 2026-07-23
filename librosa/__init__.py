#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Librosa is a python package for music and audio analysis. It provides the building
blocks necessary to create audio retrieval systems.

Core functionality (loading audio, spectral representations, etc.) is accessible
from the top level of the package, while more specialized functionality (beat tracking,
visualization, etc.) is organized into submodules.
"""

import lazy_loader as lazy

from .version import version as __version__

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
