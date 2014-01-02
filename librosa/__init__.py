#!/usr/bin/env python
"""Top-level module for librosa"""


# And all the librosa sub-modules
from . import chords, core, beat, decompose, display, feature
from . import filters, onset, output, segment, util

# Exporting all core functions is okay here: suppress the import warning
from librosa.core import *  # pylint: disable=wildcard-import

__version__ = '0.3.0-dev'
