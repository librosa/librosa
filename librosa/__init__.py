#!/usr/bin/env python
"""Top-level module for librosa."""


# And all the librosa sub-modules
from . import core, beat, decompose, feature, segment, output, onset
from librosa.core import *

__version__ = '0.2.0dev'
