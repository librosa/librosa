#!/usr/bin/env python
"""Top-level module for librosa"""


# And all the librosa sub-modules
from . import core, beat, decompose, display, feature
from . import filters, onset, output, segment, util
from librosa.core import *  # pylint: disable=wildcard-import

__version__ = '0.2.1-dev'
