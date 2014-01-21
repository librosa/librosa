#!/usr/bin/env python
"""Top-level module for librosa"""


# And all the librosa sub-modules
from . import core
from . import chords
from . import beat
from . import decompose
from . import display
from . import feature
from . import filters
from . import onset
from . import output
from . import segment
from . import util

# Exporting all core functions is okay here: suppress the import warning
from librosa.core import *  # pylint: disable=wildcard-import

__version__ = '0.3.0-dev'
