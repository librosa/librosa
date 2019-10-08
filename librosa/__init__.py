#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for librosa"""

import warnings
import sys
from .version import version as __version__
from .version import show_versions

# And all the librosa sub-modules
from ._cache import cache
from . import core
from . import beat
from . import decompose
from . import effects
from . import feature
from . import filters
from . import onset
from . import output
from . import segment
from . import sequence
from . import util

# Exporting exception classes at the top level
from .util.exceptions import *  # pylint: disable=wildcard-import

# Exporting all core functions is okay here: suppress the import warning
from .core import *  # pylint: disable=wildcard-import

# Throw a deprecation warning if we're on legacy python
if sys.version_info < (3,):
    warnings.warn('You are using librosa with Python 2. '
                  'Please note that librosa 0.7 will be the last version to support '
                  'Python 2, after which it will require Python 3 or later.',
                  FutureWarning)
