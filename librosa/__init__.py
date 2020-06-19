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
from . import segment
from . import sequence
from . import util

# Exporting exception classes at the top level
from .util.exceptions import *  # pylint: disable=wildcard-import

# Exporting data loader at the top level
from .util.files import example, ex


# Exporting all core functions is okay here: suppress the import warning
from .core import *  # pylint: disable=wildcard-import
