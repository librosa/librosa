#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for librosa"""

import warnings
import re
from .version import version as __version__
from .version import show_versions

# And all the librosa sub-modules
from . import cache
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

warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}'.format(re.escape(__name__)))
