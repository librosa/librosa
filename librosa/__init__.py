#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Top-level module for librosa"""

import warnings
import re
warnings.filterwarnings('always',
                        category=DeprecationWarning,
                        module='^{0}'.format(re.escape(__name__)))

from .version import version as __version__

# And all the librosa sub-modules
from . import cache
from . import core
from . import beat
from . import decompose
from . import display
from . import effects
from . import feature
from . import filters
from . import onset
from . import output
from . import segment
from . import util

# Exporting exception classes at the top level
from .util.exceptions import * # pylint: disable=wildcart-import

# Exporting all core functions is okay here: suppress the import warning
from librosa.core import *  # pylint: disable=wildcard-import



