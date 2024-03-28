#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lazy_loader as lazy
from .version import version as __version__

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
