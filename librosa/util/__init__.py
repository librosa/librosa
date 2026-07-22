#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Skip pydocstyle checks that erroneously trigger on "example"
"""Utilities"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
