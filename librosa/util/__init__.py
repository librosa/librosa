#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
librosa.util contains a variety of utility functions used throughout the
package.

Utility functions are broadly categorized into the following groups:

    - Numerical operations (value-based processing, optimization, etc.)
    - Array shape manipulation (slicing, framing, padding, etc.).
    - Matching (events or time intervals)
    - Files and data (packaged examples, metadata, etc.)
    - Data validation (type and range checking)
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
