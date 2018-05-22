#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence Alignment with Dynamic Time Warping."""

import numpy as np
from numba import jit
import six
from ..util.exceptions import ParameterError

__all__ = ['dtw', 'fill_off_diagonal']








