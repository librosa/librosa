#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosa (backward-compatibility alias for librosa.exceptions)"""

from ..exceptions import (
    LibrosaError,
    LibrosaNotImplementedError,
    ParameterError,
    ParameterKeyError,
    ParameterShapeError,
    ParameterTypeError,
    StreamBufferError,
    StreamClosedError,
    StreamError,
    StreamTimeoutError,
)

__all__ = [
    "LibrosaError",
    "ParameterError",
    "ParameterTypeError",
    "ParameterShapeError",
    "ParameterKeyError",
    "StreamError",
    "StreamClosedError",
    "StreamBufferError",
    "StreamTimeoutError",
    "LibrosaNotImplementedError",
]
