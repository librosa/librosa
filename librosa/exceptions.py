#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Exception classes for librosa"""

from __future__ import annotations

import queue

__all__ = [
    "LibrosaError",
    "ParameterError",
    "ParameterValueError",
    "ParameterTypeError",
    "ParameterShapeError",
    "ParameterKeyError",
    "StreamError",
    "StreamClosedError",
    "StreamBufferError",
    "StreamTimeoutError",
    "LibrosaNotImplementedError",
]


class LibrosaError(Exception):
    """The root librosa exception class"""


class ParameterError(LibrosaError):
    """Base exception class for mal-formed inputs or parameter validation errors"""


class ParameterValueError(ParameterError, ValueError):
    """Exception class for invalid parameter values"""


class ParameterTypeError(ParameterError, TypeError):
    """Exception class for parameter type mismatches"""


class ParameterShapeError(ParameterValueError):
    """Exception class for array shape, dimension, or channel mismatches"""


class ParameterKeyError(ParameterError, KeyError):
    """Exception class for invalid lookup keys, catalog entries, or note/interval names"""


class StreamError(LibrosaError):
    """Base exception class for streaming runtime errors"""


class StreamClosedError(RuntimeError, StreamError):
    """Exception raised when an operation is attempted on a closed stream bridge"""


class StreamBufferError(BufferError, StreamError):
    """Exception raised when a stream buffer overflows under the error policy"""


class StreamTimeoutError(queue.Empty, TimeoutError, StreamError):
    """Exception raised when waiting for a stream buffer chunk times out"""


class LibrosaNotImplementedError(NotImplementedError, LibrosaError):
    """Exception class for abstract or unimplemented interface methods"""
