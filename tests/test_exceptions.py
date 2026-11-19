#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Unit tests for librosa exception hierarchy"""

import pickle
import queue

import numba
import pytest

import librosa
import librosa.exceptions
import librosa.util.exceptions
from librosa.exceptions import (
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

ALL_EXCEPTIONS = [
    LibrosaError,
    ParameterError,
    ParameterTypeError,
    ParameterShapeError,
    ParameterKeyError,
    StreamError,
    StreamClosedError,
    StreamBufferError,
    StreamTimeoutError,
    LibrosaNotImplementedError,
]


def test_exception_inheritance():
    """Test inheritance relationships across the exception hierarchy."""
    # Root
    assert issubclass(LibrosaError, Exception)

    # Parameter error hierarchy
    assert issubclass(ParameterError, LibrosaError)
    assert issubclass(ParameterError, ValueError)

    assert issubclass(ParameterTypeError, ParameterError)
    assert issubclass(ParameterTypeError, TypeError)
    assert issubclass(ParameterTypeError, ValueError)
    assert issubclass(ParameterTypeError, LibrosaError)

    assert issubclass(ParameterShapeError, ParameterError)
    assert issubclass(ParameterShapeError, ValueError)
    assert issubclass(ParameterShapeError, LibrosaError)

    assert issubclass(ParameterKeyError, ParameterError)
    assert issubclass(ParameterKeyError, KeyError)
    assert issubclass(ParameterKeyError, LookupError)
    assert issubclass(ParameterKeyError, ValueError)
    assert issubclass(ParameterKeyError, LibrosaError)

    # Streaming error hierarchy
    assert issubclass(StreamError, LibrosaError)

    assert issubclass(StreamClosedError, StreamError)
    assert issubclass(StreamClosedError, RuntimeError)
    assert issubclass(StreamClosedError, LibrosaError)

    assert issubclass(StreamBufferError, StreamError)
    assert issubclass(StreamBufferError, BufferError)
    assert issubclass(StreamBufferError, LibrosaError)

    assert issubclass(StreamTimeoutError, StreamError)
    assert issubclass(StreamTimeoutError, queue.Empty)
    assert issubclass(StreamTimeoutError, TimeoutError)
    assert issubclass(StreamTimeoutError, LibrosaError)

    # Not implemented
    assert issubclass(LibrosaNotImplementedError, LibrosaError)
    assert issubclass(LibrosaNotImplementedError, NotImplementedError)


def test_exception_mro():
    """Verify that specific builtin exceptions appear before general ones in MRO."""
    type_mro = ParameterTypeError.__mro__
    assert type_mro.index(TypeError) < type_mro.index(ValueError)

    key_mro = ParameterKeyError.__mro__
    assert key_mro.index(KeyError) < key_mro.index(ValueError)

    closed_mro = StreamClosedError.__mro__
    assert closed_mro.index(RuntimeError) < closed_mro.index(LibrosaError)

    buffer_mro = StreamBufferError.__mro__
    assert buffer_mro.index(BufferError) < buffer_mro.index(LibrosaError)

    timeout_mro = StreamTimeoutError.__mro__
    assert timeout_mro.index(queue.Empty) < timeout_mro.index(LibrosaError)
    assert timeout_mro.index(TimeoutError) < timeout_mro.index(LibrosaError)


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_exception_instances(exc_cls):
    """Test instantiating each exception with a message."""
    msg = f"Test message for {exc_cls.__name__}"
    instance = exc_cls(msg)

    assert isinstance(instance, LibrosaError)
    assert str(instance) == msg or msg in str(instance)


def test_parameter_error_backward_compatibility():
    """Verify that catching ParameterError catches all parameter error subclasses."""
    with pytest.raises(ParameterError):
        raise ParameterTypeError("type error")

    with pytest.raises(ParameterError):
        raise ParameterShapeError("shape error")

    with pytest.raises(ParameterError):
        raise ParameterKeyError("key error")


def test_builtin_catch_compatibility():
    """Verify standard builtin exception catch clauses work as expected."""
    with pytest.raises(ValueError):
        raise ParameterError("value error")

    with pytest.raises(TypeError):
        raise ParameterTypeError("type error")

    with pytest.raises(KeyError):
        raise ParameterKeyError("key error")

    with pytest.raises(RuntimeError):
        raise StreamClosedError("closed stream")

    with pytest.raises(BufferError):
        raise StreamBufferError("buffer overflow")

    with pytest.raises(queue.Empty):
        raise StreamTimeoutError("timed out")

    with pytest.raises(TimeoutError):
        raise StreamTimeoutError("timed out")

    with pytest.raises(NotImplementedError):
        raise LibrosaNotImplementedError("not implemented")


@pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
def test_exception_pickle(exc_cls):
    """Verify all exceptions can be pickled and unpickled."""
    msg = f"Pickle test for {exc_cls.__name__}"
    original = exc_cls(msg)
    serialized = pickle.dumps(original)
    restored = pickle.loads(serialized)  # noqa: S301

    assert isinstance(restored, exc_cls)
    assert isinstance(restored, LibrosaError)
    assert str(restored) == str(original)


def test_numba_exception_compatibility():
    """Test that ParameterError can be raised from numba-jitted code."""
    @numba.jit(nopython=True)
    def jitted_validator(x):
        if x < 0:
            raise ParameterError
        return x * 2

    assert jitted_validator(5) == 10
    with pytest.raises(ParameterError):
        jitted_validator(-1)


def test_export_parity():
    """Verify export parity across librosa, librosa.exceptions, and librosa.util.exceptions."""
    for exc_cls in ALL_EXCEPTIONS:
        name = exc_cls.__name__
        assert getattr(librosa, name) is exc_cls
        assert getattr(librosa.exceptions, name) is exc_cls
        assert getattr(librosa.util.exceptions, name) is exc_cls
