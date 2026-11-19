.. _exceptions:

Exceptions
==========
.. automodule:: librosa.exceptions

Librosa provides a structured exception hierarchy rooted in :class:`LibrosaError`.
Exceptions raised by Librosa inherit from both :class:`LibrosaError` and relevant Python standard library exception classes where appropriate, allowing callers to catch errors using either Librosa-specific classes or familiar built-in exception types.

Exception Hierarchy
-------------------

.. mermaid::

    flowchart TD
        LibrosaError["LibrosaError"]
        ParameterError["ParameterError"]
        ParameterValueError["ParameterValueError<br>(ValueError)"]
        ParameterShapeError["ParameterShapeError"]
        ParameterTypeError["ParameterTypeError<br>(TypeError)"]
        ParameterKeyError["ParameterKeyError<br>(KeyError)"]
        StreamError["StreamError"]
        StreamClosedError["StreamClosedError<br>(RuntimeError)"]
        StreamBufferError["StreamBufferError<br>(BufferError)"]
        StreamTimeoutError["StreamTimeoutError<br>(TimeoutError, queue.Empty)"]
        LibrosaNotImplementedError["LibrosaNotImplementedError<br>(NotImplementedError)"]

        LibrosaError --> ParameterError
        LibrosaError --> StreamError
        LibrosaError --> LibrosaNotImplementedError

        ParameterError --> ParameterValueError
        ParameterError --> ParameterTypeError
        ParameterError --> ParameterKeyError

        ParameterValueError --> ParameterShapeError

        StreamError --> StreamClosedError
        StreamError --> StreamBufferError
        StreamError --> StreamTimeoutError

Usage Guidance
--------------

- **Catching specific validation failures**:
  Use specific parameter exception classes to respond to targeted errors:

  .. code-block:: python

      try:
          D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
      except librosa.ParameterShapeError:
          # Handle array shape, dimensionality, or channel mismatch
          ...
      except librosa.ParameterValueError:
          # Handle out-of-range numerical parameters or invalid options
          ...
      except librosa.ParameterTypeError:
          # Handle incorrect argument types
          ...

- **Interoperability with standard library exceptions**:
  Because specific parameter exceptions inherit from standard built-ins (:exc:`ValueError`, :exc:`TypeError`, :exc:`KeyError`), existing exception handlers in user code will catch them naturally:

  .. code-block:: python

      try:
          chroma = librosa.feature.chroma_stft(y=y, sr=sr)
      except ValueError:
          # Catches both ParameterValueError and ParameterShapeError
          ...

- **Catching all parameter validation errors**:
  Catch :class:`ParameterError` to intercept any parameter or input validation error regardless of subtype. Existing code using ``except librosa.ParameterError:`` continues to work unchanged.

- **Catching any Librosa error**:
  Catch :class:`LibrosaError` as a top-level catch-all for any exception originating within Librosa.

- **Upstream and backend errors**:
  Exceptions originating in underlying backend libraries (such as :mod:`soundfile`, :mod:`pooch`, or network errors from :mod:`requests`) are allowed to propagate directly without interception.

.. currentmodule:: librosa.exceptions

.. autosummary::
    :toctree: generated
    :nosignatures:

    LibrosaError
    ParameterError
    ParameterValueError
    ParameterTypeError
    ParameterShapeError
    ParameterKeyError
    StreamError
    StreamClosedError
    StreamBufferError
    StreamTimeoutError
    LibrosaNotImplementedError
