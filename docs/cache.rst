Caching
^^^^^^^

This section covers the *librosa* function cache.  This allows you
to store and reuse intermediate computations across sessions.

Enabling the cache
------------------
By default, caching is disabled.  To enable caching, the environment 
variable `LIBROSA_CACHE_DIR` must be set prior to loading *librosa*.
This can be done on the command line prior to instantiating a python interpreter::

    $ export LIBROSA_CACHE_DIR=/tmp/librosa_cache
    $ ipython

or from within python, prior to importing *librosa*::

    >>> import os
    >>> os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
    >>> import librosa

.. warning::
    The cache does not implement any eviction policy.  As such, 
    it can grow without bound on disk if not purged.
    To purge the cache directly, call::

        >>> librosa.cache.clear()



Cache configuration
-------------------
The cache is implemented on top of `joblib.Memory`.
The default configuration can be overridden by setting the following environment variables

  - `LIBROSA_CACHE_DIR` : path (on disk) to the cache directory
  - `LIBROSA_CACHE_MMAP` : optional memory mapping mode `{None, 'r+', 'r', 'w+', 'c'}`
  - `LIBROSA_CACHE_COMPRESS` : flag to enable compression of data on disk `{0, 1}`
  - `LIBROSA_CACHE_VERBOSE` : controls how much debug info is displayed. `{int, non-negative}`
  - `LIBROSA_CACHE_LEVEL` : controls the caching level: the larger this value, the more data is cached. `{int}`

Please refer to the `joblib.Memory` documentation for a detailed explanation of these parameters.

As of 0.7, librosa's cache wraps (rather than extends) the `joblib.Memory` object.
The memory object can be directly accessed by `librosa.cache.memory`.


Cache levels
------------

Cache levels operate in a fashion similar to logging levels.
For small values of `LIBROSA_CACHE_LEVEL`, only the most important (frequently used) data are cached.
As the cache level increases, broader classes of functions are cached.
As a result, application code may run faster at the expense of larger disk usage.

The caching levels are described as follows:

    - 10: filter bases, independent of audio data (mel, chroma, constant-q)
    - 20: low-level features (cqt, stft, zero-crossings, etc)
    - 30: high-level features (tempo, beats, decomposition, recurrence, etc)
    - 40: post-processing (delta, stack_memory, normalize, sync)

The default cache level is 10.
