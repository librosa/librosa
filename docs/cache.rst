Caching
^^^^^^^

This section covers the *librosa* function cache.  This allows you
to store and re-use intermediate computations across sessions.

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
The cache is implemented on top of `joblib.Memory <https://pythonhosted.org/joblib/memory.html>`_.
The default configuration can be overridden by setting the following environment variables

  - `LIBROSA_CACHE_DIR` : path (on disk) to the cache directory
  - `LIBROSA_CACHE_MMAP` : optional memory mapping mode `{None, 'r+', 'r', 'w+', 'c'}`
  - `LIBROSA_CACHE_COMPRESS` : flag to enable compression of data on disk `{0, 1}`
  - `LIBROSA_CACHE_VERBOSE` : controls how much debug info is displayed. `{int, non-negative}`
  - `LIBROSA_CACHE_LEVEL` : controls the caching level: the larger this value, the more data is cached. `{int}`

Please refer to the `joblib.Memory` `documentation
<https://pythonhosted.org/joblib/memory.html#memory-reference>`_ for a detailed explanation of these
parameters.


Cache levels
------------

Cache levels operate in a fashion similar to logging levels.
For small values of `LIBROSA_CACHE_LEVEL`, only the most important (frequently used) data are cached.
As the cache level increases, broader classes of functions are cached.
As a result, application code may run faster at the expense of larger disk usage.

The caching levels are described as follows:

    - 10: filter bases, independent of audio data (dct, mel, chroma, constant-q)
    - 20: low-level features (cqt, stft, zero-crossings, etc)
    - 30: high-level features (tempo, beats, decomposition, recurrence, etc)
    - 40: post-processing (delta, stack_memory, normalize, sync)

The default cache level is 10.


Example
-------
To demonstrate how to use the cache, we'll first call an example script twice without caching::

    $ time -p ./estimate_tuning.py ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg 
    Loading  ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg
    Separating harmonic component ... 
    Estimating tuning ... 
    +9.00 cents
    real 6.74
    user 6.03
    sys 1.09

    $ time -p ./estimate_tuning.py ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg 
    Loading  ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg
    Separating harmonic component ... 
    Estimating tuning ... 
    +9.00 cents
    real 6.68
    user 6.04
    sys 1.05


Next, we'll enable caching to `/tmp/librosa`::

    $ export LIBROSA_CACHE_DIR=/tmp/librosa

and set the cache level to 50::

    $ export LIBROSA_CACHE_LEVEL=50

And now we'll re-run the example script twice.  The first time, there will be no cached values, so the time
should be similar to running without cache.  The second time, we'll be able to reuse intermediate values, so
it should be significantly faster.::

    $ time -p ./estimate_tuning.py ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg 
    Loading  ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg
    Separating harmonic component ... 
    Estimating tuning ... 
    +9.00 cents
    real 7.60
    user 6.79
    sys 1.15

    $ time -p ./estimate_tuning.py ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg 
    Loading  ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg
    Separating harmonic component ... 
    Estimating tuning ... 
    +9.00 cents
    real 1.64
    user 1.30
    sys 0.74

Reducing the cache level to 20 yields an intermediate acceleration::

    $ export LIBROSA_CACHE_LEVEL=20

    $ time -p ./estimate_tuning.py ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg 
    Loading  ../librosa/util/example_data/Kevin_MacLeod_-_Vibe_Ace.ogg
    Separating harmonic component ... 
    Estimating tuning ... 
    +9.00 cents
    real 4.98
    user 4.17
    sys 1.22
