Caching
=======

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

Please refer to the `joblib.Memory` `documentation
<https://pythonhosted.org/joblib/memory.html#memory-reference>`_ for a detailed explanation of these
parameters.

Example
-------
To demonstrate how to use the cache, we'll first call an example script twice without caching::

    [~/git/librosa/examples]$ time ./estimate_tuning.py ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3 
    Loading  ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3
    Separating harmonic component ... 
    Estimating tuning ... 
    +6.00 cents
    
    real    0m4.369s
    user    0m4.065s
    sys     0m0.350s
    
    [~/git/librosa/examples]$ time ./estimate_tuning.py ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3 
    Loading  ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3
    Separating harmonic component ... 
    Estimating tuning ... 
    +6.00 cents
    
    real    0m4.414s
    user    0m4.013s
    sys     0m0.440s
    

Next, we'll enable caching to `/tmp/librosa`::

    [~/git/librosa/examples]$ export LIBROSA_CACHE_DIR=/tmp/librosa

And now we'll re-run the example script twice.  The first time, there will be no cached values, so the time
should be similar to running without cache.  The second time, we'll be able to reuse intermediate values, so
it should be significantly faster.::

    [~/git/librosa/examples]$ time ./estimate_tuning.py ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3 
    Loading  ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3
    Separating harmonic component ... 
    Estimating tuning ... 
    +6.00 cents
    
    real    0m4.859s
    user    0m4.471s
    sys     0m0.429s
    
    [~/git/librosa/examples]$ time ./estimate_tuning.py ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3 
    Loading  ../librosa/example_data/Kevin_MacLeod_-_Vibe_Ace.mp3
    Separating harmonic component ... 
    Estimating tuning ... 
    +6.00 cents
    
    real    0m0.931s
    user    0m0.862s
    sys     0m0.112s

