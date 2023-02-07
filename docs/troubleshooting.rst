Troubleshooting
===============

If you have questions about how to use librosa, please consult the `discussion forum
<https://groups.google.com/forum/#!forum/librosa>`_.
For bug reports and other, more technical issues, consult the `github issues
<https://github.com/librosa/librosa/issues>`_.

Here are a few of the most common problems that users encounter.

no attribute 'display'
^^^^^^^^^^^^^^^^^^^^^^

If you're trying to run some example code, and encounter the following error
message::


    AttributeError: module 'librosa' has no attribute 'display'


it is because the `librosa.display` submodule needed to be imported explicitly in librosa versions
earlier than 0.10.
This is because `matplotlib` is an optional dependency for librosa, so we do not
assume that all users have it installed, or want plotting capability.

To fix the problem, add the line

.. code-block:: python

    import librosa.display

to the beginning of your program.

**NOTE**: this is no longer a problem since librosa 0.10, but it won't hurt to include
the explicit import statement.

PySoundFile failed
^^^^^^^^^^^^^^^^^^

If you're loading an audio file, and see the following message::

    UserWarning: PySoundFile failed. Trying audioread instead.


Do not worry.  This is a warning, not an error.  Odds are that your code is working
just fine.

This warning is most often triggered by loading files encoded with `mp3` format,
which are not supported by `libsndfile` prior to version 1.1.
When this situation is detected, librosa falls back to use the slower, but more
flexible `audioread`-based file loader.


`import librosa` hangs indefinitely
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are trying to execute `import librosa` in either a script or interactive
environment, and it "stalls" or fails to complete, the problem is most likely
that `numba` is trying to compile librosa functions for more efficient execution,
and is unable to write the compiled functions to disk for later usage.
(See: `numba caching documentation <https://numba.readthedocs.io/en/stable/developer/caching.html>`_ for more details on this.)
This might occur if `librosa` was installed by an administrator or super-user,
and ordinary users (i.e. you) may not have permission to write files in the same folder.

There are two ways to address this issue:

1. Install `librosa` as the same user who will be executing code, e.g., in a
   virtual environment.
2. Change the `NUMBA_CACHE_DIR` environment variable to a folder which the
   user does have write permissions to.  See `numba environment variables <https://numba.readthedocs.io/en/stable/reference/envvars.html#numba-envvars-caching>`_ for details.

Note that in librosa 0.10 and later, you may not encounter this issue when importing the library, but it may arise later when executing functions.
The solutions above are applicable in either case.
