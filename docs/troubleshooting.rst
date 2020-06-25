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


it is because the `librosa.display` submodule needs to be imported explicitly.
This is because `matplotlib` is an optional dependency for librosa, so we do not
assume that all users have it installed, or want plotting capability.

To fix the problem, add the line

.. code-block:: python

    import librosa.display

to the beginning of your program.

PySoundFile failed
^^^^^^^^^^^^^^^^^^

If you're loading an audio file, and see the following message::

    UserWarning: PySoundFile failed. Trying audioread instead.


Do not worry.  This is a warning, not an error.  Odds are that your code is working
just fine.

This warning is most often triggered by loading files encoded with `mp3` format,
which are not currently supported by `libsndfile`.
When this situation is detected, librosa falls back to use the slower, but more
flexible `audioread`-based file loader.
