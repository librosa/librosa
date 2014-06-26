Tutorial
========

This section covers the fundamentals of developing with *librosa*, including
a package overview, basic and advanced usage, and integration with the *scikit-learn*
package.  We will assume basic familiarity with Python and NumPy/SciPy.


Overview
--------

The *librosa* package is structured as collection of submodules:

  - librosa

    - :ref:`beat`
        Functions for estimating tempo and detecting beat events.

    - :ref:`chord`
        This submodule contains a generic class which implements supervised training
        of Gaussian-emission Hidden Markov Models (HMM) commonly used in chord
        recognition. 

    - :ref:`core`
        Core functionality includes functions to load audio from disk, compute various
        spectrogram representations, and a variety of commonly used tools for
        music analysis.  For convenience, all functionality in this submodule is
        directly accessible from the top-level ``librosa.*`` namespace.
        
    - :ref:`decompose`
        Functions for harmonic-percussive source separation (HPSS) and generic
        spectrogram decomposition using matrix decomposition methods implemented in
        *scikit-learn*.

    - :ref:`display`
        Visualization and display routines using `matplotlib`.  

    - :ref:`effects`
        Time-domain audio processing, such as pitch shifting and time stretching.
        This submodule also provides time-domain wrappers for the `decompose`
        submodule.

    - :ref:`feature`
        Feature extraction and manipulation.  This includes low-level feature
        extraction, such as chromagrams, pseudo-constant-Q (log-frequency) transforms,
        Mel spectrogram, MFCC, and tuning estimation.  Also provided are feature
        manipulation methods, such as delta features, memory embedding, and
        event-synchronous feature alignment.

    - :ref:`filters`
        Filter-bank generation (chroma, pseudo-CQT, CQT, etc.).  These are primarily
        internal functions used by other parts of *librosa*.

    - :ref:`onset`
        Onset detection and onset strength computation.

    - :ref:`output`
        Text- and wav-file output.

    - :ref:`segment`
        Functions useful for structural segmentation, such as recurrence matrix
        construction, time-lag representation, and sequentially constrained
        clustering.

    - :ref:`util`
        Helper utilities (normalization, padding, centering, etc.)



Quickstart
----------
Before diving into the details, we'll walk through a brief example program

.. code-block:: python
    :linenos:

    # Beat tracking example

    # 1. Get the file path to the included audio example
    filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # 3. Run the default beat tracker, using a hop length of 64 frames
    #    (64 frames at sr=22.050KHz ~= 2.9ms)
    hop_length = 64
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    print 'Estimated tempo: %0.2f beats per minute' % tempo

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    print 'Saving output to beat_times.csv'
    librosa.output.times_csv('beat_times.csv', beat_times)


The first step of the program::

    filename = librosa.util.example_audio_file()

gets the path to the audio example file included with *librosa*.  After this step,
``filename`` will be a string variable containing the path to the example mp3.

The second step::

    y, sr = librosa.load(filename)
    
loads and decodes the audio as a time series ``y``, represented as a one-dimensional
NumPy floating point array.  The variable ``sr`` contains the *sampling rate* of
``y``, that is, the number of samples per second of audio.  By default, all audio is
mixed to mono and resampled to 22050 Hz at load time.  This behavior can be overridden
by supplying additional arguments to ``librosa.load()``.

The next line::

    hop_length = 64

sets the *hop length* for the subsequent analysis.  This is number of samples to
advance between subsequent audio frames.  Here, we've set the hop length to 64
samples, which at 22KHz, comes to ``64.0 / 22050 ~= 2.9ms``.  

Next, we run the beat tracker using the specified hop length::

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

The output of the beat tracker is an estimate of the tempo (in beats per minute), 
and an array of frame numbers corresponding to detected beat events.

*Frames* here correspond to short windows of the signal (``y``), each separated by 
``hop_length`` samples.  Since v0.3, *librosa* uses centered frames, so that the
*k*\ th frame is centered around sample ``k * hop_length``.

The next operation converts the frame numbers ``beat_frames`` into timings::

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

Now, ``beat_times`` will be an array of timestamps (in seconds) corresponding to
detected beat events.

Finally, we can store the detected beat timestamps as a comma-separated values (CSV)
file::

    librosa.output.times_csv('beat_times.csv', beat_times)

The contents of ``beat_times.csv`` will look something like this::

    0.067
    0.514
    0.990
    1.454
    1.910
    ...

This is primarily useful for visualization purposes (e.g., using 
`Sonic Visualiser <http://www.sonicvisualiser.org>`_) or evaluation (e.g., using
`mir_eval <https://github.com/craffel/mir_eval>`_).

Advanced usage
--------------


SciKit-learn integration
------------------------


External references
-------------------
