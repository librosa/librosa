Glossary
========

.. glossary::

    time series
        Typically an audio signal, denoted by ``y``, and represented as a
        one-dimensional *numpy.ndarray* of floating-point values.  ``y[t]`` 
        corresponds to amplitude of the waveform at sample ``t``.
    
    sampling rate
        The (positive integer) number of samples per second of a time series.  
        This is denoted by an integer variable ``sr``.

    frame
        A short slice of a :term:`time series` used for analysis purposes.  This
        usually corresponds to a single column of a spectrogram matrix.

    window
        A vector or function used to weight samples within a frame when computing
        a spectrogram.

    frame length
        The (positive integer) number of samples in an analysis window (or
        :term:`frame`).
        This is denoted by an integer variable ``n_fft``.

    hop length
        The number of samples between successive frames, e.g., the columns
        of a spectrogram.  This is denoted as a positive integer ``hop_length``.

    window length
        The length (width) of the window function (e.g., Hann window).  Note that this
        can be smaller than the :term:`frame length` used in a short-time Fourier
        transform.  Typically denoted as a positive integer variable ``win_length``.

    spectrogram
        A matrix ``S`` where the rows index frequency bins, and the columns index
        frames (time).  Spectrograms can be either real-valued or complex-valued.  By
        convention, real-valued spectrograms are denoted as *numpy.ndarray*\ s ``S``,
        while complex-valued STFT matrices are denoted as ``D``.

    onset (strength) envelope
        An onset envelope ``onset_env[t]`` measures the strength of note onsets at
        frame ``t``.  Typically stored as a one-dimensional *numpy.ndarray* of
        floating-point values ``onset_envelope``.

    chroma
        Also known as pitch class profile (PCP).  Chroma representations measure the
        amount of relative energy in each pitch class (e.g., the 12 notes in the 
        chromatic scale) at a given frame/time.
