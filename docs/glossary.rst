Glossary
========

.. glossary::

    sample
        A single value in a :term:`time series`.  In audio, this corresponds to the
        amplitude of the waveform at a given point in time.

    sampling rate
        The number of samples per second of a time series.  
        This is denoted by a positive number `sr`.

    time series
        Typically an audio signal, denoted by `y`, and represented as a
        one-dimensional *numpy.ndarray* of floating-point values.  `y[t]` 
        corresponds to amplitude of the waveform at sample `t`.
    
    frame
        A short slice of a :term:`time series` used for analysis purposes.  This
        usually corresponds to a single column of a spectrogram matrix.

    window
        A vector or function used to weight samples within a frame when computing
        a spectrogram.

    frame length
        The (positive integer) number of samples in an analysis window (or
        :term:`frame`).
        This is denoted by an integer variable `n_fft`.

    hop length
        The number of samples between successive frames, e.g., the columns
        of a spectrogram.  This is denoted as a positive integer `hop_length`.

    window length
        The length (width) of the window function (e.g., Hann window).  Note that this
        can be smaller than the :term:`frame length` used in a short-time Fourier
        transform.  Typically denoted as a positive integer variable `win_length`.

    Fourier transform
    DFT
    FFT
        A mathematical operation that transforms a time-domain signal into a
        frequency-domain representation.  The discrete Fourier transform (DFT) is
        typically computed using the fast Fourier transform (FFT) algorithm.

    STFT
        The short-time Fourier transform (STFT) is a sequence of Fourier transforms
        of windowed frames of a time series.  The STFT is typically represented as a
        complex-valued matrix `D`, where the rows index frequency bins, and the
        columns index frames (time).

    spectrogram
        A matrix `S` where the rows index frequency bins, and the columns index
        frames (time).  Spectrograms can be either real-valued or complex-valued.  By
        convention, real-valued spectrograms are denoted as *numpy.ndarray*\ s `S`,
        while complex-valued STFT matrices are denoted as `D`.

    magnitude
        A complex number `z` can be expressed as a combination of real and imaginary parts,
        :math:`z = a + jb` (where `j` represents the imaginary unit).
        The magnitude of `z` is the distance from the origin as computed
        by the Pythagorean theorem: :math:`|z| = \sqrt{a^2 + b^2}`. 

    phase
        The phase of a complex number :math:`z = a + jb` is the angle between the positive
        real axis and the line connecting the origin to the point `(a, b)` in the complex plane.
        The phase is computed as :math:`\phi = \arctan(b / a)`.

    decibel
    dB
        A logarithmic unit used to express the ratio of two values, typically power or
        intensity.  In audio, decibels are often used to represent the amplitude of a
        signal relative to a reference level.

    dBFS
        Decibels relative to full scale (dBFS) is a unit of measurement for amplitude
        levels in digital audio systems.  0 dBFS represents the maximum possible digital
        level, while negative values indicate lower amplitudes.

    frequency bin
        In a discrete Fourier transform (DFT) representation, the signal is represented as a
        combination of sinusoidal components at a finite set of frequencies.
        Each frequency index `k` corresponds to a frequency bin.

    fundamental frequency
    f0
        The fundamental frequency (f0) is the lowest frequency of a periodic waveform,
        and is often perceived as the pitch of the sound.  Note that not all waveforms
        are periodic.
    
    pitch
        The *perceived* frequency of a sound, which is often associated with the fundamental
        frequency (f0) of a periodic waveform.  In music, pitch is typically represented
        as a note name (e.g., A4, C#5) or as a MIDI note number.

    pitch class
        A pitch class is a set of all pitches that are a whole number of octaves apart.
        For example, the pitch classes for the note C include C, C1, C2, C3, etc.  In
        Western music, there are 12 pitch classes corresponding to the 12 notes in the
        chromatic scale.

    chroma
    pitch class profile
        Chroma representations measure the amount of relative energy in each pitch class
        (e.g., the 12 notes in the chromatic scale) at a given frame/time.

    onset
        The beginning time of a musical note or sound event.

    onset (strength) envelope
    novelty function
    novelty curve
        An onset envelope `onset_env[t]` measures the strength of note onsets at
        frame `t`.  Typically stored as a one-dimensional *numpy.ndarray* of
        floating-point values `onset_envelope`.

    tempo
        The speed of a piece of music, typically measured in beats per minute (BPM).

    beat
        A beat is a basic unit of time in music, often corresponding to a regular pulse
        that listeners can tap their foot to.

    cqt
        The constant-Q transform (CQT) is a time-frequency representation of a signal
        that uses logarithmically spaced frequency bins with window lengths scaled inversely
        with frequency.

    mel scale
        The mel scale is a perceptual scale of pitches that approximates the human ear's
        response to different frequencies.  The mel scale is approximately linear below
        1 kHz and logarithmic above 1 kHz.

    mel spectrogram
        A mel spectrogram is a time-frequency representation of a signal where the frequency
        axis is transformed to the mel scale.

    mel frequency cepstral coefficient
    MFCC
        Mel-frequency cepstral coefficients (MFCCs) are a representation of the short-term
        power spectrum of a sound, based on a linear cosine transform of a log power
        mel spectrogram.  MFCCs are commonly used as features in speech and audio processing.

    tempogram
        A representation of the rhythmic structure of a piece of music over time.
        It measures the strength of periodicities in the onset envelope, using either
        autocorrelation or Fourier analysis.

    recurrence matrix
    self-similarity matrix
        A square matrix that measures the similarity between features at all pairs `(i, j)` 
        of time or frame indices.

    beat synchronous features
        Beat-synchronous features are computed by aligning and aggregating features over
        detected inter-beat intervals.

    multi-channel audio
        Audio signals that contain more than one channel, such as stereo (2 channels) or
        surround sound (5.1 channels).  Multi-channel audio is typically represented as a
        two-dimensional *numpy.ndarray* of shape `(n_channels, n_samples)`.

    MIDI
        Musical Instrument Digital Interface (MIDI) is a technical standard that describes
        a protocol, digital interface, and connectors for communicating musical performance
        data between electronic musical instruments, computers, and other related devices.

    MIDI number
        A MIDI number is an integer value that represents a specific musical note in the
        MIDI standard.  For example, the MIDI number for middle C (C4) is 60.
        This is sometimes extended to fractional values to allow representation of all
        frequencies.

    Scientific Pitch Notation
    SPN
        Scientific Pitch Notation (SPN) is a standardized system for naming musical notes
        based on their pitch class and octave number.

    accidental
        An accidental is a musical notation symbol that indicates the alteration of a pitch.
        Common accidentals include sharps (#), flats (b), and naturals (♮).

    cent 
        One one-hundredth (1/100) of a semitone in the 12-tone equal temperament scale.

    A440
        A tuning standard that defines the pitch of the musical note A4 (the A above middle C)
        as 440 Hz.

    12TET
        A musical tuning system that divides the octave into 12 equal parts (semitones),
        commonly used in Western scales.

    Just intonation
        A musical tuning system that uses simple whole-number ratios to determine the
        frequencies of notes, resulting in consonant intervals.

    svara
        A pitch in Indian classical music, which can be thought of as a note in the context
        of a raga.

    Sa
        The tonic svara in Indian classical music.

    thaat
        A system of classifying ragas in Hindustani classical music based on their scale
        and note patterns.

    melakarta
    mela
        A system of classifying ragas in Carnatic classical music based on their scale
        and note patterns.
