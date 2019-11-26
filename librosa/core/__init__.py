#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core IO and DSP
===============

Audio processing
----------------
.. autosummary::
    :toctree: generated/

    load
    stream
    to_mono
    resample
    get_duration
    get_samplerate
    autocorrelate
    lpc
    zero_crossings
    clicks
    tone
    chirp
    mu_compress
    mu_expand

Spectral representations
------------------------
.. autosummary::
    :toctree: generated/

    stft
    istft
    reassigned_spectrogram

    cqt
    icqt
    hybrid_cqt
    pseudo_cqt
    iirt
    fmt

    griffinlim
    griffinlim_cqt

    interp_harmonics
    salience

    phase_vocoder
    magphase

    get_fftlib
    set_fftlib

Magnitude scaling
-----------------
.. autosummary::
    :toctree: generated/

    amplitude_to_db
    db_to_amplitude
    power_to_db
    db_to_power

    perceptual_weighting
    A_weighting

    pcen

Time and frequency conversion
-----------------------------
.. autosummary::
    :toctree: generated/

    frames_to_samples
    frames_to_time
    samples_to_frames
    samples_to_time
    time_to_frames
    time_to_samples

    blocks_to_frames
    blocks_to_samples
    blocks_to_time

    hz_to_note
    hz_to_midi
    midi_to_hz
    midi_to_note
    note_to_hz
    note_to_midi

    hz_to_mel
    hz_to_octs
    mel_to_hz
    octs_to_hz

    fft_frequencies
    cqt_frequencies
    mel_frequencies
    tempo_frequencies
    fourier_tempo_frequencies

    samples_like
    times_like


Pitch and tuning
----------------
.. autosummary::
    :toctree: generated/

    estimate_tuning
    pitch_tuning
    piptrack

Deprecated
----------
.. autosummary::
    :toctree: generated/

    ifgram
"""

from .time_frequency import *  # pylint: disable=wildcard-import
from .audio import *  # pylint: disable=wildcard-import
from .spectrum import *  # pylint: disable=wildcard-import
from .pitch import *  # pylint: disable=wildcard-import
from .constantq import *  # pylint: disable=wildcard-import
from .harmonic import *  # pylint: disable=wildcard-import
from .fft import *  # pylint: disable=wildcard-import


__all__ = [_ for _ in dir() if not _.startswith('_')]
