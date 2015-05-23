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
    to_mono
    resample
    get_duration
    autocorrelate
    zero_crossings


Spectral representations
------------------------
.. autosummary::
    :toctree: generated/

    stft
    istft
    ifgram
    cqt
    hybrid_cqt
    pseudo_cqt

    phase_vocoder

    magphase
    logamplitude
    perceptual_weighting
    A_weighting

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

    ifptrack

"""

from .time_frequency import *  # pylint: disable=wildcard-import
from .audio import *  # pylint: disable=wildcard-import
from .spectrum import *  # pylint: disable=wildcard-import
from .pitch import *  # pylint: disable=wildcard-import
from .constantq import *  # pylint: disable=wildcard-import

__all__ = [_ for _ in dir() if not _.startswith('_')]

