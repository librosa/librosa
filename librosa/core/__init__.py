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
    clicks
    tone
    chirp

Spectral representations
------------------------
.. autosummary::
    :toctree: generated/

    stft
    istft
    ifgram
    cqt
    icqt
    hybrid_cqt
    pseudo_cqt
    iirt
    fmt

    interp_harmonics
    salience

    phase_vocoder
    magphase

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

    samples_like
    times_like


Pitch and tuning
----------------
.. autosummary::
    :toctree: generated/

    estimate_tuning
    pitch_tuning
    piptrack

Deprecated (moved)
------------------
.. autosummary::
    :toctree: generated/

    dtw
    fill_off_diagonal
"""

from .time_frequency import *  # pylint: disable=wildcard-import
from .audio import *  # pylint: disable=wildcard-import
from .spectrum import *  # pylint: disable=wildcard-import
from .pitch import *  # pylint: disable=wildcard-import
from .constantq import *  # pylint: disable=wildcard-import
from .harmonic import *  # pylint: disable=wildcard-import

from ..util.decorators import moved as _moved
from ..util import fill_off_diagonal as _fod
from ..sequence import dtw as _dtw

dtw = _moved('librosa.sequence.dtw', '0.6.1', '0.7')(_dtw)
fill_off_diagonal = _moved('librosa.util.fill_off_diagonal', '0.6.1', '0.7')(_fod)

__all__ = [_ for _ in dir() if not _.startswith('_')]
