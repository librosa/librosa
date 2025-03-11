#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal generation
=================
.. autosummary::
    :toctree: generated/

    clicks
    tone
    chirp
    load
    stream
    to_mono
    resample
    get_duration
    get_samplerate
    autocorrelate
    lpc
    zero_crossings
    mu_compress
    mu_expand
    fft_frequencies
    cqt_frequencies
    mel_frequencies
    tempo_frequencies
    fourier_tempo_frequencies


Spectral representations
========================
.. autosummary::
    :toctree: generated/

    stft
    istft
    reassigned_spectrogram

    cqt
    icqt
    hybrid_cqt
    pseudo_cqt

    vqt

    iirt

    fmt

    magphase


Magnitude scaling
=================
.. autosummary::
    :toctree: generated/

    amplitude_to_db
    db_to_amplitude
    power_to_db
    db_to_power

    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    pcen

    
Unit conversion
===============
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
    hz_to_svara_h
    hz_to_svara_c
    hz_to_fjs
    midi_to_hz
    midi_to_note
    midi_to_svara_h
    midi_to_svara_c
    note_to_hz
    note_to_midi
    note_to_svara_h
    note_to_svara_c

    hz_to_mel
    hz_to_octs
    mel_to_hz
    octs_to_hz

    A4_to_tuning
    tuning_to_A4


Music notation
==============
.. autosummary::
    :toctree: generated/

    key_to_notes
    key_to_degrees

    mela_to_svara
    mela_to_degrees

    thaat_to_degrees

    list_mela
    list_thaat

    fifths_to_note
    interval_to_fjs
    interval_frequencies
    pythagorean_intervals
    plimit_intervals


Pitch and tuning
================
.. autosummary::
    :toctree: generated/

    pyin
    yin

    estimate_tuning
    pitch_tuning
    piptrack


Harmonics
=========
.. autosummary::
    :toctree: generated/

    interp_harmonics
    salience
    f0_harmonics

    phase_vocoder

    
Phase recovery
==============
.. autosummary::
    :toctree: generated/

    griffinlim
    griffinlim_cqt


Miscellaneous
=============
.. autosummary::
    :toctree: generated/

    samples_like
    times_like

    get_fftlib
    set_fftlib
"""

import lazy_loader as lazy
from .version import version as __version__

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
