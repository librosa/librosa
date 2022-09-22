#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core IO and DSP
===============

Audio loading
-------------
.. autosummary::
    :toctree: generated/

    load
    stream
    to_mono
    resample
    get_duration
    get_samplerate


Time-domain processing
----------------------
.. autosummary::
    :toctree: generated/

    autocorrelate
    lpc
    zero_crossings
    mu_compress
    mu_expand


Signal generation
-----------------
.. autosummary::
    :toctree: generated/

    clicks
    tone
    chirp


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

    vqt

    iirt

    fmt

    magphase


Phase recovery
--------------
.. autosummary::
    :toctree: generated/

    griffinlim
    griffinlim_cqt


Harmonics
---------
.. autosummary::
    :toctree: generated/

    interp_harmonics
    salience

    phase_vocoder


Magnitude scaling
-----------------
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


Time unit conversion
--------------------
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


Frequency unit conversion
-------------------------
.. autosummary::
    :toctree: generated/

    hz_to_note
    hz_to_midi
    hz_to_svara_h
    hz_to_svara_c
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
--------------
.. autosummary::
    :toctree: generated/

    key_to_notes
    key_to_degrees

    mela_to_svara
    mela_to_degrees

    thaat_to_degrees

    list_mela
    list_thaat


Frequency range generation
--------------------------
.. autosummary::
    :toctree: generated/

    fft_frequencies
    cqt_frequencies
    mel_frequencies
    tempo_frequencies
    fourier_tempo_frequencies


Pitch and tuning
----------------
.. autosummary::
    :toctree: generated/

    pyin
    yin

    estimate_tuning
    pitch_tuning
    piptrack


Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    samples_like
    times_like

    get_fftlib
    set_fftlib
"""

import lazy_loader as lazy
from .version import version as __version__


__getattr__, __lazy_dir__, _ = lazy.attach(
    __name__,
    submodules=[
        "core",
        "beat",
        "decompose",
        "display",
        "effects",
        "feature",
        "filters",
        "onset",
        "segment",
        "sequence",
        "util",
    ],
    submod_attrs={
        "_cache": ["cache"],
        "util.exceptions": ["LibrosaError", "ParameterError"],
        "util.files": ["example", "ex"],
        "core": [
            "A4_to_tuning",
            "A_weighting",
            "B_weighting",
            "C_weighting",
            "D_weighting",
            "Z_weighting",
            "amplitude_to_db",
            "audio",
            "autocorrelate",
            "blocks_to_frames",
            "blocks_to_samples",
            "blocks_to_time",
            "chirp",
            "clicks",
            "constantq",
            "convert",
            "cqt",
            "cqt_frequencies",
            "db_to_amplitude",
            "db_to_power",
            "estimate_tuning",
            "fft",
            "fft_frequencies",
            "fmt",
            "fourier_tempo_frequencies",
            "frames_to_samples",
            "frames_to_time",
            "frequency_weighting",
            "get_duration",
            "get_fftlib",
            "get_samplerate",
            "griffinlim",
            "griffinlim_cqt",
            "harmonic",
            "hybrid_cqt",
            "hz_to_mel",
            "hz_to_midi",
            "hz_to_note",
            "hz_to_octs",
            "hz_to_svara_c",
            "hz_to_svara_h",
            "icqt",
            "iirt",
            "interp_harmonics",
            "istft",
            "key_to_degrees",
            "key_to_notes",
            "list_mela",
            "list_thaat",
            "load",
            "lpc",
            "magphase",
            "mel_frequencies",
            "mel_to_hz",
            "mela_to_degrees",
            "mela_to_svara",
            "midi_to_hz",
            "midi_to_note",
            "midi_to_svara_c",
            "midi_to_svara_h",
            "mu_compress",
            "mu_expand",
            "multi_frequency_weighting",
            "notation",
            "note_to_hz",
            "note_to_midi",
            "note_to_svara_c",
            "note_to_svara_h",
            "octs_to_hz",
            "pcen",
            "perceptual_weighting",
            "phase_vocoder",
            "piptrack",
            "pitch",
            "pitch_tuning",
            "power_to_db",
            "pseudo_cqt",
            "pyin",
            "reassigned_spectrogram",
            "resample",
            "salience",
            "samples_like",
            "samples_to_frames",
            "samples_to_time",
            "set_fftlib",
            "spectrum",
            "stft",
            "stream",
            "tempo_frequencies",
            "thaat_to_degrees",
            "time_to_frames",
            "time_to_samples",
            "times_like",
            "to_mono",
            "tone",
            "tuning_to_A4",
            "vqt",
            "yin",
            "zero_crossings",
        ],
        "version": ["show_versions"],
    },
)


def __dir__():
    return __lazy_dir__() + ["__version__"]
