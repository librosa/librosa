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


Frequency range generation
--------------------------
.. autosummary::
    :toctree: generated/

    fft_frequencies
    cqt_frequencies
    mel_frequencies
    tempo_frequencies
    fourier_tempo_frequencies


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

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
