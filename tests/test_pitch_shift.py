#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import warnings

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

from contextlib2 import nullcontext as dnr
import numpy as np
import pytest

import librosa

__EXAMPLE_FILE_STEREO = os.path.join("tests", "data", "test1_44100.wav")

def test_pitch_shift():
    y, sr = librosa.load(__EXAMPLE_FILE_STEREO, 44100, mono=False)
    ys = librosa.effects.pitch_shift(y, sr, 5, bins_per_octave=12)
    orig_duration = librosa.get_duration(y, sr=sr)
    new_duration = librosa.get_duration(ys, sr=sr)
