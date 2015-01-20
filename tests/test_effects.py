#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np

from nose.tools import raises


def test_time_stretch():

    def __test(infile, rate):
        y, sr = librosa.load('data/test1_22050.wav', duration=4.0)
        ys = librosa.effects.time_stretch(y, rate)

        orig_duration = librosa.get_duration(y, sr=sr)
        new_duration = librosa.get_duration(ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert np.allclose(orig_duration, rate * new_duration,
                           rtol=1e-2, atol=1e-3)

    for rate in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yield __test, 'data/test1_22050.wav', rate
