#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Unit tests for the effects module'''

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

from nose.tools import raises, eq_

import librosa
import numpy as np

__EXAMPLE_FILE = 'data/test1_22050.wav'


def test_time_stretch():

    def __test(infile, rate):
        y, sr = librosa.load(infile, duration=4.0)
        ys = librosa.effects.time_stretch(y, rate)

        orig_duration = librosa.get_duration(y, sr=sr)
        new_duration = librosa.get_duration(ys, sr=sr)

        # We don't have to be too precise here, since this goes through an STFT
        assert np.allclose(orig_duration, rate * new_duration,
                           rtol=1e-2, atol=1e-3)

    for rate in [0.25, 0.5, 1.0, 2.0, 4.0]:
        yield __test, 'data/test1_22050.wav', rate

    for rate in [-1, 0]:
        yield raises(ValueError)(__test), 'data/test1_22050.wav', rate


def test_hpss():

    y, sr = librosa.load(__EXAMPLE_FILE)

    y_harm, y_perc = librosa.effects.hpss(y)

    # Make sure that the residual energy is generally small
    y_residual = y - y_harm - y_perc

    rms_orig = librosa.feature.rms(y=y)
    rms_res = librosa.feature.rms(y=y_residual)

    assert np.percentile(rms_orig, 0.01) > np.percentile(rms_res, 0.99)


def test_percussive():

    y, sr = librosa.load('data/test1_22050.wav')

    yh1, yp1 = librosa.effects.hpss(y)

    yp2 = librosa.effects.percussive(y)

    assert np.allclose(yp1, yp2)


def test_harmonic():

    y, sr = librosa.load('data/test1_22050.wav')

    yh1, yp1 = librosa.effects.hpss(y)

    yh2 = librosa.effects.harmonic(y)

    assert np.allclose(yh1, yh2)
