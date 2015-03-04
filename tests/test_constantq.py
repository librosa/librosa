#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq

Run me as follows:
    cd tests/
    nosetests -v --with-coverage --cover-package=librosa
"""
from __future__ import division

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import librosa
import numpy as np

from nose.tools import nottest, eq_, raises

def __test_cqt_size(y, sr, hop_length, fmin, n_bins, bins_per_octave,
            tuning, resolution, aggregate, norm, sparsity):

    cqt_output = librosa.cqt(y,
                             sr=sr,
                             hop_length=hop_length,
                             fmin=fmin,
                             n_bins=n_bins,
                             bins_per_octave=bins_per_octave,
                             tuning=tuning,
                             resolution=resolution,
                             aggregate=aggregate,
                             norm=norm,
                             sparsity=sparsity)

    assert cqt_output.shape[0] == n_bins

    return cqt_output


def test_cqt():

    sr = 11025


    # Impulse train
    y = np.zeros(int(5.0 * sr))
    y[::sr] = 1.0


    # Hop size not long enough for num octaves
    # num_octaves = 6, 2**6 = 64 > 32
    yield (raises(ValueError)(__test_cqt_size), y, sr, 32, None, 72,
           12, None, 2, None, 1, 0.01)

    # Filters go beyond Nyquist. 500 Hz -> 4 octaves = 8000 Hz > 11000 Hz
    yield (raises(ValueError)(__test_cqt_size), y, sr, 512, 500, 48,
           12, None, 2, None, 1, 0.01)


    # Test for no errors and correct output size
    for fmin in [None, librosa.note_to_hz('C3')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [0, 0.25]:
                    for resolution in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                                bins_per_octave, tuning,
                                resolution, None, norm, 0.01)




