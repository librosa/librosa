#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq

Run me as follows:
    cd tests/
    nosetests -v --with-coverage --cover-package=librosa
"""


from __future__ import print_function
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


def test_cqt():

    def __test(y, sr, hop_length, fmin, n_bins, bins_per_octave,
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


    sr = 11025


    # Impulse train
    y = np.zeros(int(10.0 * sr))
    y[::sr] = 1.0


    hop_length = 512
    fmin = None
    n_bins = 48
    bins_per_octave = 12
    tuning = None
    resolution = 2
    aggregate = None
    norm = 1
    sparsity = 0.01


    # Hop size not long enough for num octaves
    yield (raises(ValueError)(__test), y, sr, 32, fmin, 72,
           bins_per_octave, tuning, resolution, aggregate, norm, sparsity)

    # Filters go beyond Nyquist
    yield (raises(ValueError)(__test), y, sr, hop_length, 500, 48,
           bins_per_octave, tuning, resolution, aggregate, norm, sparsity)

    # TODO Test that upper bands of CQT are equivalent to a CQT that
    # starts at upper bands.

    # TODO Test that lower bands of CQT are equivalent to a CQT with
    # same bins_per_octave but greater n_bins


    hop_length = 512
    aggregate = None
    sparsity = 0.01

    for fmin in [None, librosa.note_to_hz('C3')]:
        for n_bins in [12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [0, 0.25]:
                    for resolution in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test, y, sr, hop_length, fmin, n_bins,
                                bins_per_octave, tuning,
                                resolution, aggregate, norm, sparsity)



