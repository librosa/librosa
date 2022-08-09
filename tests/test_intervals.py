#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2022-08-09 06:34:21 by Brian McFee <brian.mcfee@nyu.edu>
"""Unit tests for just intonation and friends"""

import os
import sys

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import warnings
import numpy as np
import pytest
import librosa


def test_pythagorean():
    ivals = librosa.pythagorean_intervals(bins_per_octave=6, sort=False)
    assert np.allclose(ivals, [1, 3/2, 9/8, 27/16, 81/64, 243/128])
    ivals2 = librosa.pythagorean_intervals(bins_per_octave=6, sort=True)
    assert np.allclose(sorted(ivals), ivals2)


@pytest.mark.parametrize('n_bins', [6, 12, 24, 30])
@pytest.mark.parametrize('intervals', ['equal', 'ji3', 'ji5', 'ji7', [1, 4/3, 3/2, 5/4]])
@pytest.mark.parametrize('bins_per_octave', [6, 12, 15])
def test_interval_frequencies(n_bins, intervals, bins_per_octave):
    freqs = librosa.interval_frequencies(n_bins, fmin=10, intervals=intervals, bins_per_octave=bins_per_octave)

    assert len(freqs) == n_bins
    assert min(freqs) == 10
