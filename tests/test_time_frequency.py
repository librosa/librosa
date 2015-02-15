#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-14 19:13:49 by Brian McFee <brian.mcfee@nyu.edu>
'''Unit tests for time and frequency conversion''' 

import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import numpy as np
import librosa
from nose.tools import raises


def test_frames_to_samples():

    def __test(x, y, hop_length, n_fft):
        y_test = librosa.frames_to_samples(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)

    x = np.arange(2)
    for hop_length in [512, 1024]:
        for n_fft in [None, 1024]:
            y = x * hop_length
            if n_fft is not None:
                y += n_fft // 2
            yield __test, x, y, hop_length, n_fft


def test_samples_to_frames():

    def __test(x, y, hop_length, n_fft):
        y_test = librosa.samples_to_frames(x,
                                           hop_length=hop_length,
                                           n_fft=n_fft)
        assert np.allclose(y, y_test)

    x = np.arange(2)
    for hop_length in [512, 1024]:
        for n_fft in [None, 1024]:
            y = x * hop_length
            if n_fft is not None:
                y += n_fft // 2
            yield __test, y, x, hop_length, n_fft


def test_time_to_samples():

    def __test(sr):
        assert np.allclose(librosa.time_to_samples([0, 1, 2], sr=sr),
                           [0, sr, 2 * sr])

    for sr in [22050, 44100]:
        yield __test, sr


def test_samples_to_time():

    def __test(sr):
        assert np.allclose(librosa.samples_to_time([0, sr, 2 * sr], sr=sr),
                           [0, 1, 2])

    for sr in [22050, 44100]:
        yield __test, sr


def test_cqt_frequencies():

    def __test(n_bins, fmin, bins_per_octave, tuning):

        freqs = librosa.cqt_frequencies(n_bins,
                                        fmin,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning)

        # Make sure we get the right number of bins
        assert len(freqs) == n_bins

        # And that the first bin matches fmin by tuning
        assert np.allclose(freqs[0],
                           fmin * 2.0**(float(tuning) / bins_per_octave))

        # And that we have constant Q
        Q = np.diff(np.log2(freqs))
        assert np.allclose(Q, 1./bins_per_octave)

    for n_bins in [12, 24, 36]:
        for fmin in [440.0]:
            for bins_per_octave in [12, 24, 36]:
                for tuning in [-0.25, 0.0, 0.25]:
                    yield __test, n_bins, fmin, bins_per_octave, tuning
