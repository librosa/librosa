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
except KeyError:
    pass

import librosa
import numpy as np

from nose.tools import raises, eq_


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

    eq_(cqt_output.shape[0], n_bins)

    return cqt_output


def test_cqt():

    sr = 11025

    # Impulse train
    y = np.zeros(int(5.0 * sr))
    y[::sr] = 1.0


    # incorrect hop length for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 16
    for hop_length in [-1, 0, 16, 63, 65]:
        yield (raises(librosa.ParameterError)(__test_cqt_size), y, sr, hop_length, None, 72,
               12, 0.0, 2, None, 1, 0.01)

    # Filters go beyond Nyquist. 500 Hz -> 4 octaves = 8000 Hz > 11000 Hz
    yield (raises(librosa.ParameterError)(__test_cqt_size), y, sr, 512, 500, 48,
           12, 0.0, 2, None, 1, 0.01)

    # Test with fmin near Nyquist
    for fmin in [3000, 4800]:
        for n_bins in [1, 2]:
            for bins_per_octave in [12]:
                yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                       bins_per_octave, 0.0, 2, None, 1, 0.01)

    # Test for no errors and correct output size
    for fmin in [None, librosa.note_to_hz('C2')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [None, 0, 0.25]:
                    for resolution in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                                   bins_per_octave, tuning,
                                   resolution, None, norm, 0.01)


def test_hybrid_cqt():

    sr = 11025

    # Impulse train
    y = np.zeros(int(5.0 * sr))
    y[::sr] = 1.0

    def __test(hop_length, fmin, n_bins, bins_per_octave,
               tuning, resolution, norm, sparsity):

        C2 = librosa.hybrid_cqt(y, sr=sr,
                                hop_length=hop_length,
                                fmin=fmin, n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning, resolution=resolution,
                                norm=norm,
                                sparsity=sparsity)

        C1 = librosa.cqt(y, sr=sr,
                         hop_length=hop_length,
                         fmin=fmin, n_bins=n_bins,
                         bins_per_octave=bins_per_octave,
                         tuning=tuning, resolution=resolution,
                         norm=norm,
                         sparsity=sparsity)

        eq_(C1.shape, C2.shape)

        # Check for numerical comparability
        assert np.mean(np.abs(C1 - C2)) < 1e-3

    for fmin in [None, librosa.note_to_hz('C2')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [None, 0, 0.25]:
                    for resolution in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test, 512, fmin, n_bins,
                                   bins_per_octave, tuning,
                                   resolution, norm, 0.01)


def test_cqt_position():

    # synthesize a two second sine wave at midi note 60

    sr = 22050
    f = librosa.midi_to_hz(60)

    y = np.sin(2 * np.pi * f * np.linspace(0, 2.0, 2 * sr))

    def __test(note_min):

        C = librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min))

        # Average over time
        Cbar = np.median(C, axis=1)

        # Find the peak
        idx = np.argmax(Cbar)

        eq_(idx, 60 - note_min)

        # Make sure that the max outside the peak is sufficiently small
        Cscale = Cbar / Cbar[idx]
        Cscale[idx] = np.nan

        assert np.nanmax(Cscale) < 1e-1

        Cscale[idx-1:idx+2] = np.nan
        assert np.nanmax(Cscale) < 1e-2

    for note_min in [12, 18, 24, 30, 36]:
        yield __test, note_min
