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
                    tuning, filter_scale, norm, sparsity):

    cqt_output = np.abs(librosa.cqt(y,
                                    sr=sr,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    n_bins=n_bins,
                                    bins_per_octave=bins_per_octave,
                                    tuning=tuning,
                                    filter_scale=filter_scale,
                                    norm=norm,
                                    sparsity=sparsity))

    eq_(cqt_output.shape[0], n_bins)

    return cqt_output


def make_signal(sr, duration, fmax='C8'):
    ''' Generates a linear sine sweep '''

    fmin = librosa.note_to_hz('C1') / sr
    if fmax is None:
        fmax = 0.5
    else:
        fmax = librosa.note_to_hz(fmax) / sr

    return np.sin(np.cumsum(2 * np.pi * np.logspace(np.log10(fmin), np.log10(fmax),
                                                    num=duration * sr)))

def test_cqt():

    sr = 11025
    duration = 5.0

    y = make_signal(sr, duration)

    # incorrect hop length for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 16
    for hop_length in [-1, 0, 16, 63, 65]:
        yield (raises(librosa.ParameterError)(__test_cqt_size), y, sr, hop_length, None, 72,
               12, 0.0, 2, 1, 0.01)

    # Filters go beyond Nyquist. 500 Hz -> 4 octaves = 8000 Hz > 11000 Hz
    yield (raises(librosa.ParameterError)(__test_cqt_size), y, sr, 512, 500, 4 * 12,
           12, 0.0, 2, 1, 0.01)

    # Test with fmin near Nyquist
    for fmin in [3000, 4800]:
        for n_bins in [1, 2]:
            for bins_per_octave in [12]:
                yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                       bins_per_octave, 0.0, 2, 1, 0.01)

    # Test for no errors and correct output size
    for fmin in [None, librosa.note_to_hz('C2')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [None, 0, 0.25]:
                    for filter_scale in [1, 2]:
                        for norm in [1, 2]:
                            yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                                   bins_per_octave, tuning,
                                   filter_scale, norm, 0.01)


def test_hybrid_cqt():
    # This test verifies that hybrid and full cqt agree down to 1e-4
    # on 99% of bins which are nonzero (> 1e-8) in either representation.

    sr = 11025
    duration = 5.0

    y = make_signal(sr, duration, None)

    def __test(hop_length, fmin, n_bins, bins_per_octave,
               tuning, resolution, norm, sparsity):

        C2 = librosa.hybrid_cqt(y, sr=sr,
                                hop_length=hop_length,
                                fmin=fmin, n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning, filter_scale=resolution,
                                norm=norm,
                                sparsity=sparsity)

        C1 = np.abs(librosa.cqt(y, sr=sr,
                                hop_length=hop_length,
                                fmin=fmin, n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning, filter_scale=resolution,
                                norm=norm,
                                sparsity=sparsity, real=False))

        eq_(C1.shape, C2.shape)

        # Check for numerical comparability
        idx1 = (C1 > 1e-4 * C1.max())
        idx2 = (C2 > 1e-4 * C2.max())

        perc = 0.99

        thresh = 1e-3

        idx = idx1 | idx2

        assert np.percentile(np.abs(C1[idx] - C2[idx]),
                             perc) < thresh * max(C1.max(), C2.max())

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
    freq = librosa.midi_to_hz(60)

    y = np.sin(2 * np.pi * freq * np.linspace(0, 2.0, 2 * sr))

    def __test(note_min):

        C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min), real=False))

        # Average over time
        Cbar = np.median(C, axis=1)

        # Find the peak
        idx = np.argmax(Cbar)

        eq_(idx, 60 - note_min)

        # Make sure that the max outside the peak is sufficiently small
        Cscale = Cbar / Cbar[idx]
        Cscale[idx] = np.nan
        assert np.nanmax(Cscale) < 6e-1

        Cscale[idx-1:idx+2] = np.nan
        assert np.nanmax(Cscale) < 5e-2

    for note_min in [12, 18, 24, 30, 36]:
        yield __test, note_min


@raises(librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36, real=False)


@raises(librosa.ParameterError)
def test_cqt_fail_short_late():

    y = np.zeros(64)
    librosa.cqt(y, sr=22050, real=False)

def test_cqt_impulse():
    # Test to resolve issue #348
    def __test(sr, hop_length, y):

        C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, real=False))

        max_response = np.max(C, axis=1)


        ref_response = np.max(max_response)
        continuity = np.abs(np.diff(max_response))

        # Test that continuity is never violated by more than 15% point-wise energy
        assert np.max(continuity) < 1.5e-1 * ref_response, np.max(continuity) / ref_response

        # Test that peak-energy deviation is bounded
        assert np.std(max_response) < 0.5 * ref_response, np.std(max_response) / ref_response

    for sr in [11025, 16384, 22050, 32000, 44100]:
        # Generate an impulse
        x = np.zeros(sr)

        for hop_scale in range(1, 9):
            hop_length = 64 * hop_scale
            # Center the impulse response on a frame
            center = (len(x) / (2 * float(hop_length))) * hop_length
            x[center] = 1
            yield __test, sr, hop_length, x


def test_hybrid_cqt_scale():
    # Test to resolve issue #341
    def __test(sr, hop_length, y):

        hcqt = librosa.hybrid_cqt(y=y, sr=sr, hop_length=hop_length, tuning=0)

        max_response = np.max(np.abs(hcqt), axis=1)


        ref_response = np.max(max_response)
        continuity = np.abs(np.diff(max_response))

        # Test that continuity is never violated by more than 75% point-wise energy
        assert np.max(continuity) <= 0.6 * ref_response, np.max(continuity)

        # Test that peak-energy deviation is bounded
        assert np.std(max_response) < 0.5 * ref_response, np.std(max_response)

    for sr in [11025, 16384, 22050, 32000, 44100]:
        # Generate an impulse
        x = np.zeros(sr)

        for hop_scale in range(1, 9):
            hop_length = 64 * hop_scale
            # Center the impulse response on a frame
            center = (len(x) / (2 * float(hop_length))) * hop_length
            x[center] = 1
            yield __test, sr, hop_length, x
