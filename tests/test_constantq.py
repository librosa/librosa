#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq

Run me as follows:
    cd tests/
    nosetests -v --with-coverage --cover-package=librosa
"""
from __future__ import division

import warnings

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except KeyError:
    pass

import librosa
import numpy as np

from nose.tools import raises, eq_

from test_core import srand

warnings.resetwarnings()
warnings.simplefilter('always')


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
                                sparsity=sparsity))

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

        C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min)))**2

        # Average over time
        Cbar = np.median(C, axis=1)

        # Find the peak
        idx = np.argmax(Cbar)

        eq_(idx, 60 - note_min)

        # Make sure that the max outside the peak is sufficiently small
        Cscale = Cbar / Cbar[idx]
        Cscale[idx] = np.nan
        assert np.nanmax(Cscale) < 6e-1, Cscale

        Cscale[idx-1:idx+2] = np.nan
        assert np.nanmax(Cscale) < 5e-2, Cscale

    for note_min in [12, 18, 24, 30, 36]:
        yield __test, note_min


@raises(librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36, real=False)


@raises(librosa.ParameterError)
def test_cqt_fail_short_late():

    y = np.zeros(16)
    librosa.cqt(y, sr=22050, real=False)


def test_cqt_impulse():
    # Test to resolve issue #348
    # Updated in #417 to use integrated energy, rather than frame-wise max
    def __test(sr, hop_length, y):

        C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length))

        response = np.mean(C**2, axis=1)

        continuity = np.abs(np.diff(response))

        # Test that integrated energy is approximately constant
        assert np.max(continuity) < 5e-4, continuity

    for sr in [11025, 16384, 22050, 32000, 44100]:
        # Generate an impulse
        x = np.zeros(sr)

        for hop_scale in range(1, 9):
            hop_length = 64 * hop_scale
            # Center the impulse response on a frame
            center = int((len(x) / (2.0 * float(hop_length))) * hop_length)
            x[center] = 1
            yield __test, sr, hop_length, x


def test_hybrid_cqt_scale():
    # Test to resolve issue #341
    # Updated in #417 to ise integrated energy instead of pointwise max
    def __test(sr, hop_length, y):

        hcqt = librosa.hybrid_cqt(y=y, sr=sr, hop_length=hop_length, tuning=0)

        response = np.mean(np.abs(hcqt)**2, axis=1)

        continuity = np.abs(np.diff(response))

        assert np.max(continuity) < 5e-4, continuity

    for sr in [11025, 16384, 22050, 32000, 44100]:
        # Generate an impulse
        x = np.zeros(sr)

        for hop_scale in range(1, 9):
            hop_length = 64 * hop_scale
            # Center the impulse response on a frame
            center = int((len(x) / (2.0 * float(hop_length))) * hop_length)
            x[center] = 1
            yield __test, sr, hop_length, x


def test_cqt_white_noise():

    def __test(fmin, n_bins, scale, sr, y):

        C = np.abs(librosa.cqt(y=y, sr=sr,
                               fmin=fmin,
                               n_bins=n_bins,
                               scale=scale))

        if not scale:
            lengths = librosa.filters.constant_q_lengths(sr, fmin,
                                                         n_bins=n_bins)
            C /= np.sqrt(lengths[:, np.newaxis])

        # Only compare statistics across the time dimension
        # we want ~ constant mean and variance across frequencies
        assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
        assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)

    srand()
    for sr in [22050]:
        y = np.random.randn(30 * sr)

        for scale in [False, True]:
            for fmin in librosa.note_to_hz(['C1', 'C2']):
                for n_octaves in range(2, 4):
                    yield __test, fmin, n_octaves * 12, scale, sr, y


def test_hcqt_white_noise():

    def __test(fmin, n_bins, scale, sr, y):

        C = librosa.hybrid_cqt(y=y, sr=sr,
                               fmin=fmin,
                               n_bins=n_bins,
                               scale=scale)

        if not scale:
            lengths = librosa.filters.constant_q_lengths(sr, fmin,
                                                         n_bins=n_bins)
            C /= np.sqrt(lengths[:, np.newaxis])

        assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
        assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)

    srand()
    for sr in [22050]:
        y = np.random.randn(30 * sr)

        for scale in [False, True]:
            for fmin in librosa.note_to_hz(['C1', 'C2']):
                for n_octaves in [6, 7]:
                    yield __test, fmin, n_octaves * 12, scale, sr, y


def test_cqt_real_warning():

    def __test(real):
        warnings.resetwarnings()
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True) as out:
            C = librosa.cqt(y=y, sr=sr, real=real)
            assert len(out) > 0
            assert out[0].category is DeprecationWarning

            if real:
                assert np.isrealobj(C)
            else:
                assert np.iscomplexobj(C)

    sr = 22050
    y = np.zeros(2 * sr)

    yield __test, False
    yield __test, True
