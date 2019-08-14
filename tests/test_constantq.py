#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq
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

import pytest

from test_core import srand


def __test_cqt_size(y, sr, hop_length, fmin, n_bins, bins_per_octave,
                    tuning, filter_scale, norm, sparsity, res_type):

    cqt_output = np.abs(librosa.cqt(y,
                                    sr=sr,
                                    hop_length=hop_length,
                                    fmin=fmin,
                                    n_bins=n_bins,
                                    bins_per_octave=bins_per_octave,
                                    tuning=tuning,
                                    filter_scale=filter_scale,
                                    norm=norm,
                                    sparsity=sparsity,
                                    res_type=res_type))

    assert cqt_output.shape[0] == n_bins

    return cqt_output


def make_signal(sr, duration, fmin='C1', fmax='C8'):
    ''' Generates a linear sine sweep '''

    if fmin is None:
        fmin = 0.01
    else:
        fmin = librosa.note_to_hz(fmin) / sr

    if fmax is None:
        fmax = 0.5
    else:
        fmax = librosa.note_to_hz(fmax) / sr

    return np.sin(np.cumsum(2 * np.pi * np.logspace(np.log10(fmin),
                                                    np.log10(fmax),
                                                    num=int(duration * sr))))


def test_cqt():

    sr = 11025
    duration = 5.0

    y = make_signal(sr, duration)

    # incorrect hop length for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 16
    for hop_length in [-1, 0, 16, 63, 65]:
        yield (pytest.mark.xfail(__test_cqt_size, raises=librosa.ParameterError), y, sr, hop_length, None, 72,
               12, 0.0, 2, 1, 0.01, None)

    # Filters go beyond Nyquist. 500 Hz -> 4 octaves = 8000 Hz > 11000 Hz
    yield (pytest.mark.xfail(__test_cqt_size, raises=librosa.ParameterError), y, sr, 512, 500, 4 * 12,
           12, 0.0, 2, 1, 0.01, None)

    # Test with fmin near Nyquist
    for fmin in [3000, 4800]:
        for n_bins in [1, 2]:
            for bins_per_octave in [12]:
                yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                       bins_per_octave, 0.0, 2, 1, 0.01, None)

    # Test for no errors and correct output size
    for fmin in [None, librosa.note_to_hz('C2')]:
        for n_bins in [1, 12, 24, 48, 72, 74, 76]:
            for bins_per_octave in [12, 24]:
                for tuning in [None, 0, 0.25]:
                    for filter_scale in [1, 2]:
                        for norm in [1, 2]:
                            for res_type in [None, 'polyphase']:
                                yield (__test_cqt_size, y, sr, 512, fmin, n_bins,
                                        bins_per_octave, tuning, filter_scale, norm, 0.01, res_type)


def test_hybrid_cqt():
    # This test verifies that hybrid and full cqt agree down to 1e-4
    # on 99% of bins which are nonzero (> 1e-8) in either representation.

    sr = 11025
    duration = 5.0

    y = make_signal(sr, duration, None)

    def __test(hop_length, fmin, n_bins, bins_per_octave,
               tuning, resolution, norm, sparsity, res_type):

        C2 = librosa.hybrid_cqt(y, sr=sr,
                                hop_length=hop_length,
                                fmin=fmin, n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning, filter_scale=resolution,
                                norm=norm,
                                sparsity=sparsity, res_type=res_type)

        C1 = np.abs(librosa.cqt(y, sr=sr,
                                hop_length=hop_length,
                                fmin=fmin, n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning, filter_scale=resolution,
                                norm=norm,
                                sparsity=sparsity, res_type=res_type))

        assert C1.shape == C2.shape

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
                            for res_type in [None, 'polyphase']:
                                yield (__test, 512, fmin, n_bins,
                                        bins_per_octave, tuning,
                                        resolution, norm, 0.01, res_type)


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

        assert idx == 60 - note_min

        # Make sure that the max outside the peak is sufficiently small
        Cscale = Cbar / Cbar[idx]
        Cscale[idx] = np.nan
        assert np.nanmax(Cscale) < 6e-1, Cscale

        Cscale[idx-1:idx+2] = np.nan
        assert np.nanmax(Cscale) < 5e-2, Cscale

    for note_min in [12, 18, 24, 30, 36]:
        yield __test, note_min


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_late():

    y = np.zeros(16)
    librosa.cqt(y, sr=22050)


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


def test_hybrid_cqt_white_noise():

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


def test_icqt():

    def __test(sr, scale, hop_length, over_sample, length, res_type, dtype, y):

        bins_per_octave = over_sample * 12
        n_bins = 7 * bins_per_octave

        C = librosa.cqt(y, sr=sr, n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        scale=scale,
                        hop_length=hop_length)

        if length:
            _len = len(y)
        else:
            _len = None
        yinv = librosa.icqt(C, sr=sr,
                            scale=scale,
                            hop_length=hop_length,
                            bins_per_octave=bins_per_octave,
                            length=_len,
                            res_type=res_type,
                            dtype=dtype)

        assert yinv.dtype == dtype

        # Only test on the middle section
        if length:
            assert len(y) == len(yinv)
        else:
            yinv = librosa.util.fix_length(yinv, len(y))

        y = y[sr//2:-sr//2]
        yinv = yinv[sr//2:-sr//2]

        residual = np.abs(y - yinv)
        # We'll tolerate 10% RMSE
        # error is lower on more recent numpy/scipy builds

        resnorm = np.sqrt(np.mean(residual**2))
        assert resnorm <= 0.1, resnorm

    for sr in [22050, 44100]:
        y = make_signal(sr, 1.5, fmin='C2', fmax='C4')
        for over_sample in [1, 3]:
            for scale in [False, True]:
                for hop_length in [384, 512]:
                    for length in [None, True]:
                        for res_type in ['scipy', 'kaiser_fast', 'polyphase']:
                            for dtype in [np.float32, np.float64]:
                                yield __test, sr, scale, hop_length, over_sample, length, res_type, dtype, y



@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(55, 55 * 2**3, length=sr//8, sr=sr)
    return y


@pytest.mark.parametrize('hop_length', [512, 1024])
@pytest.mark.parametrize('window', ['hann', 'hamming'])
@pytest.mark.parametrize('use_length', [False, True])
@pytest.mark.parametrize('over_sample', [1, 3])
@pytest.mark.parametrize('res_type', ['polyphase'])
@pytest.mark.parametrize('pad_mode', ['reflect'])
@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('momentum', [0, 0.99])
@pytest.mark.parametrize('random_state', [None, 0, np.random.RandomState()])
@pytest.mark.parametrize('fmin', [40.0])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('init', [None, 'random'])
def test_griffinlim_cqt(y_chirp, hop_length, window, use_length, over_sample, fmin,
                        res_type, pad_mode, scale, momentum, init, random_state, dtype):

    if use_length:
        length = len(y_chirp)
    else:
        length = None

    sr = 22050
    bins_per_octave = 12 * over_sample
    n_bins = 6 * bins_per_octave
    C = librosa.cqt(y_chirp, sr=sr, hop_length=hop_length, window=window, fmin=fmin,
                    bins_per_octave=bins_per_octave, n_bins=n_bins,
                    scale=scale,
                    pad_mode=pad_mode,
                    res_type=res_type)

    Cmag = np.abs(C)

    y_rec = librosa.griffinlim_cqt(Cmag, hop_length=hop_length, window=window,
                                   sr=sr,
                                   fmin=fmin,
                                   bins_per_octave=bins_per_octave,
                                   scale=scale,
                                   pad_mode=pad_mode,
                                   n_iter=3,
                                   momentum=momentum,
                                   random_state=random_state,
                                   length=length,
                                   res_type=res_type,
                                   init=init,
                                   dtype=dtype)

    y_inv = librosa.icqt(Cmag, sr=sr, fmin=fmin, hop_length=hop_length,
                         window=window, bins_per_octave=bins_per_octave, scale=scale,
                         length=length, res_type=res_type)

    # First check for length
    if use_length:
        assert len(y_rec) == length

    assert y_rec.dtype == dtype

    # Check that the data is okay
    assert np.all(np.isfinite(y_rec))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_badinit():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, init='garbage')


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_momentum():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, momentum=-1)


def test_griffinlim_cqt_momentum_warn():
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim_cqt(x, momentum=2)
