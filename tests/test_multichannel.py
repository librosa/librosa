#!/usr/bin/env python
# CREATED:2013-03-08 15:25:18 by Brian McFee <brm2132@columbia.edu>
#  unit tests for multi-channel functionality
#

from __future__ import print_function

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except:
    pass

import librosa
import glob
import numpy as np
import scipy.io
import pytest
import warnings
from unittest import mock


@pytest.fixture(scope="module", params=["test1_44100.wav"])
def y_multi(request):
    infile = request.param
    return librosa.load(os.path.join("tests", "data", infile),
                        sr=None, mono=False)


def test_stft_multi(y_multi):

    # Verify that a stereo STFT matches on
    # each channel individually
    y, sr = y_multi

    D = librosa.stft(y)

    D0 = librosa.stft(y[0])
    D1 = librosa.stft(y[1])

    # Check each channel
    assert np.allclose(D[0], D0)
    assert np.allclose(D[1], D1)

    # Check that they're not both the same
    assert not np.allclose(D0, D1)


def test_istft_multi(y_multi):

    # Verify that a stereo ISTFT matches on each channel
    y, sr = y_multi

    # Assume the forward transform works properly in stereo
    D = librosa.stft(y)

    # Invert per channel
    y0m = librosa.istft(D[0])
    y1m = librosa.istft(D[1])

    # Invert both channels at once
    ys = librosa.istft(D)

    # Check each channel
    assert np.allclose(y0m, ys[0])
    assert np.allclose(y1m, ys[1])

    # Check that they're not both the same
    assert not np.allclose(ys[0], ys[1])


def test_griffinlim_multi(y_multi):
    y, sr = y_multi

    # Compute the stft
    D = librosa.stft(y)

    # Run a couple of iterations of griffin-lim
    yout = librosa.griffinlim(np.abs(D), n_iter=2, length=y.shape[-1])

    # Check the lengths
    assert np.allclose(y.shape, yout.shape)



@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('res_type', [None, 'polyphase'])
def test_cqt_multi(y_multi, scale, res_type):

    y, sr = y_multi

    # Assuming single-channel CQT is well behaved
    C0 = librosa.cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.cqt(y=y, sr=sr, scale=scale, res_type=res_type)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('res_type', [None, 'polyphase'])
def test_hybrid_cqt_multi(y_multi, scale, res_type):

    y, sr = y_multi

    # Assuming single-channel CQT is well behaved
    C0 = librosa.hybrid_cqt(y=y[0], sr=sr, scale=scale, res_type=res_type)
    C1 = librosa.hybrid_cqt(y=y[1], sr=sr, scale=scale, res_type=res_type)
    Call = librosa.hybrid_cqt(y=y, sr=sr, scale=scale, res_type=res_type)

    # Check each channel
    assert np.allclose(C0, Call[0])
    assert np.allclose(C1, Call[1])

    # Verify that they're not all the same
    assert not np.allclose(Call[0], Call[1])


@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('length', [None, 22050])
def test_icqt_multi(y_multi, scale, length):

    y, sr = y_multi

    # Assuming the forward transform is well-behaved
    C = librosa.cqt(y=y, sr=sr, scale=scale)

    yboth = librosa.icqt(C, sr=sr, scale=scale, length=length)
    y0 = librosa.icqt(C[0], sr=sr, scale=scale, length=length)
    y1 = librosa.icqt(C[1], sr=sr, scale=scale, length=length)

    if length is not None:
        assert yboth.shape[-1] == length

    # Check each channel
    assert np.allclose(yboth[0], y0)
    assert np.allclose(yboth[1], y1)

    # Check that they're not the same
    assert not np.allclose(yboth[0], yboth[1])


def test_griffinlim_cqt_multi(y_multi):
    y, sr = y_multi

    # Compute the stft
    C = librosa.cqt(y, sr=sr)

    # Run a couple of iterations of griffin-lim
    yout = librosa.griffinlim_cqt(np.abs(C), n_iter=2, length=y.shape[-1])

    # Check the lengths
    assert np.allclose(y.shape, yout.shape)


@pytest.mark.parametrize('rate', [0.5, 2])
def test_phase_vocoder(y_multi, rate):
    y, sr = y_multi
    D = librosa.stft(y)

    D0 = librosa.phase_vocoder(D[0], rate)
    D1 = librosa.phase_vocoder(D[1], rate)
    D2 = librosa.phase_vocoder(D, rate)

    assert np.allclose(D2[0], D0)
    assert np.allclose(D2[1], D1)
    assert not np.allclose(D2[0], D2[1])

