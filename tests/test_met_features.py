#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# CREATED:2015-02-16 13:10:05 by Brian McFee <brian.mcfee@nyu.edu>
"""Regression tests on metlab features"""

# Disable cache
import os
import pytest

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import numpy as np
import scipy.io
import scipy.signal

from test_core import load, files

import librosa

__EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")


def met_stft(y, n_fft, hop_length, win_length, normalize):

    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=scipy.signal.windows.hamming,
            center=False,
        )
    )

    if normalize:
        S = S / (S[0] + np.sum(2 * S[1:], axis=0))

    return S


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "met-centroid-*.mat"))
)
def test_spectral_centroid(infile):
    DATA = load(infile)

    y, sr = librosa.load(os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True)

    n_fft = DATA["nfft"][0, 0].astype(int)
    hop_length = DATA["hop_length"][0, 0].astype(int)

    # spectralCentroid uses normalized spectra
    S = met_stft(y, n_fft, hop_length, n_fft, True)

    centroid = librosa.feature.spectral_centroid(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    assert np.allclose(centroid, DATA["centroid"])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "met-contrast-*.mat"))
)
def test_spectral_contrast(infile):
    DATA = load(infile)

    y, sr = librosa.load(os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True)

    n_fft = DATA["nfft"][0, 0].astype(int)
    hop_length = DATA["hop_length"][0, 0].astype(int)

    # spectralContrast uses normalized spectra
    S = met_stft(y, n_fft, hop_length, n_fft, True)

    contrast = librosa.feature.spectral_contrast(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, linear=True
    )

    assert np.allclose(contrast, DATA["contrast"], rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "met-rolloff-*.mat"))
)
def test_spectral_rolloff(infile):
    DATA = load(infile)

    y, sr = librosa.load(os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True)

    n_fft = DATA["nfft"][0, 0].astype(int)
    hop_length = DATA["hop_length"][0, 0].astype(int)
    pct = DATA["pct"][0, 0]

    # spectralRolloff uses normalized spectra
    S = met_stft(y, n_fft, hop_length, n_fft, True)

    rolloff = librosa.feature.spectral_rolloff(
        S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=pct
    )

    assert np.allclose(rolloff, DATA["rolloff"])


@pytest.mark.parametrize(
    "infile", files(os.path.join("tests", "data", "met-bandwidth-*.mat"))
)
def test_spectral_bandwidth(infile):
    DATA = load(infile)

    y, sr = librosa.load(os.path.join("tests", DATA["wavfile"][0]), sr=None, mono=True)

    n_fft = DATA["nfft"][0, 0].astype(int)
    hop_length = DATA["hop_length"][0, 0].astype(int)

    S = DATA["S"]

    # normalization is disabled here, since the precomputed S is already
    # normalized
    # metlab uses p=1, other folks use p=2
    bw = librosa.feature.spectral_bandwidth(
        S=S,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        centroid=DATA["centroid"],
        norm=False,
        p=1,
    )

    # METlab implementation takes the mean, not the sum
    assert np.allclose(bw, S.shape[0] * DATA["bw"])
