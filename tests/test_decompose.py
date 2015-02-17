#!/usr/bin/env python
# CREATED: 2013-10-06 22:31:29 by Dawen Liang <dl2771@columbia.edu>
# unit tests for librosa.decompose

# Disable cache
import os
try:
    os.environ.pop('LIBROSA_CACHE_DIR')
except:
    pass

import matplotlib
matplotlib.use('Agg')
import numpy as np
import librosa
import sklearn.decomposition


def test_default_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_given_decompose():

    D = sklearn.decomposition.NMF(random_state=0)

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, transformer=D)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_sorted_decompose():

    X = np.array([[1, 2, 3, 4, 5, 6], [1, 1, 1.2, 1, 0.8, 1]])

    (W, H) = librosa.decompose.decompose(X, sort=True, random_state=0)

    assert np.allclose(X, W.dot(H), rtol=1e-2, atol=1e-2)


def test_real_hpss():

    # Load an audio signal
    y, sr = librosa.load('data/test1_22050.wav')

    D = np.abs(librosa.stft(y))

    def __hpss_test(w, p, m):
        H, P = librosa.decompose.hpss(D, kernel_size=w, power=p, mask=m)

        if m:
            assert np.allclose(H + P, np.ones_like(D))
        else:
            assert np.allclose(H + P, D)

    for window in [31, (5, 5)]:
        for power in [0, 1, 2]:
            for mask in [False, True]:
                yield __hpss_test, window, power, mask


def test_complex_hpss():

    # Load an audio signal
    y, sr = librosa.load('data/test1_22050.wav')

    D = librosa.stft(y)

    H, P = librosa.decompose.hpss(D)

    assert np.allclose(H + P, D)
