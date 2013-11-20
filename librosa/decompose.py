#!/usr/bin/env python
"""Spectrogram decomposition"""

import numpy as np
import scipy
import scipy.signal

import sklearn.decomposition

import librosa.core


def decompose(X, n_components=None, transformer=None):
    """Decompose the feature matrix with non-negative matrix factorization

    :parameters:
        - X : np.ndarray
            feature matrix (d-by-t)
        - n_components : int > 0 or None
            number of components, if None then all d components are used
        - transformer : any instance which implements fit_transform()
            If None, use sklearn.decomposition.NMF by default
            Otherwise, because of scikit-learn convention where the input data
            is (n_samples, n_features), NMF.fit_transform() should take X.T as
            input, and returns transformed X_new, where:
            ``X.T ~= X_new.dot(transformer.components_)``
            or equivalently:
            ``X ~= transformer.components_.T.dot(X_new.T)``

    :returns:
        - components: np.ndarray
            dictionary matrix (d-by-n_components)
        - X_new: np.ndarray
            transformed matrix/activation matrix (n_components-by-t)

    """

    if transformer is None:
        transformer = sklearn.decomposition.NMF(n_components=n_components)
    X_new = transformer.fit_transform(X.T)
    return (transformer.components_.T, X_new.T)

def hpss(S, win_P=19, win_H=19, p=1.0):
    """Median-filtering harmonic percussive separation

    :parameters:
      - S : np.ndarray
          input spectrogram. May be real (magnitude) or complex.

      - win_P : int        
          window size for percussive filter

      - win_H : int
          window size for harmonic filter 

      - p : float
          masking exponent

    :returns:
      - harmonic : np.ndarray
          harmonic component

      - percussive : np.ndarray
          percussive component

      .. note:: harmonic + percussive = S

    .. note::
      - Fitzgerald, D. (2010). 
        Harmonic/percussive separation using median filtering.

    """

    if np.iscomplex(S).any():
        S, phase = librosa.core.magphase(S)
    else:
        phase = 1

    # Compute median filters
    P = scipy.signal.medfilt2d(S, [win_P, 1])
    H = scipy.signal.medfilt2d(S, [1, win_H])

    if p == 0:
        Mh = (H > P).astype(float)
        Mp = 1 - Mh
    else:
        zP = (P == 0)
        P = P ** p
        P[zP] = 0.0
    
        zH = (H == 0)
        H = H ** p
        H[zH] = 0.0

        # Find points where both are zero, equalize
        H[zH & zP] = 0.5
        P[zH & zP] = 0.5

        # Compute harmonic mask
        Mh = H / (H + P)
        Mp = P / (H + P)

    return (Mh * S * phase, Mp * S * phase)

