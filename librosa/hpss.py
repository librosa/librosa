#!/usr/bin/env python
"""Harmonic-percussive source separation"""

import numpy as np
import scipy
import scipy.signal

def hpss(S, alpha=0.5, max_iter=50):
    """Harmonic-percussive source separation

    :parameters:
      - S : np.ndarray
            input spectrogram

      - alpha : float > 0
            balance parameter

      - max_iter : int
            maximum iteration bound

    :returns: 

      - harmonic : np.ndarray
            harmonic component

      - percussive : np.ndarray
            percussive component

      .. note:: harmonic + percussive = S

    .. note::

      - Ono, N., Miyamoto, K., Kameoka, H., & Sagayama, S.
        A real-time equalizer of harmonic and percussive components in music signals. 
        In ISMIR 2008 (pp. 139-144).

    """
    # Initialize H/P iterates
    harmonic    = S * 0.5
    percussive  = harmonic.copy()
    
    filt    = np.array([[0.25, -0.5, 0.25]])

    for _ in range(max_iter):
        # Compute delta
        Dh = scipy.signal.convolve2d(harmonic, filt, mode='same')
        Dp = scipy.signal.convolve2d(percussive, filt.T, mode='same')
        D  = alpha * Dh - (1-alpha) * Dp
        harmonic   = np.minimum(np.maximum(harmonic + D, 0.0), S)
        percussive = S - harmonic
    
    return (harmonic, percussive)


def hpss_median(S, win_P=9, win_H=9, p=0.0):
    """Median-filtering harmonic percussive separation

    :parameters:
      - S : np.ndarray
          input spectrogram

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

    # Compute median filters
    P = scipy.signal.medfilt2d(S, [win_P, 1])
    H = scipy.signal.medfilt2d(S, [1, win_H])

    if p == 0:
        Mh = (H > P).astype(float)
        Mp = 1 - Mh
    else:
        zP = P == 0
        P = P ** p
        P[zP] = 0.0
    
        zH = H == 0
        H = H ** p
        H[zH] = 0.0

        # Find points where both are zero, equalize
        H[zH & zP] = 0.5
        P[zH & zP] = 0.5

        # Compute harmonic mask
        Mh = H / (H + P)
        Mp = P / (H + P)

    return (Mh * S, Mp * S)

