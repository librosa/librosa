#!/usr/bin/env python
'''
CREATED:2013-02-12 13:49:08 by Brian McFee <brm2132@columbia.edu>

Harmonic-percussive source separation

'''

import numpy, scipy, scipy.signal

def hpss(S, alpha=0.5, max_iter=50):
    '''
    Harmonic-percussive source separation

    Ono, N., Miyamoto, K., Kameoka, H., & Sagayama, S. (2008, September). 
    A real-time equalizer of harmonic and percussive components in music signals. 
    In Proc. ISMIR (pp. 139-144).

    Input:
        S:          spectrogram
        alpha:      balance parameter           | default: 0.5
        max_iter:   maximum iteration bound     | default: 50

    Output:
        H:      harmonic component of S
        P:      percussive component of S

        Note: S = H + P
    '''
    # Initialize H/P iterates
    harmonic    = S * 0.5
    percussive  = harmonic.copy()
    
    filt    = numpy.array([[0.25, -0.5, 0.25]])

    for _ in range(max_iter):
        # Compute delta
        Dh = scipy.signal.convolve2d(harmonic, filt, mode='same')
        Dp = scipy.signal.convolve2d(percussive, filt.T, mode='same')
        D  = alpha * Dh - (1-alpha) * Dp
        harmonic   = numpy.minimum(numpy.maximum(harmonic + D, 0.0), S)
        percussive = S - harmonic
    
    return (harmonic, percussive)

def hpss_median(S, win_P=9, win_H=9, p=0.0):
    '''
    Median-filtering harmonic percussive separation

    Fitzgerald, D. (2010). 
    Harmonic/percussive separation using median filtering.

    Input:
        S:      spectrogram
        win_P:  window size for percussive median filtering     | default: 7
        win_H:  window size for harmonic median filtering       | default: 7
        p:      masking exponent                                | default: 0 (hard mask)
    '''

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
