#!/usr/bin/env python
'''
CREATED:2013-02-12 13:49:08 by Brian McFee <brm2132@columbia.edu>

Harmonic-percussive source separation

'''

import numpy, scipy, scipy.signal
import librosa

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
    H = S * 0.5
    P = H.copy()
    
    for t in range(max_iter):
        # Compute delta
        Dh = scipy.signal.convolve2d(H, numpy.array([[0.25, -.5, 0.25]]), mode='same')
        Dp = scipy.signal.convolve2d(P, numpy.array([[0.25], [-.5], [0.25]]), mode='same')
        D  = alpha * Dh - (1-alpha) * Dp
        H  = numpy.minimum(numpy.maximum(H + D, 0.0), S)
        P  = S - H
        pass
    
    return (H, P)

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
        z = P == 0
        P = P ** p
        P[z] = 0.0
    
        z = H == 0
        H = H ** p
        H[z] = 0.0
        # Compute harmonic mask
        Mh = H / (H + P)
        Mp = P / (H + P)
        pass

    return (Mh * S, Mp * S)
