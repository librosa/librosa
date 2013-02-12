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

