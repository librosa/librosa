#!/usr/bin/env python
'''
CREATED:2012-10-20 18:27:18 by Brian McFee <brm2132@columbia.edu>
 
Porting over STFT/ISTFT from dpwe's code

Surely this is redundant, but who cares
'''

import numpy, scipy

def __make_hann_window(w, f):
    # force window length to be odd
    if w % 2 == 0:
        w += 1
        pass

    half_length = (w - 1) / 2
    half_f      = f/2
    half_window = 0.5 * (1 + numpy.cos(numpy.pi / half_length * numpy.arange(half_length)))

    window                                  = numpy.zeros((f,))
    acthalflen                              = numpy.minimum(half_f, half_length)
    window[half_f:(half_f+acthalflen)]      = half_window[:acthalflen]
    window[half_f:(half_f-acthalflen):-1]   = half_window[:acthalflen]

    return window



def stft(x, n_fft=256, hann_window=None, hop_size=None, sample_rate=8000):

    if hann_window is None:
        hann_window = n_fft
        pass

    
    num_samples = len(x)

    if hann_window == 0:
        window = numpy.ones((n_fft,))
        pass
    elif hann_window > 0:
        window = __make_hann_window(hann_window, n_fft)
        pass

    # Set the default hop_size, if it's not already specified
    if hop_size is None:
        hop_size = int(window.shape[0] / 2)
        pass

    # allocate output array
    D = numpy.zeros( (int(1 + n_fft / 2), 1 + int( ( num_samples - n_fft) / hop_size) ), dtype=numpy.complex)

    for (i, b) in enumerate(xrange(0, num_samples - n_fft, hop_size)):
        u           = window * x[b:(b+n_fft)]
        t           = scipy.fft(u)
        D[:,i]      = t[:1+n_fft/2]
        pass

    return D
