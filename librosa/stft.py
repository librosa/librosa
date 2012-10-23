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



def stft(x, n_fft=256, hann_window=None, hop=None, sample_rate=8000):

    num_samples = len(x)

    if hann_window is None:
        hann_window = n_fft
        pass

    if hann_window == 0:
        window = numpy.ones((n_fft,))
    else:
        window = __make_hann_window(hann_window, n_fft)
        pass

    # Set the default hop, if it's not already specified
    if hop is None:
        hop = int(window.shape[0] / 2)
        pass

    # allocate output array
    D = numpy.zeros( (int(1 + n_fft / 2), 1 + int( ( num_samples - n_fft) / hop) ), dtype=numpy.complex)

    for (i, b) in enumerate(xrange(0, 1+num_samples - n_fft, hop)):
        u           = window * x[b:(b+n_fft)]
        t           = scipy.fft(u)
        D[:,i]      = t[:1+n_fft/2]
        pass

    return D


def istft(d, n_fft=None, hann_window=None, hop=None):
    

    num_frames = d.shape[1]

    if n_fft is None:
        n_fft = 2 * (d.shape[0] - 1)
        pass

    if hann_window is None:
        hann_window = n_fft
        pass

    if hann_window == 0:
        window = numpy.ones((n_fft,))
    else:
        # FIXME:   2012-10-20 18:58:56 by Brian McFee <brm2132@columbia.edu>
        #      there's a magic number 2/3 in istft.m ... not sure about this one
        window = __make_hann_window(hann_window, n_fft) * 2.0 / 3
        pass

    # Set the default hop, if it's not already specified
    if hop is None:
        hop = int(window.shape[0] / 2 )
        pass

    x_length    = n_fft + (num_frames - 1) * hop
    x           = numpy.zeros((x_length,))

    for b in xrange(0, hop * (num_frames), hop):
        ft              = d[:, b/hop]
        ft              = numpy.concatenate((ft, numpy.conj(ft[(n_fft/2 -1):0:-1])), 0)
        px              = numpy.real(scipy.ifft(ft))
        x[b:(b+n_fft)] += px * window
        pass

    return x
