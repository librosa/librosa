#!/usr/bin/env python
'''
CREATED:2012-10-20 11:09:30 by Brian McFee <brm2132@columbia.edu>

Top-level class for librosa

Includes constants, core utility functions, etc

'''

import numpy, scipy

def hann_window(w, f):
    '''
    Construct a Hann window

    Inputs:
        w       = window size
        f       = frame size

    Output:
        window  = f-by-1 array containing Hann weights
    '''

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


def stft(x, n_fft=256, hann_w=None, hop=None, sample_rate=8000):
    '''
    Short-time fourier transform

    Inputs:
        x           = the input signal
        n_fft       = the number of FFT components  | default: 256
        hann_w      = size of hann window           | default: = n_fft
        hop         = hop length                    | default: = hann_w / 2
        sample_rate = sampling rate of x            | default: 8000

    Output:
        D           = complex-valued STFT matrix of x
    '''
    num_samples = len(x)

    if hann_w is None:
        hann_w = n_fft
        pass

    if hann_w == 0:
        window = numpy.ones((n_fft,))
    else:
        window = hann_window(hann_w, n_fft)
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


def istft(d, n_fft=None, hann_w=None, hop=None):
    '''
    Inverse short-time fourier transform

    Inputs:
        d       = STFT matrix
        n_fft   = number of FFT components          | default: 2 * (d.shape[0] -1
        hann_w  = size of hann window               | default: n_fft
        hop     = hop size                          | default: hann_w / 2

    Outputs:
        y       = time domain signal reconstructed from d
    '''
    num_frames = d.shape[1]

    if n_fft is None:
        n_fft = 2 * (d.shape[0] - 1)
        pass

    if hann_w is None:
        hann_w = n_fft
        pass

    if hann_window == 0:
        window = numpy.ones((n_fft,))
    else:
        # FIXME:   2012-10-20 18:58:56 by Brian McFee <brm2132@columbia.edu>
        #      there's a magic number 2/3 in istft.m ... not sure about this one
        window = hann_window(hann_w, n_fft) * 2.0 / 3
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
