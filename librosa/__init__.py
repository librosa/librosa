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

# Dead-simple mel spectrum conversion
def hz_to_mel(f):
    return 2595.0 * numpy.log10(1.0 + f / 700.0)

def mel_to_hz(z):
    return 700.0 * (10.0**(z / 2595.0) - 1.0)

# Stolen from ronw's mfcc.py
# https://github.com/ronw/frontend/blob/master/mfcc.py

def melfb(samplerate, nfft, nfilts=20, width=1.0, fmin=0, fmax=None):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    samplerate : int
        Sampling rate of the incoming signal.
    nfft : int
        FFT length to use.
    nfilts : int
        Number of Mel bands to use.
    width : float
        The constant width of each band relative to standard Mel. Defaults 1.0.
    fmin : float
        Frequency in Hz of the lowest edge of the Mel bands. Defaults to 0.
    fmax : float
        Frequency in Hz of the upper edge of the Mel bands. Defaults
        to `samplerate` / 2.

    See Also
    --------
    Filterbank
    MelSpec
    """

    if fmax is None:
        fmax = samplerate / 2
        pass

    # Initialize the weights
#     wts = numpy.zeros((nfilts, nfft / 2 + 1))
    wts         = numpy.zeros( (nfilts, nfft) )

    # Center freqs of each FFT bin
#     fftfreqs = numpy.arange(nfft / 2 + 1, dtype=numpy.double) / nfft * samplerate
    fftfreqs    = numpy.arange( wts.shape[1], dtype=numpy.double ) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel      = hz_to_mel(fmin)
    maxmel      = hz_to_mel(fmax)
    binfreqs    = mel_to_hz(minmel + numpy.arange((nfilts+2), dtype=numpy.double) / (nfilts+1) * (maxmel - minmel))

    for i in xrange(nfilts):
        freqs       = binfreqs[i + numpy.arange(3)]
        
        # scale by width
        freqs       = freqs[1] + width * (freqs - freqs[1])

        # lower and upper slopes for all bins
        loslope     = (fftfreqs - freqs[0]) / (freqs[1] - freqs[0])
        hislope     = (freqs[2] - fftfreqs) / (freqs[2] - freqs[1])

        # .. then intersect them with each other and zero
        wts[i,:]    = numpy.maximum(0, numpy.minimum(loslope, hislope))

        pass

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm   = 2.0 / (binfreqs[2:nfilts+2] - binfreqs[:nfilts])
    wts     = numpy.dot(numpy.diag(enorm), wts)
    
    return wts


