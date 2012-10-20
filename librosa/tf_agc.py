#!/usr/bin/env python
'''
CREATED:2012-10-20 12:20:33 by Brian McFee <brm2132@columbia.edu>

Time-frequency automatic gain control

Ported from tf_agc.m by DPWE
    http://www.ee.columbia.edu/~dpwe/resources/matlab/tf_agc/ 
'''

import numpy
import scipy

## Cut-pasted stft code from stackoverflow
#   http://stackoverflow.com/questions/2459295/stft-and-istft-in-python 
#
# Probably redundant
# def stft(x, fs, framesz, hop):
#     framesamp = int(framesz*fs)
#     hopsamp = int(hop*fs)
#     w = scipy.hamming(framesamp)
#     X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
#                      for i in range(0, len(x)-framesamp, hopsamp)])
#     return X

# def istft(X, fs, T, hop):
#     x = scipy.zeros(T*fs)
#     framesamp = X.shape[1]
#     hopsamp = int(hop*fs)
#     for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
#         x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
#     return x

###
# Stolen from ronw's mfcc.py
#
def _hz_to_mel(f):
    return 2595.0 * numpy.log10(1 + f / 700.0)

def _mel_to_hz(z):
    return 700.0 * (10.0**(z / 2595.0) - 1.0)

def melfb(samplerate, nfft, nfilts=20, width=1.0, fmin=0, fmax=None):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    samplerate : int
        Sampling rate of the incoming signal.
    nfft : int
        FFT length to use.
    nwin : int
        Length of each window in samples.  Defaults to `nfft`.
    nhop : int
        Number of samples to skip between adjacent frames (hopsize).
        Defaults to `nwin`.
    winfun : function of the form fun(winlen), returns array of length winlen
        Function to generate a window of a given length.  Defaults to
        numpy.hamming.
    nmel : int
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

#     wts = numpy.zeros((nfilts, nfft / 2 + 1))
    wts = numpy.zeros((nfilts, nfft))
    # Center freqs of each FFT bin
#     fftfreqs = numpy.arange(nfft / 2 + 1, dtype=numpy.double) / nfft * samplerate
    fftfreqs = numpy.arange(wts.shape[1], dtype=numpy.double) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = _hz_to_mel(fmin)
    maxmel = _hz_to_mel(fmax)
    binfreqs = _mel_to_hz(minmel
                          + numpy.arange((nfilts+2), dtype=numpy.double) / (nfilts+1)
                          * (maxmel - minmel))

    for i in xrange(nfilts):
        freqs = binfreqs[i + numpy.arange(3)]
        # scale by width
        freqs = freqs[1] + width * (freqs - freqs[1])
        # lower and upper slopes for all bins
        loslope = (fftfreqs - freqs[0]) / (freqs[1] - freqs[0])
        hislope = (freqs[2] - fftfreqs) / (freqs[2] - freqs[1])
        # .. then intersect them with each other and zero
        wts[i,:] = numpy.maximum(0, numpy.minimum(loslope, hislope))
        pass

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm   = 2.0 / (binfreqs[2:nfilts+2] - binfreqs[:nfilts])
    wts     = numpy.dot(numpy.diag(enorm), wts)
    
    return wts


def tf_agc(frameIterator, **kwargs):
    '''
    frameIterator               iterates over audio frames, duhr
                                information we need from the audio buffer:
                                    ???

    Optional arguments:
        frequency_scale     (f_scale from dpwe)     (default: 1.0)
        time_scale          (t_scale from dwpe)     (default: 0.5 sec) 
                                                    XXX redo in terms of frame count
        gaussian_smoothing  (type)                  (default: False)
                                                    Use time-symmetric, non-causal Gaussian window smoothing        XXX: not supported
                                                    Otherwise, defaults to infinite-attack, exponential release
    '''

    ##
    # Parse arguments
    frequency_scale = 1.0
    if 'frequency_scale' in kwargs:
        frequency_scale = kwargs['frequency_scale']
        if not isinstance(frequency_scale, float):
            raise TypeError('Argument frequency_scale must be of type float')
        if frequency_scale <= 0.0:
            raise ValueError('Argument frequency_scale must be a non-negative float')
        pass
    

    time_scale      = 0.5
    if 'time_scale'     in kwargs:
        time_scale      = kwargs['time_scale']
        if not isinstance(time_scale, float):
            raise TypeError('Argument time_scale must be of type float')
        if time_scale <= 0.0:
            raise ValueError('Argument time_scale must be a non-negative float')
        pass

    gaussian_smoothing = False
    if 'gaussian_smoothing' in kwargs:
        gaussian_smoothing = kwargs['gaussian_smoothing']
        if not isinstance(gaussian_smoothing, bool):
            raise TypeError('Argument gaussian_smoothing must be of type bool')
        if gaussian_smoothing:
            raise ValueError('Gaussian smoothing currently unsupported')
        pass


    # Setup

    # How many frequency bands do we need?
    num_frequency_bands = max(10, round(20 / frequency_scale))
    mel_filter_width    = round(frequency_scale  * num_frequency_bands / 10);


    # Iterate over frames
    for frame in frameIterator:
        if gaussian_smoothing:
            pass
        else:

            pass
    pass
