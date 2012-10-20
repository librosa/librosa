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

    # Initialize the weights
#     wts = numpy.zeros((nfilts, nfft / 2 + 1))
    wts = numpy.zeros( (nfilts, nfft) )

    # Center freqs of each FFT bin
#     fftfreqs = numpy.arange(nfft / 2 + 1, dtype=numpy.double) / nfft * samplerate
    fftfreqs = numpy.arange( wts.shape[1], dtype=numpy.double ) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel      = _hz_to_mel(fmin)
    maxmel      = _hz_to_mel(fmax)
    binfreqs    = _mel_to_hz(minmel + numpy.arange((nfilts+2), dtype=numpy.double) / (nfilts+1) * (maxmel - minmel))

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


def tf_agc(frame_iterator, sample_rate=22050, **kwargs):
    '''
    frame_iterator              iterates over audio frames, duhr
    sample_rate                 sampling rate of the  audio stream

    Optional arguments:
        frequency_scale     (f_scale from dpwe)     (default: 1.0)
        alpha                                       (default: 0.95 ) 
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
    

    alpha      = 0.95
    if 'alpha'     in kwargs:
        alpha      = kwargs['alpha']
        if not isinstance(alpha, float):
            raise TypeError('Argument alpha must be of type float')
        if not (0.0 < alpha < 1.0):
            raise ValueError('Argument alpha must be between 0.0 and 1.0')
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
    num_frequency_bands = max(10, int(20 / frequency_scale))
    mel_filter_width    = int(frequency_scale  * num_frequency_bands / 10);

    # initialize the mel filterbank to None; 
    # do the real initialization after grabbing the first frame
    f2a = None

    if gaussian_smoothing:
        pass
    else:
        # Else, use infinite-attack exp-release


        # Iterate over frames
        for frame in frame_iterator:
            frame = numpy.frombuffer(frame, 'h')

            if f2a is None: 
                # initialize the mel filter bank after grabbing the first frame

                f2a = melfb(sample_rate, len(frame), num_frequency_bands, mel_filter_width)
#                 f2a = f2a[:,:(round(len(frame)/2) + 1)]
                

                #% map back to FFT grid, flatten bark loop gain
                #sf2a = sum(f2a);

                normalize_f2a                       = numpy.sum(f2a, axis=0)
                normalize_f2a[normalize_f2a == 0]   = 1.0
                normalize_f2a                       = 1.0 / normalize_f2a

                # initialze the state vector
                state   = numpy.zeros( (num_frequency_bands, 1) )[0]

                pass

            # FFT each frame
            D = scipy.fft(frame)

            # multiply by f2a
            audiogram = numpy.dot(f2a, numpy.abs(D))

            ## DPWE
            #             state = max([alpha*state,audgram(:,i)],[],2);
            #             fbg(:,i) = state;
            # ...
            #
            state = numpy.maximum(alpha * state, audiogram)

            #E = diag(1./(sf2a+(sf2a==0))) * f2a' * fbg;
            E   = normalize_f2a * numpy.dot(f2a.T, state);

            #% Remove any zeros in E (shouldn't be any, but who knows?)
            #E(E(:)<=0) = min(E(E(:)>0));

            #% invert back to waveform
            #y = istft(D./E);

            y = scipy.ifft(D/E)

            pass
        pass

    pass
