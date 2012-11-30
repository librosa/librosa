#!/usr/bin/env python
'''
CREATED:2012-10-20 11:09:30 by Brian McFee <brm2132@columbia.edu>

Top-level class for librosa

Includes constants, core utility functions, etc

'''

import numpy, scipy
import beat, framegenerator, _chroma, _mfcc, tf_agc
import audioread

def load(path, mono=True, frame_size=1024):
    '''
    Load an audio file into a single, long time series

    Input:
        path:       path to the input file
        mono:       convert to mono?        | Default: True
        frame_size: buffer size             | Default: 1024 samples
    Output:
        y:          the time series
        sr:         the sampling rate
    '''

    with audioread.audio_open(path) as f:
        sr  = f.samplerate
        y   = numpy.concatenate([frame for frame in framegenerator.audioread_timeseries(f, frame_size)], axis=0)
        pass

    return (y, sr)

def pad(w, d_pad, v=0.0, center=True):
    '''
    Pad a vector w out to d dimensions, using value v

    if center is True, w will be centered in the output vector
    otherwise, w will be at the beginning
    '''
    # FIXME:  2012-11-27 11:08:54 by Brian McFee <brm2132@columbia.edu>
    #  This function will be deprecated by numpy 1.7.0    

    d = len(w)
    if d > d_pad:
        raise ValueError('Insufficient pad space')

    q = v * numpy.ones(d_pad)
    q[:d] = w

    if center:
        q = numpy.roll(q, numpy.floor((d_pad - d) / 2.0).astype(int), axis=0)
        pass
    return q



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
        window = pad(scipy.signal.hanning(hann_w), n_fft)
        pass

    # Set the default hop, if it's not already specified
    if hop is None:
        hop = int(window.shape[0] / 2.0)
        pass

    # allocate output array
    D = numpy.zeros( (int(1 + n_fft / 2.0), 1 + int( ( num_samples - n_fft) / hop) ), dtype=numpy.complex)

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

    if hann_w == 0:
        window = numpy.ones((n_fft,))
    else:
        # FIXME:   2012-10-20 18:58:56 by Brian McFee <brm2132@columbia.edu>
        #      there's a magic number 2/3 in istft.m ... not sure about this one
        window = pad(scipy.signal.hanning(hann_w) * 2.0 / 3, n_fft)
        pass

    # Set the default hop, if it's not already specified
    if hop is None:
        hop = int(window.shape[0] / 2.0 )
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
def hz_to_mel(f, htk=False):
    #     TODO:   2012-11-27 11:28:43 by Brian McFee <brm2132@columbia.edu>
    #  too many magic numbers in these functions
    #   redo with informative variable names
    #   then make them into parameters

    if numpy.isscalar(f):
        f = numpy.array([f],dtype=float)
        pass
    if htk:
        return 2595.0 * numpy.log10(1.0 + f / 700.0)
    else:
        f           = f.astype(float)
        # Oppan Slaney style
        f_0         = 0.0
        f_sp        = 200.0 / 3
        brkfrq      = 1000.0
        brkpt       = (brkfrq - f_0) / f_sp
        logstep     = numpy.exp(numpy.log(6.4) / 27.0)
        linpts      = f < brkfrq

        nlinpts     = numpy.invert(linpts)

        z           = numpy.zeros_like(f)
        # Fill in parts separately
        z[linpts]   = (f[linpts] - f_0) / f_sp
        z[nlinpts]  = brkpt + numpy.log(f[nlinpts] / brkfrq) / numpy.log(logstep)
        return z
    pass

def mel_to_hz(z, htk=False):
    if numpy.isscalar(z):
        z = numpy.array([z], dtype=float)
        pass
    if htk:
        return 700.0 * (10.0**(z / 2595.0) - 1.0)
    else:
        z           = z.astype(float)
        f_0         = 0.0
        f_sp        = 200.0 / 3
        brkfrq      = 1000
        brkpt       = (brkfrq - f_0) / f_sp
        logstep     = numpy.exp(numpy.log(6.4) / 27.0)
        f           = numpy.zeros_like(z)
        linpts      = z < brkpt
        nlinpts     = numpy.invert(linpts)

        f[linpts]   = f_0 + f_sp * z[linpts]
        f[nlinpts]  = brkfrq * numpy.exp(numpy.log(logstep) * (z[nlinpts]-brkpt))
        return f
    pass


def dctfb(nfilts, d):
    '''
    Build a discrete cosine transform basis

    Input:
        nfilts  :       number of output components
        d       :       number of input components

    Output:
        D       :       nfilts-by-d DCT matrix
    '''
    DCT = numpy.empty((nfilts, d))

    q = numpy.arange(1, 2*d, 2) * numpy.pi / (2.0 * d)
    DCT[0,:] = 1.0 / numpy.sqrt(d)
    for i in xrange(1,nfilts):
        DCT[i,:] = numpy.cos(i*q) * numpy.sqrt(2.0/d)
        pass

    return DCT 


def mfcc(S, d=20):
    '''
    Mel-frequency cepstral coefficients

    Input:
        S   :   k-by-n      log-amplitude spectrogram
        d   :   number of MFCCs to return               | default: 20
    Output:
        M   :   d-by-n      MFCC sequence
    '''

    return numpy.dot(dctfb(d, S.shape[0]), S)

# Stolen from ronw's mfcc.py
# https://github.com/ronw/frontend/blob/master/mfcc.py
def melfb(samplerate, nfft, nfilts=40, width=1.0, fmin=None, fmax=None, use_htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    samplerate : int
        Sampling rate of the incoming signal.
    nfft : int
        FFT length to use.
    nfilts : int
        Number of Mel bands to use.  Defaults to 40.
    width : float
        The constant width of each band relative to standard Mel. Defaults 1.0
    fmin : float
        Frequency in Hz of the lowest edge of the Mel bands. Defaults to 0.
    fmax : float
        Frequency in Hz of the upper edge of the Mel bands. Defaults
        to `samplerate` / 2.
    use_htk: bool
        Use HTK mels instead of Slaney's version? Defaults to false.

    See Also
    --------
    Filterbank
    MelSpec
    """

    if fmin is None:
        fmin = 0
        pass

    if fmax is None:
        fmax = samplerate / 2.0
        pass

    # Initialize the weights
    wts         = numpy.zeros( (nfilts, nfft) )

    # Center freqs of each FFT bin
    fftfreqs    = numpy.arange( wts.shape[1], dtype=numpy.double ) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel      = hz_to_mel(fmin, htk=use_htk)
    maxmel      = hz_to_mel(fmax, htk=use_htk)
    binfreqs    = mel_to_hz(numpy.arange(minmel, minmel + nfilts + 2, dtype=float)  * (maxmel - minmel) / (nfilts+1.0), htk=use_htk)

    for i in xrange(nfilts):
        freqs       = binfreqs[range(i, i+3)]
        
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

def melspectrogram(y, sampling_rate=8000, window_length=256, hop_length=32, mel_channels=40, htk=False, width=3.0):
    '''
    Compute a mel spectrogram from a time series

    Input:
        y                   =   the audio signal
        sampling_rate       =   the sampling rate of y                      | default: 8000
        window_length       =   FFT window size                             | default: 256
        hop_length          =   hop size                                    | default: 32
        mel_channels        =   number of Mel filters to use                | default: 40
        htk                 =   use HTK mels instead of Slaney              | default: False
        width               =   width of mel bins                           | default: 3

    Output:
        S                   =   Mel amplitude spectrogram
    '''

    # Compute the STFT
    S = stft(y, window_length, window_length, hop_length)

    # Build a Mel filter
    M = melfb(sampling_rate, window_length / 2 + 1, nfilts=mel_channels, width=width, use_htk=htk)
    
    S = numpy.dot(M, numpy.abs(S))

    return S

def logamplitude(S, amin=1e-10, gain_threshold=-80.0):
    '''
    Log-scale an ampltitude spectrogram

    Input:
        S                   =   the input spectrogram
        amin                =   minimum allowed amplitude                   | default: 1e-10
        gain_threshold      =   minimum output value                        | default: -80 (None to disable)
    Output:
        D                   =   S in dBs
    '''

    SCALE   =   20.0
    D       =   SCALE * numpy.log10(numpy.maximum(amin, S))

    if gain_threshold is not None:
        D[D < gain_threshold] = gain_threshold
        pass
    return D

def localmax(x):
    '''
        Return 1 where there are local maxima in x (column-wise)
        left edges do not fire, right edges might.
    '''

    return numpy.logical_and(x > numpy.hstack([x[0], x[:-1]]), x >= numpy.hstack([x[1:], x[-1]]))


def autocorrelate(x, max_size):
    #   TODO:   2012-11-07 14:05:42 by Brian McFee <brm2132@columbia.edu>
    #  maybe could be done faster by directly implementing a clipped correlate
#     result = numpy.correlate(x, x, mode='full')
    result = scipy.signal.fftconvolve(x, x[::-1], mode='full')

    result = result[len(result)/2:]
    if max_size is None:
        return result
    return result[:max_size]
