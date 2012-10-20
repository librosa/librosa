#!/usr/bin/env python
'''
Frame-level mel-frequency cepstrum coefficients

Ported from Ron's Python implementation, which is ported from DPWE's Matlab
http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/
'''
import numpy as np

def mfcc(framevector, samplerate, winfun=np.hamming, nmel=20, width=1.0, fmin=0, fmax=None):
    '''Given a frame of arbitrary length and sample rate, compute the MFCCs

    framevector: 1 * N numpy array
        The audio frame
    samplerate: int
        Sampling rate of the incoming signal
    winfun: func
        The windowing function applying to the input frame, Hamming window by
        default
    nmel: int
        The number of mel bins
    width : float
        The constant width of each band relative to standard Mel. Defaults 1.0.
    fmin : float
        Frequency in Hz of the lowest edge of the Mel bands. Defaults to 0.
    fmax : float
        Frequency in Hz of the upper edge of the Mel bands. Defaults
        to `samplerate` / 2.
    '''
    nfft = len(framevector)
    F = np.abs(np.fft.fft(framevector * winfun(nfft)))
    # transfermation matrix from FFT bin to mel bin
    fft2melmx = melfb(samplerate, nfft, nmel, width, fmin, fmax)
    # hope the dimension not messed up
    return np.dot(fft2melmx, F) 
	
# Stolen from ronw's mfcc.py
# https://github.com/ronw/frontend/blob/master/mfcc.py
def _hz_to_mel(f):
    return 2595.0 * np.log10(1 + f / 700.0)

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
#     wts = np.zeros((nfilts, nfft / 2 + 1))
    wts = np.zeros( (nfilts, nfft) )

    # Center freqs of each FFT bin
#     fftfreqs = np.arange(nfft / 2 + 1, dtype=np.double) / nfft * samplerate
    fftfreqs = np.arange( wts.shape[1], dtype=np.double ) / nfft * samplerate

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel      = _hz_to_mel(fmin)
    maxmel      = _hz_to_mel(fmax)
    binfreqs    = _mel_to_hz(minmel + np.arange((nfilts+2), dtype=np.double) / (nfilts+1) * (maxmel - minmel))

    for i in xrange(nfilts):
        freqs       = binfreqs[i + np.arange(3)]
        
        # scale by width
        freqs       = freqs[1] + width * (freqs - freqs[1])

        # lower and upper slopes for all bins
        loslope     = (fftfreqs - freqs[0]) / (freqs[1] - freqs[0])
        hislope     = (freqs[2] - fftfreqs) / (freqs[2] - freqs[1])

        # .. then intersect them with each other and zero
        wts[i,:]    = np.maximum(0, np.minimum(loslope, hislope))

        pass

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm   = 2.0 / (binfreqs[2:nfilts+2] - binfreqs[:nfilts])
    wts     = np.dot(np.diag(enorm), wts)
    
    return wts

