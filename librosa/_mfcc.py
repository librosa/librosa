#!/usr/bin/env python
'''
Frame-level mel-frequency cepstrum coefficients

Ported from Ron's Python implementation, which is ported from DPWE's Matlab
http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/
'''
import numpy as np
import librosa

def mfcc(framevector, samplerate, winfun=np.hamming, nmel=20, width=1.0, fmin=0, fmax=None):
    '''Given a frame of arbitrary length and sample rate, compute the MFCCs

    framevector: 1 * N numpy array
        The audio frame to be used.
    samplerate: int
        Sampling rate of the incoming signal.
    winfun: func
        The windowing function applying to the input frame, Hamming window by
        default.
    nmel: int
        The number of mel bins.
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
    fft2melmx = librosa.melfb(samplerate, nfft, nmel, width, fmin, fmax)
    # hope the dimension not messed up
    return np.dot(fft2melmx, F) 
	
