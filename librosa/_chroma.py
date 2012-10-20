#!/usr/bin/env python

import numpy as np

def chroma(framevector, samplerate, nchroma=12, A440=440.0, ctroct=5.0,
        octwidth=0, order=None):
    nfft = len(framevector)
    F = np.abs(np.fft.fft(framevector))
    fft2chmx = chromafb(samplerate, nfft, nchroma, A440, ctroct, octwidth)
# this is unnormalized chroma
    unchroma = np.dot(fft2chmx, F[:nfft/2 + 1])
    return unchroma / np.linalg.norm(unchroma, order)

# Stolen from ronw's chroma.py
# https://github.com/ronw/frontend/blob/master/chroma.py
def _hz2octs(freq, A440):
    return np.log2(freq / (A440 / 16.0))

def chromafb(samplerate, nfft, nchroma, A440=440.0, ctroct=5.0, octwidth=0):
    """Create a Filterbank matrix to convert FFT to Chroma.

    Based on Dan Ellis's fft2chromamx.m

    Parameters
    ----------
    samplerate : int
        Sampling rate of the incoming signal.
    nfft : int
        FFT length to use.
    nchroma : int
        Number of chroma dimensions to return (number of bins per octave).
    A440 : float
        Reference frequency in Hz for A.  Defaults to 440.
    ctroct, octwidth : float
        These parameters specify a dominance window - Gaussian
        weighting centered on ctroct (in octs, re A0 = 27.5Hz) and
        with a gaussian half-width of octwidth.  Defaults to
        halfwidth = inf i.e. flat.
    """
    wts = np.zeros((nchroma, nfft))

    fftfrqbins = nchroma * _hz2octs(np.arange(1, nfft, dtype='d') / nfft
                                    * samplerate, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = np.concatenate(([fftfrqbins[0] - 1.5 * nchroma], fftfrqbins))

    binwidthbins = np.concatenate(
        (np.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = np.tile(fftfrqbins, (nchroma,1))  \
        - np.tile(np.arange(0, nchroma, dtype='d')[:,np.newaxis], (1,nfft))

    nchroma2 = round(nchroma / 2.0);

    # Project into range -nchroma/2 .. nchroma/2
    # add on fixed offset of 10*nchroma to ensure all values passed to
    # rem are +ve
    D = np.remainder(D + nchroma2 + 10*nchroma, nchroma) - nchroma2;

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (nchroma,1)))**2)

    # normalize each column
    wts /= np.tile(np.sqrt(np.sum(wts**2, 0)), (nchroma,1))

    # Maybe apply scaling for fft bins
    if octwidth > 0:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/nchroma - ctroct)/octwidth)**2)),
            (nchroma, 1))

    # remove aliasing columns
    return wts[:,:nfft/2+1]

