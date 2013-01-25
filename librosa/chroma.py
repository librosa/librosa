#!/usr/bin/env python

import librosa
import numpy

# FIXME:  2013-01-25 09:39:26 by Brian McFee <brm2132@columbia.edu>
# this needs to be much more efficient
#   framevector should be a frame array, or spectrogram, not a single frame
def chroma(framevector, sr, nchroma=12, A440=440.0, ctroct=5.0, octwidth=0, order=None):
    '''
    Extract chroma from an audio frame

    Input:


    '''
    nfft        = len(framevector)
    F           = numpy.abs(numpy.fft.fft(framevector))
    fft2chmx    = chromafb(sr, nfft, nchroma, A440, ctroct, octwidth)

    # this is unnormalized chroma
    unchroma = numpy.dot(fft2chmx, F[:nfft/2 + 1])
    return unchroma / numpy.linalg.norm(unchroma, order)

def chromafb(sr, nfft, nchroma, A440=440.0, ctroct=5.0, octwidth=0):
    """Create a Filterbank matrix to convert FFT to Chroma.

    Based on Dan Ellis's fft2chromamx.m

    Parameters
    ----------
    sr: int
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
    wts = numpy.zeros((nchroma, nfft))

    fftfrqbins = nchroma * librosa.hz2octs(numpy.arange(1, nfft, dtype='d') / nfft
                                    * sr, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = numpy.concatenate(([fftfrqbins[0] - 1.5 * nchroma], fftfrqbins))

    binwidthbins = numpy.concatenate(
        (numpy.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = numpy.tile(fftfrqbins, (nchroma,1))  \
        - numpy.tile(numpy.arange(0, nchroma, dtype='d')[:,numpy.newaxis], (1,nfft))

    nchroma2 = round(nchroma / 2.0);

    # Project into range -nchroma/2 .. nchroma/2
    # add on fixed offset of 10*nchroma to ensure all values passed to
    # rem are +ve
    D = numpy.remainder(D + nchroma2 + 10*nchroma, nchroma) - nchroma2;

    # Gaussian bumps - 2*D to make them narrower
    wts = numpy.exp(-0.5 * (2*D / numpy.tile(binwidthbins, (nchroma,1)))**2)

    # normalize each column
    wts /= numpy.tile(numpy.sqrt(numpy.sum(wts**2, 0)), (nchroma,1))

    # Maybe apply scaling for fft bins
    if octwidth > 0:
        wts *= numpy.tile(
            numpy.exp(-0.5 * (((fftfrqbins/nchroma - ctroct)/octwidth)**2)),
            (nchroma, 1))

    # remove aliasing columns
    return wts[:,:nfft/2+1]

