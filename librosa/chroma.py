#!/usr/bin/env python

import librosa
import numpy


def chromagram(S, sr, nchroma=12, A440=440.0, ctroct=5.0, octwidth=0, norm='inf'):

    nfft        = (S.shape[0] -1 ) * 2.0

    spec2chroma = chromafb(sr, nfft, nchroma, A440=A440, ctroct=ctroct, octwidth=octwidth)

    # Compute raw chroma
    U           = numpy.dot(spec2chroma, S)

    # Compute normalization factor for each frame
    if norm == 'inf':
        Z       = numpy.max(numpy.abs(U), axis=0)
    elif norm == 1:
        Z       = numpy.sum(numpy.abs(U), axis=0)
    elif norm == 2:
        Z       = numpy.sum( (U**2), axis=0) ** 0.5
    else:
        raise ValueError("norm must be one of: 'inf', 1, 2")

    # Tile the normalizer to match U's shape
    Z[Z == 0] = 1.0
    Z   = numpy.tile(1.0/Z, (U.shape[0], 1))

    return Z * U


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

    fftfrqbins = nchroma * librosa.hz_to_octs(numpy.arange(1, nfft, dtype='d') / nfft
                                    * sr, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = numpy.concatenate(([fftfrqbins[0] - 1.5 * nchroma], fftfrqbins))

    binwidthbins = numpy.concatenate(
        (numpy.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = numpy.tile(fftfrqbins, (nchroma, 1))  \
        - numpy.tile(numpy.arange(0, nchroma, dtype='d')[:,numpy.newaxis], (1,nfft))

    nchroma2 = round(nchroma / 2.0);

    # Project into range -nchroma/2 .. nchroma/2
    # add on fixed offset of 10*nchroma to ensure all values passed to
    # rem are +ve
    D = numpy.remainder(D + nchroma2 + 10*nchroma, nchroma) - nchroma2;

    # Gaussian bumps - 2*D to make them narrower
    wts = numpy.exp(-0.5 * (2*D / numpy.tile(binwidthbins, (nchroma, 1)))**2)

    # normalize each column
    wts /= numpy.tile(numpy.sqrt(numpy.sum(wts**2, 0)), (nchroma, 1))

    # Maybe apply scaling for fft bins
    if octwidth > 0:
        wts *= numpy.tile(
            numpy.exp(-0.5 * (((fftfrqbins/nchroma - ctroct)/octwidth)**2)),
            (nchroma, 1))

    # remove aliasing columns
    return wts[:, :(nfft/2+1)]

