#!/usr/bin/env python
"""Commonly used filter banks: DCT, Chroma, Mel..."""

import numpy as np
import librosa.core

def dct(n_filts, d):
    """Discrete cosine transform basis

    :parameters:
      - n_filts   : int
          number of output components
      - d         : int
          number of input components

    :returns:
      - D         : np.ndarray, shape=(n_filts, d)
          DCT basis vectors

    """

    basis       = np.empty((n_filts, d))
    basis[0, :] = 1.0 / np.sqrt(d)

    samples     = np.arange(1, 2*d, 2) * np.pi / (2.0 * d)

    for i in xrange(1, n_filts):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/d)

    return basis

def chroma(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=None):
    """Create a Filterbank matrix to convert STFT to chroma

    :parameters:
      - sr        : int
          sampling rate
      - n_fft     : int
          number of FFT components
      - n_chroma  : int
          number of chroma dimensions   
      - A440      : float
          Reference frequency for A
      - ctroct    : float
      - octwidth  : float
          These parameters specify a dominance window - Gaussian
          weighting centered on ctroct (in octs, re A0 = 27.5Hz) and
          with a gaussian half-width of octwidth.  
          Defaults to halfwidth = inf, i.e. flat.

    :returns:
      wts       : ndarray, shape=(n_chroma, n_fft) 
          Chroma filter matrix

    """

    wts         = np.zeros((n_chroma, n_fft))

    fft_res     = float(sr) / n_fft

    frequencies = np.arange(fft_res, sr, fft_res)

    fftfrqbins  = n_chroma * librosa.core.hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = np.concatenate( (   [fftfrqbins[0] - 1.5 * n_chroma],
                                        fftfrqbins))

    binwidthbins = np.concatenate(
        (np.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = np.tile(fftfrqbins, (n_chroma, 1))  \
        - np.tile(np.arange(0, n_chroma, dtype='d')[:, np.newaxis], 
        (1, n_fft))

    n_chroma2 = round(n_chroma / 2.0)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are +ve
    D = np.remainder(D + n_chroma2 + 10*n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (n_chroma, 1)))**2)

    # normalize each column
    wts /= np.tile(np.sqrt(np.sum(wts**2, 0)), (n_chroma, 1))

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/n_chroma - ctroct)/octwidth)**2)),
            (n_chroma, 1))

    # remove aliasing columns
    wts[:, (1 + n_fft/2):] = 0.0
    return wts

def mel(sr, n_fft, n_mels=40, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :parameters:
      - sr        : int
          sampling rate of the incoming signal
      - n_fft     : int
          number of FFT components
      - n_mels    : int
          number of Mel bands 
      - fmin      : float
          lowest frequency (in Hz) 
      - fmax      : float
          highest frequency (in Hz)
      - htk       : boolean
          use HTK formula instead of Slaney

    :returns:
      - M         : np.ndarray, shape=(n_mels, n_fft)
          Mel transform matrix

    .. note:: coefficients above 1 + n_fft/2 are set to 0.

    """

    if fmax is None:
        fmax = sr / 2.0

    # Initialize the weights
    weights     = np.zeros( (n_mels, n_fft) )

    # Center freqs of each FFT bin
    size        = 1 + n_fft / 2
    fftfreqs    = np.arange( size, dtype=float ) * sr / n_fft

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs       = librosa.core.mel_frequencies(n_mels, fmin, fmax, htk)

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm       = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in xrange(n_mels):
        # lower and upper slopes for all bins
        lower   = (fftfreqs - freqs[i])     / (freqs[i+1] - freqs[i])
        upper   = (freqs[i+2] - fftfreqs)   / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i, :size]   = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
   
    return weights


def constantq(sr, n_fft, n_bins=12, fmin=None, fmax=None):
    """Create a Filterbank matrix to combine FFT bins to form the constant Q
    transform.

    Based on B. Blankertz, "The Constant Q Transform"
    http://doc.ml.tu-berlin.de/bbci/material/publications/Bla_constQ.pdf

    :parameters:
        - sr    : int
            Sampling rate of the incoming signal.
        - n_fft : int
            FFT length to use.
        - n_bins : int
            Number of bins per octave.
        - fmin : float
            Frequency in Hz of the lowest edge. Defaults to minimum resolution
            available with sr, n_fft and n_bins.
        - fmax : float
            Frequency in Hz of the upper edge. Defaults to ``sr / 2``.

    :returns:
        - Q : np.ndarray, dtype=complex
            Complex-valued filter bank for CQT.
    """

    # Set the upper frequency limit
    if fmax is None:
        fmax = sr / 2.0

    # Bin-spacing
    Q = (2 ** (1.0 / n_bins) - 1)**-1

    # Compute minimum feasible fmin from input parameters
    if fmin is None:
        fmin = Q * sr / n_fft

    fmin        = max(fmin, Q * sr / n_fft)
    n_filters   = np.ceil(n_bins * np.log2(float(fmax) / fmin))
    
    CQT         = np.zeros((n_filters, n_fft), dtype=np.complex)

    for k in np.arange(n_filters - 1, -1, -1, dtype=np.float):
        ilen = np.ceil(Q * sr / (fmin * 2.0**(k / n_bins)))

        # calculate offsets so that kernels are centered 
        start = (n_fft - ilen - np.mod(ilen, 2)) / 2

        CQT[k, start:start+len] = (np.hamming(ilen) / ilen
                                        * np.exp(2 * np.pi * 1j * Q * np.linspace(0, 1.0, ilen, endpoint=False)))

    CQT = np.fft.rfft2(CQT, axis=1) / n_fft
    return CQT 
