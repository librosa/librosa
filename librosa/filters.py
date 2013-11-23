#!/usr/bin/env python
"""Commonly used filter banks: DCT, Chroma, Mel, CQT"""

import numpy as np
import librosa.core

def dct(n_filts, n_input):
    """Discrete cosine transform basis

    :parameters:
      - n_filts   : int
          number of output components
      - n_input   : int
          number of input components

    :returns:
      - D         : np.ndarray, shape=(n_filts, n_input)
          DCT basis vectors

    """

    basis       = np.empty((n_filts, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples     = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in xrange(1, n_filts):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

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
      wts       : ndarray, shape=(n_chroma, 1 + n_fft / 2) 
          Chroma filter matrix

    """

    wts         = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

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
    return wts[:, :(1 + n_fft/2)]

def constantq(sr, n_fft, bins_per_octave=12, tuning=0.0, fmin=None, fmax=None, spread=0.125):
    '''Approximate a constant-Q filterbank for a fixed-window STFT.
    
    Each filter is a log-normal window centered at the corresponding pitch frequency.
    
    :parameters:
      - sr : int > 0
        audio sampling rate
        
      - n_fft : int > 0
        FFT window size
        
      - bins_per_octave : int > 0
        Number of bins per octave. Defaults to 12 (semitones).
        
      - tuning : None or float in [-0.5, +0.5]
        Tuning correction parameter, in fractions of a bin.
        
      - fmin : float > 0
        Minimum frequency bin. Defaults to ``C1 ~= 16.35``
        
      - fmax : float > 0
        Maximum frequency bin. Defaults to ``C9 = 4816.01``
        
      - spread : float > 0
        Spread of each filter, as a fraction of a bin.
        
    :returns:
      - C : np.ndarray, shape=(ceil(log(fmax/fmin)) * bins_per_octave, 1 + n_fft/2)
        CQT filter bank.
    '''
    
    if fmin is None:
        fmin = librosa.core.midi_to_hz(librosa.note_to_midi('C1'))
        
    if fmax is None:
        fmax = librosa.core.midi_to_hz(librosa.note_to_midi('C9'))
    
    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    
    # How many bins can we get?
    n_filters = int(np.ceil(bins_per_octave * np.log2(float(fmax) / fmin)))
    
    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave
    
    # Construct the output matrix
    C = np.zeros( (n_filters, n_fft /2  + 1) )
    
    # Get bin frequencies
    fftfreqs = librosa.fft_freq(sr, n_fft)
                                
    for i in range(n_filters):
        # What's the center (median) frequency of this filter?
        center_freq = correction * fmin * (2.0**(float(i)/bins_per_octave))
        
        # Place a log-normal window around center_freq
        # We skip the sigma*sqrt(2*pi) normalization because it will wash out below anyway
        C[i, 1:] = np.exp(-0.5 * ((np.log2(fftfreqs[1:]) - np.log2(center_freq)) /sigma)**2) / fftfreqs[1:]
                                  
        # Normalize each filter
        C[i] = C[i] / np.sqrt(np.sum(C[i]**2))
        
    return C

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
      - M         : np.ndarray, shape=(n_mels, 1+ n_fft/2)
          Mel transform matrix

    """

    if fmax is None:
        fmax = sr / 2.0

    # Initialize the weights
    size        = 1 + n_fft / 2
    weights     = np.zeros( (n_mels, size) )

    # Center freqs of each FFT bin
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
        weights[i]   = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
   
    return weights
