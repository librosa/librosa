#!/usr/bin/env python
"""Commonly used filter banks: DCT, Chroma, Mel, CQT"""

import numpy as np
import librosa

def dct(n_filts, n_input):
    """Discrete cosine transform basis

    :usage:
        >>> # Compute MFCCs
        >>> S           = librosa.melspectrogram(y, sr)
        >>> dct_filters = librosa.filters.dct(13, S.shape[0])
        >>> mfcc        = dct_filters.dot(librosa.logamplitude(S))

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

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    :usage:
        >>> mel_fb = librosa.filters.mel(22050, 2048)

        >>> # Or clip the maximum frequency to 8KHz
        >>> mel_fb = librosa.filters.mel(22050, 2048, fmax=8000)

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
      - htk       : bool
          use HTK formula instead of Slaney

    :returns:
      - M         : np.ndarray, shape=(n_mels, 1+ n_fft/2)
          Mel transform matrix

    """

    if fmax is None:
        fmax = sr / 2.0

    # Initialize the weights
    size        = int(1 + n_fft / 2)
    n_mels      = int(n_mels)
    weights     = np.zeros( (n_mels, size) )

    # Center freqs of each FFT bin
    fftfreqs    = np.arange( size, dtype=float ) * sr / n_fft

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs       = librosa.mel_frequencies(n_mels, fmin=fmin, fmax=fmax, htk=htk, extra=True)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm       = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in xrange(n_mels):
        # lower and upper slopes for all bins
        lower   = (fftfreqs - freqs[i])     / (freqs[i+1] - freqs[i])
        upper   = (freqs[i+2] - fftfreqs)   / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i]   = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
   
    return weights

def chroma(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=2):
    """Create a Filterbank matrix to convert STFT to chroma

    :usage:
        >>> # Build a simple chroma filter bank
        >>> chroma_fb   = librosa.filters.chroma(22050, 4096)

        >>> # Use quarter-tones instead of semitones
        >>> chroma_fbq  = librosa.filters.chroma(22050, 4096, n_chroma=24)

        >>> # Equally weight all octaves
        >>> chroma_fb   = librosa.filters.chroma(22050, 4096, octwidth=None)

    :parameters:
      - sr        : int
          audio sampling rate
      - n_fft     : int
          FFT window size
      - n_chroma  : int
          number of chroma bins
      - A440      : float
          Reference frequency for A440
      - ctroct    : float
      - octwidth  : float or None
          These parameters specify a dominance window - Gaussian
          weighting centered on `ctroct` (in octs, re A0 = 27.5Hz) and
          with a gaussian half-width of `octwidth`.  
          Set `octwidth` to `None` to use a flat weighting.

    :returns:
      - wts       : ndarray, shape=(n_chroma, 1 + n_fft / 2) 
          Chroma filter matrix

    """

    wts         = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    fftfrqbins  = n_chroma * librosa.hz_to_octs(frequencies, A440)

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
    wts = librosa.util.normalize(wts, norm=2, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/n_chroma - ctroct)/octwidth)**2)),
            (n_chroma, 1))

    # remove aliasing columns
    return wts[:, :(1 + n_fft/2)]

def logfrequency(sr, n_fft, bins_per_octave=12, tuning=0.0, fmin=None, fmax=None, spread=0.125):
    '''Approximate a constant-Q filterbank for a fixed-window STFT.
    
    Each filter is a log-normal window centered at the corresponding pitch frequency.
    
    :usage:
        >>> # Simple log frequency filters
        >>> logfs_fb = librosa.filters.logfrequency(22050, 4096)

        >>> # Use a narrower frequency range
        >>> logfs_fb = librosa.filters.logfrequency(22050, 4096, fmin=110, fmax=880)

        >>> # Use narrower filters for sparser response: 5% of a semitone
        >>> logfs_fb = librosa.filters.logfrequency(22050, 4096, spread=0.05)
        >>> # Or wider: 50% of a semitone
        >>> logfs_fb = librosa.filters.logfrequency(22050, 4096, spread=0.5)

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
        fmin = librosa.midi_to_hz(librosa.note_to_midi('C1'))
        
    if fmax is None:
        fmax = librosa.midi_to_hz(librosa.note_to_midi('C9'))
    
    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    
    # How many bins can we get?
    n_filters = int(np.ceil(bins_per_octave * np.log2(float(fmax) / fmin)))
    
    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave
    
    # Construct the output matrix
    basis = np.zeros( (n_filters, n_fft /2  + 1) )
    
    # Get log frequencies of bins
    log_freqs = np.log2(librosa.fft_frequencies(sr, n_fft)[1:])
                                
    for i in range(n_filters):
        # What's the center (median) frequency of this filter?
        center_freq = correction * fmin * (2.0**(float(i)/bins_per_octave))
        
        # Place a log-normal window around center_freq
        # We skip the sqrt(2*pi) normalization because it will wash out below anyway
        basis[i, 1:] = np.exp(-0.5 * ((log_freqs - np.log2(center_freq)) /sigma)**2 - np.log2(sigma) - log_freqs)
                                  
        # Normalize each filter
        c_norm = np.sqrt(np.sum(basis[i]**2))
        if c_norm > 0:
            basis[i] = basis[i] / c_norm
        
    return basis

def constant_q(sr, fmin=None, fmax=None, bins_per_octave=12, tuning=0.0, window=None, resolution=2, pad=False):
    '''Construct a constant-Q basis.

    :usage:
        >>> # Get the CQT basis for C1 to C9, standard tuning
        >>> basis   = librosa.filters.constant_q(22050)
        >>> CQT     = librosa.cqt(y, sr, basis=basis)
        
        >>> # Change the windowing function to Hanning instead of Hamming
        >>> basis   = librosa.filters.constant_q(22050, window=np.hanning)

        >>> # Use a longer window for each filter
        >>> basis   = librosa.filters.constant_q(22050, resolution=2)

    :parameters:
      - sr : int > 0
          Audio sampling rate

      - fmin : float > 0
          Minimum frequency bin. Defaults to ``C1 ~= 16.35``
        
      - fmax : float > 0
          Maximum frequency bin. Defaults to ``C9 = 4816.01``

      - bins_per_octave : int > 0
          Number of bins per octave

      - tuning : float in [-0.5, +0.5)
          Tuning deviation from A440 in fractions of a bin
      
      - window : function or None
          Windowing function to apply to filters. 
          If None, no window is applied.
          Default: np.hamming

      - resolution : float > 0
          Resolution of filter windows. Larger values use longer windows.

      - pad : boolean
          Zero-pad all filters to have a constant width (equal to the longest filter).

      .. note::
            @phdthesis{mcvicar2013,
              title  = {A machine learning approach to automatic chord extraction},
              author = {McVicar, M.},
              year   = {2013},
              school = {University of Bristol}}

    :returns:
      - filters : list of np.ndarray
          filters[i] is the time-domain representation of the i'th CQT basis.
    '''
    

    if fmin is None:
        fmin = librosa.midi_to_hz(librosa.note_to_midi('C1'))
        
    if fmax is None:
        fmax = librosa.midi_to_hz(librosa.note_to_midi('C9'))

    if window is None:
        window = np.hamming

    correction = 2.0**(float(tuning) / bins_per_octave)

    fmin       = correction * fmin
    fmax       = correction * fmax
    
    # Q should be capitalized here, so we suppress the name warning
    Q = float(resolution) / (2.0**(1./bins_per_octave) - 1) # pylint: disable=invalid-name
    
    # How many bins can we get?
    n_filters = int(np.ceil(bins_per_octave * np.log2(float(fmax) / fmin)))

    filters = []
    for i in np.arange(n_filters, dtype=float):
        
        # Length of this filter
        ilen = np.ceil(Q * sr / (fmin * 2.0**(i / bins_per_octave)))

        # Build the filter 
        win = np.exp(Q * 1j * np.linspace(0, 2 * np.pi, ilen, endpoint=False))

        # Apply the windowing function
        if window is not None:
            win = win * window(ilen) 

        # Normalize
        win = librosa.util.normalize(win, norm=2)
        
        filters.append(win)
    
    if pad:
        max_len = max(map(len, filters))
        
        for i in range(len(filters)):
            filters[i] = librosa.util.pad_center(filters[i], max_len)

    return filters

def cq_to_chroma(n_input, bins_per_octave=12, n_chroma=12, roll=0):
    '''Convert a Constant-Q basis to Chroma.

    :usage:
        >>> # Get a CQT, and wrap bins to chroma
        >>> CQT         = librosa.cqt(y, sr)
        >>> chroma_map  = librosa.filters.cq_to_chroma(CQT.shape[0])
        >>> chromagram  = chroma_map.dot(CQT)

    :parameters:
      - n_input : int > 0
          Number of input components (CQT bins)

      - bins_per_octave : int > 0
          How many bins per octave in the CQT

      - n_chroma : int > 0
          Number of output bins (per octave) in the chroma

      - roll : int
          Number of bins to offset the output by.
          For example, if the 0-bin of the CQT is C, and
          the desired 0-bin for the chroma is A, then roll=-3.

    :returns:
      - cq_to_chroma : np.ndarray, shape=(n_chroma, n_input)
          Transformation matrix: ``Chroma = np.dot(cq_to_chroma, CQT)``      
        
    :raises:
      - ValueError
          If n_input is not an integer multiple of n_chroma
    '''

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if np.mod(n_merge, 1) != 0:
        raise ValueError('Incompatible CQ merge: input bins must be an integer multiple of output bins.')

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(np.float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, -roll, axis=0)

    return cq_to_ch
