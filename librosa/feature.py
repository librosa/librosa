#!/usr/bin/env python
"""Feature extraction routines."""

import numpy as np

import librosa.core

#-- Chroma --#
def logfsgram(y, sr, n_fft=4096, hop_length=512, **kwargs):
    '''Compute a log-frequency spectrogram (piano roll) using a fixed-window STFT.

    :parameters:
      - y : np.ndarray
        audio time series

      - sr : int > 0
        sampling rate of ``y``

      - n_fft : int > 0
        FFT window size

      - hop_length : int > 0
        hop length for STFT

      - bins_per_octave : int > 0
        Number of bins per octave. Defaults to 12.

      - tuning : float in [-0.5,  0.5)
        Deviation (in fractions of a bin) from A440 tuning.
        If not provided, it will be automatically estimated from ``y``.

      - kwargs : additional arguments
        See ``librosa.filters.logfrequency()`` 

    :returns:
      - P : np.ndarray, shape = (n_pitches, t)
        P(f, t) contains the energy at pitch bin f, frame t.

    '''
    
    # First, get the spectrogram and track pitches
    pitches, magnitudes, D = ifptrack(y, sr, n_fft=n_fft, hop_length=hop_length)

    # Normalize, retain magnitude
    D = np.abs(D / D.max())

    # If the user didn't specify tuning, do it ourselves
    if 'tuning' not in kwargs:
        bins_per_octave = kwargs.get('bins_per_octave', 12)
        tuning = estimate_tuning(pitches[magnitudes > np.median(magnitudes)], 
                                 bins_per_octave=bins_per_octave)


    # Build the CQ basis
    cq_basis = librosa.filters.logfrequency(sr, n_fft=n_fft, tuning=tuning, **kwargs)
    
    return cq_basis.dot(D)

def chromagram(y=None, sr=22050, S=None, norm='inf', n_fft=2048, hop_length=512, tuning=0.0, **kwargs):
    """Compute a chromagram from a spectrogram or waveform

    :parameters:

      - y          : np.ndarray or None
          audio time series
      - sr         : int
          audio sampling rate 
      - S          : np.ndarray or None
          spectrogram (STFT magnitude)
      - norm       : {'inf', 1, 2, None}
          column-wise normalization:

             'inf' :  max norm

             1 :  l_1 norm 
             
             2 :  l_2 norm
             
             None :  do not normalize
      - n_fft      : int  > 0
          FFT window size if working with waveform data

      - hop_length : int > 0
          hop length if working with waveform data

      - tuning : float in [-0.5, 0.5)
          Deviation from A440 tuning in fractional bins (cents)

      - kwargs
          Parameters to build the chroma filterbank.
          See librosa.filters.chroma() for details.

    .. note:: One of either ``S`` or ``y`` must be provided.
          If y is provided, the magnitude spectrogram is computed automatically given
          the parameters ``n_fft`` and ``hop_length``.
          If S is provided, it is used as the input spectrogram, and n_fft is inferred
          from its shape.
      
    :returns:
      - chromagram  : np.ndarray
          Normalized energy for each chroma bin at each frame.

    :raises:
      - ValueError 
          if an improper value is supplied for norm

    """
    
    n_chroma = kwargs.get('n_chroma', 12)

    # Build the chroma filterbank, estimate tuning
    if S is None:
        S = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
        pitches, magnitudes, S = ifptrack(y, sr, n_fft)
        S = np.abs(S / S.max())
        
        tuning = estimate_tuning(pitches[magnitudes > np.median(magnitudes)], 
                                 bins_per_octave=n_chroma)

    else:
        n_fft       = (S.shape[0] -1 ) * 2

    if 'A440' not in kwargs:
        kwargs['A440'] = 440.0 * 2.0**(tuning/n_chroma)

    chromafb = librosa.filters.chroma( sr, n_fft, **kwargs)

    # Compute raw chroma
    raw_chroma  = np.dot(chromafb, S)

    # Compute normalization factor for each frame
    if norm == 'inf':
        chroma_norm = np.max(np.abs(raw_chroma), axis=0)

    elif norm == 1:
        chroma_norm = np.sum(np.abs(raw_chroma), axis=0)
    
    elif norm == 2:
        chroma_norm = np.sum( (raw_chroma**2), axis=0) ** 0.5
    
    elif norm is None:
        return raw_chroma
    
    else:
        raise ValueError("norm must be one of: 'inf', 1, 2, None")

    # Tile the normalizer to match raw_chroma's shape
    chroma_norm[chroma_norm == 0] = 1.0

    return raw_chroma / chroma_norm

def perceptual_weighting(S, frequencies, ref_power=1e-12):
    '''Perceptual weighting of a power spectrogram:
    
    S_p[f] = A_weighting(f) + 10*log(S[f] / ref_power)
    
    :parameters:
      - S : np.ndarray, shape=(d,t)
        Power spectrogram
        
      - frequencies : np.ndarray, shape=(d,)
        Center frequency for each row of S
        
      - ref_power : float > 0
        Reference power
        
    :returns:
      - S_p : np.ndarray, shape=(d,t)
        perceptually weighted version of S, in dB.
    '''
    
    offset = librosa.A_weighting(frequencies).reshape((-1, 1))
    
    return librosa.logamplitude(S) - 10.0 * np.log10(ref_power) + offset

#-- Pitch and tuning --#
def estimate_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    '''Given a collection of pitches, estimate its tuning offset
    (in fractions of a bin) relative to A440=440.0Hz.
    
    :parameters:
      - frequencies : array-like, float
        Detected frequencies in the signal

      - resolution : float in (0, 1)
        Resolution of the tuning
        
      - bins_per_octave : int > 0
        How many bins per octave?
        
    :returns:
      - semisoff: float in [-0.5, 0.5]
        estimated tuning in cents (fractions of a bin)
                  
    '''

    frequencies = np.asarray([frequencies], dtype=float).flatten()

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave * librosa.core.hz_to_octs(frequencies) , 1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0
    
    bins     = np.linspace(-0.5, 0.5, np.ceil(1./resolution), endpoint=False)
  
    counts, cents = np.histogram(residual, bins)
    
    # return the histogram peak
    return cents[np.argmax(counts)]

def ifptrack(y, sr=22050, n_fft=4096, hop_length=None, fmin=(150.0, 300.0), fmax=(2000.0, 4000.0), threshold=0.75):
    '''Instantaneous pitch frequency tracking.

    :parameters:
      - y: np.ndarray
        audio signal
      
      - sr : int
        audio sample rate of y
        
      - n_fft: int
        DFT length.
        
      - threshold : float in (0, 1)
        Maximum fraction of expected frequency increment to tolerate
      
      - fmin : float or tuple of float
        Ramp parameter for lower frequency cutoff
        If scalar, the ramp has 0 width.
        If tuple, a linear ramp is applied from fmin[0] to fmin[1]
        
      - fmax : float or tuple of float
        Ramp parameter for upper frequency cutoff
        If scalar, the ramp has 0 width.
        If tuple, a linear ramp is applied from fmax[0] to fmax[1]
        
    :returns:
      - pitches : np.ndarray, shape=(d,t)
      - magnitudes : np.ndarray, shape=(d,t)
        Where 'd' is the subset of FFT bins within fmin and fmax.
        
        pitches[i, t] contains instantaneous frequencies at time t
        magnitudes[i, t] contains their magnitudes.
        
      - D : np.ndarray, dtype=complex
        STFT matrix
    '''

    
    fmin = np.asarray([fmin]).squeeze()
    fmax = np.asarray([fmax]).squeeze()
    
    # Truncate to feasible region
    fmin = np.maximum(0, fmin)
    fmax = np.minimum(fmax, sr / 2)
    
    # What's our DFT bin resolution?
    fft_res = float(sr) / n_fft
    
    # Only look at bins up to 2 kHz
    max_bin = int(round(fmax[-1] / fft_res))
  
    if hop_length is None:
        hop_length = n_fft / 4

    # Calculate the inst freq gram
    if_gram, D = librosa.core.ifgram(y, sr=sr, 
                                     n_fft=n_fft, 
                                     win_length=n_fft/2, 
                                     hop_length=hop_length)

    # Find plateaus in ifgram - stretches where delta IF is < thr:
    # ie, places where the same frequency is spread across adjacent bins
    idx_above  = range(1, max_bin) + [max_bin - 1]
    idx_below  = [0] + range(0, max_bin - 1)
    
    # expected increment per bin = sr/w, threshold at 3/4 that
    matches    = abs(if_gram[idx_above] - if_gram[idx_below]) < threshold * fft_res
  
    # mask out any singleton bins (where both above and below are zero)
    matches    = matches * ((matches[idx_above] > 0) | (matches[idx_below] > 0))

    pitches    = np.zeros_like(matches, dtype=float)
    magnitudes = np.zeros_like(matches, dtype=float)

    # For each frame, extract all harmonic freqs & magnitudes
    for t in range(matches.shape[1]):
        
        # find nonzero regions in this vector
        # The mask selects out constant regions + active borders
        mask   = ~np.pad(matches[:, t], 1, mode='constant')
        
        starts = np.argwhere(matches[:, t] & mask[:-2])
        ends   = 1 + np.argwhere(matches[:, t] & mask[2:])
        
        # Set up inner loop    
        frqs = np.zeros_like(starts, dtype=float)
        mags = np.zeros_like(starts, dtype=float)
        
        for i in range(len(starts)):
            # Weight frequencies by energy
            weights = np.abs(D[starts[i]:ends[i], t])
            mags[i] = weights.sum()
            
            # Compute the weighted average frequency.
            # FIXME: is this the right thing to do? 
            # These are frequencies... shouldn't this be a 
            # weighted geometric average?
            frqs[i] = weights.dot(if_gram[starts[i]:ends[i], t])
            if mags[i] > 0:
                frqs[i] /= mags[i]
            
        # Clip outside the ramp zones
        idx        = (fmax[-1] < frqs) | (frqs < fmin[0])
        mags[idx]  = 0
        frqs[idx]  = 0
        
        # Ramp down at the high end
        idx        = (fmax[-1] > frqs) & (frqs > fmax[0])
        mags[idx] *= (fmax[-1] - frqs[idx]) / (fmax[-1] - fmax[0])
        
        # Ramp up from the bottom end
        idx        = (fmin[-1] > frqs) & (frqs > fmin[0])
        mags[idx] *= (frqs[idx] - fmin[0]) / (fmin[-1] - fmin[0])
        
        # Assign pitch and magnitude to their center bin
        bins                = (starts + ends) / 2
        pitches[bins, t]    = frqs
        magnitudes[bins, t] = mags

    return pitches, magnitudes, D
  
#-- Mel spectrogram and MFCCs --#
def mfcc(S=None, y=None, sr=22050, n_mfcc=20):
    """Mel-frequency cepstral coefficients

    :parameters:
      - S     : np.ndarray or None
          log-power Mel spectrogram
      - y     : np.ndarray or None
          audio time series
      - sr    : int > 0
          audio sampling rate of y
      - n_mfcc: int
          number of MFCCs to return

    .. note::
        One of ``S`` or ``y, sr`` must be provided.
        If ``S`` is not given, it is computed from ``y, sr`` using
        the default parameters of ``melspectrogram``.

    :returns:
      - M     : np.ndarray 
          MFCC sequence

    """

    if S is None:
        S = librosa.logamplitude(melspectrogram(y=y, sr=sr))
    
    return np.dot(librosa.filters.dct(n_mfcc, S.shape[0]), S)

def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, **kwargs):
    """Compute a Mel-scaled power spectrogram.

    :parameters:
      - y : np.ndarray
          audio time-series
      - sr : int
          audio sampling rate of y  
      - S : np.ndarray
          magnitude or power spectrogram
      - n_fft : int
          number of FFT components
      - hop_length : int
          frames to hop

      - kwargs
          Mel filterbank parameters
          See librosa.filters.mel() documentation for details.

    .. note:: One of either ``S`` or ``y, sr`` must be provided.
        If the pair y, sr is provided, the power spectrogram is computed.
        If S is provided, it is used as the spectrogram, and the parameters ``y, n_fft,
        hop_length`` are ignored.

    :returns:
      - S : np.ndarray
          Mel spectrogram

    """

    # Compute the STFT
    if S is None:
        S       = np.abs(librosa.core.stft(y,   
                                            n_fft       =   n_fft, 
                                            hop_length  =   hop_length))**2
    else:
        n_fft = (S.shape[0] - 1) * 2

    # Build a Mel filter
    mel_basis   = librosa.filters.mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)

#-- miscellaneous utilities --#
def sync(data, frames, aggregate=np.mean):
    """Synchronous aggregation of a feature matrix

    :parameters:
      - data      : np.ndarray, shape=(d, T)
          matrix of features
      - frames    : np.ndarray
          (ordered) array of frame segment boundaries
      - aggregate : function
          aggregation function (defualt: mean)

    :returns:
      - Y         : ndarray 
          ``Y[:, i] = aggregate(data[:, F[i-1]:F[i]], axis=1)``

    .. note:: In order to ensure total coverage, boundary points are added to frames

    """
    if data.ndim < 2:
        data = np.asarray([data])
    elif data.ndim > 2:
        raise ValueError('Synchronized data has ndim=%d, must be 1 or 2.' % data.ndim)

    (dimension, n_frames) = data.shape

    frames      = np.unique(np.concatenate( ([0], frames, [n_frames]) ))

    if min(frames) < 0:
        raise ValueError('Negative frame index.')
    elif max(frames) > n_frames:
        raise ValueError('Frame index exceeds data length.')

    data_agg    = np.empty( (dimension, len(frames)-1) )

    start       = frames[0]

    for (i, end) in enumerate(frames[1:]):
        data_agg[:, i] = aggregate(data[:, start:end], axis=1)
        start = end

    return data_agg
