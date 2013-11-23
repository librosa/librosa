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

def chromagram(y=None, sr=22050, S=None, norm='inf', n_fft=2048, hop_length=512, **kwargs):
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
    
    # Build the chroma filterbank
    if S is None:
        S = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
    else:
        n_fft       = (S.shape[0] -1 ) * 2

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

def loudness_chroma(x, sr, beat_times, tuning, fmin=55.0, fmax=1661.0, resolution_fact=5):
    """Compute a loudness-based chromagram, for use in chord estimation 

    :parameters:
      - x : np.ndarray
           audio time-series
      - sr : int
        audio sampling rate of x
      - beat_times: np.ndarray
        estimated beat locations (in seconds). 
      - tuning: float in [-0.5, 0.5]
        estimated tuning in cents. 
      - fmin: float
          minimum frequency of spectrum to consider. Will be rounded to
          Closest pitch frequency in Hz (accounting for tuning)
        fmax: float
          maximum frequency of spectrum to consider. For balanced results,
          make this one pitch less than an octave multiple of fmin
          (for example, default value is 4 octaves + 11 pitches above fmin = 55)
      - resolution_fact: int
          multiplying factor of power in window (see PhD thesis "A Machine Learning approach
          to automatic chord extraction", Matt McVicar, University of Bristol 2013)

    :returns:
      - raw_chroma : np.ndarray 12 x T
        Loudness-based chromagram, 12 pitches by number of beats + 1
      - normal_chroma: np.ndarray 12 x T
        Normalised chromagram, each row normalised to range [0, 1]
      - sample_times: np.ndarray
        start and end points of chroma windows (in seconds)
      - tuning: float in [-0.5, 0.5]
        estimated tuning of piece, returned in case None was supplied
                  
      """
      
    # Get hamming windows for convolution
    hamming_k = librosa.filters.constant_q(sr, 
                                         fmin=fmin, fmax=fmax, 
                                         bins_per_octave=12,
                                         tuning=tuning,
                                         resolution=resolution_fact)

    half_winLenK = map(lambda x: np.ceil(0.5 * len(x)), hamming_k)
    freq_bins = librosa.cqt_frequencies(len(hamming_k), fmin, tuning=tuning)

    # Extract chroma
    raw_chroma, normal_chroma, sample_times = CQ_chroma_loudness(x, 
                   sr, beat_times, hamming_k, half_winLenK, freq_bins)

    return raw_chroma, normal_chroma, sample_times, tuning   

# FIXME:  2013-09-25 18:08:53 by Brian McFee <brm2132@columbia.edu>
# why is this "cal_"? 
# FIXED: 2013-09-29 by Matt.
# It was short for 'calculate' ;-) removed here and wherever it's called
def CQ_chroma_loudness(x, sr, beat_times, hammingK, half_winLenK, freqK, refLabel='s', A_weightLabel=1, q_value=0):
    """Compute a loudness-based chromagram

    :parameters:
      - x : np.ndarray
            audio time-series

      - sr : int
            audio sampling rate of x

      - beat_times: np.ndarray
            estimated beat locations (in seconds). 

      - hammingK: complex array
            hamming window to use in convolution, generated by cal_hamming_window

      - half_winLenK: complex array
            half the length of the above windows, generated by cal_hamming_window

      - freqK: np.ndarray
            Frequency of the kth window

      - refLabel: {'n','s','mean','median','q'}
            reference power level.
            'n'       - no reference, i.e. 1
            's'       - standard human reference power of 10**-12
            'mean'    - reference of each frequency is the mean of this
                        frequency over the song
            'median'  - as above with median
            'q'       - regard the qth quantile of the signal to be silence
                        (see q_value argument)
      - q_value: float in [0.0, 1.0]
            quantile to consider silence if refLabel = 'q'

    :returns:
      - raw_chroma : np.ndarray 12 x T
        Loudness-based chromagram, 12 pitches by number of beats + 1

      - normal_chroma: np.ndarray 12 x T
        Normalised chromagram, each row normalised to range [0, 1]

      - sample_times: np.ndarray
        start and end points of chroma windows (in seconds)

      - tuning: float in [-0.5, 0.5]
        estimated tuning of piece, returned in case None was supplied
                  
      """
 
    # 1. configuration. Pad x to be a power of 2, get length parameters
    bins    = 12
    Nxorig  = len(x)
    
    # add the end to make the length to be 2^N 
    #     TODO:   2013-09-25 17:51:57 by Brian McFee <brm2132@columbia.edu>
    # this should be done with np.pad, not hstack
    # FIXED: 2013-09-29 by Matt
    # amended to use np.pad
    # FIXED: 2013-10-06 by Dawen
    # Apparently there was a parentheses mismatch, fixed by looking back at 
    # previous version
    n_pad = 2.0**np.ceil(np.log2(Nxorig))-Nxorig
    x = np.lib.pad(x, (0,n_pad), 'constant', constant_values=(0.0,0.0))
    #x   = np.hstack([x, np.zeros(2.0**np.ceil(np.log2(Nxorig))-Nxorig)]) 
    Nx  = len(x)                                                       
    
    # number of frequency bins
    K = len(hammingK)                                                 

    # full fft of signal
    xf = np.fft.fft(x)                                                

    # check whether hamming window length is > length(xf) and issue a warning
    warning_flag = np.zeros(K)
    for k in range(K):
        if len(hammingK[k]) > Nx:
            print('Warning: signalskye is shorter than one of the analysis windows')
            warning_flag[k] = 1

    # Beat-time interval

    # Get the beat time (transform it into sample indices)
    beat_sr = np.ceil(np.multiply(beat_times, sr))                     

    # delete those samples that have exceeded the end of the song
    beat_sr = np.delete(beat_sr, np.nonzero(beat_sr>=Nxorig))            
 
    # Pad 0 to start, length of song to end
    if beat_sr[0] is 0:
        beat_sr = np.hstack([beat_sr, Nxorig])
    else:
        beat_sr = np.hstack([0.0, beat_sr, Nxorig])

    num_F = len(beat_sr)-1
 
    # Process reference powers. Create storage if needed
    if refLabel is 'n':
        ref_power       = 1

    elif refLabel is 's':
        ref_power       = 10.0**(-12.0)

    elif refLabel is 'mean':
        meanPowerK      = np.zeros(K)

    elif refLabel is 'median':
        medianPowerK    = np.zeros(K)

    elif refLabel is 'q':
        # Need to store the average power of each frame
        quantile_matrix  = np.zeros(Nxorig)
        if q_value < 0.0 or q_value > 1.0:
            raise ValueError("Quantile must be in range [0.1, 1.0]")
    else:
        raise ValueError("Reference power must be one of: ['n', 's', 'mean', 'median', 'q']")

    # A-weight parameters
    if A_weightLabel is 1:
        Ap1 = 12200.0   ** 2.0
        Ap2 = 20.6      ** 2.0
        Ap3 = 107.7     ** 2.0
        Ap4 = 737.9     ** 2.0

    # Compute the CQ matrix for each point (row) and each frequency bin (column)
    A_offsets = np.zeros(K)
    CQ = np.zeros([K, num_F])
 
    for k in range(K):
        # Get the constant Q tranformation efficiently via convolution. 
        # First create hamming window for this frequency
        half_len = int(half_winLenK[k])
        w = np.hstack([hammingK[k][half_len-1:], np.zeros(Nx-len(hammingK[k])), hammingK[k][:half_len-1]])

        # Take fft of window and convolve, then invert
        wf = np.fft.fft(w)
        convolf = xf*wf
        convol = np.fft.ifft(convolf)
        
        # add A-weighting value for this frequency?
        if A_weightLabel is 1:
            frequency_k2 = freqK[k]**2.0
            A_scale = Ap1*frequency_k2**2.0/((frequency_k2+Ap2)*np.sqrt((frequency_k2+Ap3)*(frequency_k2+Ap4))*(frequency_k2+Ap1))
            A_offsets[k] = 2.0+20.0*np.log10(A_scale)
        
        # Reference power and A weighting.
        # Compute abs(X)**2 and calculate offsets if needed
        if refLabel is 'mean':
            convol = np.abs(convol[:Nxorig])**2.0
            meanPowerK[k] = np.mean(convol)

        elif refLabel is 'median':
            convol = np.abs(convol[:Nxorig])**2.0
            medianPowerK[k] = np.median(convol)

        elif refLabel is 'q':
            convol = np.abs(convol[:Nxorig])**2.0
            quantile_matrix = np.add(quantile_matrix, convol)

        else:
            convol = (np.abs(convol[:Nxorig]))**2.0
        
        # Get the beat interval (median)
        for t in range(num_F):
            t1 = int(beat_sr[t])+1
            t2 = int(beat_sr[t+1])
            CQ[k, t] = np.median(convol[t1-1:t2])   
          
    # Add the reference power (for mean/median/q-quantiles)
    # FIXME:  2013-09-25 18:00:20 by Brian McFee <brm2132@columbia.edu>
    # unreadable     
    # FIXED: 2013-09-29 by Matt
    # Expanded/broken up with comments
    if refLabel is 'mean':

        # Compute mean power
        ref_power = np.mean(meanPowerK)

        # convert to dB, minus the reference power
        CQ = np.add(10.0*np.log10(CQ), -10.0*np.log10(ref_power))

        # Add offsets according to A-weighting
        Aweights = np.transpose(np.tile(A_offsets, (num_F, 1)))
        CQ = np.add(CQ, Aweights)
    elif refLabel is 'median':

        # Compute median power
        ref_power = np.median(medianPowerK)

        # convert to dB, minus the reference power
        CQ = np.add(10.0*np.log10(CQ), -10.0*np.log10(ref_power))

        # Add offsets according to A-weighting
        Aweights  = np.transpose(np.tile(A_offsets, (num_F, 1)))
        CQ = np.add(CQ, Aweights)

    elif refLabel is 'q':
        # sort the values, set reference as the value that falls in the qth quantile
        quantile_value = np.sort(quantile_matrix) 
        ref_power = quantile_value[int(np.floor(q_value*Nxorig))-1]/K

        # FIXME:  2013-09-25 17:57:39 by Brian McFee <brm2132@columbia.edu>
        # these should use librosa.logamplitude        
        # Matt: I'm not sure it should. I don't want to throw anything
        # away, so would have to call
        #
        # librosa.core.logamplitude(S, amin=1e-10, top_db=None) 
        #
        # twice in each line. Think it would make the code pretty dense. 
        # Happy to change it if you like though.
        CQ = np.add(10.0*np.log10(CQ), -10*np.log10(ref_power))

        # Add offsets accoring to A-weighting
        Aweights = np.transpose(np.tile(A_offsets, (num_F, 1)))
        CQ = np.add(CQ, np.transpose(np.tile(A_offsets, (num_F, 1))))
    else:
        CQ = np.add(10.0*np.log10(CQ), -10.0*np.log10(ref_power))
        CQ = np.add(CQ, np.transpose(np.tile(A_offsets, (num_F, 1))))
  
    # Beat synchronise
    # FIXME:  2013-09-25 18:01:39 by Brian McFee <brm2132@columbia.edu>
    # do not use chromagram as a variable: it is a function in this module  
    # FIXED 2013-09-29 by Matt
    # Renamed to output_chromagram   
    output_chromagram = np.zeros((bins, num_F))
    normal_chromagram = np.zeros((bins, num_F))
    
    for i in range(bins):
        output_chromagram[i, :] = np.sum(CQ[i::bins, :], 0)
     
    # Normalise
    for i in range(output_chromagram.shape[1]):
        maxCol = np.max(output_chromagram[:, i])
        minCol = np.min(output_chromagram[:, i])
        if (maxCol>minCol):
            normal_chromagram[:, i] = np.true_divide(output_chromagram[:, i] - minCol, maxCol - minCol)
        else:
            normal_chromagram[:, i] = 0.0   

    # Shift to be C-based
    shift_pos = round(12.0*np.log2(freqK[0]/27.5)) # The relative position to A0
    shift_pos = int(np.mod(shift_pos, 12)-3)        # since A0 should shift -3
    if not (shift_pos is 0):
        output_chromagram = np.roll(output_chromagram, shift_pos, 0)
        normal_chromagram = np.roll(normal_chromagram, shift_pos, 0)

    # 5. return the sample times
    beat_sr = beat_sr/sr
    sample_times = np.vstack([beat_sr[:-1], beat_sr[1:]])

    return output_chromagram, normal_chromagram, sample_times

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
        ends   = np.argwhere(matches[:, t] & mask[2:])
        
        # Set up inner loop    
        frqs = np.zeros_like(starts, dtype=float)
        mags = np.zeros_like(starts, dtype=float)
        
        for i, (u, v) in enumerate(zip(starts, ends)):
            # Weight frequencies by energy
            weights = np.abs(D[u:v+1, t])
            mags[i] = weights.sum()
            
            # Compute the weighted average frequency.
            # FIXME: is this the right thing to do? 
            # These are frequencies... shouldn't this be a 
            # weighted geometric average?
            frqs[i] = weights.dot(if_gram[u:v+1, t])
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
        bins                = np.round(0.5 * (starts+ends)).astype(int)
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
