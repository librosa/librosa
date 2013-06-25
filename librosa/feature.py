#!/usr/bin/env python
"""Feature extraction routines."""

import numpy as np

import librosa.core

#-- Frequency conversions --#
def midi_to_hz( notes ):
    """Get the frequency (Hz) of MIDI note(s)

    :parameters:
      - note_num      : int, np.ndarray
          number of the note(s)

    :returns:
      - frequency     : float, np.ndarray
          frequency of the note in Hz
    """

    return 440.0 * (2.0 ** ((notes - 69)/12.0))

def hz_to_midi( frequency ):
    """Get the closest MIDI note number(s) for given frequencies

    :parameters:
      - frequencies   : float, np.ndarray
          target frequencies

    :returns:
      - note_nums     : int, np.ndarray
          closest MIDI notes

    """

    return 12 * (np.log2(frequency) - np.log2(440.0)) + 69


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :parameters:
      - frequencies   : np.ndarray, float
          scalar or array of frequencies
      - htk           : boolean
          use HTK formula instead of Slaney

    :returns:
      - mels        : np.ndarray
          input frequencies in Mels

    """

    if np.isscalar(frequencies):
        frequencies = np.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    
    # Fill in the linear part
    f_min   = 0.0
    f_sp    = 200.0 / 3

    mels    = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    
    min_log_hz  = 1000.0                        # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep     = np.log(6.4) / 27.0            # step size for log region

    log_t       = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    :parameters:
      - mels          : np.ndarray, float
          mel bins to convert
      - htk           : boolean
          use HTK formula instead of Slaney

    :returns:
      - frequencies   : np.ndarray
          input mels in Hz

    """

    if np.isscalar(mels):
        mels = np.array([mels], dtype=float)
    else:
        mels = mels.astype(float)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min       = 0.0
    f_sp        = 200.0 / 3
    freqs       = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz  = 1000.0                        # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep     = np.log(6.4) / 27.0            # step size for log region
    log_t       = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def hz_to_octs(frequencies, A440=440.0):
    """Convert frquencies (Hz) to octave numbers

    :parameters:
      - frequencies   : np.ndarray, float
          scalar or vector of frequencies
      - A440          : float
          frequency of A440

    :returns:
      - octaves       : np.ndarray
          octave number for each frequency

    """
    return np.log2(frequencies / (A440 / 16.0))


#-- CHROMA --#
def chromagram(S, sr, norm='inf', **kwargs):
    """Compute a chromagram from a spectrogram

    :parameters:
      - S          : np.ndarray
          spectrogram
      - sr         : int
          audio sampling rate of S
      - norm       : {'inf', 1, 2, None}
          column-wise normalization:

             'inf' :  max norm

             1 :  l_1 norm 
             
             2 :  l_2 norm
             
             None :  do not normalize

      - kwargs
          Parameters to build the chroma filterbank
          See chromafb() for details.

    :returns:
      - chromagram  : np.ndarray
          Normalized energy for each chroma bin at each frame.

    :raises:
      - ValueError 
          if an improper value is supplied for norm

    """
    n_fft       = (S.shape[0] -1 ) * 2

    spec2chroma = chromafb( sr, n_fft, **kwargs)[:, :S.shape[0]]

    # Compute raw chroma
    raw_chroma  = np.dot(spec2chroma, S)

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


def chromafb(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0, octwidth=None):
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

    fftfrqbins  = n_chroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = np.concatenate( (   [fftfrqbins[0] - 1.5 * n_chroma],
                                        fftfrqbins))

    binwidthbins = np.concatenate(
        (np.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = np.tile(fftfrqbins, (n_chroma, 1))  \
        - np.tile(np.arange(0, n_chroma, dtype='d')[:,np.newaxis], 
        (1,n_fft))

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


#-- Mel spectrogram and MFCCs --#

def dctfb(n_filts, d):
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


def mfcc(S, d=20):
    """Mel-frequency cepstral coefficients

    :parameters:
      - S     : np.ndarray
          log-power Mel spectrogram
      - d     : int
          number of MFCCs to return

    :returns:
      - M     : np.ndarray
          MFCC sequence

    """

    return np.dot(dctfb(d, S.shape[0]), S)


def mel_frequencies(n_mels=40, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands

    :parameters:
      - n_mels    : int
          number of Mel bins  
      - fmin      : float
          minimum frequency (Hz)
      - fmax      : float
          maximum frequency (Hz)
      - htk       : boolean
          use HTK formula instead of Slaney

    :returns:
      - bin_frequencies : ndarray
          ``n_mels+1``-dimensional vector of Mel frequencies

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel  = hz_to_mel(fmin, htk=htk)
    maxmel  = hz_to_mel(fmax, htk=htk)

    mels    = np.arange(minmel, maxmel + 1, (maxmel - minmel)/(n_mels + 1.0))
    
    return  mel_to_hz(mels, htk=htk)


def melfb(sr, n_fft, n_mels=40, fmin=0.0, fmax=None, htk=False):
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
    freqs       = mel_frequencies(n_mels, fmin, fmax, htk)

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm       = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in xrange(n_mels):
        # lower and upper slopes for all bins
        lower   = (fftfreqs - freqs[i])     / (freqs[i+1] - freqs[i])
        upper   = (freqs[i+2] - fftfreqs)   / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i, :size]   = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
   
    return weights


def melspectrogram(y, sr=22050, n_fft=256, hop_length=128, **kwargs):
    """Compute a mel spectrogram from a time series

    :parameters:
      - y : np.ndarray
          audio time-series
      - sr : int
          audio sampling rate of y  
      - n_fft : int
          number of FFT components
      - hop_length : int
          frames to hop

      - kwargs
          Mel filterbank parameters
          See melfb() documentation for details.

    :returns:
      - S : np.ndarray
          Mel spectrogram

    """

    # Compute the STFT
    powspec     = np.abs(librosa.core.stft(y,   
                                      n_fft       =   n_fft, 
                                      hann_w      =   n_fft, 
                                      hop_length  =   hop_length))**2

    # Build a Mel filter
    mel_basis   = melfb(sr, n_fft, **kwargs)

    # Remove everything past the nyquist frequency
    mel_basis   = mel_basis[:, :(n_fft/ 2  + 1)]
    
    return np.dot(mel_basis, powspec)


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

