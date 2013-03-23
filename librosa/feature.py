#!/usr/bin/env python
'''
CREATED:2013-03-13 21:34:58 by Brian McFee <brm2132@columbia.edu>

Feature extraction code:

    Mel spectrogram, chromagram, MFCCs and helper utilities

'''


import librosa
import numpy as np

#-- Frequency conversions --#
def hz_to_mel(frequencies, htk=False):
    '''
    Convert Hz to Mels

    Input:
        frequencies:    scalar or array of frequencies
        htk:            use HTK mel conversion          | default: False 

    Output:
        mels:           input frequencies in Mels
    '''

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
    '''
    Convert mel numbers to frequencies

    '''
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
    '''
    Convert frquencies (Hz) to octave numbers

    Input:
        frequencies:    scalar or vector of frequencies
        A440:           frequency of A440 (in Hz)                   | Default: 440.0

    Output:
        octaves:        octave number for each frequency
    '''
    return np.log2(frequencies / (A440 / 16.0))


#-- CHROMA --#
def chromagram(S, sr, norm='inf', **kwargs):
    '''
    Compute a chromagram from a spectrogram

    Input:
        S:      spectrogram
        sr:     sampling rate of S
        norm:   column-wise chroma normalization
                'inf':  l_infinity norm (max)       | default
                1:      l_1 norm (sum)
                2:      l_2 norm

        **kwargs:   Parameters to build the chroma filterbank
                    See chromafb() for details.
    Output:
        C:      chromagram
    '''
    n_fft        = (S.shape[0] -1 ) * 2

    spec2chroma = chromafb( sr, n_fft, **kwargs)

    # Compute raw chroma
    raw_chroma  = np.dot(spec2chroma, S)

    # Compute normalization factor for each frame
    if norm == 'inf':
        chroma_norm = np.max(np.abs(raw_chroma), axis=0)
    elif norm == 1:
        chroma_norm = np.sum(np.abs(raw_chroma), axis=0)
    elif norm == 2:
        chroma_norm = np.sum( (raw_chroma**2), axis=0) ** 0.5
    else:
        raise ValueError("norm must be one of: 'inf', 1, 2")

    # Tile the normalizer to match raw_chroma's shape
    chroma_norm[chroma_norm == 0] = 1.0
    chroma_norm     = np.tile(1.0/chroma_norm, (raw_chroma.shape[0], 1))

    return chroma_norm * raw_chroma


def chromafb(sr, n_fft, nchroma, A440=440.0, ctroct=5.0, octwidth=None):
    """Create a Filterbank matrix to convert FFT to Chroma.

    Based on Dan Ellis's fft2chromamx.m

    Parameters
    ----------
    sr: int
        Sampling rate of the incoming signal.
    n_fft : int
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

    wts         = np.zeros((nchroma, n_fft))

    fft_res     = float(sr) / n_fft

    frequencies = np.arange(fft_res, sr, fft_res)

    fftfrqbins  = nchroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = np.concatenate( (   [fftfrqbins[0] - 1.5 * nchroma],
                                        fftfrqbins))

    binwidthbins = np.concatenate(
        (np.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = np.tile(fftfrqbins, (nchroma, 1))  \
        - np.tile(np.arange(0, nchroma, dtype='d')[:,np.newaxis], 
        (1,n_fft))

    nchroma2 = round(nchroma / 2.0)

    # Project into range -nchroma/2 .. nchroma/2
    # add on fixed offset of 10*nchroma to ensure all values passed to
    # rem are +ve
    D = np.remainder(D + nchroma2 + 10*nchroma, nchroma) - nchroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (nchroma, 1)))**2)

    # normalize each column
    wts /= np.tile(np.sqrt(np.sum(wts**2, 0)), (nchroma, 1))

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((fftfrqbins/nchroma - ctroct)/octwidth)**2)),
            (nchroma, 1))

    # remove aliasing columns
    wts[:, (1 + n_fft/2):] = 0.0
    return wts


#-- Mel spectrogram and MFCCs --#

def dctfb(n_filts, d):
    '''
    Build a discrete cosine transform basis

    Input:
        n_filts :       number of output components
        d       :       number of input components

    Output:
        D       :       n_filts-by-d DCT basis
    '''

    basis       = np.empty((n_filts, d))
    basis[0, :] = 1.0 / np.sqrt(d)

    samples     = np.arange(1, 2*d, 2) * np.pi / (2.0 * d)

    for i in xrange(1, n_filts):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/d)

    return basis


def mfcc(S, d=20):
    '''
    Mel-frequency cepstral coefficients

    Input:
        S   :   k-by-n      log-amplitude Mel spectrogram
        d   :   number of MFCCs to return               | default: 20
    Output:
        M   :   d-by-n      MFCC sequence
    '''

    return np.dot(dctfb(d, S.shape[0]), S)


def mel_frequencies(n_filts=40, fmin=0, fmax=11025, htk=False):
    '''
    Compute the center frequencies of mel bands

    Input:
        n_filts:    number of Mel bins                  | Default: 40
        fmin:       minimum frequency (Hz)              | Default: 0
        fmax:       maximum frequency (Hz)              | Default: 11025
        htk:        use HTK mels instead of  Slaney     | Default: False

    Output:
        bin_frequencies:    n_filts+1 vector of Mel frequencies
    '''

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel  = hz_to_mel(fmin, htk=htk)
    maxmel  = hz_to_mel(fmax, htk=htk)

    mels    = np.arange(minmel, maxmel + 1, (maxmel - minmel)/(n_filts + 1.0))
    
    return  mel_to_hz(mels, htk=htk)


def melfb(sr, n_fft, n_filts=40, fmin=0.0, fmax=None, htk=False):
    '''
    Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Input:
        sr:         Sampling rate of the incoming signal.
        n_fft:      FFT length to use.
        n_filts:    Number of Mel bands to use.             | default:  40
        fmin:       lowest edge of the Mel bands (in Hz)    | default:  0.0
        fmax:       upper edge of the Mel bands (in Hz)     | default:  sr / 2
        htk:        Use HTK mels instead of Slaney's        | default:  False

    Output:
        M:          (n_filts * n_fft)   Mel transform matrix
                    Note: coefficients above 1+n_fft/2 are 0.

    '''

    if fmax is None:
        fmax = sr / 2.0

    # Initialize the weights
    weights     = np.zeros( (n_filts, n_fft) )

    # Center freqs of each FFT bin
    size        = 1 + n_fft / 2
    fftfreqs    = np.arange( size, dtype=float ) * sr / n_fft

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs       = mel_frequencies(n_filts, fmin, fmax, htk)

    # Slaney-style mel is scaled to be approx constant E per channel
    enorm       = 2.0 / (freqs[2:n_filts+2] - freqs[:n_filts])

    for i in xrange(n_filts):
        # lower and upper slopes for all bins
        lower   = (fftfreqs - freqs[i])     / (freqs[i+1] - freqs[i])
        upper   = (freqs[i+2] - fftfreqs)   / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i, :size]   = np.maximum(0, np.minimum(lower, upper)) * enorm[i]
   
    return weights

def melspectrogram(y, sr=22050, n_fft=256, hop_length=128, **kwargs):
    '''
    Compute a mel spectrogram from a time series

    Input:
        y                   =   the audio signal
        sr                  =   the sampling rate of        | default: 22050
        n_fft               =   FFT window size             | default: 256
        hop_length          =   hop size                    | default: 128

        **kwargs:           =   Mel filterbank parameters
                                See melfb() documentation for details.

    Output:
        S                   =   Mel amplitude spectrogram
    '''

    # Compute the STFT
    specgram    = librosa.stft(y,   n_fft       =   n_fft, 
                                    hann_w      =   n_fft, 
                                    hop_length  =   hop_length)

    # Build a Mel filter
    mel_basis   = melfb(sr, n_fft, **kwargs)

    # Remove everything past the nyquist frequency
    mel_basis   = mel_basis[:, :(n_fft/ 2  + 1)]
    
    return np.dot(mel_basis, np.abs(specgram))


#-- miscellaneous utilities --#
def sync(data, frames, aggregate=np.mean):
    '''
    Synchronous aggregation of a feature matrix

    Input:
        data:       d-by-T              | feature matrix 
        frames:     t-vector            | (ordered) array of frame numbers
        aggregate:  aggregator function | default: np.mean

    Output:
        Y:      d-by-(<=t+1) vector
        where 
                Y[:, i] = aggregate(data[:, F[i-1]:F[i]], axis=1)

        In order to ensure total coverage, boundary points are added to frames
    '''

    (dimension, n_frames) = data.shape

    frames      = np.unique(np.concatenate( ([0], frames, [n_frames]) ))

    data_agg    = np.empty( (dimension, len(frames)-1) )

    start       = frames[0]

    for (i, end) in enumerate(frames[1:]):
        data_agg[:, i] = aggregate(data[:, start:end], axis=1)
        start = end

    return data_agg

