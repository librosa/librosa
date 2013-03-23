#!/usr/bin/env python
'''
CREATED:2013-03-13 21:34:58 by Brian McFee <brm2132@columbia.edu>

Feature extraction code:

    Mel spectrogram, chromagram, MFCCs and helper utilities

'''


import librosa
import numpy

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

    #     TODO:   2012-11-27 11:28:43 by Brian McFee <brm2132@columbia.edu>
    #  too many magic numbers in these functions
    #   redo with informative variable names
    #   then make them into parameters

    if numpy.isscalar(frequencies):
        frequencies = numpy.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * numpy.log10(1.0 + frequencies / 700.0)
    
    # Fill in the linear part
    f_min   = 0.0
    f_sp    = 200.0 / 3

    mels    = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    brkfrq  = 1000.0
    brkpt   = (brkfrq - f_min) / f_sp
    logstep = numpy.log(6.4) / 27.0
    nonlin  = frequencies >= brkfrq

    mels[nonlin]  = brkpt + numpy.log(frequencies[nonlin] / brkfrq) / logstep

    return mels

def mel_to_hz(mels, htk=False):
    '''
    Convert mel numbers to frequencies

    '''
    if numpy.isscalar(mels):
        mels = numpy.array([mels], dtype=float)
    else:
        mels = mels.astype(float)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min       = 0.0
    f_sp        = 200.0 / 3
    frequencies = f_min + f_sp * mels

    # And now the nonlinear scale
    brkfrq      = 1000.0
    brkpt       = (brkfrq - f_min) / f_sp
    logstep     = numpy.log(6.4) / 27.0
    nonlin      = mels >= brkpt

    frequencies[nonlin] = brkfrq * numpy.exp(logstep * (mels[nonlin] - brkpt))

    return frequencies

def hz_to_octs(frequencies, A440=440.0):
    '''
    Convert frquencies (Hz) to octave numbers

    Input:
        frequencies:    scalar or vector of frequencies
        A440:           frequency of A440 (in Hz)                   | Default: 440.0

    Output:
        octaves:        octave number for each frequency
    '''
    return numpy.log2(frequencies / (A440 / 16.0))


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
    raw_chroma  = numpy.dot(spec2chroma, S)

    # Compute normalization factor for each frame
    if norm == 'inf':
        chroma_norm = numpy.max(numpy.abs(raw_chroma), axis=0)
    elif norm == 1:
        chroma_norm = numpy.sum(numpy.abs(raw_chroma), axis=0)
    elif norm == 2:
        chroma_norm = numpy.sum( (raw_chroma**2), axis=0) ** 0.5
    else:
        raise ValueError("norm must be one of: 'inf', 1, 2")

    # Tile the normalizer to match raw_chroma's shape
    chroma_norm[chroma_norm == 0] = 1.0
    chroma_norm     = numpy.tile(1.0/chroma_norm, (raw_chroma.shape[0], 1))

    return chroma_norm * raw_chroma


def chromafb(sr, n_fft, nchroma, A440=440.0, ctroct=5.0, octwidth=0):
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

    wts         = numpy.zeros((nchroma, n_fft))

    frequencies = numpy.arange(float(sr) / n_fft, sr, float(sr) / n_fft)
    fftfrqbins  = nchroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    fftfrqbins = numpy.concatenate( (   [fftfrqbins[0] - 1.5 * nchroma],
                                        fftfrqbins))

    binwidthbins = numpy.concatenate(
        (numpy.maximum(fftfrqbins[1:] - fftfrqbins[:-1], 1.0), [1]))

    D = numpy.tile(fftfrqbins, (nchroma, 1))  \
        - numpy.tile(numpy.arange(0, nchroma, dtype='d')[:,numpy.newaxis], 
        (1,n_fft))

    nchroma2 = round(nchroma / 2.0)

    # Project into range -nchroma/2 .. nchroma/2
    # add on fixed offset of 10*nchroma to ensure all values passed to
    # rem are +ve
    D = numpy.remainder(D + nchroma2 + 10*nchroma, nchroma) - nchroma2

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
    return wts[:, :(n_fft/2+1)]


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

    basis       = numpy.empty((n_filts, d))
    basis[0, :] = 1.0 / numpy.sqrt(d)

    samples     = numpy.arange(1, 2*d, 2) * numpy.pi / (2.0 * d)

    for i in xrange(1, n_filts):
        basis[i, :] = numpy.cos(i*samples) * numpy.sqrt(2.0/d)

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

    return numpy.dot(dctfb(d, S.shape[0]), S)


def mel_frequencies(n_filts=40, fmin=0, fmax=11025, htk=False):
    '''
    Compute the center frequencies of mel bands

    Input:
        n_filts:     number of Mel bins                  | Default: 40
        fmin:       minimum frequency (Hz)              | Default: 0
        fmax:       maximum frequency (Hz)              | Default: 11025
        htk:    use HTK mels instead of  Slaney     | Default: False

    Output:
        bin_frequencies:    n_filts+1 vector of Mel frequencies
    '''

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel      = hz_to_mel(fmin, htk=htk)
    maxmel      = hz_to_mel(fmax, htk=htk)

    mels        = numpy.arange( minmel,     
                                maxmel + 1, 
                                (maxmel - minmel) / (n_filts + 1))
    
    return      mel_to_hz(mels, htk=htk)


def melfb(sr, n_fft, n_filts=40, width=1.0, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Parameters
    ----------
    sr : int
        Sampling rate of the incoming signal.
    n_fft : int
        FFT length to use.
    n_filts : int
        Number of Mel bands to use.  Defaults to 40.
    width : float
        The constant width of each band relative to standard Mel. Defaults 1.0
    fmin : float
        Frequency in Hz of the lowest edge of the Mel bands. Defaults to 0.
    fmax : float
        Frequency in Hz of the upper edge of the Mel bands. Defaults
        to `sr` / 2.
    htk: bool
        Use HTK mels instead of Slaney's version? Defaults to false.

    """

    if fmax is None:
        fmax = sr / 2.0

    # Initialize the weights
    wts         = numpy.zeros( (n_filts, n_fft) )

    # Center freqs of each FFT bin
    fftfreqs    = numpy.arange( 1 + n_fft / 2, dtype=numpy.double ) / n_fft * sr

    # 'Center freqs' of mel bands - uniformly spaced between limits
    binfreqs    = mel_frequencies(n_filts, fmin, fmax, htk)

    for i in xrange(n_filts):
        freqs       = binfreqs[range(i, i+3)]
        
        # scale by width
        freqs       = freqs[1] + width * (freqs - freqs[1])

        # lower and upper slopes for all bins
        loslope     = (fftfreqs - freqs[0]) / (freqs[1] - freqs[0])
        hislope     = (freqs[2] - fftfreqs) / (freqs[2] - freqs[1])

        # .. then intersect them with each other and zero
        wts[i, :(1 + n_fft/2)]    = numpy.maximum(0, 
                                            numpy.minimum(loslope, hislope))


    # Slaney-style mel is scaled to be approx constant E per channel
    enorm   = 2.0 / (binfreqs[2:n_filts+2] - binfreqs[:n_filts])
    wts     = numpy.dot(numpy.diag(enorm), wts)
    
    return wts

def melspectrogram(y, sr=22050, n_fft=256, hop_length=128, **kwargs):
    '''
    Compute a mel spectrogram from a time series

    Input:
        y                   =   the audio signal
        sr                  =   the sampling rate of y                      | default: 22050
        n_fft               =   FFT window size                             | default: 256
        hop_length          =   hop size                                    | default: 128

        **kwargs:           =   Mel filterbank parameters

                                See melfb() documentation for details.

    Output:
        S                   =   Mel amplitude spectrogram
    '''

    # Compute the STFT
    S = librosa.stft(y, n_fft=n_fft, hann_w=n_fft, hop_length=hop_length)

    # Build a Mel filter
    mel_basis   = melfb(sr, n_fft, **kwargs)

    # Remove everything past the nyquist frequency
    mel_basis   = mel_basis[:, :(n_fft/ 2  + 1)]
    
    return numpy.dot(mel_basis, numpy.abs(S))


#-- miscellaneous utilities --#
def sync(data, frames, aggregate=numpy.mean):
    '''
    Synchronous aggregation of a feature matrix

    Input:
        data:       d-by-T              | feature matrix 
        frames:     t-vector            | (ordered) array of frame numbers
        aggregate:  aggregator function | default: numpy.mean

    Output:
        Y:      d-by-(<=t+1) vector
        where 
                Y[:, i] = aggregate(data[:, F[i-1]:F[i]], axis=1)

        In order to ensure total coverage, boundary points are added to frames
    '''

    (dimension, n_frames) = data.shape

    frames      = numpy.unique(numpy.concatenate( ([0], frames, [n_frames]) ))

    data_agg    = numpy.empty( (dimension, len(frames)-1) )

    start       = frames[0]

    for (i, end) in enumerate(frames[1:]):
        data_agg[:, i] = aggregate(data[:, start:end], axis=1)
        start = end

    return data_agg

