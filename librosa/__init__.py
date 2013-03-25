#!/usr/bin/env python
'''
CREATED:2012-10-20 11:09:30 by Brian McFee <brm2132@columbia.edu>

Top-level class for librosa

Includes constants, core utility functions, etc

'''

import numpy as np
import numpy.fft as fft
import scipy.signal
import os.path
import audioread

# And all the librosa sub-modules
from . import beat, feature, hpss, output

#-- CORE ROUTINES --#
def load(path, sr=22050, mono=True):
    '''
    Load an audio file into a single, long time series

    Input:
        path:       path to the input file
        sr:         target sample rate                      | default: 22050 
                    'None' uses the native sampling rate

        mono:       convert to mono?                        | default: True

    Output:
        y:          the time series
        sr:         the sampling rate
    '''

    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate

        y = [np.frombuffer(frame, '<i2').astype(float) / float(1<<15) 
                for frame in input_file]

        y = np.concatenate(y)
        if input_file.channels > 1:
            if mono:
                y = 0.5 * (y[::2] + y[1::2])
            else:
                y = y.reshape( (-1, 2)).T

    if sr is not None:
        y = resample(y, sr_native, sr)
    else:
        sr = sr_native

    return (y, sr)

def resample(y, orig_sr, target_sr):
    '''
    Resample a signal from orig_sr to target_sr

    Input:
        y:          time series (either mono or stereo)
        orig_sr:    original sample rate of y
        target_sr:  target sample rate
    
    Output:
        y_hat:      resampled signal
    '''

    if orig_sr == target_sr:
        return y

    axis = y.ndim-1

    n_samples = len(y) * target_sr / orig_sr

    y_hat = scipy.signal.resample(y, n_samples, axis=axis)

    return y_hat

def stft(y, n_fft=256, hann_w=None, hop_length=None):
    '''
    Short-time fourier transform

    Inputs:
        y           = the input signal
        n_fft       = the number of FFT components  | default: 256
        hann_w      = size of hann window           | default: = n_fft
        hop_length  = hop length                    | default: = hann_w / 2

    Output:
        D           = complex-valued STFT matrix of y
    '''
    num_samples = len(y)

    if hann_w is None:
        hann_w = n_fft

    if hann_w == 0:
        window = np.ones((n_fft,))
    else:
        lpad = (n_fft - hann_w)/2
        window = np.pad( scipy.signal.hann(hann_w), 
                            (lpad, n_fft - hann_w - lpad), 
                            mode='constant')

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(n_fft / 2)

    n_specbins  = 1 + int(n_fft / 2)
    n_frames    = 1 + int( (num_samples - n_fft) / hop_length)

    # allocate output array
    stft_matrix = np.empty( (n_specbins, n_frames), dtype=np.complex)

    for i in xrange(n_frames):
        sample  = i * hop_length
        frame   = fft.fft(window * y[sample:(sample+n_fft)])

        # Conjugate here to match phase from DPWE code
        stft_matrix[:, i]  = frame[:n_specbins].conj()

    return stft_matrix


def istft(stft_matrix, n_fft=None, hann_w=None, hop_length=None):
    '''
    Inverse short-time fourier transform

    Inputs:
        stft_matrix = STFT matrix
        n_fft       = number of FFT components          | default: 2 * (d.shape[0] -1
        hann_w      = size of hann window               | default: n_fft
        hop_length  = hop length                        | default: hann_w / 2

    Outputs:
        y       = time domain signal reconstructed from d
    '''

    # n = Number of stft frames
    n_frames    = stft_matrix.shape[1]

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[0] - 1)

    if hann_w is None:
        hann_w = n_fft

    if hann_w == 0:
        window = np.ones(n_fft)
    else:
        #   magic number alert!
        #   2/3 scaling is to make stft(istft(.)) identity for 25% hop
        lpad = (n_fft - hann_w)/2
        window = np.pad( scipy.signal.hann(hann_w) * 2.0 / 3.0, 
                            (lpad, n_fft - hann_w - lpad), 
                            mode='constant')

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = n_fft / 2

    y           = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)

        y[sample:(sample+n_fft)]    = (y[sample:(sample+n_fft)] 
                                    + window * fft.ifft(spec).real)

    return y

def logamplitude(S, amin=1e-10, top_db=80.0):
    '''
    Log-scale the amplitude of a spectrogram

    Input:
        S       =   spectrogram
        amin    =   amplitude threshold                     | default: 1e-10
        top_db  =   threshold below max(log(S)) - top_db    | default: 80 

    Output:
        log_S   =   S in dBs
    '''

    log_S   =   20.0 * np.log10(np.maximum(amin, np.abs(S)))

    if top_db is not None:
        log_S = np.maximum(log_S, log_S.max() - top_db)

    return log_S


#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=64):
    '''
    Converts frame counts to time (seconds)

    Input:
        frames:         scalar or n-by-1 vector of frame numbers
        sr:             sampling rate                               | 22050 Hz
        hop_length:     hop length of the frames                    | 64 frames

    Output:
        times:          time (in seconds) of each given frame number
    '''
    return frames * float(hop_length) / float(sr)

def autocorrelate(y, max_size=None):
    '''
        Bounded auto-correlation

        Input:
            y:          t-by-1  vector
            max_size:   (optional) maximum lag                  | None

        Output:
            z:          y's autocorrelation (up to max_size if given)
    '''

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[len(result)/2:]

    if max_size is None:
        return result
    
    return result[:max_size]

def localmax(x):
    '''
        Return 1 where there are local maxima in x (column-wise)
        left edges do not fire, right edges might.
    '''

    return np.logical_and(x > np.hstack([x[0], x[:-1]]), 
                             x >= np.hstack([x[1:], x[-1]]))

