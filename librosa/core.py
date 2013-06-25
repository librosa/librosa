#!/usr/bin/env python
"""Core IO, DSP and utility functions."""

import os.path
import audioread

import numpy as np
import numpy.fft as fft
import scipy.signal

# Do we have scikits.samplerate?
try:
    import scikits.samplerate as samplerate
    _HAS_SAMPLERATE = True
except ImportError:
    _HAS_SAMPLERATE = False



#-- CORE ROUTINES --#
def load(path, sr=22050, mono=True):
    """Load an audio file as a floating point time series.

    :parameters:
      - path : string
          path to the input file

      - sr   : int > 0
          target sample rate.
          'None' uses the native sampling rate

      - mono : boolean
          convert signal to mono

    :returns:
      - y    : np.ndarray
          audio time series

      - sr   : int  
          sampling rate of y

    """

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

def resample(y, orig_sr, target_sr, res_type='sinc_fastest'):
    """Resample a signal from orig_sr to target_sr

    :parameters:
      - y           : np.ndarray
          audio time series 

      - orig_sr     : int
          original sample rate of y

      - target_sr   : int
          target sample rate

      - res_type    : str
          resample type (see note)
    
    :returns:
      - y_hat       : np.ndarray
          y resampled from orig_sr to target_sr

    .. note::
        If scikits.samplerate is installed, resample will use res_type
        otherwise, it will fall back on scipy.signal.resample

    """

    if orig_sr == target_sr:
        return y

    if _HAS_SAMPLERATE:
        y_hat = samplerate.resample(y, float(target_sr) / orig_sr, res_type)
    else:
        n_samples = len(y) * target_sr / orig_sr
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)

    return y_hat

def stft(y, n_fft=256, hann_w=None, hop_length=None, window=None):
    """Short-time fourier transform

    :parameters:
      - y           : np.ndarray
          the input signal

      - n_fft       : int
          number of FFT components

      - hann_w      : int
          The size of Hann window. 
          If unspecified, defaults to n_fft

      - hop_length  : int
          number audio of frames between STFT columns.
          If unspecified, defaults hann_w / 2.

      - window      : np.ndarray
          (optional) user-specified window

    :returns:
      - D           : np.ndarray, dtype=complex
          STFT matrix

    """
    num_samples = len(y)

    # if there is no user-specified window, construct it
    if window is None:
        if hann_w is None:
            hann_w = n_fft

        if hann_w == 0:
            window = np.ones((n_fft,))
        else:
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w, sym=False), 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')
    else:
        window = np.asarray(window)
        if window.shape != n_fft:
            raise ValueError('Size mismatch between n_fft and window shape')

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


def istft(stft_matrix, n_fft=None, hann_w=None, hop_length=None, window=None):
    """
    Inverse short-time fourier transform

    :parameters:
      - stft_matrix : np.ndarray
          STFT matrix from stft()

      - n_fft       : int
          number of FFT components.
          If unspecified, n_fft is inferred from the shape of stft_matrix.

      - hann_w      : int
          size of Hann window
          If unspecified, defaults to the value of n_fft.

      - hop_length  : int
          Number of audio frames between STFT columns.
          If unspecified, defaults to hann_w / 2.

      - window      : np.ndarray
          (optional) user-specified window

    :returns:
      - y           : np.ndarray
          time domain signal reconstructed from stft_matrix

    """

    # n = Number of stft frames
    n_frames    = stft_matrix.shape[1]

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[0] - 1)

    # if there is no user-specified window, construct it
    if window is None: 
        if hann_w is None:
            hann_w = n_fft

        if hann_w == 0:
            window = np.ones(n_fft)
        else:
            #   magic number alert!
            #   2/3 scaling is to make stft(istft(.)) identity for 25% hop
            lpad = (n_fft - hann_w)/2
            window = np.pad( scipy.signal.hann(hann_w, sym=False) * 2.0 / 3.0, 
                                (lpad, n_fft - hann_w - lpad), 
                                mode='constant')

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = n_fft / 2

    y = np.zeros(n_fft + hop_length * (n_frames - 1))

    for i in xrange(n_frames):
        sample  = i * hop_length
        spec    = stft_matrix[:, i].flatten()
        spec    = np.concatenate((spec.conj(), spec[-2:0:-1] ), 0)
        ytmp    = window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    return y

def logamplitude(S, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram

    :parameters:
      - S       : np.ndarray
          input spectrogram

      - amin    : float
          minimum amplitude threshold 

      - top_db  : float
          threshold ``max(log(S)) - top_db``

    :returns:
      log_S   : np.ndarray
          ``log_S ~= 10 * log10(S)``

    """

    log_S   =   10.0 * np.log10(np.maximum(amin, np.abs(S)))

    if top_db is not None:
        log_S = np.maximum(log_S, log_S.max() - top_db)

    return log_S


#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=64):
    """Converts frame counts to time (seconds)

    :parameters:
      - frames     : np.ndarray
          vector of frame numbers

      - sr         : int
          audio sampling rate 

      - hop_length : int
          hop length

    :returns:
      - times : np.ndarray 
          time (in seconds) of each given frame number:
          ``times[i] = frames[i] * hop_length / sr``

    """
    return (frames * hop_length) / float(sr)

def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    :parameters:
      - y         : np.ndarray
          vector to autocorrelate

      - max_size  : int
          maximum correlation lag.
          If unspecified, defaults to ``len(y)``

    :returns:
      - z         : np.ndarray
          truncated autocorrelation ``y*y``

    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[len(result)/2:]

    if max_size is None:
        return result
    
    return result[:max_size]

def localmax(x):
    """Return 1 where there are local maxima in x (column-wise)
       left edges do not fire, right edges might.

    :parameters:
      - x     : np.ndarray
          input vector

    :returns:
      - m     : np.ndarray, dtype=boolean
          boolean indicator vector of local maxima:
          m[i] <=> x[i] is a local maximum
    """

    return np.logical_and(x > np.hstack([x[0], x[:-1]]), 
                             x >= np.hstack([x[1:], x[-1]]))

