#!/usr/bin/env python
"""Top-level class for librosa

Includes constants, core utility functions, etc.

See also:
  - librosa.beat
  - librosa.feature
  - librosa.hpss
  - librosa.output

CREATED:2012-10-20 11:09:30 by Brian McFee <brm2132@columbia.edu>

"""

import os.path
import audioread

import numpy as np
import numpy.fft as fft
import scipy.signal

# And all the librosa sub-modules
from . import beat, feature, hpss, output

#-- CORE ROUTINES --#
def load(path, sr=22050, mono=True):
    """Load an audio file into a single, long time series

    Arguments:
      path -- (string)    path to the input file
      sr   -- (int > 0)   target sample rate              | default: 22050 
                          'None' uses the native sampling rate
      mono -- (boolean)   convert to mono                 | default: True


    Returns (y, sr):
      y    -- (ndarray)   audio time series
      sr   -- (int)       sampling rate of y

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

def resample(y, orig_sr, target_sr):
    """Resample a signal from orig_sr to target_sr

    Arguments:
      y           -- (ndarray)    audio time series 
      orig_sr     -- (int)        original sample rate of y
      target_sr   -- (int)        target sample rate
    
    Returns y_hat:
      y_hat       -- (ndarray)    y resampled from orig_sr to target_sr

    """

    if orig_sr == target_sr:
        return y

    axis = y.ndim-1

    n_samples = len(y) * target_sr / orig_sr

    y_hat = scipy.signal.resample(y, n_samples, axis=axis)

    return y_hat

def stft(y, n_fft=256, hann_w=None, hop_length=None):
    """Short-time fourier transform

    Arguments:
      y           -- (ndarray)  the input signal
      n_fft       -- (int)      number of FFT components  | default: 256
      hann_w      -- (int)      size of Hann window       | default: n_fft
      hop_length  -- (int)      number audio of frames 
                                between STFT columns      | default: hann_w / 2

    Returns D:
      D           -- (ndarray)  complex-valued STFT matrix

    """
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
    """
    Inverse short-time fourier transform

    Arguments:
      stft_matrix -- (ndarray)  STFT matrix from stft()
      n_fft       -- (int)      number of FFT components   | default: inferred
      hann_w      -- (int)      size of Hann window        | default: n_fft
      hop_length  -- (int)      audio frames between STFT                       
                                columns                    | default: hann_w / 2

    Returns y:
      y           -- (ndarray)  time domain signal reconstructed from d

    """

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
    """Log-scale the amplitude of a spectrogram

    Arguments:
      S       -- (ndarray)  spectrogram
      amin    -- (float)    minimum amplitude threshold     | default: 1e-10
      top_db  -- (float)    threshold max(log(S)) - top_db  | default: 80.0


    Returns log_S:

      log_S   -- (ndarray)   S in dBs, ~= 10 * log10(S)

    """

    log_S   =   10.0 * np.log10(np.maximum(amin, np.abs(S)))

    if top_db is not None:
        log_S = np.maximum(log_S, log_S.max() - top_db)

    return log_S


#-- UTILITIES --#
def frames_to_time(frames, sr=22050, hop_length=64):
    """Converts frame counts to time (seconds)

    Arguments:
      frames     -- (ndarray) vector of frame numbers
      sr         -- (int)     sampling rate             | default: 22050
      hop_length -- (int)     hop length                | default: 64


    Returns times:
      times      -- time (in seconds) of each given frame number
                    times[i] = frames[i] * hop_length / sr
    """
    return frames * float(hop_length) / float(sr)

def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    Arguments:
      y         -- (ndarray) vector to autocorrelate
      max_size  -- (int)     maximum correlation lag    | default: len(y)


    Returns z:
      z         -- (ndarray) truncated autocorrelation y*y

    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[len(result)/2:]

    if max_size is None:
        return result
    
    return result[:max_size]

def localmax(x):
    """Return 1 where there are local maxima in x (column-wise)
       left edges do not fire, right edges might.

    Arguments:
      x     -- (ndarray)    input vector


    Returns m:
      m     -- (ndarray)    boolean indicator vector
                            m[i] <=> x[i] is a local maximum
    """

    return np.logical_and(x > np.hstack([x[0], x[:-1]]), 
                             x >= np.hstack([x[1:], x[-1]]))

