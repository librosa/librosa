#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core IO, DSP and utility functions."""

import os
import re
import warnings

import audioread
import numpy as np
import numpy.fft as fft
import scipy.signal
import scipy.ndimage

from . import cache
from . import filters
from . import feature
from . import util

# Do we have scikits.samplerate?
try:
    # Pylint won't handle dynamic imports, so we suppress this warning
    import scikits.samplerate as samplerate     # pylint: disable=import-error
    _HAS_SAMPLERATE = True
except ImportError:
    warnings.warn('Could not import scikits.samplerate. ' +
                  'Falling back to scipy.signal')
    _HAS_SAMPLERATE = False

# Constrain STFT block sizes to 128 MB
MAX_MEM_BLOCK = 2**7 * 2**20

SMALL_FLOAT = 1e-20


# -- CORE ROUTINES --#
# Load should never be cached, since we cannot verify that the contents of
# 'path' are unchanged across calls.
def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32):
    """Load an audio file as a floating point time series.

    :usage:
        >>> # Load a wav file
        >>> filename = librosa.util.example_audio_file()
        >>> y, sr = librosa.load(filename)
        >>> y
        array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32)
        >>> sr
        22050

        >>> # Load a wav file and resample to 11 KHz
        >>> filename = librosa.util.example_audio_file()
        >>> y, sr = librosa.load(filename, sr=11025)
        >>> y
        array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32)
        >>> sr
        11025

        >>> # Load 5 seconds of a wav file, starting 15 seconds in
        >>> filename = librosa.util.example_audio_file()
        >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
        >>> y
        array([ 0.066,  0.101,  0.089, ..., -0.097, -0.109,  0.   ],
              dtype=float32)
        >>> sr
        22050

    :parameters:
      - path : string
          path to the input file.
          Any format supported by ``audioread`` will work.

      - sr   : int > 0 [scalar]
          target sampling rate
          'None' uses the native sampling rate

      - mono : bool
          convert signal to mono

      - offset : float
          start reading after this time (in seconds)

      - duration : float
          only load up to this much audio (in seconds)

      - dtype : numeric type
          data type of ``y``

    :returns:
      - y    : np.ndarray [shape=(n,) or (2, n)]
          audio time series

      - sr   : int > 0 [scalar]
          sampling rate of ``y``
    """

    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate

        s_start = int(np.floor(sr_native * offset) * input_file.channels)

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(np.ceil(sr_native * duration)
                                  * input_file.channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start < n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

        if not len(y):
            # Zero-length read
            y = np.zeros(0, dtype=dtype)

        else:
            y = np.concatenate(y)

            if input_file.channels > 1:
                y = y.reshape((-1, 2)).T
                if mono:
                    y = to_mono(y)

            if sr is not None:
                if y.ndim > 1:
                    y = np.vstack([resample(yi, sr_native, sr) for yi in y])
                else:
                    y = resample(y, sr_native, sr)

            else:
                sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)


@cache
def to_mono(y):
    '''Force an audio signal down to mono.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
        >>> y.shape
        (2, 1354752)
        >>> y_mono = librosa.to_mono(y)
        >>> y_mono.shape
        (1354752,)

    :parameters:
        - y : np.ndarray [shape=(2,n) or shape=(n,)]

    :returns:
        - y_mono : np.ndarray [shape=(n,)]
    '''

    # Validate the buffer.  Stereo is ok here.
    util.valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


@cache
def resample(y, orig_sr, target_sr, res_type='sinc_fastest', fix=True,
             **kwargs):
    """Resample a time series from orig_sr to target_sr

    :usage:
        >>> # Downsample from 22 KHz to 8 KHz
        >>> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
        >>> y_8k = librosa.resample(y, sr, 8000)
        >>> y.shape, y_8k.shape
        ((1354752,), (491520,))

    :parameters:
      - y           : np.ndarray [shape=(n,)]
          audio time series

      - orig_sr     : int > 0 [scalar]
          original sampling rate of ``y``

      - target_sr   : int > 0 [scalar]
          target sampling rate

      - res_type    : str
          resample type (see note)

      - fix         : bool
          adjust the length of the resampled signal to be of size exactly
          ``ceil(target_sr * len(y) / orig_sr)``

      - *kwargs*
          If ``fix==True``, additional keyword arguments to pass to
          :func:`librosa.util.fix_length()`.

    :returns:
      - y_hat       : np.ndarray [shape=(n * target_sr / orig_sr,)]
          ``y`` resampled from ``orig_sr`` to ``target_sr``

    .. note::
        If `scikits.samplerate` is installed, :func:`librosa.core.resample`
        will use ``res_type``.
        Otherwise, it will fall back on `scipy.signal.resample`

    """

    # First, validate the audio buffer
    util.valid_audio(y)

    if orig_sr == target_sr:
        return y

    n_samples = int(np.ceil(y.shape[-1] * float(target_sr) / orig_sr))

    if _HAS_SAMPLERATE:
        y_hat = samplerate.resample(y.T,
                                    float(target_sr) / orig_sr,
                                    res_type).T
    else:
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                 center=True):
    """Compute the duration (in seconds) of an audio time series or STFT matrix.

    :usage:
        >>> # Load the example audio file
        >>> y, sr = librosa.load(librosa.util.example_audio())
        >>> librosa.get_duration(y=y, sr=sr)
        61.44

        >>> # Or compute duration from an STFT matrix
        >>> y, sr = librosa.load(librosa.util.example_audio())
        >>> S = librosa.stft(y)
        >>> librosa.get_duration(S=S, sr=sr)
        61.44

        >>> # Or a non-centered STFT matrix
        >>> S_left = librosa.stft(y, center=False)
        >>> librosa.get_duration(S=S_left, sr=sr)
        61.3471201814059

    :parameters:
      - y : np.ndarray [shape=(n,), (2, n)] or None
          Audio time series

      - sr : int > 0 [scalar]
          Audio sampling rate

      - S : np.ndarray [shape=(d, t)] or None
          STFT matrix, or any STFT-derived matrix (e.g., chromagram
          or mel spectrogram).

      - n_fft       : int > 0 [scalar]
          FFT window size for ``S``

      - hop_length  : int > 0 [ scalar]
          number of audio samples between columns of ``S``

      - center      : boolean
          - If ``True``, ``S[:, t]`` is centered at ``y[t * hop_length]``.
          - If ``False``, then ``S[f, t]`` *begins* at ``y[t * hop_length]``

    :returns:
      - d : float >= 0
          Duration (in seconds) of the input time series or spectrogram.
    """

    if y is None:
        assert S is not None

        n_frames = S.shape[1]
        n_samples = n_fft + hop_length * (n_frames - 1)

        # If centered, we lose half a window from each end of S
        if center:
            n_samples = n_samples - 2 * int(n_fft / 2)

    else:
        # Validate the audio buffer.  Stereo is okay here.
        util.valid_audio(y, mono=False)
        if y.ndim == 1:
            n_samples = len(y)
        else:
            n_samples = y.shape[-1]

    return float(n_samples) / sr


@cache
def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
      - ``np.abs(D[f, t])`` is the magnitude of frequency bin ``f``
        at frame ``t``
      - ``np.angle(D[f, t])`` is the phase of frequency bin ``f``
        at frame ``t``

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> D = librosa.stft(y)
        >>> D
        array([[  2.515e-02 -0.000e+00j,   7.316e-02 -0.000e+00j, ...,
                  2.517e-04 -0.000e+00j,   1.452e-04 -0.000e+00j],
               [  5.897e-02 +2.488e-17j,   4.895e-02 +1.744e-02j, ...,
                 -2.114e-04 +1.046e-04j,   9.238e-05 -1.012e-06j],
               ...,
               [ -4.351e-09 -2.131e-17j,   1.778e-08 +8.089e-09j, ...,
                  1.227e-10 +5.685e-11j,  -3.968e-11 -4.419e-13j],
               [ -1.805e-08 -0.000e+00j,  -1.289e-08 -0.000e+00j, ...,
                 -1.181e-10 -0.000e+00j,  -6.003e-11 -0.000e+00j]],
              dtype=complex64)

        >>> # Use left-aligned frames
        >>> D_left = librosa.stft(y, center=False)

        >>> # Use a shorter hop length
        >>> D_short = librosa.stft(y, hop_length=64)


    :parameters:
      - y           : np.ndarray [shape=(n,)], real-valued
          the input signal (audio time series)

      - n_fft       : int > 0 [scalar]
          FFT window size

      - hop_length  : int > 0 [scalar]
          number audio of frames between STFT columns.
          If unspecified, defaults ``win_length / 4``.

      - win_length  : int <= n_fft [scalar]
          Each frame of audio is windowed by ``window()``.
          The window will be of length ``win_length`` and then padded
          with zeros to match ``n_fft``.

          If unspecified, defaults to ``win_length = n_fft``.

      - window      : None, function, np.ndarray [shape=(n_fft,)]
          - None (default): use an asymmetric Hann window
          - a window function, such as ``scipy.signal.hanning``
          - a vector or array of length ``n_fft``

      - center      : boolean
          - If ``True``, the signal ``y`` is padded so that frame
            ``D[f, t]`` is centered at ``y[t * hop_length]``.
          - If ``False``, then ``D[f, t]`` *begins* at ``y[t * hop_length]``

      - dtype       : numeric type
          Complex numeric type for ``D``.  Default is 64-bit complex.

    :returns:
      - D           : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
          STFT matrix
    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window
        fft_window = scipy.signal.hann(win_length, sym=False)

    elif hasattr(window, '__call__'):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        util.valid_audio(y)
        y = np.pad(y, int(n_fft / 2), mode='reflect')

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft / 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0]
                                     * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window
                                             * y_frames[:, bl_s:bl_t],
                                             axis=0).conj()

    return stft_matrix


@cache
def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform.

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> D = librosa.stft(y)
        >>> y_hat = librosa.istft(D)
        >>> y_hat
        array([  1.121e-10,   1.093e-10, ...,   4.644e-14,   3.913e-14],
              dtype=float32)

    :parameters:
      - stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
          STFT matrix from :func:`librosa.core.stft()`

      - hop_length  : int > 0 [scalar]
          Number of frames between STFT columns.
          If unspecified, defaults to ``win_length / 4``.

      - win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
          When reconstructing the time series, each frame is windowed
          according to the ``window`` function (see below).

          If unspecified, defaults to ``n_fft``.

      - window      : None, function, np.ndarray [shape=(n_fft,)]
          - None (default): use an asymmetric Hann window * 2/3
          - a window function, such as ``scipy.signal.hanning``
          - a user-specified window vector of length ``n_fft``

      - center      : boolean
          - If `True`, `D` is assumed to have centered frames.
          - If `False`, `D` is assumed to have left-aligned frames.

      - dtype       : numeric type
          Real numeric type for ``y``.  Default is 32-bit float.

    :returns:
      - y           : np.ndarray [shape=(n,)]
          time domain signal reconstructed from ``stft_matrix``
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        # 2/3 scaling is to make stft(istft(.)) identity for 25% hop
        ifft_window = scipy.signal.hann(win_length, sym=False) * (2.0 / 3)

    elif hasattr(window, '__call__'):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ValueError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = util.pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    y = np.zeros(n_fft + hop_length * (n_frames - 1), dtype=dtype)

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample+n_fft)] = y[sample:(sample+n_fft)] + ytmp

    if center:
        y = y[int(n_fft / 2):-int(n_fft / 2)]

    return y


@cache
def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           norm=False, center=True, dtype=np.complex64):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by Toshihiro Abe et al. in ICASSP'95, Eurospeech'97.

    Calculates regular STFT as a side effect.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> frequencies, D = librosa.ifgram(y, sr=sr)
        >>> frequencies
        array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
               [  2.613e+01,   3.606e+01, ...,   8.199e+00,   3.845e+01],
               ...,
               [  1.096e+04,   5.650e+03, ...,   1.101e+04,   1.101e+04],
               [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])

    :parameters:
      - y       : np.ndarray [shape=(n,)]
          audio time series

      - sr      : int > 0 [scalar]
          sampling rate of ``y``

      - n_fft   : int > 0 [scalar]
          FFT window size

      - hop_length : int > 0 [scalar]
          hop length, number samples between subsequent frames.
          If not supplied, defaults to ``win_length / 4``.

      - win_length : int > 0, <= n_fft
          Window length. Defaults to ``n_fft``.
          See :func:`librosa.core.stft()` for details.

      - norm : bool
          Normalize the STFT.

      - center      : boolean
          - If ``True``, the signal ``y`` is padded so that frame
            ``D[f, t]`` is centered at ``y[t * hop_length]``.
          - If ``False``, then ``D[f, t]`` *begins* at ``y[t * hop_length]``

      - dtype       : numeric type
          Complex numeric type for ``D``.  Default is 64-bit complex.

    :returns:
      - if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
          Instantaneous frequency spectrogram:
          ``if_gram[f, t]`` is the frequency at bin ``f``, time ``t``

      - D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
          Short-time Fourier transform

    .. note::
        - Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
          "Harmonics tracking and pitch extraction based on
          instantaneous frequency."
          Acoustics, Speech, and Signal Processing, 1995. ICASSP-95.,
          1995 International Conference on. Vol. 1. IEEE, 1995.
    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length / 4)

    # Construct a padded hann window
    window = util.pad_center(scipy.signal.hann(win_length, sym=False), n_fft)

    # Window for discrete differentiation
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)

    d_window = np.sin(-freq_angular) * np.pi / n_fft

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       window=window, center=center, dtype=dtype)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center, dtype=dtype).conj()

    # Compute power normalization. Suppress zeros.
    power = np.abs(stft_matrix)**2
    power[power < SMALL_FLOAT] = 1.0

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1))

    if_gram = ((freq_angular[:n_fft/2 + 1]
                + (stft_matrix * diff_stft).imag / power)
               * float(sr) / (2.0 * np.pi))

    if norm:
        stft_matrix = stft_matrix * 2.0 / window.sum()

    return if_gram, stft_matrix


@cache
def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> D = librosa.stft(y)
        >>> magnitude, phase = librosa.magphase(D)
        >>> magnitude
        array([[  2.515e-02,   7.316e-02, ...,   2.517e-04,   1.452e-04],
               [  5.897e-02,   5.196e-02, ...,   2.359e-04,   9.238e-05],
               ...,
               [  4.351e-09,   1.953e-08, ...,   1.352e-10,   3.969e-11],
               [  1.805e-08,   1.289e-08, ...,   1.181e-10,   6.003e-11]],
              dtype=float32)
        >>> phase
        array([[ 1.000 +0.000e+00j,  1.000 +0.000e+00j, ...,
                 1.000 +0.000e+00j,  1.000 +0.000e+00j],
               [ 1.000 +4.220e-16j,  0.942 +3.356e-01j, ...,
                -0.896 +4.435e-01j,  1.000 -1.096e-02j],
               ...,
               [-1.000 +8.742e-08j,  0.910 +4.141e-01j, ...,
                 0.907 +4.205e-01j, -1.000 -1.114e-02j],
               [-1.000 +8.742e-08j, -1.000 +8.742e-08j, ...,
                -1.000 +8.742e-08j, -1.000 +8.742e-08j]], dtype=complex64)
        >>> # Or get the phase angle (in radians)
        >>> np.angle(phase)
        array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
               [  4.220e-16,   3.422e-01, ...,   2.682e+00,  -1.096e-02],
               ...,
               [  3.142e+00,   4.270e-01, ...,   4.339e-01,  -3.130e+00],
               [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]],
              dtype=float32)

    :parameters:
      - D       : np.ndarray [shape=(d, t), dtype=complex]
          complex-valued spectrogram

    :returns:
      - D_mag   : np.ndarray [shape=(d, t), dtype=real]
          magnitude of ``D``
      - D_phase : np.ndarray [shape=(d, t), dtype=complex]
          ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``
    """

    mag = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


@cache
def cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, tuning=None, resolution=2, res_type='sinc_best',
        aggregate=None):
    '''Compute the constant-Q transform of an audio signal.

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> C = librosa.cqt(y, sr=sr)
        >>> C
        array([[  3.985e-01,   4.696e-01, ...,   7.009e-04,   8.497e-04],
               [  1.135e+00,   1.220e+00, ...,   1.669e-03,   1.691e-03],
               ...,
               [  6.036e-04,   3.765e-02, ...,   3.100e-14,   0.000e+00],
               [  4.690e-04,   8.762e-02, ...,   1.995e-14,   0.000e+00]])

        >>> # Limit the frequency range
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C3'),
                            n_bins=60)
        >>> C
        array([[  8.936e+01,   9.573e+01, ...,   1.333e-02,   1.443e-02],
               [  8.181e+01,   8.231e+01, ...,   2.552e-02,   2.147e-02],
               ...,
               [  2.791e-03,   2.463e-02, ...,   9.306e-04,   0.000e+00],
               [  2.687e-03,   1.446e-02, ...,   8.878e-04,   0.000e+00]])

        >>> # Use higher resolution
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C3'),
                            n_bins=60 * 2, bins_per_octave=12 * 2)
        >>> C
        array([[  1.000e+02,   1.094e+02, ...,   8.701e-02,   8.098e-02],
               [  2.222e+02,   2.346e+02, ...,   5.625e-02,   4.770e-02],
               ...,
               [  5.177e-02,   1.710e-02, ...,   4.670e-03,   7.403e-12],
               [  1.981e-02,   2.721e-03, ...,   1.943e-03,   7.246e-12]])

    :parameters:
      - y : np.ndarray [shape=(n,)]
          audio time series

      - sr : int > 0 [scalar]
          sampling rate of ``y``

      - hop_length : int > 0 [scalar]
          number of samples between successive CQT columns.

          .. note:: ``hop_length`` must be at least
            ``2**(n_bins / bins_per_octave)``

      - fmin : float > 0 [scalar]
          Minimum frequency. Defaults to C2 ~= 32.70 Hz

      - n_bins : int > 0 [scalar]
          Number of frequency bins, starting at `fmin`

      - bins_per_octave : int > 0 [scalar]
          Number of bins per octave

      - tuning : None or float in ``[-0.5, 0.5)``
          Tuning offset in fractions of a bin (cents)
          If ``None``, tuning will be automatically estimated.

      - resolution : float > 0
          Filter resolution factor. Larger values use longer windows.

      - res_type : str
          Resampling type, see :func:`librosa.core.resample()` for details.

      - aggregate : None or function
          Aggregation function for time-oversampling energy aggregation.
          By default, ``np.mean``.  See :func:`librosa.feature.sync()`.

    :returns:
      - CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
          Constant-Q energy for each frequency at each time.

    .. note:: This implementation is based on the recursive sub-sampling method
        described by Schoerkhuber and Klapuri, 2010.

        - Schoerkhuber, Christian, and Anssi Klapuri.
            "Constant-Q transform toolbox for music processing."
            7th Sound and Music Computing Conference, Barcelona, Spain. 2010.
    '''

    if fmin is None:
        # C2 by default
        fmin = note_to_hz('C2')

    if tuning is None:
        tuning = feature.estimate_tuning(y=y, sr=sr)

    # First thing, get the fmin of the top octave
    freqs = cqt_frequencies(n_bins + 1, fmin, bins_per_octave=bins_per_octave)
    fmin_top = freqs[-bins_per_octave-1]

    # Generate the basis filters
    basis = np.asarray(filters.constant_q(sr,
                                          fmin=fmin_top,
                                          n_bins=bins_per_octave,
                                          bins_per_octave=bins_per_octave,
                                          tuning=tuning,
                                          resolution=resolution,
                                          pad=True))

    # FFT the filters
    max_filter_length = basis.shape[1]
    n_fft = int(2.0**(np.ceil(np.log2(max_filter_length))))

    # Conjugate-transpose the basis
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1).conj()

    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    # Make sure our hop is long enough to support the bottom octave
    assert hop_length >= 2**n_octaves

    def __variable_hop_response(my_y, target_hop):
        '''Compute the filter response with a target STFT hop.
        If the hop is too large (more than half the frame length),
        then over-sample at a smaller hop, and aggregate the results
        to the desired resolution.
        '''

        # If target_hop <= n_fft / 2:
        #   my_hop = target_hop
        # else:
        #   my_hop = target_hop * 2**(-k)

        zoom_factor = int(np.maximum(0,
                                     1 + np.ceil(np.log2(target_hop)
                                                 - np.log2(n_fft))))

        my_hop = int(target_hop / (2**(zoom_factor)))

        assert my_hop > 0

        # Compute the STFT matrix
        D = stft(my_y, n_fft=n_fft, hop_length=my_hop)

        D = np.vstack([D.conj(), D[-2:0:-1]])

        # And filter response energy
        my_cqt = np.abs(fft_basis.dot(D))

        if zoom_factor > 0:
            # We need to aggregate.  Generate the boundary frames
            bounds = list(np.arange(0, my_cqt.shape[1], 2**(zoom_factor)))
            my_cqt = feature.sync(my_cqt, bounds,
                                  aggregate=aggregate)

        return my_cqt

    cqt_resp = []

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for _ in range(n_octaves):
        # Compute a dynamic hop based on n_fft
        my_cqt = __variable_hop_response(my_y, my_hop)

        # Convolve
        cqt_resp.append(my_cqt)

        # Resample
        my_y = resample(my_y, my_sr, my_sr/2.0, res_type=res_type)
        my_sr = my_sr / 2.0
        my_hop = int(my_hop / 2.0)

    # cleanup any framing errors at the boundaries
    max_col = min([x.shape[1] for x in cqt_resp])

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    cqt_resp = cqt_resp[-n_bins:]

    # Transpose magic here to ensure column-contiguity
    return np.ascontiguousarray(cqt_resp.T).T


@cache
def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of ``rate``

    :usage:
        >>> # Play at double speed
        >>> y, sr   = librosa.load(librosa.util.example_audio_file())
        >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
        >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
        >>> y_fast  = librosa.istft(D_fast, hop_length=512)

        >>> # Or play at 1/3 speed
        >>> y, sr   = librosa.load(librosa.util.example_audio_file())
        >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
        >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
        >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    :parameters:
      - D       : np.ndarray [shape=(d, t), dtype=complex]
          STFT matrix

      - rate    :  float > 0 [scalar]
          Speed-up factor: ``rate > 1`` is faster, ``rate < 1`` is slower.

      - hop_length : int > 0 [scalar] or None
          The number of samples between successive columns of ``D``.
          If None, defaults to ``n_fft/4 = (D.shape[0]-1)/2``

    :returns:
      - D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
          time-stretched STFT

    .. note::
      - Ellis, D. P. W. "A phase vocoder in Matlab." Columbia University
        (http://www.ee.columbia.edu/dpwe/resources/matlab/pvoc/) (2002).
    """

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft / 4)

    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0])
               + alpha * np.abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1])
                  - np.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


# -- FREQUENCY UTILITIES AND CONVERTERS -- #
def note_to_hz(note, **kwargs):
    '''Convert one or more note names to frequency (Hz)

    :usage:
        >>> # Get the frequency of a note
        >>> librosa.note_to_hz('C')
        array([ 8.176])
        >>> # Or multiple notes
        >>> librosa.note_to_hz(['A3', 'A4', 'A5'])
        array([ 110.,  220.,  440.])
        >>> # Or notes with tuning deviations
        >>> librosa.note_to_hz('C2-32', round_midi=False)
        array([ 32.104])

    :parameters:
        - note : str or iterable of str
            One or more note names to convert

        - kwargs : keyword arguments
            Additional parameters to :func:`note_to_midi`

    :returns:
        - frequencies : np.ndarray [shape=(len(note),)]
            Array of frequencies (in Hz) corresponding to ``note``

    .. seealso:: :func:`midi_to_hz`, :func:`note_to_midi`, :func:`note_to_hz`

    '''
    return midi_to_hz(note_to_midi(note, **kwargs))


def note_to_midi(note, round_midi=True):
    '''Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with `#`, flats may be indicated with `!` or `b`.

    :usage:
        >>> librosa.note_to_midi('C')
        0
        >>> librosa.note_to_midi('C#3')
        37
        >>> librosa.note_to_midi('f4')
        53
        >>> librosa.note_to_midi('Bb-1')
        -2
        >>> librosa.note_to_midi('A!8')
        104
        >>> # Lists of notes also work
        >>> librosa.note_to_midi(['C', 'E', 'G'])
        array([0, 4, 7])


    :parameters:
      - note : str or iterable of str
          One or more note names.

      - round_midi : bool
          If True, allow for fractional midi notes
          Otherwise, round cent deviations to the nearest note
    :returns:
      - midi : float or np.array
          Midi note numbers corresponding to inputs.
    '''

    if not isinstance(note, str):
        return np.array([note_to_midi(n) for n in note])

    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1}

    try:
        match = re.match(r'^(?P<n>[A-Ga-g])'
                         r'(?P<off>[#b!]?)'
                         r'(?P<oct>[+-]?\d*)'
                         r'(?P<cents>[+-]?\d*)$',
                         note)

        pitch = match.group('n').upper()
        offset = acc_map[match.group('off')]
        octave = match.group('oct')
        cents = match.group('cents')
        if not octave:
            octave = 0
        else:
            octave = int(octave)

        if round_midi or not cents:
            cents = 0
        else:
            cents = int(cents) * 1e-2

    except:
        raise ValueError('Improper note format: {:s}'.format(note))

    return 12 * octave + pitch_map[pitch] + offset + cents


def midi_to_note(midi, octave=True, cents=False):
    '''Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C#0', 'D0', ...

    :usage:
        >>> librosa.midi_to_note(0)
        'C0'
        >>> librosa.midi_to_note(37)
        'C#3'
        >>> librosa.midi_to_note(-2)
        'A#-1'
        >>> librosa.midi_to_note(104.7)
        'A8'
        >>> librosa.midi_to_note(104.7, cents=True)
        'A8-30'
        >>> librosa.midi_to_note(range(12))
        ['C0', 'C#0', 'D0', 'D#0', 'E0', 'F0',
         'F#0', 'G0', 'G#0', 'A0', 'A#0', 'B0']

    :parameters:
      - midi : int or iterable of int
          Midi numbers to convert.

      - octave: bool
          If True, include the octave number

      - cents: bool
          If true, cent markers will be appended for fractional notes.
          Eg, ``midi_to_note(69.3, cents=True)`` == ``A5+03``

    :returns:
      - notes : str or iterable of str
          Strings describing each midi note.
    '''

    if not np.isscalar(midi):
        return [midi_to_note(x, octave=octave, cents=cents) for x in midi]

    note_map = ['C', 'C#', 'D', 'D#',
                'E', 'F', 'F#', 'G',
                'G#', 'A', 'A#', 'B']

    note_num = int(np.round(midi))
    note_cents = int(100 * np.around(midi - note_num, 2))

    note = note_map[note_num % 12]

    if octave:
        note = '{:s}{:0d}'.format(note, int(note_num / 12))
    if cents:
        note = '{:s}{:+02d}'.format(note, note_cents)

    return note


def midi_to_hz(notes):
    """Get the frequency (Hz) of MIDI note(s)

    :usage:
        >>> librosa.midi_to_hz(36)
        array([ 65.406])

        >>> librosa.midi_to_hz(np.arange(36, 48))
        array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
                 92.499,   97.999,  103.826,  110.   ,  116.541,  123.471])

    :parameters:
      - notes       : int or np.ndarray [shape=(n,), dtype=int]
          midi number(s) of the note(s)

    :returns:
      - frequency   : np.ndarray [shape=(n,), dtype=float]
          frequency (frequencies) of ``notes`` in Hz
    """

    notes = np.asarray([notes]).flatten()
    return 440.0 * (2.0 ** ((notes - 69.0)/12.0))


def hz_to_midi(frequencies):
    """Get the closest MIDI note number(s) for given frequencies

    :usage:
        >>> librosa.hz_to_midi(60)
        array([ 34.506])
        >>> librosa.hz_to_midi([110, 220, 440])
        array([ 45.,  57.,  69.])

    :parameters:
      - frequencies   : float or np.ndarray [shape=(n,), dtype=float]
          frequencies to convert

    :returns:
      - note_nums     : np.ndarray [shape=(n,), dtype=int]
          closest MIDI notes to ``frequencies``
    """

    frequencies = np.asarray([frequencies]).flatten()
    return 12 * (np.log2(frequencies) - np.log2(440.0)) + 69


def hz_to_note(frequencies, **kwargs):
    '''Convert one or more frequencies (in Hz) to the nearest note names.

    :usage:
        >>> # Get a single note name for a frequency
        >>> librosa.hz_to_note(440.0)
        ['A5']
        >>> # Get multiple notes with cent deviation
        >>> librosa.hz_to_note([32, 64], cents=True)
        ['C2-38', 'C3-38']
        >>> # Get multiple notes, but suppress octave labels
        >>> librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 12)),
                               octave=False)
        ['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#', 'A']

    :parameters:
        - frequencies : float or iterable of float
            Input frequencies, specified in Hz

        - kwargs : additional keyword arguments
            Arguments passed through to :func:`midi_to_note`

    :returns:
        - notes : list of str
            ``notes[i]`` is the closest note name to ``frequency[i]``
            (or ``frequency`` if the input is scalar)

    .. seealso:: :func:`hz_to_midi`, :func:`midi_to_note`, :func:`note_to_hz`
    '''
    return midi_to_note(hz_to_midi(frequencies), **kwargs)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    :usage:
        >>> librosa.hz_to_mel(60)
        array([0.9])
        >>> librosa.hz_to_mel([110, 220, 440])
        array([ 1.65,  3.3 ,  6.6 ])

    :parameters:
      - frequencies   : np.ndarray [shape=(n,)] , float
          scalar or array of frequencies
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - mels        : np.ndarray [shape=(n,)]
          input frequencies in Mels
    """

    frequencies = np.asarray([frequencies]).flatten()

    if np.isscalar(frequencies):
        frequencies = np.array([frequencies], dtype=float)
    else:
        frequencies = frequencies.astype(float)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    :usage:
        >>> librosa.mel_to_hz(3)
        array([ 200.])

        >>> librosa.mel_to_hz([1,2,3,4,5])
        array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    :parameters:
      - mels          : np.ndarray [shape=(n,)], float
          mel bins to convert
      - htk           : bool
          use HTK formula instead of Slaney

    :returns:
      - frequencies   : np.ndarray [shape=(n,)]
          input mels in Hz
    """

    mels = np.asarray([mels], dtype=float).flatten()

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def hz_to_octs(frequencies, A440=440.0):
    """Convert frequencies (Hz) to (fractional) octave numbers.

    :usage:
        >>> librosa.hz_to_octs(440.0)
        array([ 4.])
        >>> librosa.hz_to_octs([32, 64, 128, 256])
        array([ 0.219,  1.219,  2.219,  3.219])

    :parameters:
      - frequencies   : np.ndarray [shape=(n,)] or float
          scalar or vector of frequencies
      - A440          : float
          frequency of A440

    :returns:
      - octaves       : np.ndarray [shape=(n,)]
          octave number for each frequency

    """
    frequencies = np.asarray([frequencies]).flatten()
    return np.log2(frequencies / (float(A440) / 16))


def octs_to_hz(octs, A440=440.0):
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    :usage:
        >>> librosa.octs_to_hz(1)
        array([ 55.])
        >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
        array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    :parameters:
      - octaves       : np.ndarray [shape=(n,)] or float
          octave number for each frequency
      - A440          : float
          frequency of A440

    :returns:
      - frequencies   : np.ndarray [shape=(n,)]
          scalar or vector of frequencies
    """
    octs = np.asarray([octs]).flatten()
    return (float(A440) / 16)*(2.0**octs)


def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of ``np.fft.fftfreqs``

    :usage:
        >>> librosa.fft_frequencies(sr=22050, n_fft=16)
        array([     0.   ,   1378.125,   2756.25 ,   4134.375,   5512.5  ,
                 6890.625,   8268.75 ,   9646.875,  11025.   ])

    :parameters:
      - sr : int > 0 [scalar]
          Audio sampling rate

      - n_fft : int > 0 [scalar]
          FFT window size

    :returns:
      - freqs : np.ndarray [shape=(1 + n_fft/2,)]
          Frequencies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2)
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft/2),
                       endpoint=True)


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.

    :usage:
        >>> # Get the CQT frequencies for 24 notes, starting at C2
        >>> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
        array([  32.703,   34.648,   36.708,   38.891,   41.203,   43.654,
                 46.249,   48.999,   51.913,   55.   ,   58.27 ,   61.735,
                 65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
                 92.499,   97.999,  103.826,  110.   ,  116.541,  123.471])

    :parameters:
      - n_bins  : int > 0 [scalar]
          Number of constant-Q bins

      - fmin    : float > 0 [scalar]
          Minimum frequency

      - bins_per_octave : int > 0 [scalar]
          Number of bins per octave

      - tuning : float in ``[-0.5, +0.5)``
          Deviation from A440 tuning in fractional bins (cents)

    :returns:
      - frequencies : np.ndarray [shape=(n_bins,)]
          Center frequency for each CQT bin
    """

    correction = 2.0**(float(tuning) / bins_per_octave)
    frequencies = 2.0**(np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False,
                    extra=False):
    """Compute the center frequencies of mel bands

    :usage:
        >>> librosa.mel_frequencies(n_mels=40)
        array([    0.   ,    81.155,   162.311,   243.466,   324.622,
                 405.777,   486.933,   568.088,   649.244,   730.399,
                 811.554,   892.71 ,   973.865,  1058.382,  1150.775,
                1251.232,  1360.46 ,  1479.222,  1608.352,  1748.754,
                1901.413,  2067.399,  2247.874,  2444.104,  2657.464,
                2889.45 ,  3141.687,  3415.943,  3714.14 ,  4038.369,
                4390.902,  4774.209,  5190.978,  5644.128,  6136.837,
                6672.557,  7255.043,  7888.378,  8577.001,  9325.737])

    :parameters:
      - n_mels    : int > 0 [scalar]
          number of Mel bins

      - fmin      : float >= 0 [scalar]
          minimum frequency (Hz)

      - fmax      : float >= 0 [scalar]
          maximum frequency (Hz)

      - htk       : bool
          use HTK formula instead of Slaney

      - extra     : bool
          include extra frequencies necessary for building Mel filters

    :returns:
      - bin_frequencies : ndarray [shape=(n_mels,)]
          vector of Mel frequencies
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz_to_mel(fmin, htk=htk)
    maxmel = hz_to_mel(fmax, htk=htk)

    mels = np.arange(minmel, maxmel + 1, (maxmel - minmel) / (n_mels + 1.0))

    if not extra:
        mels = mels[:n_mels]

    return mel_to_hz(mels, htk=htk)


# A-weighting should be capitalized: suppress the naming warning
def A_weighting(frequencies, min_db=-80.0):     # pylint: disable=invalid-name
    '''Compute the A-weighting of a set of frequencies.

    :usage:
        >>> # Get the A-weighting for 20 Mel frequencies
        >>> freqs = librosa.mel_frequencies(20)
        >>> librosa.A_weighting(freqs)
        array([-80.   , -13.355,  -6.594,  -3.574,  -1.877,  -0.835,  -0.16 ,
                 0.316,   0.684,   0.953,   1.135,   1.239,   1.271,   1.232,
                 1.116,   0.916,   0.615,   0.193,  -0.374,  -1.113])

    :parameters:
      - frequencies : scalar or np.ndarray [shape=(n,)]
          One or more frequencies (in Hz)

      - min_db : float [scalar] or None
          Clip weights below this threshold.
          If ``None``, no clipping is performed.

    :returns:
      - A_weighting : scalar or np.ndarray [shape=(n,)]
          ``A_weighting[i]`` is the A-weighting of ``frequencies[i]``
    '''

    # Vectorize to make our lives easier
    frequencies = np.asarray([frequencies]).flatten()

    # Pre-compute squared frequeny
    f_sq = frequencies**2.0

    const = np.array([12200, 20.6, 107.7, 737.9])**2.0

    r_a = const[0] * f_sq**2
    r_a /= (f_sq + const[0]) * (f_sq + const[1])
    r_a /= np.sqrt((f_sq + const[2]) * (f_sq + const[3]))

    weights = 2.0 + 20 * np.log10(r_a)

    if min_db is not None:
        weights = np.maximum(min_db, weights)

    return weights


# -- Magnitude scaling -- #
@cache
def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    :usage:
        >>> # Get a power spectrogram from a waveform y
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> S = np.abs(librosa.stft(y)) ** 2
        >>> librosa.logamplitude(S)
        array([[-31.988, -22.714, ..., -33.325, -33.325],
               [-24.587, -25.686, ..., -33.325, -33.325],
               ...,
               [-33.325, -33.325, ..., -33.325, -33.325],
               [-33.325, -33.325, ..., -33.325, -33.325]], dtype=float32)

        >>> # Compute dB relative to peak power
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> S = np.abs(librosa.stft(y)) ** 2
        >>> librosa.logamplitude(S, ref_power=np.max)
        array([[-78.663, -69.389, ..., -80.   , -80.   ],
               [-71.262, -72.361, ..., -80.   , -80.   ],
               ...,
               [-80.   , -80.   , ..., -80.   , -80.   ],
               [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)

        >>> # Or compare to median power
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> S = np.abs(librosa.stft(y)) ** 2
        >>> librosa.logamplitude(S, ref_power=np.median)
        array([[  3.279,  12.552, ...,   1.942,   1.942],
               [ 10.68 ,   9.581, ...,   1.942,   1.942],
               ...,
               [  1.942,   1.942, ...,   1.942,   1.942],
               [  1.942,   1.942, ...,   1.942,   1.942]], dtype=float32)

    :parameters:
      - S       : np.ndarray [shape=(d, t)]
          input spectrogram

      - ref_power : scalar or function
          If scalar, ``log(abs(S))`` is compared to ``log(ref_power)``.
          If a function, ``log(abs(S))`` is compared to
          ``log(ref_power(abs(S)))``.

          This is primarily useful for comparing to the maximum value of ``S``.

      - amin    : float [scalar]
          minimum amplitude threshold for ``abs(S)`` and ``ref_power``

      - top_db  : float [scalar]
          threshold log amplitude at top_db below the peak:
          ``max(log(S)) - top_db``

    :returns:
      log_S   : np.ndarray [shape=(d, t)]
          ``log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))``
    """

    magnitude = np.abs(S)

    if hasattr(ref_power, '__call__'):
        # User supplied a function to calculate reference power
        __ref = ref_power(magnitude)
    else:
        __ref = np.abs(ref_power)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, __ref))

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@cache
def perceptual_weighting(S, frequencies, **kwargs):
    '''Perceptual weighting of a power spectrogram:

    ``S_p[f] = A_weighting(f) + 10*log(S[f] / ref_power)``

    :usage:
        >>> # Re-weight a CQT representation, using peak power as reference
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> CQT = librosa.cqt(y, sr=sr, fmin=55)
        >>> freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=55)
        >>> librosa.perceptual_weighting(CQT, freqs, ref_power=np.max)
        array([[-50.206, -50.024, ..., -97.805, -90.37 ],
               [-48.665, -48.196, ..., -81.633, -81.323],
               ...,
               [-55.747, -38.991, ..., -80.098, -80.098],
               [-55.259, -40.938, ..., -80.311, -80.311]])

    :parameters:
      - S : np.ndarray [shape=(d, t)]
          Power spectrogram

      - frequencies : np.ndarray [shape=(d,)]
          Center frequency for each row of ``S``

      - *kwargs*
          Additional keyword arguments to :func:`librosa.core.logamplitude`.

    :returns:
      - S_p : np.ndarray [shape=(d, t)]
          perceptually weighted version of ``S``
    '''

    offset = A_weighting(frequencies).reshape((-1, 1))

    return offset + logamplitude(S, **kwargs)


# -- UTILITIES -- #
def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> beat_samples = librosa.frames_to_samples(beats, hop_length=64)
        >>> beat_samples[:20]
        array([  1472,  11328,  21824,  32064,  42112,  52160,  62464,  72448,
                82688,  92608, 103168, 113472, 123584, 133568, 143872, 153856,
               163904, 174336, 184704, 195200])

    :parameters:
      - frames     : np.ndarray [shape=(n,)]
          vector of frame indices

      - hop_length : int > 0 [scalar]
          number of samples between successive frames

      - n_fft : None or int > 0 [scalar]
          Optional: length of the FFT window.
          If given, time conversion will include an offset of ``n_fft / 2``
          to counteract windowing effects when using a non-centered STFT.

    :returns:
      - times : np.ndarray [shape=(n,)]
          time (in seconds) of each given frame number:
          ``times[i] = frames[i] * hop_length``

    .. seealso:: :func:`frames_to_time`
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft / 2)

    return (frames * hop_length + offset).astype(int)


def samples_to_frames(samples, hop_length=512, n_fft=None):
    """Converts sample indices into STFT frames.

    :usage:
        >>> # Get the frame numbers for every 256 samples
        >>> librosa.samples_to_frames(np.arange(0, 22050, 256))
        array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
                8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15,
               16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,
               24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31,
               32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39,
               40, 40, 41, 41, 42, 42, 43])

    :parameters:
      - samples : np.ndarray [shape=(n,)]
          vector of sample indices

      - hop_length : int > 0 [scalar]
          number of samples between successive frames

      - n_fft : None or int > 0 [scalar]
          Optional: length of the FFT window.
          If given, time conversion will include an offset of ``- n_fft / 2``
          to counteract windowing effects in STFT.

          .. note:: This may result in negative frame indices.

    :returns:
      - frames : np.ndarray [shape=(n,), dtype=int]
          Frame numbers corresponding to the given times:
          ``frames[i] = floor( samples[i] / hop_length )``
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft / 2)

    return np.floor((samples - offset) / hop_length).astype(int)


def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds)

    :usage:
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=64)
        >>> beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=64)
        >>> beat_times[:20]
        array([ 0.067,  0.514,  0.99 ,  1.454,  1.91 ,  2.366,  2.833,  3.286,
                3.75 ,  4.2  ,  4.679,  5.146,  5.605,  6.058,  6.525,  6.978,
                7.433,  7.906,  8.377,  8.853])

    :parameters:
      - frames     : np.ndarray [shape=(n,)]
          vector of frame numbers

      - sr         : int > 0 [scalar]
          audio sampling rate

      - hop_length : int > 0 [scalar]
          number of samples between successive frames

      - n_fft : None or int > 0 [scalar]
          Optional: length of the FFT window.
          If given, time conversion will include an offset of ``n_fft / 2``
          to counteract windowing effects when using a non-centered STFT.

    :returns:
      - times : np.ndarray [shape=(n,)]
          time (in seconds) of each given frame number:
          ``times[i] = frames[i] * hop_length / sr``
    """

    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


def time_to_frames(times, sr=22050, hop_length=512, n_fft=None):
    """Converts time stamps into STFT frames.

    :usage:
        >>> # Get the frame numbers for every 100ms
        >>> librosa.time_to_frames(np.arange(0, 1, 0.1),
                                   sr=22050, hop_length=512)
        array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    :parameters:
      - times : np.ndarray [shape=(n,)]
          vector of time stamps

      - sr : int > 0 [scalar]
          audio sampling rate

      - hop_length : int > 0 [scalar]
          number of samples between successive frames

      - n_fft : None or int > 0 [scalar]
          Optional: length of the FFT window.
          If given, time conversion will include an offset of ``- n_fft / 2``
          to counteract windowing effects in STFT.

          .. note:: This may result in negative frame indices.

    :returns:
      - frames : np.ndarray [shape=(n,), dtype=int]
          Frame numbers corresponding to the given times:
          ``frames[i] = floor( times[i] * sr / hop_length )``
    """

    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


def time_to_samples(times, sr=22050):
    '''Convert timestamps (in seconds) to sample indices.

    :usage:
        >>> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
        array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435, 17640,
               19845])

    :parameters:
        - times : np.ndarray
            Array of time values (in seconds)

        - sr : int > 0
            Sampling rate

    :returns:
        - samples : np.ndarray [shape=times.shape, dtype=int]
            Sample indices corresponding to values in ``times``
    '''

    return (times * sr).astype(int)


def samples_to_time(samples, sr=22050):
    '''Convert sample indices to time (in seconds).

    :usage:
        >>> # Get timestamps corresponding to every 512 samples
        >>> librosa.samples_to_time(np.arange(0, 22050, 512))
        array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,  0.163,
                0.186,  0.209,  0.232,  0.255,  0.279,  0.302,  0.325,  0.348,
                0.372,  0.395,  0.418,  0.441,  0.464,  0.488,  0.511,  0.534,
                0.557,  0.58 ,  0.604,  0.627,  0.65 ,  0.673,  0.697,  0.72 ,
                0.743,  0.766,  0.789,  0.813,  0.836,  0.859,  0.882,  0.906,
                0.929,  0.952,  0.975,  0.998])

    :parameters:
        - samples : np.ndarray
            Array of sample indices

        - sr : int > 0
            Sampling rate

    :returns:
        - times : np.ndarray [shape=samples.shape, dtype=int]
            Time values corresponding to ``samples`` (in seconds)
    '''

    return samples / float(sr)


@cache
def autocorrelate(y, max_size=None):
    """Bounded auto-correlation

    :usage:
        >>> # Compute full autocorrelation of y
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.autocorrelate(y)
        array([  1.573e+04,   1.569e+04, ...,   1.090e-13,   1.090e-13])

        >>> # Compute autocorrelation up to 4 seconds lag
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> librosa.autocorrelate(y, max_size=4 * sr)
        array([ 15734.031,  15689.047, ...,   -410.197,   -436.05 ])

    :parameters:
      - y         : np.ndarray [shape=(n,)]
          vector to autocorrelate

      - max_size  : int > 0 or None
          maximum correlation lag.
          If unspecified, defaults to ``len(y)`` (unbounded)

    :returns:
      - z         : np.ndarray [shape=(n,) or (max_size,)]
          truncated autocorrelation ``y*y``
    """

    result = scipy.signal.fftconvolve(y, y[::-1], mode='full')

    result = result[int(len(result)/2):]

    if max_size is None:
        return result
    else:
        max_size = int(max_size)

    return result[:max_size]


@cache
def localmax(x, axis=0):
    """Find local maxima in an array ``x``.

    :usage:
        >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
        >>> librosa.localmax(x)
        array([False, False, False,  True, False,  True, False, True],
              dtype=bool)

        >>> # Two-dimensional example
        >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
        >>> librosa.localmax(x, axis=0)
        array([[False, False, False],
               [ True, False, False],
               [False,  True,  True]], dtype=bool)
        >>> librosa.localmax(x, axis=1)
        array([[False, False,  True],
               [False, False,  True],
               [False, False,  True]], dtype=bool)

    :parameters:
      - x     : np.ndarray [shape=(d1,d2,...)]
          input vector or array

      - axis : int
          axis along which to compute local maximality

    :returns:
      - m     : np.ndarray [shape=x.shape, dtype=bool]
          indicator vector of local maxima:
          ``m[i] == True`` if ``x[i]`` is a local maximum
    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [Ellipsis] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [Ellipsis] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


@cache
def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    '''Uses a flexible heuristic to pick peaks in a signal.

    :usage:
        >>> # Look +-3 steps
        >>> # compute the moving average over +-5 steps
        >>> # peaks must be > avg + 0.5
        >>> # skip 10 steps before taking another peak
        >>> y, sr = librosa.load(librosa.util.example_audio_file())
        >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=64)
        >>> librosa.peak_pick(onset_env, 3, 3, 5, 6, 0.5, 10)
        array([ 2558,  4863,  5259,  5578,  5890,  6212,  6531,  6850,  7162,
                7484,  7804,  8434,  8756,  9076,  9394,  9706, 10028, 10350,
               10979, 11301, 11620, 12020, 12251, 12573, 12894, 13523, 13846,
               14164, 14795, 15117, 15637, 15837, 16274, 16709, 16910, 17109,
               17824, 18181, 18380, 19452, 19496, 19653, 20369])

    :parameters:
      - x         : np.ndarray [shape=(n,)]
          input signal to peak picks from

      - pre_max   : int >= 0 [scalar]
          number of samples before n over which max is computed

      - post_max  : int >= 0 [scalar]
          number of samples after n over which max is computed

      - pre_avg   : int >= 0 [scalar]
          number of samples before n over which mean is computed

      - post_avg  : int >= 0 [scalar]
          number of samples after n over which mean is computed

      - delta     : float >= 0 [scalar]
          threshold offset for mean

      - wait      : int >= 0 [scalar]
          number of samples to wait after picking a peak

    :returns:
      - peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
          indices of peaks in x

    .. note::
      A sample n is selected as an peak if the corresponding x[n]
      fulfills the following three conditions:

        1. ``x[n] == max(x[n - pre_max:n + post_max])``
        2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
        3. ``n - previous_n > wait``

      where ``previous_n`` is the last sample picked as a peak (greedily).

    .. note::
      Implementation based on
      https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

      - Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.
    '''

    # Get the maximum of the signal over a sliding window
    max_length = pre_max + post_max + 1
    max_origin = 0.5 * (pre_max - post_max)
    mov_max = scipy.ndimage.filters.maximum_filter1d(x, int(max_length),
                                                     mode='constant',
                                                     origin=int(max_origin))

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg + 1
    avg_origin = 0.5 * (pre_avg - post_avg)
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, int(avg_length),
                                                     mode='constant',
                                                     origin=int(avg_origin))

    # First mask out all entries not equal to the local max
    detections = x*(x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections*(detections >= mov_avg + delta)

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)


@cache
def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True,
                   zero_pos=True, axis=-1):
    '''Find the zero-crossings of a signal ``y``: indices `i` such that
    ``sign(y[i]) != sign(y[j])``.

    If ``y`` is multi-dimensional, then zero-crossings are computed along
    the specified ``axis``.

    :usage:
        >>> # Generate a time-series
        >>> y = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
        >>> y
        array([  0.000e+00,   9.694e-01,   4.759e-01,  -7.357e-01,
                -8.372e-01,   3.247e-01,   9.966e-01,   1.646e-01,
                -9.158e-01,  -6.142e-01,   6.142e-01,   9.158e-01,
                -1.646e-01,  -9.966e-01,  -3.247e-01,   8.372e-01,
                 7.357e-01,  -4.759e-01,  -9.694e-01,  -9.797e-16])
        >>> # Compute zero-crossings
        >>> z = librosa.zero_crossings(y)
        >>> z
        array([ True, False, False,  True, False,  True, False, False,  True,
               False,  True, False,  True, False, False,  True, False,  True,
               False,  True], dtype=bool)
        >>> # Stack y against the zero-crossing indicator
        >>> np.vstack([y, z]).T
        array([[  0.000e+00,   1.000e+00],
               [  9.694e-01,   0.000e+00],
               [  4.759e-01,   0.000e+00],
               [ -7.357e-01,   1.000e+00],
               [ -8.372e-01,   0.000e+00],
               [  3.247e-01,   1.000e+00],
               [  9.966e-01,   0.000e+00],
               [  1.646e-01,   0.000e+00],
               [ -9.158e-01,   1.000e+00],
               [ -6.142e-01,   0.000e+00],
               [  6.142e-01,   1.000e+00],
               [  9.158e-01,   0.000e+00],
               [ -1.646e-01,   1.000e+00],
               [ -9.966e-01,   0.000e+00],
               [ -3.247e-01,   0.000e+00],
               [  8.372e-01,   1.000e+00],
               [  7.357e-01,   0.000e+00],
               [ -4.759e-01,   1.000e+00],
               [ -9.694e-01,   0.000e+00],
               [ -9.797e-16,   1.000e+00]])
        >>> # Find the indices of zero-crossings
        >>> np.nonzero(z)
        (array([ 0,  3,  5,  8, 10, 12, 15, 17, 19]),)


    :parameters:
      - y : np.ndarray
          The input array

      - threshold : float > 0 or None
          If specified, values where ``-threshold <= y <= threshold`` are
          clipped to 0.

      - ref_magnitude : float > 0 or callable
          If numeric, the threshold is scaled relative to ``ref_magnitude``
          If callable, the threshold is scaled relative to
          ``ref_magnitude(np.abs(y))``

      - pad : boolean
          If ``True``, then ``y[0]`` is considered a valid zero-crossing.

      - zero_pos : boolean
          If ``True`` then the value 0 is interpreted as having positive sign.
          If ``False``, then 0, -1, and +1 all have distinct signs.

      - axis : int
          Axis along which to compute zero-crossings.

    :returns:
      - zero_crossings : np.ndarray [shape=y.shape, dtype=boolean]
          Indicator array of zero-crossings in ``y`` along the selected axis.
    '''

    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if hasattr(ref_magnitude, '__call__'):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [Ellipsis] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [Ellipsis] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    return np.pad((y_sign[slice_post] != y_sign[slice_pre]),
                  padding,
                  mode='constant',
                  constant_values=pad)
