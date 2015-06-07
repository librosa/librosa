#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Utilities for spectral processing'''

import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.signal
import six

from . import time_frequency
from .. import cache
from .. import util
from ..util.exceptions import ParameterError

__all__ = ['stft', 'istft', 'magphase',
           'ifgram',
           'phase_vocoder',
           'logamplitude', 'perceptual_weighting']


@cache
def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix


    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length `n_fft`.


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> D
    array([[  2.576e-03 -0.000e+00j,   4.327e-02 -0.000e+00j, ...,
              3.189e-04 -0.000e+00j,  -5.961e-06 -0.000e+00j],
           [  2.441e-03 +2.884e-19j,   5.145e-02 -5.076e-03j, ...,
             -3.885e-04 -7.253e-05j,   7.334e-05 +3.868e-04j],
          ..., 
           [ -7.120e-06 -1.029e-19j,  -1.951e-09 -3.568e-06j, ...,
             -4.912e-07 -1.487e-07j,   4.438e-06 -1.448e-05j],
           [  7.136e-06 -0.000e+00j,   3.561e-06 -0.000e+00j, ...,
             -5.144e-07 -0.000e+00j,  -1.514e-05 -0.000e+00j]], dtype=complex64)


    Use left-aligned frames, instead of centered frames


    >>> D_left = librosa.stft(y, center=False)


    Use a shorter hop length


    >>> D_short = librosa.stft(y, hop_length=64)


    Display a spectrogram


    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

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

    elif six.callable(window):
        # User supplied a window function
        fft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make sure it's an array:
        fft_window = np.asarray(window)

        # validate length compatibility
        if fft_window.size != n_fft:
            raise ParameterError('Size mismatch between n_fft and len(window)')

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        util.valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode='reflect')

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


@cache
def istft(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """
    Inverse short-time Fourier transform.

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window * 2/3
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length `n_fft`

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

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

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ParameterError('Size mismatch between n_fft and window size')

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
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y


def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           norm=False, center=True, ref_power=1e-6, clip=True, dtype=np.complex64):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_.

    Calculates regular STFT as a side effect.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : int > 0 [scalar]
        sampling rate of `y`

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.

    norm : bool
        Normalize the STFT.

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
            `D[:, t]` (and `if_gram`) is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` at `y[t * hop_length]`

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating instantaneous frequency.
        Any bin with `np.abs(D[f, t])**2 < ref_power` will receive the
        default frequency estimate.

        If callable, the threshold is set to `ref_power(np.abs(D)**2)`.

    clip : boolean
        - If `True`, clip estimated frequencies to the range `[0, 0.5 * sr]`.
        - If `False`, estimated frequencies can be negative or exceed
          `0.5 * sr`.

    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    Returns
    -------
    if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
        Instantaneous frequency spectrogram:
        `if_gram[f, t]` is the frequency at bin `f`, time `t`

    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frequencies, D = librosa.ifgram(y, sr=sr)
    >>> frequencies
    array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
           [  3.150e+01,   3.070e+01, ...,   1.077e+01,   1.077e+01],
           ..., 
           [  1.101e+04,   1.101e+04, ...,   1.101e+04,   1.101e+04],
           [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])

    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

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
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1))
    bin_offset = (phase * diff_stft).imag / mag

    bin_offset[mag < ref_power**0.5] = 0

    if_gram = freq_angular[:n_fft//2 + 1] + bin_offset

    if norm:
        stft_matrix = stft_matrix * 2.0 / window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix


def magphase(D):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram


    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ..., 
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ..., 
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)


    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ..., 
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)

    """

    mag = np.abs(D)
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


@cache
def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Examples
    --------
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

    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        STFT matrix

    rate :  float > 0 [scalar]
        Speed-up factor: `rate > 1` is faster, `rate < 1` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.

        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    Returns
    -------
    D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
        time-stretched STFT
    """

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

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


@cache
def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        input spectrogram

    ref_power : scalar or function
        If scalar, `log(abs(S))` is compared to `log(ref_power)`.

        If a function, `log(abs(S))` is compared to `log(ref_power(abs(S)))`.

        This is primarily useful for comparing to the maximum value of `S`.

    amin    : float > 0[scalar]
        minimum amplitude threshold for `abs(S)` and `ref_power`

    top_db  : float >= 0 [scalar]
        threshold log amplitude at top_db below the peak:
        ``max(log(S)) - top_db``

    Returns
    -------
    log_S   : np.ndarray [shape=(d, t)]
        ``log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))``

    See Also
    --------
    perceptual_weighting

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.logamplitude(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.logamplitude(S**2, ref_power=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.logamplitude(S**2, ref_power=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()

    """

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    magnitude = np.abs(S)

    if six.callable(ref_power):
        # User supplied a function to calculate reference power
        __ref = ref_power(magnitude)
    else:
        __ref = np.abs(ref_power)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, __ref))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative positive')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@cache
def perceptual_weighting(S, frequencies, **kwargs):
    '''Perceptual weighting of a power spectrogram:

    `S_p[f] = A_weighting(f) + 10*log(S[f] / ref_power)`

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        Power spectrogram

    frequencies : np.ndarray [shape=(d,)]
        Center frequency for each row of `S`

    kwargs : additional keyword arguments
        Additional keyword arguments to `logamplitude`.

    Returns
    -------
    S_p : np.ndarray [shape=(d, t)]
        perceptually weighted version of `S`

    See Also
    --------
    logamplitude

    Examples
    --------
    Re-weight a CQT power spectrum, using peak power as reference

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1'))
    >>> freqs = librosa.cqt_frequencies(CQT.shape[0],
    ...                                 fmin=librosa.note_to_hz('A1'))
    >>> perceptual_CQT = librosa.perceptual_weighting(CQT**2,
    ...                                               freqs,
    ...                                               ref_power=np.max)
    >>> perceptual_CQT
    array([[ -80.076,  -80.049, ..., -104.735, -104.735],
           [ -78.344,  -78.555, ..., -103.725, -103.725],
           ..., 
           [ -76.272,  -76.272, ...,  -76.272,  -76.272],
           [ -76.485,  -76.485, ...,  -76.485,  -76.485]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(CQT**2,
    ...                                               ref_power=np.max),
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          y_axis='cqt_hz',
    ...                          x_axis='time')
    >>> plt.title('Log CQT power')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          x_axis='time')
    >>> plt.title('Perceptually weighted log CQT')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    '''

    offset = time_frequency.A_weighting(frequencies).reshape((-1, 1))

    return offset + logamplitude(S, **kwargs)


@cache
def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.


    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, n_fft=n_fft, hop_length=hop_length)|**power`

    n_fft : int > 0
        - If `S` is provided, then `n_fft` is inferred from `S`
        - Else, copied from input
    '''

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft
