#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Utilities for spectral processing'''

import numpy as np
import numpy.fft as fft
import scipy
import scipy.signal

from . import time_frequency
from .. import cache
from .. import util


@cache
def stft(y, n_fft=2048, hop_length=None, win_length=None, window=None,
         center=True, dtype=np.complex64):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
      - `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`
      - `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Examples
    --------
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

        If unspecified, defaults to `win_length = n_fft`.

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
    ValueError
        If `window` is supplied as a vector of length `!= n_fft`.

    See Also
    --------
    istft
        Inverse STFT
    ifgram
        Instantaneous frequency spectrogram
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
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0]
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

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([  1.121e-10,   1.093e-10, ...,   4.644e-14,   3.913e-14],
          dtype=float32)

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
    ValueError
        If `window` is supplied as a vector of length `!= n_fft`

    See Also
    --------
    stft : Short-time Fourier Transform
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
    described by [1]_.

    Calculates regular STFT as a side effect.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frequencies, D = librosa.ifgram(y, sr=sr)
    >>> frequencies
    array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
           [  2.613e+01,   3.606e+01, ...,   8.199e+00,   3.845e+01],
           ...,
           [  1.096e+04,   5.650e+03, ...,   1.101e+04,   1.101e+04],
           [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])

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
    power[power < util.SMALL_FLOAT] = 1.0

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
    and phase (P) components, so that `D = S * P`.

    Examples
    --------
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
        http://www.ee.columbia.edu/dpwe/resources/matlab/pvoc/

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


@cache
def logamplitude(S, ref_power=1.0, amin=1e-10, top_db=80.0):
    """Log-scale the amplitude of a spectrogram.


    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        input spectrogram

    ref_power : scalar or function
        - If scalar, `log(abs(S))` is compared to `log(ref_power)`.
        - If a function, `log(abs(S))` is compared to
            `log(ref_power(abs(S)))`.

        This is primarily useful for comparing to the maximum value of `S`.

    amin    : float [scalar]
        minimum amplitude threshold for `abs(S)` and `ref_power`

    top_db  : float [scalar]
        threshold log amplitude at top_db below the peak:
        `max(log(S)) - top_db`

    Returns
    -------
    log_S   : np.ndarray [shape=(d, t)]
        `log_S ~= 10 * log10(S) - 10 * log10(abs(ref_power))`

    See Also
    --------
    perceptual_weighting

    Examples
    --------
    Get a power spectrogram from a waveform `y`

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y)) ** 2
    >>> librosa.logamplitude(S)
    array([[-31.988, -22.714, ..., -33.325, -33.325],
           [-24.587, -25.686, ..., -33.325, -33.325],
           ...,
           [-33.325, -33.325, ..., -33.325, -33.325],
           [-33.325, -33.325, ..., -33.325, -33.325]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.logamplitude(S, ref_power=np.max)
    array([[-78.663, -69.389, ..., -80.   , -80.   ],
           [-71.262, -72.361, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)

    Or compare to median power

    >>> librosa.logamplitude(S, ref_power=np.median)
    array([[  3.279,  12.552, ...,   1.942,   1.942],
           [ 10.68 ,   9.581, ...,   1.942,   1.942],
           ...,
           [  1.942,   1.942, ...,   1.942,   1.942],
           [  1.942,   1.942, ...,   1.942,   1.942]], dtype=float32)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S, y_axis='log', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram: $|S|^2$')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max),
    ...                                               y_axis='log',
    ...                                               x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Log-Power spectrogram: $\log |S|^2$')
    >>> plt.tight_layout()
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

    `S_p[f] = A_weighting(f) + 10*log(S[f] / ref_power)`

    Examples
    --------
    >>> # Re-weight a CQT representation, using peak power as reference
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A2'))
    >>> freqs = librosa.cqt_frequencies(CQT.shape[0],
                                        fmin=librosa.note_to_hz('A2'))
    >>> librosa.perceptual_weighting(CQT, freqs, ref_power=np.max)
    array([[-50.113, -49.887, ..., -86.524, -88.887],
           [-48.525, -48.043, ..., -81.741, -81.102],
           ...,
           [-41.162, -44.677, ..., -61.139, -64.137],
           [-40.303, -44.41 , ..., -59.63 , -65.519]])

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
