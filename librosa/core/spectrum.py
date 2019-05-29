#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Utilities for spectral processing'''
import warnings

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
import six

from numba import jit

from . import time_frequency
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare

__all__ = ['stft', 'istft', 'magphase', 'iirt',
           'ifgram', 'phase_vocoder',
           'perceptual_weighting',
           'power_to_db', 'db_to_power',
           'amplitude_to_db', 'db_to_amplitude',
           'fmt', 'pcen', 'griffinlim']


@cache(level=20)
def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
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

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram

    np.pad : array padding

    Notes
    -----
    This function caches at level 20.


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = np.abs(librosa.stft(y))
    >>> D
    array([[2.58028018e-03, 4.32422794e-02, 6.61255598e-01, ...,
            6.82710262e-04, 2.51654536e-04, 7.23036574e-05],
           [2.49403086e-03, 5.15930466e-02, 6.00107312e-01, ...,
            3.48026224e-04, 2.35853557e-04, 7.54836728e-05],
           [7.82410789e-04, 1.05394892e-01, 4.37517226e-01, ...,
            6.29352580e-04, 3.38571583e-04, 8.38094638e-05],
           ...,
           [9.48568513e-08, 4.74725084e-07, 1.50052492e-05, ...,
            1.85637656e-08, 2.89708542e-08, 5.74304337e-09],
           [1.25165826e-07, 8.58259284e-07, 1.11157215e-05, ...,
            3.49099771e-08, 3.11740926e-08, 5.29926236e-09],
           [1.70630571e-07, 8.92518756e-07, 1.23656537e-05, ...,
            5.33256745e-08, 3.33264900e-08, 5.13272980e-09]], dtype=float32)


    Use left-aligned frames, instead of centered frames

    >>> D_left = np.abs(librosa.stft(y, center=False))


    Use a shorter hop length

    >>> D_short = np.abs(librosa.stft(y, hop_length=64))


    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.amplitude_to_db(D,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    >>> plt.show()

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    util.valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(fft_window *
                                             y_frames[:, bl_s:bl_t],
                                             axis=0)
    return stft_matrix


@cache(level=30)
def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, dtype=np.float32, length=None):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    length : int > 0, optional
        If provided, the output `y` is zero-padded or clipped to exactly
        `length` samples.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of `y` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    1.4901161e-07
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = util.pad_center(ifft_window, n_fft)[:, np.newaxis]

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = int(util.MAX_MEM_BLOCK // (stft_matrix.shape[0] *
                                           stft_matrix.itemsize))

    fft = get_fftlib()

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length:], ytmp, hop_length)

        frame += (bl_t - bl_s)

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(window,
                                       n_frames,
                                       win_length=win_length,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       dtype=dtype)

    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = util.fix_length(y[start:], length)

    return y


@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample:(sample + n_fft)] += ytmp[:, frame]


def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           window='hann', norm=False, center=True, ref_power=1e-6,
           clip=True, dtype=np.complex64, pad_mode='reflect'):
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

    sr : number > 0 [scalar]
        sampling rate of `y`

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        See `stft` for details.

        .. see also:: `filters.get_window`

    norm : bool
        Normalize the STFT.

    center : boolean
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

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


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
    fft_window = util.pad_center(get_window(window, win_length,
                                            fftbins=True),
                                 n_fft)

    # Window for discrete differentiation
    # we use a cyclic gradient calculation here to ensure that the edges of
    # the window are handled properly
    d_window = - util.cyclic_gradient(fft_window)

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length,
                       window=window, center=center,
                       dtype=dtype, pad_mode=pad_mode)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center,
                     dtype=dtype, pad_mode=pad_mode).conj()

    # Compute power normalization. Suppress zeros.
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)
    freq_angular = freq_angular.reshape((-1, 1))

    bin_offset = (-phase * diff_stft).imag / mag
    bin_offset[mag < ref_power**0.5] = 0

    if_gram = freq_angular[:n_fft//2 + 1] + bin_offset

    if norm:
        stft_matrix *= 2.0 / fft_window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix


def magphase(D, power=1):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.


    Returns
    -------
    D_mag : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`, raised to `power`
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
    mag **= power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


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
    D_stretched : np.ndarray [shape=(d, t / rate), dtype=complex]
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


@cache(level=20)
def iirt(y, sr=22050, win_length=2048, hop_length=None, center=True,
         tuning=0.0, pad_mode='reflect', flayout=None, **kwargs):
    r'''Time-frequency representation using IIR filters [1]_.

    This function will return a time-frequency representation
    using a multirate filter bank consisting of IIR filters.
    First, `y` is resampled as needed according to the provided `sample_rates`.
    Then, a filterbank with with `n` band-pass filters is designed.
    The resampled input signals are processed by the filterbank as a whole.
    (`scipy.signal.filtfilt` resp. `sosfiltfilt` is used to make the phase linear.)
    The output of the filterbank is cut into frames.
    For each band, the short-time mean-square power (STMSP) is calculated by
    summing `win_length` subsequent filtered time samples.

    When called with the default set of parameters, it will generate the TF-representation
    as described in [1]_ (pitch filterbank):

        * 85 filters with MIDI pitches [24, 108] as `center_freqs`.
        * each filter having a bandwith of one semitone.

    .. [1] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    win_length : int > 0, <= n_fft
        Window length.

    hop_length : int > 0 [scalar]
        Hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin.

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, this function uses reflection padding.

    flayout : string
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
        - If `sos`, a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.

    kwargs : additional keyword arguments
        Additional arguments for `librosa.filters.semitone_filterbank()`
        (e.g., could be used to provide another set of `center_freqs` and `sample_rates`).

    Returns
    -------
    bands_power : np.ndarray [shape=(n, t), dtype=dtype]
        Short-time mean-square power for the input signal.

    Raises
    ------
    ParameterError
        If `flayout` is not None, `ba`, or `sos`.

    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters._multirate_fb
    librosa.filters.mr_frequencies
    librosa.core.cqt
    scipy.signal.filtfilt
    scipy.signal.sosfiltfilt

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = np.abs(librosa.iirt(y))
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='cqt_hz', x_axis='time')
    >>> plt.title('Semitone spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    >>> plt.show()
    '''

    if flayout is None:
        warnings.warn('Default filter layout for `iirt` is `ba`, but will be `sos` in 0.7.',
                      FutureWarning)
        flayout = 'ba'

    elif flayout not in ('ba', 'sos'):
        raise ParameterError('Unsupported flayout={}'.format(flayout))

    # check audio input
    util.valid_audio(y)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(hop_length), mode=pad_mode)

    # get the semitone filterbank
    filterbank_ct, sample_rates = semitone_filterbank(tuning=tuning, flayout=flayout, **kwargs)

    # create three downsampled versions of the audio signal
    y_resampled = []

    y_srs = np.unique(sample_rates)

    for cur_sr in y_srs:
        y_resampled.append(resample(y, sr, cur_sr))

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - win_length) / float(hop_length))

    bands_power = []

    for cur_sr, cur_filter in zip(sample_rates, filterbank_ct):
        factor = float(sr) / float(cur_sr)
        win_length_STMSP = int(np.round(win_length / factor))
        hop_length_STMSP = int(np.round(hop_length / factor))

        # filter the signal
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]

        if flayout == 'ba':
            cur_filter_output = scipy.signal.filtfilt(cur_filter[0], cur_filter[1],
                                                      y_resampled[cur_sr_idx])
        elif flayout == 'sos':
            cur_filter_output = scipy.signal.sosfiltfilt(cur_filter,
                                                         y_resampled[cur_sr_idx])

        # frame the current filter output
        cur_frames = util.frame(np.ascontiguousarray(cur_filter_output),
                                frame_length=win_length_STMSP,
                                hop_length=hop_length_STMSP)

        bands_power.append(factor * np.sum(cur_frames**2, axis=0)[:n_frames])

    return np.asarray(bands_power)


@cache(level=30)
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()
    >>> plt.show()

    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@cache(level=30)
def db_to_power(S_db, ref=1.0):
    '''Convert a dB-scale spectrogram to a power spectrogram.

    This effectively inverts `power_to_db`:

        `db_to_power(S_db) ~= ref * 10.0**(S_db / 10)`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return ref * np.power(10.0, 0.1 * S_db)


@cache(level=30)
def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    '''Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2)``, but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `20 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `S` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(20 * log10(S)) - top_db``


    Returns
    -------
    S_db : np.ndarray
        ``S`` measured in dB

    See Also
    --------
    power_to_db, db_to_amplitude

    Notes
    -----
    This function caches at level 30.
    '''

    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('amplitude_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call amplitude_to_db(np.abs(S)) instead.')

    magnitude = np.abs(S)

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value**2, amin=amin**2,
                       top_db=top_db)


@cache(level=30)
def db_to_amplitude(S_db, ref=1.0):
    '''Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`:

        `db_to_amplitude(S_db) ~= 10.0**(0.5 * (S_db + log10(ref)/10))`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref: number > 0
        Optional reference power.

    Returns
    -------
    S : np.ndarray
        Linear magnitude spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return db_to_power(S_db, ref=ref**2)**0.5


@cache(level=30)
def perceptual_weighting(S, frequencies, **kwargs):
    '''Perceptual weighting of a power spectrogram:

    `S_p[f] = A_weighting(f) + 10*log(S[f] / ref)`

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        Power spectrogram

    frequencies : np.ndarray [shape=(d,)]
        Center frequency for each row of `S`

    kwargs : additional keyword arguments
        Additional keyword arguments to `power_to_db`.

    Returns
    -------
    S_p : np.ndarray [shape=(d, t)]
        perceptually weighted version of `S`

    See Also
    --------
    power_to_db

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Re-weight a CQT power spectrum, using peak power as reference

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
    >>> freqs = librosa.cqt_frequencies(C.shape[0],
    ...                                 fmin=librosa.note_to_hz('A1'))
    >>> perceptual_CQT = librosa.perceptual_weighting(C**2,
    ...                                               freqs,
    ...                                               ref=np.max)
    >>> perceptual_CQT
    array([[ -80.076,  -80.049, ..., -104.735, -104.735],
           [ -78.344,  -78.555, ..., -103.725, -103.725],
           ...,
           [ -76.272,  -76.272, ...,  -76.272,  -76.272],
           [ -76.485,  -76.485, ...,  -76.485,  -76.485]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                  ref=np.max),
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          y_axis='cqt_hz')
    >>> plt.title('Log CQT power')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          x_axis='time')
    >>> plt.title('Perceptually weighted log CQT')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    >>> plt.show()
    '''

    offset = time_frequency.A_weighting(frequencies).reshape((-1, 1))

    return offset + power_to_db(S, **kwargs)


@cache(level=30)
def fmt(y, t_min=0.5, n_fmt=None, kind='cubic', beta=0.5, over_sample=1, axis=-1):
    """The fast Mellin transform (FMT) [1]_ of a uniformly sampled signal y.

    When the Mellin parameter (beta) is 1/2, it is also known as the scale transform [2]_.
    The scale transform can be useful for audio analysis because its magnitude is invariant
    to scaling of the domain (e.g., time stretching or compression).  This is analogous
    to the magnitude of the Fourier transform being invariant to shifts in the input domain.


    .. [1] De Sena, Antonio, and Davide Rocchesso.
        "A fast Mellin and scale transform."
        EURASIP Journal on Applied Signal Processing 2007.1 (2007): 75-75.

    .. [2] Cohen, L.
        "The scale representation."
        IEEE Transactions on Signal Processing 41, no. 12 (1993): 3275-3292.

    Parameters
    ----------
    y : np.ndarray, real-valued
        The input signal(s).  Can be multidimensional.
        The target axis must contain at least 3 samples.

    t_min : float > 0
        The minimum time spacing (in samples).
        This value should generally be less than 1 to preserve as much information as
        possible.

    n_fmt : int > 2 or None
        The number of scale transform bins to use.
        If None, then `n_bins = over_sample * ceil(n * log((n-1)/t_min))` is taken,
        where `n = y.shape[axis]`

    kind : str
        The type of interpolation to use when re-sampling the input.
        See `scipy.interpolate.interp1d` for possible values.

        Note that the default is to use high-precision (cubic) interpolation.
        This can be slow in practice; if speed is preferred over accuracy,
        then consider using `kind='linear'`.

    beta : float
        The Mellin parameter.  `beta=0.5` provides the scale transform.

    over_sample : float >= 1
        Over-sampling factor for exponential resampling.

    axis : int
        The axis along which to transform `y`

    Returns
    -------
    x_scale : np.ndarray [dtype=complex]
        The scale transform of `y` along the `axis` dimension.

    Raises
    ------
    ParameterError
        if `n_fmt < 2` or `t_min <= 0`
        or if `y` is not finite
        or if `y.shape[axis] < 3`.

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    >>> # Generate a signal and time-stretch it (with energy normalization)
    >>> scale = 1.25
    >>> freq = 3.0
    >>> x1 = np.linspace(0, 1, num=1024, endpoint=False)
    >>> x2 = np.linspace(0, 1, num=scale * len(x1), endpoint=False)
    >>> y1 = np.sin(2 * np.pi * freq * x1)
    >>> y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)
    >>> # Verify that the two signals have the same energy
    >>> np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)
        (255.99999999999997, 255.99999999999969)
    >>> scale1 = librosa.fmt(y1, n_fmt=512)
    >>> scale2 = librosa.fmt(y2, n_fmt=512)
    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.plot(y1, label='Original')
    >>> plt.plot(y2, linestyle='--', label='Stretched')
    >>> plt.xlabel('time (samples)')
    >>> plt.title('Input signals')
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.subplot(1, 2, 2)
    >>> plt.semilogy(np.abs(scale1), label='Original')
    >>> plt.semilogy(np.abs(scale2), linestyle='--', label='Stretched')
    >>> plt.xlabel('scale coefficients')
    >>> plt.title('Scale transform magnitude')
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.tight_layout()
    >>> plt.show()

    >>> # Plot the scale transform of an onset strength autocorrelation
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10.0, duration=30.0)
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr)
    >>> # Auto-correlate with up to 10 seconds lag
    >>> odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)
    >>> # Normalize
    >>> odf_ac = librosa.util.normalize(odf_ac, norm=np.inf)
    >>> # Compute the scale transform
    >>> odf_ac_scale = librosa.fmt(librosa.util.normalize(odf_ac), n_fmt=512)
    >>> # Plot the results
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> plt.plot(odf, label='Onset strength')
    >>> plt.axis('tight')
    >>> plt.xlabel('Time (frames)')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.subplot(3, 1, 2)
    >>> plt.plot(odf_ac, label='Onset autocorrelation')
    >>> plt.axis('tight')
    >>> plt.xlabel('Lag (frames)')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.subplot(3, 1, 3)
    >>> plt.semilogy(np.abs(odf_ac_scale), label='Scale transform magnitude')
    >>> plt.axis('tight')
    >>> plt.xlabel('scale coefficients')
    >>> plt.legend(frameon=True)
    >>> plt.tight_layout()
    >>> plt.show()
    """

    n = y.shape[axis]

    if n < 3:
        raise ParameterError('y.shape[{:}]=={:} < 3'.format(axis, n))

    if t_min <= 0:
        raise ParameterError('t_min must be a positive number')

    if n_fmt is None:
        if over_sample < 1:
            raise ParameterError('over_sample must be >= 1')

        # The base is the maximum ratio between adjacent samples
        # Since the sample spacing is increasing, this is simply the
        # ratio between the positions of the last two samples: (n-1)/(n-2)
        log_base = np.log(n - 1) - np.log(n - 2)

        n_fmt = int(np.ceil(over_sample * (np.log(n - 1) - np.log(t_min)) / log_base))

    elif n_fmt < 3:
        raise ParameterError('n_fmt=={:} < 3'.format(n_fmt))
    else:
        log_base = (np.log(n_fmt - 1) - np.log(n_fmt - 2)) / over_sample

    if not np.all(np.isfinite(y)):
        raise ParameterError('y must be finite everywhere')

    base = np.exp(log_base)
    # original grid: signal covers [0, 1).  This range is arbitrary, but convenient.
    # The final sample is positioned at (n-1)/n, so we omit the endpoint
    x = np.linspace(0, 1, num=n, endpoint=False)

    # build the interpolator
    f_interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=axis)

    # build the new sampling grid
    # exponentially spaced between t_min/n and 1 (exclusive)
    # we'll go one past where we need, and drop the last sample
    # When over-sampling, the last input sample contributions n_over samples.
    # To keep the spacing consistent, we over-sample by n_over, and then
    # trim the final samples.
    n_over = int(np.ceil(over_sample))
    x_exp = np.logspace((np.log(t_min) - np.log(n)) / log_base,
                        0,
                        num=n_fmt + n_over,
                        endpoint=False,
                        base=base)[:-n_over]

    # Clean up any rounding errors at the boundaries of the interpolation
    # The interpolator gets angry if we try to extrapolate, so clipping is necessary here.
    if x_exp[0] < t_min or x_exp[-1] > float(n - 1.0) / n:
        x_exp = np.clip(x_exp, float(t_min) / n, x[-1])

    # Make sure that all sample points are unique
    # This should never happen!
    if len(np.unique(x_exp)) != len(x_exp):
        raise RuntimeError('Redundant sample positions in Mellin transform')

    # Resample the signal
    y_res = f_interp(x_exp)

    # Broadcast the window correctly
    shape = [1] * y_res.ndim
    shape[axis] = -1

    # Apply the window and fft
    # Normalization is absorbed into the window here for expedience
    fft = get_fftlib()
    return fft.rfft(y_res * ((x_exp**beta).reshape(shape) * np.sqrt(n) / n_fmt),
                    axis=axis)


@cache(level=30)
def pcen(S, sr=22050, hop_length=512, gain=0.98, bias=2, power=0.5,
         time_constant=0.400, eps=1e-6, b=None, max_size=1, ref=None,
         axis=-1, max_axis=None, zi=None, return_zf=False):
    '''Per-channel energy normalization (PCEN) [1]_

    This function normalizes a time-frequency representation `S` by
    performing automatic gain control, followed by nonlinear compression:

        P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power

    where `M` is the result of applying a low-pass, temporal IIR filter
    to `S`:

        M[f, t] = (1 - b) * M[f, t - 1] + b * S[f, t]

    If `b` is not provided, it is calculated as:

        b = (sqrt(1 + 4* T**2) - 1) / (2 * T**2)

    where `T = time_constant * sr / hop_length`.

    This normalization is designed to suppress background noise and
    emphasize foreground signals, and can be used as an alternative to
    decibel scaling (`amplitude_to_db`).

    This implementation also supports smoothing across frequency bins
    by specifying `max_size > 1`.  If this option is used, the filtered
    spectrogram `M` is computed as

        M[f, t] = (1 - b) * M[f, t - 1] + b * R[f, t]

    where `R` has been max-filtered along the frequency axis, similar to
    the SuperFlux algorithm implemented in `onset.onset_strength`:

        R[f, t] = max(S[f - max_size//2: f + max_size//2, t])

    This can be used to perform automatic gain control on signals that cross
    or span multiple frequency bans, which may be desirable for spectrograms
    with high frequency resolution.

    .. [1] Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.
       (2017, March). Trainable frontend for robust and far-field keyword spotting.
       In Acoustics, Speech and Signal Processing (ICASSP), 2017
       IEEE International Conference on (pp. 5670-5674). IEEE.

    Parameters
    ----------
    S : np.ndarray (non-negative)
        The input (magnitude) spectrogram

    sr : number > 0 [scalar]
        The audio sampling rate

    hop_length : int > 0 [scalar]
        The hop length of `S`, expressed in samples

    gain : number >= 0 [scalar]
        The gain factor.  Typical values should be slightly less than 1.

    bias : number >= 0 [scalar]
        The bias point of the nonlinear compression (default: 2)

    power : number > 0 [scalar]
        The compression exponent.  Typical values should be between 0 and 1.
        Smaller values of `power` result in stronger compression.

    time_constant : number > 0 [scalar]
        The time constant for IIR filtering, measured in seconds.

    eps : number > 0 [scalar]
        A small constant used to ensure numerical stability of the filter.

    b : number in [0, 1]  [scalar]
        The filter coefficient for the low-pass filter.
        If not provided, it will be inferred from `time_constant`.

    max_size : int > 0 [scalar]
        The width of the max filter applied to the frequency axis.
        If left as `1`, no filtering is performed.

    ref : None or np.ndarray (shape=S.shape)
        An optional pre-computed reference spectrum (`R` in the above).
        If not provided it will be computed from `S`.

    axis : int [scalar]
        The (time) axis of the input spectrogram.

    max_axis : None or int [scalar]
        The frequency axis of the input spectrogram.
        If `None`, and `S` is two-dimensional, it will be inferred
        as the opposite from `axis`.
        If `S` is not two-dimensional, and `max_size > 1`, an error
        will be raised.

    zi : np.ndarray
        The initial filter delay values.

        This may be the `zf` (final delay values) of a previous call to `pcen`, or
        computed by `scipy.signal.lfilter_zi`.

    return_zf : bool
        If `True`, return the final filter delay values along with the PCEN output `P`.
        This is primarily useful in streaming contexts, where the final state of one
        block of processing should be used to initialize the next block.

        If `False` (default) only the PCEN values `P` are returned.


    Returns
    -------
    P : np.ndarray, non-negative [shape=(n, m)]
        The per-channel energy normalized version of `S`.

    zf : np.ndarray (optional)
        The final filter delay values.  Only returned if `return_zf=True`.

    See Also
    --------
    amplitude_to_db
    librosa.onset.onset_strength

    Examples
    --------

    Compare PCEN to log amplitude (dB) scaling on Mel spectra

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10, duration=10)

    >>> # We'll use power=1 to get a magnitude spectrum
    >>> # instead of a power spectrum
    >>> S = librosa.feature.melspectrogram(y, sr=sr, power=1)
    >>> log_S = librosa.amplitude_to_db(S, ref=np.max)
    >>> pcen_S = librosa.pcen(S)
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(log_S, x_axis='time', y_axis='mel')
    >>> plt.title('log amplitude (dB)')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel')
    >>> plt.title('Per-channel energy normalization')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()

    Compare PCEN with and without max-filtering

    >>> pcen_max = librosa.pcen(S, max_size=3)
    >>> plt.figure()
    >>> plt.subplot(2,1,1)
    >>> librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel')
    >>> plt.title('Per-channel energy normalization (no max-filter)')
    >>> plt.colorbar()
    >>> plt.subplot(2,1,2)
    >>> librosa.display.specshow(pcen_max, x_axis='time', y_axis='mel')
    >>> plt.title('Per-channel energy normalization (max_size=3)')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    >>> plt.show()

    '''

    if power <= 0:
        raise ParameterError('power={} must be strictly positive'.format(power))

    if gain < 0:
        raise ParameterError('gain={} must be non-negative'.format(gain))

    if bias < 0:
        raise ParameterError('bias={} must be non-negative'.format(bias))

    if eps <= 0:
        raise ParameterError('eps={} must be strictly positive'.format(eps))

    if time_constant <= 0:
        raise ParameterError('time_constant={} must be strictly positive'.format(time_constant))

    if max_size < 1 or not isinstance(max_size, int):
        raise ParameterError('max_size={} must be a positive integer'.format(max_size))

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    if not 0 <= b <= 1:
        raise ParameterError('b={} must be between 0 and 1'.format(b))

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('pcen was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call pcen(np.abs(D)) instead.')
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError('Max-filtering cannot be applied to 1-dimensional input')
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError('Max-filtering a {:d}-dimensional spectrogram '
                                         'requires you to specify max_axis'.format(S.ndim))
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    if zi is None:
        # Make sure zi matches dimension to input
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]

    S_smooth, zf = scipy.signal.lfilter([b], [1, b - 1], ref, zi=zi,
                                        axis=axis)

    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))
    S_out = (S * smooth + bias)**power - bias**power

    if return_zf:
        return S_out, zf
    else:
        return S_out


def griffinlim(S, n_iter=32, hop_length=None, win_length=None, window='hann',
               center=True, dtype=np.float32, length=None, pad_mode='reflect',
               momentum=0.99, random_state=None):

    '''Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm [1,2]_

    Given a short-time Fourier transform magnitude matrix (`S`), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations.
    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that `S` contains only the non-negative frequencies (as computed by
    `core.stft`).

    .. [1] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    .. [2] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    S : np.ndarray [shape=(n_fft / 2 + 1, t), non-negative]
        An array of short-time Fourier transform magnitudes as produced by
        `core.stft`.

    n_iter : int > 0
        The number of iterations to run

    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to `n_fft // 4`

    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal `n_fft`

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`

    center : boolean
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.

    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is 32-bit float.

    length : None or int > 0
        If provided, the output `y` is zero-padded or clipped to exactly `length`
        samples.

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` instance, the random number generator itself;

        If `None`, defaults to the current `np.random` object.


    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time-domain signal reconstructed from `S`

    See Also
    --------
    stft
    istft
    magphase
    filters.get_window

    Examples
    --------
    A basic STFT inverse example

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=30)
    >>> # Get the magnitude spectrogram
    >>> S = np.abs(librosa.stft(y))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim(S)
    >>> # Invert without estimating phase
    >>> y_istft = librosa.istft(S)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> ax = plt.subplot(3,1,1)
    >>> librosa.display.waveplot(y, sr=sr, color='b')
    >>> plt.title('Original')
    >>> plt.xlabel('')
    >>> plt.subplot(3,1,2, sharex=ax, sharey=ax)
    >>> librosa.display.waveplot(y_inv, sr=sr, color='g')
    >>> plt.title('Griffin-Lim reconstruction')
    >>> plt.xlabel('')
    >>> plt.subplot(3,1,3, sharex=ax, sharey=ax)
    >>> librosa.display.waveplot(y_istft, sr=sr, color='r')
    >>> plt.title('Magnitude-only istft reconstruction')
    >>> plt.tight_layout()
    >>> plt.show()
    '''

    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise ParameterError('griffinlim() called with momentum={} < 0'.format(momentum))

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (S.shape[0] - 1)

    # randomly initialize the phase
    angles = np.exp(2j * np.pi * rng.rand(*S.shape))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = istft(S * angles, hop_length=hop_length, win_length=win_length,
                        window=window, center=center, dtype=dtype, length=length)

        # Rebuild the spectrogram
        rebuilt = stft(inverse, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length, window=window, center=center,
                       pad_mode=pad_mode)
        
        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16
    
    # Return the final phase estimates
    return istft(S * angles, hop_length=hop_length, win_length=win_length,
                 window=window, center=center, dtype=dtype, length=length)


def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1,
                 win_length=None, window='hann', center=True, pad_mode='reflect'):
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

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        .. see also:: `filters.get_window`

    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `t` is centered at `y[t * hop_length]`.
        - If `False`, then frame `t` begins at `y[t * hop_length]`

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, ...)|**power`

    n_fft : int > 0
        - If `S` is provided, then `n_fft` is inferred from `S`
        - Else, copied from input
    '''

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, center=center,
                        window=window, pad_mode=pad_mode))**power

    return S, n_fft
