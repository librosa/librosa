#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for spectral processing"""
from __future__ import annotations
import warnings

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate

from numba import jit

from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import (
    _WindowSpec,
    _PadMode,
    _PadModeSTFT,
    _SequenceLike,
    _ScalarOrSequence,
    _ComplexLike_co,
    _FloatLike_co
)

__all__ = [
    "stft",
    "istft",
    "magphase",
    "iirt",
    "reassigned_spectrogram",
    "phase_vocoder",
    "perceptual_weighting",
    "power_to_db",
    "db_to_power",
    "amplitude_to_db",
    "db_to_amplitude",
    "fmt",
    "pcen",
    "griffinlim",
]


@cache(level=20)
def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_samples` and `fft_frequencies`.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        input signal. Multi-channel is supported.

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.  Padding is added on
        both the left- and the right-side of the window so that the window
        is centered within the frame.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.

        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.

        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `librosa.stream`

    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.

    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.

        .. note:: Not all padding modes supported by `numpy.pad` are supported here.
            `wrap`, `mean`, `maximum`, `median`, and `minimum` are not supported.

            Other modes that depend at most on input values at the edges of the
            signal (e.g., `constant`, `edge`, `linear_ramp`) are supported.

        If ``center=False``,  this argument is ignored.

        .. see also:: `numpy.pad`

    out : np.ndarray or None
        A pre-allocated, complex-valued array to store the STFT results.
        This must be of compatible shape and dtype for the given input parameters.

        If `out` is larger than necessary for the provided input signal, then only
        a prefix slice of `out` will be used.

        If not provided, a new array is allocated and returned.

    Returns
    -------
    D : np.ndarray [shape=(..., 1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.

        If a pre-allocated `out` array is provided, then `D` will be
        a reference to `out`.

        If `out` is larger than necessary, then `D` will be a sliced
        view: `D = out[..., :n_frames]`.

    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> S_left = librosa.stft(y, center=False)

    Use a shorter hop length

    >>> D_short = librosa.stft(y, hop_length=64)

    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not util.is_positive_int(hop_length):
        raise ParameterError(f"hop_length={hop_length} must be a positive integer")

    # Check audio is valid
    util.valid_audio(y, mono=False)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = util.expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if pad_mode in ("wrap", "maximum", "mean", "median", "minimum"):
            # Note: padding with a user-provided function "works", but
            # use at your own risk.
            # Since we don't pass-through kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ParameterError(
                f"pad_mode='{pad_mode}' is not supported by librosa.stft"
            )

        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
            )

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(y.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = util.frame(
                    y_post, frame_length=n_fft, hop_length=hop_length
                )
                # How many extra frames do we have from the tail?
                extra += y_frames_post.shape[-1]
            else:
                # In this event, the first frame that touches tail padding would run off
                # the end of the padded array
                # We'll circumvent this by allocating an empty frame buffer for the tail
                # this keeps the subsequent logic simple
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        if n_fft > y.shape[-1]:
            raise ParameterError(
                f"n_fft={n_fft} is too large for uncentered analysis of input signal of length={y.shape[-1]}"
            )

        # "Middle" of the signal starts at sample 0
        start = 0
        # We have no extra frames
        extra = 0

    fft = get_fftlib()

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Window the time series.
    y_frames = util.frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ParameterError(
            f"Shape mismatch for provided output array out.shape={out.shape} and target shape={shape}"
        )
    elif not np.iscomplexobj(out):
        raise ParameterError(f"output with dtype={out.dtype} is not of complex type")
    else:
        if np.allclose(shape, out.shape):
            stft_matrix = out
        else:
            stft_matrix = out[..., : shape[-1]]

    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(
        util.MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix


@cache(level=30)
def istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``
    by minimizing the mean squared error between ``stft_matrix`` and STFT of
    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified ``stft_matrix``.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(..., 1 + n_fft//2, t)]
        STFT matrix from ``stft``

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to ``win_length // 4``.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the ``window`` function (see below).

        If unspecified, defaults to ``n_fft``.

    n_fft : int > 0 or None
        The number of samples per frame in the input spectrogram.
        By default, this will be inferred from the shape of ``stft_matrix``.
        However, if an odd frame length was used, you can specify the correct
        length by setting ``n_fft``.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, ``D`` is assumed to have centered frames.
        - If ``False``, ``D`` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for ``y``.  Default is to match the numerical
        precision of the input spectrogram.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    out : np.ndarray or None
        A pre-allocated, complex-valued array to store the reconstructed signal
        ``y``.  This must be of the correct shape for the given input parameters.

        If not provided, a new array is allocated and returned.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time domain signal reconstructed from ``stft_matrix``.
        If ``stft_matrix`` contains more than two axes
        (e.g., from a stereo input signal), then ``y`` will match shape on the leading dimensions.

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([-1.407e-03, -4.461e-04, ...,  5.131e-06, -1.417e-05],
          dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of ``y`` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    8.940697e-08
    """
    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = util.pad_center(ifft_window, size=n_fft)
    ifft_window = util.expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + 2 * (n_fft // 2)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    if dtype is None:
        dtype = util.dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if length:
        expected_signal_len = length
    elif center:
        expected_signal_len -= 2 * (n_fft // 2)

    shape.append(expected_signal_len)

    if out is None:
        y = np.zeros(shape, dtype=dtype)
    elif not np.allclose(out.shape, shape):
        raise ParameterError(
            f"Shape mismatch for provided output array out.shape={out.shape} != {shape}"
        )
    else:
        y = out
        # Since we'll be doing overlap-add here, this needs to be initialized to zero.
        y.fill(0.0)

    fft = get_fftlib()

    if center:
        # First frame that does not depend on padding
        #  k * hop_length - n_fft//2 >= 0
        # k * hop_length >= n_fft // 2
        # k >= (n_fft//2 / hop_length)

        start_frame = int(np.ceil((n_fft // 2) / hop_length))

        # Do overlap-add on the head block
        ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

        shape[-1] = n_fft + hop_length * (start_frame - 1)
        head_buffer = np.zeros(shape, dtype=dtype)

        __overlap_add(head_buffer, ytmp, hop_length)

        # If y is smaller than the head buffer, take everything
        if y.shape[-1] < shape[-1] - n_fft // 2:
            y[..., :] = head_buffer[..., n_fft // 2 : y.shape[-1] + n_fft // 2]
        else:
            # Trim off the first n_fft//2 samples from the head and copy into target buffer
            y[..., : shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2 :]

        # This offset compensates for any differences between frame alignment
        # and padding truncation
        offset = start_frame * hop_length - n_fft // 2

    else:
        start_frame = 0
        offset = 0

    n_columns = int(
        util.MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    )
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    if center:
        start = n_fft // 2
    else:
        start = 0

    ifft_window_sum = util.fix_length(ifft_window_sum[..., start:], size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y


@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample

        y[..., sample : (sample + N)] += ytmp[..., :N, frame]


def __reassign_frequencies(
    y: np.ndarray,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequencies based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.20 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="wrap"`` or else ``center=False``, rather
    than the defaults. Frequency reassignment assumes that the energy in each
    FFT bin is associated with exactly one signal component. Reflection padding
    at the edges of the signal may invalidate the reassigned estimates in the
    boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_frequencies`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See ``stft`` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the numerical precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
        ``freqs[f, t]`` is the frequency for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Frequencies with zero support will produce a divide-by-zero warning and
        will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frequencies, S = librosa.core.spectrum.__reassign_frequencies(y, sr=sr)
    >>> frequencies
    array([[0.000e+00, 0.000e+00, ..., 0.000e+00, 0.000e+00],
           [3.628e+00, 4.698e+00, ..., 1.239e+01, 1.072e+01],
           ...,
           [1.101e+04, 1.102e+04, ..., 1.105e+04, 1.102e+04],
           [1.102e+04, 1.102e+04, ..., 1.102e+04, 1.102e+04]])
    """
    # retrieve window samples if needed so that the window derivative can be
    # calculated
    if win_length is None:
        win_length = n_fft

    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)

    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)

        S_h = stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    else:
        if dtype is None:
            dtype = S.dtype

        S_h = S

    # cyclic gradient to correctly handle edges of a periodic window
    window_derivative = util.cyclic_gradient(window)

    S_dh = stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_derivative,
        center=center,
        dtype=dtype,
        pad_mode=pad_mode,
    )

    # equation 5.20 of Flandrin, Auger, & Chassande-Mottin 2002
    # the sign of the correction is reversed in some papers - see Plante,
    # Meyer, & Ainsworth 1998 pp. 283-284
    with np.errstate(invalid="ignore"):
        # We can ignore divide-by-zero here because NaN is an acceptable correction value
        correction = -np.imag(S_dh / S_h)

    freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = util.expand_to(freqs, ndim=correction.ndim, axes=-2) + correction * (
        0.5 * sr / np.pi
    )

    return freqs, S_h


def __reassign_times(
    y: np.ndarray,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Time reassignments based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.23 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        t_reassigned = t + np.real(S_th/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_th`` is the complex STFT calculated using the original window multiplied
    by the time offset from the window center.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="constant"`` (zero padding) or else
    ``center=False``, rather than the defaults. Time reassignment assumes that
    the energy in each FFT bin is associated with exactly one impulse event.
    Reflection padding at the edges of the signal may invalidate the reassigned
    estimates in the boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_times`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    times : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Reassigned times:
        ``times[f, t]`` is the time for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Time estimates with zero support will produce a divide-by-zero warning
        and will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> times, S = librosa.core.spectrum.__reassign_times(y, sr=sr)
    >>> times
    array([[ 2.268e-05,  1.144e-02, ...,  5.332e+00,  5.333e+00],
           [ 2.268e-05,  1.451e-02, ...,  5.334e+00,  5.333e+00],
           ...,
           [ 2.268e-05, -6.177e-04, ...,  5.368e+00,  5.327e+00],
           [ 2.268e-05,  1.420e-03, ...,  5.307e+00,  5.328e+00]])
    """
    # retrieve window samples if needed so that the time-weighted window can be
    # calculated
    if win_length is None:
        win_length = n_fft

    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)

    # retrieve hop length if needed so that the frame times can be calculated
    if hop_length is None:
        hop_length = int(win_length // 4)

    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S

    # calculate window weighted by time
    half_width = n_fft // 2

    window_times: np.ndarray
    if n_fft % 2:
        window_times = np.arange(-half_width, half_width + 1)

    else:
        window_times = np.arange(0.5 - half_width, half_width)

    window_time_weighted = window * window_times

    S_th = stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_time_weighted,
        center=center,
        dtype=dtype,
        pad_mode=pad_mode,
    )

    # equation 5.23 of Flandrin, Auger, & Chassande-Mottin 2002
    # the sign of the correction is reversed in some papers - see Plante,
    # Meyer, & Ainsworth 1998 pp. 283-284
    with np.errstate(invalid="ignore"):
        # We can ignore divide-by-zero here because NaN is an acceptable correction value
        correction = np.real(S_th / S_h)

    if center:
        pad_length = None

    else:
        pad_length = n_fft

    times = convert.frames_to_time(
        np.arange(S_h.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length
    )

    times = util.expand_to(times, ndim=correction.ndim, axes=-1) + correction / sr

    return times, S_h


def reassigned_spectrogram(
    y: np.ndarray,
    *,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    reassign_frequencies: bool = True,
    reassign_times: bool = True,
    ref_power: Union[float, Callable] = 1e-6,
    fill_nan: bool = False,
    clip: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Time-frequency reassigned spectrogram.

    The reassignment vectors are calculated using equations 5.20 and 5.23 in
    [#]_::

        t_reassigned = t + np.real(S_th/S_h)
        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window,
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window, and ``S_th`` is the complex STFT calculated using the original window
    multiplied by the time offset from the window center. See [#]_ for
    additional algorithms, and [#]_ and [#]_ for history and discussion of the
    method.

    .. [#] Flandrin, P., Auger, F., & Chassande-Mottin, E. (2002).
        Time-Frequency reassignment: From principles to algorithms. In
        Applications in Time-Frequency Signal Processing (Vol. 10, pp.
        179-204). CRC Press.

    .. [#] Fulop, S. A., & Fitz, K. (2006). Algorithms for computing the
        time-corrected instantaneous frequency (reassigned) spectrogram, with
        applications. The Journal of the Acoustical Society of America, 119(1),
        360. doi:10.1121/1.2133000

    .. [#] Auger, F., Flandrin, P., Lin, Y.-T., McLaughlin, S., Meignen, S.,
        Oberlin, T., & Wu, H.-T. (2013). Time-Frequency Reassignment and
        Synchrosqueezing: An Overview. IEEE Signal Processing Magazine, 30(6),
        32-41. doi:10.1109/MSP.2013.2265316

    .. [#] Hainsworth, S., Macleod, M. (2003). Time-frequency reassignment: a
        review and analysis. Tech. Rep. CUED/FINFENG/TR.459, Cambridge
        University Engineering Department

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to ``reassigned_spectrogram``

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        See `stft` for details.

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True`` (default), the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``. See `Notes` for
          recommended usage in this function.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    reassign_frequencies : boolean
        - If ``True`` (default), the returned frequencies will be instantaneous
          frequency estimates.
        - If ``False``, the returned frequencies will be a read-only view of the
          STFT bin frequencies for all frames.

    reassign_times : boolean
        - If ``True`` (default), the returned times will be corrected
          (reassigned) time estimates for each bin.
        - If ``False``, the returned times will be a read-only view of the STFT
          frame times for all bins.

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating time-frequency reassignments.
        Any bin with ``np.abs(S[f, t])**2 < ref_power`` will be returned as
        `np.nan` in both frequency and time, unless ``fill_nan`` is ``True``. If 0
        is provided, then only bins with zero power will be returned as
        `np.nan` (unless ``fill_nan=True``).

    fill_nan : boolean
        - If ``False`` (default), the frequency and time reassignments for bins
          below the power threshold provided in ``ref_power`` will be returned as
          `np.nan`.
        - If ``True``, the frequency and time reassignments for these bins will
          be returned as the bin center frequencies and frame times.

    clip : boolean
        - If ``True`` (default), estimated frequencies outside the range
          `[0, 0.5 * sr]` or times outside the range `[0, len(y) / sr]` will be
          clipped to those ranges.
        - If ``False``, estimated frequencies and times beyond the bounds of the
          spectrogram may be returned.

    dtype : numeric type
        Complex numeric type for STFT calculation. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs, times, mags : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
            ``freqs[..., f, t]`` is the frequency for bin ``f``, frame ``t``.
            If ``reassign_frequencies=False``, this will instead be a read-only array
            of the same shape containing the bin center frequencies for all frames.

        Reassigned times:
            ``times[..., f, t]`` is the time for bin ``f``, frame ``t``.
            If ``reassign_times=False``, this will instead be a read-only array of
            the same shape containing the frame times for all bins.

        Magnitudes from short-time Fourier transform:
            ``mags[..., f, t]`` is the magnitude for bin ``f``, frame ``t``.

    Warns
    -----
    RuntimeWarning
        Frequency or time estimates with zero support will produce a
        divide-by-zero warning, and will be returned as `np.nan` unless
        ``fill_nan=True``.

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    It is recommended to use ``center=False`` with this function rather than the
    librosa default ``True``. Unlike ``stft``, reassigned times are not aligned to
    the left or center of each frame, so padding the signal does not affect the
    meaning of the reassigned times. However, reassignment assumes that the
    energy in each FFT bin is associated with exactly one signal component and
    impulse event.

    If ``reassign_times`` is ``False``, the frame times that are returned will be
    aligned to the left or center of the frame, depending on the value of
    ``center``. In this case, if ``center`` is ``True``, then ``pad_mode="wrap"`` is
    recommended for valid estimation of the instantaneous frequencies in the
    boundary frames.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> amin = 1e-10
    >>> n_fft = 64
    >>> sr = 4000
    >>> y = 1e-3 * librosa.clicks(times=[0.3], sr=sr, click_duration=1.0,
    ...                           click_freq=1200.0, length=8000) +\
    ...     1e-3 * librosa.clicks(times=[1.5], sr=sr, click_duration=0.5,
    ...                           click_freq=400.0, length=8000) +\
    ...     1e-3 * librosa.chirp(fmin=200, fmax=1600, sr=sr, duration=2.0) +\
    ...     1e-6 * np.random.randn(2*sr)
    >>> freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr,
    ...                                                     n_fft=n_fft)
    >>> mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
    ...                          hop_length=n_fft//4, ax=ax[0])
    >>> ax[0].set(title="Spectrogram", xlabel=None)
    >>> ax[0].label_outer()
    >>> ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    >>> ax[1].set_title("Reassigned spectrogram")
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    if not callable(ref_power) and ref_power < 0:
        raise ParameterError("ref_power must be non-negative or callable.")

    if not reassign_frequencies and not reassign_times:
        raise ParameterError("reassign_frequencies or reassign_times must be True.")

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # frequency and time reassignment if requested
    if reassign_frequencies:
        freqs, S = __reassign_frequencies(
            y=y,
            sr=sr,
            S=S,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    if reassign_times:
        times, S = __reassign_times(
            y=y,
            sr=sr,
            S=S,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    assert S is not None

    mags: np.ndarray = np.abs(S)

    # clean up reassignment issues: divide-by-zero, bins with near-zero power,
    # and estimates outside the spectrogram bounds

    # retrieve bin frequencies and frame times to replace missing estimates
    if fill_nan or not reassign_frequencies or not reassign_times:
        if center:
            pad_length = None

        else:
            pad_length = n_fft

        bin_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)

        frame_times = convert.frames_to_time(
            frames=np.arange(S.shape[-1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=pad_length,
        )

    # find bins below the power threshold
    # reassigned bins with zero power will already be NaN
    if callable(ref_power):
        ref_p = ref_power(mags**2)
    else:
        ref_p = ref_power
    mags_low = np.less(mags, ref_p**0.5, where=~np.isnan(mags))

    # for reassigned estimates, optionally set thresholded bins to NaN, return
    # bin frequencies and frame times in place of NaN generated by
    # divide-by-zero and power threshold, and clip to spectrogram bounds
    if reassign_frequencies:
        if ref_p > 0:
            freqs[mags_low] = np.nan

        if fill_nan:
            freqs = np.where(np.isnan(freqs), bin_freqs[:, np.newaxis], freqs)

        if clip:
            np.clip(freqs, 0, sr / 2.0, out=freqs)

    # or if reassignment was not requested, return bin frequencies and frame
    # times for every cell is the spectrogram
    else:
        freqs = np.broadcast_to(bin_freqs[:, np.newaxis], S.shape)

    if reassign_times:
        if ref_p > 0:
            times[mags_low] = np.nan

        if fill_nan:
            times = np.where(np.isnan(times), frame_times[np.newaxis, :], times)

        if clip:
            np.clip(times, 0, y.shape[-1] / float(sr), out=times)

    else:
        times = np.broadcast_to(frame_times[np.newaxis, :], S.shape)

    return freqs, times, mags


def magphase(D: np.ndarray, *, power: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    D_mag : np.ndarray [shape=(..., d, t), dtype=real]
        magnitude of ``D``, raised to ``power``
    D_phase : np.ndarray [shape=(..., d, t), dtype=complex]
        ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)
    >>> phase
    array([[ 1.   +0.000e+00j,  1.   +0.000e+00j, ...,
            -1.   -8.742e-08j, -1.   -8.742e-08j],
           [-1.   -8.742e-08j, -0.775-6.317e-01j, ...,
            -0.885-4.648e-01j,  0.472-8.815e-01j],
           ...,
           [ 1.   -4.342e-12j,  0.028-9.996e-01j, ...,
            -0.222-9.751e-01j, -0.75 -6.610e-01j],
           [-1.   -8.742e-08j, -1.   -8.742e-08j, ...,
             1.   +0.000e+00j,  1.   +0.000e+00j]], dtype=complex64)

    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[ 0.000e+00,  0.000e+00, ..., -3.142e+00, -3.142e+00],
           [-3.142e+00, -2.458e+00, ..., -2.658e+00, -1.079e+00],
           ...,
           [-4.342e-12, -1.543e+00, ..., -1.794e+00, -2.419e+00],
           [-3.142e+00, -3.142e+00, ...,  0.000e+00,  0.000e+00]],
          dtype=float32)
    """
    mag = np.abs(D)

    # Prevent NaNs and return magnitude 0, phase 1+0j for zero
    zeros_to_ones = mag == 0
    mag_nonzero = mag + zeros_to_ones
    # Compute real and imaginary separately, because complex division can
    # produce NaNs when denormalized numbers are involved (< ~2e-39 for
    # complex64, ~5e-309 for complex128)
    phase = np.empty_like(D, dtype=util.dtype_r2c(D.dtype))
    phase.real = D.real / mag_nonzero + zeros_to_ones
    phase.imag = D.imag / mag_nonzero

    mag **= power

    return mag, phase


def phase_vocoder(
    D: np.ndarray,
    *,
    rate: float,
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of ``rate``

    Based on the implementation provided by [#]_.

    This is a simplified implementation, intended primarily for
    reference and pedagogical purposes.  It makes no attempt to
    handle transients, and is likely to produce many audible
    artifacts.  For a higher quality implementation, we recommend
    the RubberBand library [#]_ and its Python wrapper `pyrubberband`.

    .. [#] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        https://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    .. [#] https://breakfastquay.com/rubberband/

    Examples
    --------
    >>> # Play at double speed
    >>> y, sr   = librosa.load(librosa.ex('trumpet'))
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_fast  = librosa.phase_vocoder(D, rate=2.0, hop_length=512)
    >>> y_fast  = librosa.istft(D_fast, hop_length=512)

    >>> # Or play at 1/3 speed
    >>> y, sr   = librosa.load(librosa.ex('trumpet'))
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_slow  = librosa.phase_vocoder(D, rate=1./3, hop_length=512)
    >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        STFT matrix

    rate : float > 0 [scalar]
        Speed-up factor: ``rate > 1`` is faster, ``rate < 1`` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of ``D``.

        If None, defaults to ``n_fft//4 = (D.shape[0]-1)//2``

    n_fft : int > 0 or None
        The number of samples per frame in D.
        By default (None), this will be inferred from the shape of D.
        However, if D was constructed using an odd-length window, the correct
        frame length can be specified here.

    Returns
    -------
    D_stretched : np.ndarray [shape=(..., d, t / rate), dtype=complex]
        time-stretched STFT

    See Also
    --------
    pyrubberband
    """
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # Expected phase advance in each bin per frame
    phi_advance = hop_length * convert.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[..., 0])

    # Pad 0 columns to simplify boundary logic
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for t, step in enumerate(time_steps):
        columns = D[..., int(step) : int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

        # Store to output array
        d_stretch[..., t] = util.phasor(phase_acc, mag=mag)

        # Compute phase advance
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


@cache(level=20)
def iirt(
    y: np.ndarray,
    *,
    sr: float = 22050,
    win_length: int = 2048,
    hop_length: Optional[int] = None,
    center: bool = True,
    tuning: float = 0.0,
    pad_mode: _PadMode = "constant",
    flayout: str = "sos",
    res_type: str = "soxr_hq",
    **kwargs: Any,
) -> np.ndarray:
    r"""Time-frequency representation using IIR filters

    This function will return a time-frequency representation
    using a multirate filter bank consisting of IIR filters. [#]_

    First, ``y`` is resampled as needed according to the provided ``sample_rates``.

    Then, a filterbank with with ``n`` band-pass filters is designed.

    The resampled input signals are processed by the filterbank as a whole.
    (`scipy.signal.filtfilt` resp. `sosfiltfilt` is used to make the phase linear.)
    The output of the filterbank is cut into frames.
    For each band, the short-time mean-square power (STMSP) is calculated by
    summing ``win_length`` subsequent filtered time samples.

    When called with the default set of parameters, it will generate the TF-representation
    (pitch filterbank):

        * 85 filters with MIDI pitches [24, 108] as ``center_freqs``.
        * each filter having a bandwidth of one semitone.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    win_length : int > 0, <= n_fft
        Window length.
    hop_length : int > 0 [scalar]
        Hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``D[..., :, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then `D[..., :, t]`` begins at ``y[t * hop_length]``
    tuning : float [scalar]
        Tuning deviation from A440 in fractions of a bin.
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, this function uses zero padding.
    flayout : string
        - If `sos` (default), a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
    res_type : string
        The resampling mode.  See `librosa.resample` for details.
    **kwargs : additional keyword arguments
        Additional arguments for `librosa.filters.semitone_filterbank`
        (e.g., could be used to provide another set of ``center_freqs`` and ``sample_rates``).

    Returns
    -------
    bands_power : np.ndarray [shape=(..., n, t), dtype=dtype]
        Short-time mean-square power for the input signal.

    Raises
    ------
    ParameterError
        If ``flayout`` is not None, `ba`, or `sos`.

    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters.mr_frequencies
    librosa.cqt
    scipy.signal.filtfilt
    scipy.signal.sosfiltfilt

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.iirt(y, sr=sr))
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Constant-Q transform')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[1])
    >>> ax[1].set_title('Semitone spectrogram (iirt)')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if flayout not in ("ba", "sos"):
        raise ParameterError(f"Unsupported flayout={flayout}")

    # check audio input
    util.valid_audio(y, mono=False)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = win_length // 4

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (win_length // 2, win_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # get the semitone filterbank
    filterbank_ct, sample_rates = semitone_filterbank(
        tuning=tuning, flayout=flayout, **kwargs
    )

    # create three downsampled versions of the audio signal
    y_resampled = []

    y_srs = np.unique(sample_rates)

    for cur_sr in y_srs:
        y_resampled.append(resample(y, orig_sr=sr, target_sr=cur_sr, res_type=res_type))

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = int(1 + (y.shape[-1] - win_length) // hop_length)

    # Pre-allocate the output array
    shape = list(y.shape)
    # Time dimension reduces to n_frames
    shape[-1] = n_frames
    # Insert a new axis at position -2 for filter response
    shape.insert(-1, len(filterbank_ct))

    bands_power = np.empty_like(y, shape=shape)

    slices: List[Union[int, slice]] = [slice(None) for _ in bands_power.shape]
    for i, (cur_sr, cur_filter) in enumerate(zip(sample_rates, filterbank_ct)):
        slices[-2] = i

        # filter the signal
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]

        if flayout == "ba":
            cur_filter_output = scipy.signal.filtfilt(
                cur_filter[0], cur_filter[1], y_resampled[cur_sr_idx], axis=-1
            )
        elif flayout == "sos":
            cur_filter_output = scipy.signal.sosfiltfilt(
                cur_filter, y_resampled[cur_sr_idx], axis=-1
            )

        factor = sr / cur_sr
        hop_length_STMSP = hop_length / factor
        win_length_STMSP_round = int(round(win_length / factor))

        # hop_length_STMSP is used here as a floating-point number.
        # The discretization happens at the end to avoid accumulated rounding errors.
        start_idx = np.arange(
            0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP
        )
        if len(start_idx) < n_frames:
            min_length = (
                int(np.ceil(n_frames * hop_length_STMSP)) + win_length_STMSP_round
            )
            cur_filter_output = util.fix_length(cur_filter_output, size=min_length)
            start_idx = np.arange(
                0,
                cur_filter_output.shape[-1] - win_length_STMSP_round,
                hop_length_STMSP,
            )
        start_idx = np.round(start_idx).astype(int)[:n_frames]

        idx = np.add.outer(start_idx, np.arange(win_length_STMSP_round))

        bands_power[tuple(slices)] = factor * np.sum(
            cur_filter_output[..., idx] ** 2, axis=-1
        )

    return bands_power


@overload
def power_to_db(
    S: _ComplexLike_co,
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.floating[Any]:
    ...

@overload
def power_to_db(
    S: _SequenceLike[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.ndarray:
    ...

@overload
def power_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def power_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::

            10 * log10(S / ref)

        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S/ref)) - top_db``

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

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)

    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)

    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@overload
def db_to_power(
    S_db: _FloatLike_co,
    *,
    ref: float = ...,
) -> np.floating[Any]:
    ...

@overload
def db_to_power(
        S_db: np.ndarray,
    *,
    ref: float = ...,
) -> np.ndarray:
    ...

@overload
def db_to_power(
    S_db: Union[_FloatLike_co, np.ndarray],
    *,
    ref: float = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def db_to_power(S_db: Union[_FloatLike_co, np.ndarray], *, ref: float = 1.0) -> Union[np.floating[Any], np.ndarray]:
    """Convert dB-scale values to a power values.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power values

    Notes
    -----
    This function caches at level 30.
    """
    return ref * np.power(10.0, S_db * 0.1)


@overload
def amplitude_to_db(
    S: _ComplexLike_co,
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.floating[Any]:
    ...

@overload
def amplitude_to_db(
    S: _SequenceLike[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.ndarray:
    ...

@overload
def amplitude_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def amplitude_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,
    but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``:
        ``20 * log10(S / ref)``.
        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``S`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(20 * log10(S/ref)) - top_db``

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
    """
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = np.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    out_array = magnitude if isinstance(magnitude, np.ndarray) else None
    power = np.square(magnitude, out=out_array)

    db: np.ndarray = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db


@overload
def db_to_amplitude(
    S_db: _FloatLike_co,
    *,
    ref: float = ...,
) -> np.floating[Any]:
    ...

@overload
def db_to_amplitude(
    S_db: np.ndarray,
    *,
    ref: float = ...,
) -> np.ndarray:
    ...

@overload
def db_to_amplitude(
    S_db: Union[_FloatLike_co, np.ndarray],
    *,
    ref: float = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def db_to_amplitude(S_db: Union[_FloatLike_co, np.ndarray], *, ref: float = 1.0) -> Union[np.floating[Any], np.ndarray]:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Optional reference amplitude.

    Returns
    -------
    S : np.ndarray
        Linear magnitude values

    Notes
    -----
    This function caches at level 30.
    """
    return db_to_power(S_db, ref=ref**2) ** 0.5


@cache(level=30)
def perceptual_weighting(
    S: np.ndarray, frequencies: np.ndarray, *, kind: str = "A", **kwargs: Any
) -> np.ndarray:
    """Perceptual weighting of a power spectrogram::

        S_p[..., f, :] = frequency_weighting(f, 'A') + 10*log(S[..., f, :] / ref)

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, t)]
        Power spectrogram
    frequencies : np.ndarray [shape=(d,)]
        Center frequency for each row of` `S``
    kind : str
        The frequency weighting curve to use.
        e.g. `'A'`, `'B'`, `'C'`, `'D'`, `None or 'Z'`
    **kwargs : additional keyword arguments
        Additional keyword arguments to `power_to_db`.

    Returns
    -------
    S_p : np.ndarray [shape=(..., d, t)]
        perceptually weighted version of ``S``

    See Also
    --------
    power_to_db

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Re-weight a CQT power spectrum, using peak power as reference

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
    >>> freqs = librosa.cqt_frequencies(C.shape[0],
    ...                                 fmin=librosa.note_to_hz('A1'))
    >>> perceptual_CQT = librosa.perceptual_weighting(C**2,
    ...                                               freqs,
    ...                                               ref=np.max)
    >>> perceptual_CQT
    array([[ -96.528,  -97.101, ..., -108.561, -108.561],
           [ -95.88 ,  -96.479, ..., -107.551, -107.551],
           ...,
           [ -65.142,  -53.256, ...,  -80.098,  -80.098],
           [ -71.542,  -53.197, ...,  -80.311,  -80.311]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                        ref=np.max),
    ...                                fmin=librosa.note_to_hz('A1'),
    ...                                y_axis='cqt_hz', x_axis='time',
    ...                                ax=ax[0])
    >>> ax[0].set(title='Log CQT power')
    >>> ax[0].label_outer()
    >>> imgp = librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
    ...                                 fmin=librosa.note_to_hz('A1'),
    ...                                 x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Perceptually weighted log CQT')
    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    >>> fig.colorbar(imgp, ax=ax[1], format="%+2.0f dB")
    """
    offset = convert.frequency_weighting(frequencies, kind=kind).reshape((-1, 1))

    result: np.ndarray = offset + power_to_db(S, **kwargs)
    return result


@cache(level=30)
def fmt(
    y: np.ndarray,
    *,
    t_min: float = 0.5,
    n_fmt: Optional[int] = None,
    kind: str = "cubic",
    beta: float = 0.5,
    over_sample: float = 1,
    axis: int = -1,
) -> np.ndarray:
    """Fast Mellin transform (FMT)

    The Mellin of a signal `y` is performed by interpolating `y` on an exponential time
    axis, applying a polynomial window, and then taking the discrete Fourier transform.

    When the Mellin parameter (beta) is 1/2, it is also known as the scale transform. [#]_
    The scale transform can be useful for audio analysis because its magnitude is invariant
    to scaling of the domain (e.g., time stretching or compression).  This is analogous
    to the magnitude of the Fourier transform being invariant to shifts in the input domain.

    .. [#] De Sena, Antonio, and Davide Rocchesso.
        "A fast Mellin and scale transform."
        EURASIP Journal on Applied Signal Processing 2007.1 (2007): 75-75.

    .. [#] Cohen, L.
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
        If None, then ``n_bins = over_sample * ceil(n * log((n-1)/t_min))`` is taken,
        where ``n = y.shape[axis]``

    kind : str
        The type of interpolation to use when re-sampling the input.
        See `scipy.interpolate.interp1d` for possible values.

        Note that the default is to use high-precision (cubic) interpolation.
        This can be slow in practice; if speed is preferred over accuracy,
        then consider using ``kind='linear'``.

    beta : float
        The Mellin parameter.  ``beta=0.5`` provides the scale transform.

    over_sample : float >= 1
        Over-sampling factor for exponential resampling.

    axis : int
        The axis along which to transform ``y``

    Returns
    -------
    x_scale : np.ndarray [dtype=complex]
        The scale transform of ``y`` along the ``axis`` dimension.

    Raises
    ------
    ParameterError
        if ``n_fmt < 2`` or ``t_min <= 0``
        or if ``y`` is not finite
        or if ``y.shape[axis] < 3``.

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Generate a signal and time-stretch it (with energy normalization)
    >>> scale = 1.25
    >>> freq = 3.0
    >>> x1 = np.linspace(0, 1, num=1024, endpoint=False)
    >>> x2 = np.linspace(0, 1, num=int(scale * len(x1)), endpoint=False)
    >>> y1 = np.sin(2 * np.pi * freq * x1)
    >>> y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)
    >>> # Verify that the two signals have the same energy
    >>> np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)
        (255.99999999999997, 255.99999999999969)
    >>> scale1 = librosa.fmt(y1, n_fmt=512)
    >>> scale2 = librosa.fmt(y2, n_fmt=512)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].plot(y1, label='Original')
    >>> ax[0].plot(y2, linestyle='--', label='Stretched')
    >>> ax[0].set(xlabel='time (samples)', title='Input signals')
    >>> ax[0].legend()
    >>> ax[1].semilogy(np.abs(scale1), label='Original')
    >>> ax[1].semilogy(np.abs(scale2), linestyle='--', label='Stretched')
    >>> ax[1].set(xlabel='scale coefficients', title='Scale transform magnitude')
    >>> ax[1].legend()

    >>> # Plot the scale transform of an onset strength autocorrelation
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr)
    >>> # Auto-correlate with up to 10 seconds lag
    >>> odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)
    >>> # Normalize
    >>> odf_ac = librosa.util.normalize(odf_ac, norm=np.inf)
    >>> # Compute the scale transform
    >>> odf_ac_scale = librosa.fmt(librosa.util.normalize(odf_ac), n_fmt=512)
    >>> # Plot the results
    >>> fig, ax = plt.subplots(nrows=3)
    >>> ax[0].plot(odf, label='Onset strength')
    >>> ax[0].set(xlabel='Time (frames)', title='Onset strength')
    >>> ax[1].plot(odf_ac, label='Onset autocorrelation')
    >>> ax[1].set(xlabel='Lag (frames)', title='Onset autocorrelation')
    >>> ax[2].semilogy(np.abs(odf_ac_scale), label='Scale transform magnitude')
    >>> ax[2].set(xlabel='scale coefficients')
    """
    n = y.shape[axis]

    if n < 3:
        raise ParameterError(f"y.shape[{axis}]=={n} < 3")

    if t_min <= 0:
        raise ParameterError(f"t_min={t_min} must be a positive number")

    if n_fmt is None:
        if over_sample < 1:
            raise ParameterError(f"over_sample={over_sample} must be >= 1")

        # The base is the maximum ratio between adjacent samples
        # Since the sample spacing is increasing, this is simply the
        # ratio between the positions of the last two samples: (n-1)/(n-2)
        log_base = np.log(n - 1) - np.log(n - 2)

        n_fmt = int(np.ceil(over_sample * (np.log(n - 1) - np.log(t_min)) / log_base))

    elif n_fmt < 3:
        raise ParameterError(f"n_fmt=={n_fmt} < 3")
    else:
        log_base = (np.log(n_fmt - 1) - np.log(n_fmt - 2)) / over_sample

    if not np.all(np.isfinite(y)):
        raise ParameterError("y must be finite everywhere")

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
    x_exp = np.logspace(
        (np.log(t_min) - np.log(n)) / log_base,
        0,
        num=n_fmt + n_over,
        endpoint=False,
        base=base,
    )[:-n_over]

    # Clean up any rounding errors at the boundaries of the interpolation
    # The interpolator gets angry if we try to extrapolate, so clipping is necessary here.
    if x_exp[0] < t_min or x_exp[-1] > float(n - 1.0) / n:
        x_exp = np.clip(x_exp, float(t_min) / n, x[-1])

    # Make sure that all sample points are unique
    # This should never happen!
    if len(np.unique(x_exp)) != len(x_exp):
        raise ParameterError("Redundant sample positions in Mellin transform")

    # Resample the signal
    y_res = f_interp(x_exp)

    # Broadcast the window correctly
    shape = [1] * y_res.ndim
    shape[axis] = -1

    # Apply the window and fft
    # Normalization is absorbed into the window here for expedience
    fft = get_fftlib()
    result: np.ndarray = fft.rfft(
        y_res * ((x_exp**beta).reshape(shape) * np.sqrt(n) / n_fmt), axis=axis
    )
    return result


@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: Literal[False] = ...,
) -> np.ndarray:
    ...


@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: Literal[True],
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: bool = ...,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ...


@cache(level=30)
def pcen(
    S: np.ndarray,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    gain: float = 0.98,
    bias: float = 2,
    power: float = 0.5,
    time_constant: float = 0.400,
    eps: float = 1e-6,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[np.ndarray] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[np.ndarray] = None,
    return_zf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-channel energy normalization (PCEN)

    This function normalizes a time-frequency representation ``S`` by
    performing automatic gain control, followed by nonlinear compression [#]_ ::

        P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power

    IMPORTANT: the default values of eps, gain, bias, and power match the
    original publication, in which ``S`` is a 40-band mel-frequency
    spectrogram with 25 ms windowing, 10 ms frame shift, and raw audio values
    in the interval [-2**31; 2**31-1[. If you use these default values, we
    recommend to make sure that the raw audio is properly scaled to this
    interval, and not to [-1, 1[ as is most often the case.

    The matrix ``M`` is the result of applying a low-pass, temporal IIR filter
    to ``S``::

        M[f, t] = (1 - b) * M[f, t - 1] + b * S[f, t]

    If ``b`` is not provided, it is calculated as::

        b = (sqrt(1 + 4* T**2) - 1) / (2 * T**2)

    where ``T = time_constant * sr / hop_length``. [#]_

    This normalization is designed to suppress background noise and
    emphasize foreground signals, and can be used as an alternative to
    decibel scaling (`amplitude_to_db`).

    This implementation also supports smoothing across frequency bins
    by specifying ``max_size > 1``.  If this option is used, the filtered
    spectrogram ``M`` is computed as::

        M[f, t] = (1 - b) * M[f, t - 1] + b * R[f, t]

    where ``R`` has been max-filtered along the frequency axis, similar to
    the SuperFlux algorithm implemented in `onset.onset_strength`::

        R[f, t] = max(S[f - max_size//2: f + max_size//2, t])

    This can be used to perform automatic gain control on signals that cross
    or span multiple frequency bans, which may be desirable for spectrograms
    with high frequency resolution.

    .. [#] Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.
       (2017, March). Trainable frontend for robust and far-field keyword spotting.
       In Acoustics, Speech and Signal Processing (ICASSP), 2017
       IEEE International Conference on (pp. 5670-5674). IEEE.

    .. [#] Lostanlen, V., Salamon, J., McFee, B., Cartwright, M., Farnsworth, A.,
       Kelling, S., and Bello, J. P. Per-Channel Energy Normalization: Why and How.
       IEEE Signal Processing Letters, 26(1), 39-43.

    Parameters
    ----------
    S : np.ndarray (non-negative)
        The input (magnitude) spectrogram

    sr : number > 0 [scalar]
        The audio sampling rate

    hop_length : int > 0 [scalar]
        The hop length of ``S``, expressed in samples

    gain : number >= 0 [scalar]
        The gain factor.  Typical values should be slightly less than 1.

    bias : number >= 0 [scalar]
        The bias point of the nonlinear compression (default: 2)

    power : number >= 0 [scalar]
        The compression exponent.  Typical values should be between 0 and 0.5.
        Smaller values of ``power`` result in stronger compression.
        At the limit ``power=0``, polynomial compression becomes logarithmic.

    time_constant : number > 0 [scalar]
        The time constant for IIR filtering, measured in seconds.

    eps : number > 0 [scalar]
        A small constant used to ensure numerical stability of the filter.

    b : number in [0, 1]  [scalar]
        The filter coefficient for the low-pass filter.
        If not provided, it will be inferred from ``time_constant``.

    max_size : int > 0 [scalar]
        The width of the max filter applied to the frequency axis.
        If left as `1`, no filtering is performed.

    ref : None or np.ndarray (shape=S.shape)
        An optional pre-computed reference spectrum (``R`` in the above).
        If not provided it will be computed from ``S``.

    axis : int [scalar]
        The (time) axis of the input spectrogram.

    max_axis : None or int [scalar]
        The frequency axis of the input spectrogram.
        If `None`, and ``S`` is two-dimensional, it will be inferred
        as the opposite from ``axis``.
        If ``S`` is not two-dimensional, and ``max_size > 1``, an error
        will be raised.

    zi : np.ndarray
        The initial filter delay values.

        This may be the ``zf`` (final delay values) of a previous call to ``pcen``, or
        computed by `scipy.signal.lfilter_zi`.

    return_zf : bool
        If ``True``, return the final filter delay values along with the PCEN output ``P``.
        This is primarily useful in streaming contexts, where the final state of one
        block of processing should be used to initialize the next block.

        If ``False`` (default) only the PCEN values ``P`` are returned.

    Returns
    -------
    P : np.ndarray, non-negative [shape=(n, m)]
        The per-channel energy normalized version of ``S``.
    zf : np.ndarray (optional)
        The final filter delay values.  Only returned if ``return_zf=True``.

    See Also
    --------
    amplitude_to_db
    librosa.onset.onset_strength

    Examples
    --------
    Compare PCEN to log amplitude (dB) scaling on Mel spectra

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('robin'))

    >>> # We recommend scaling y to the range [-2**31, 2**31[ before applying
    >>> # PCEN's default parameters. Furthermore, we use power=1 to get a
    >>> # magnitude spectrum instead of a power spectrum.
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, power=1)
    >>> log_S = librosa.amplitude_to_db(S, ref=np.max)
    >>> pcen_S = librosa.pcen(S * (2**31))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(log_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='log amplitude (dB)', xlabel=None)
    >>> ax[0].label_outer()
    >>> imgpcen = librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization')
    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    >>> fig.colorbar(imgpcen, ax=ax[1])

    Compare PCEN with and without max-filtering

    >>> pcen_max = librosa.pcen(S * (2**31), max_size=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Per-channel energy normalization (no max-filter)')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(pcen_max, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization (max_size=3)')
    >>> fig.colorbar(img, ax=ax)
    """
    if power < 0:
        raise ParameterError(f"power={power} must be nonnegative")

    if gain < 0:
        raise ParameterError(f"gain={gain} must be non-negative")

    if bias < 0:
        raise ParameterError(f"bias={bias} must be non-negative")

    if eps <= 0:
        raise ParameterError(f"eps={eps} must be strictly positive")

    if time_constant <= 0:
        raise ParameterError(f"time_constant={time_constant} must be strictly positive")

    if not util.is_positive_int(max_size):
        raise ParameterError(f"max_size={max_size} must be a positive integer")

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    if not 0 <= b <= 1:
        raise ParameterError(f"b={b} must be between 0 and 1")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "pcen was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call pcen(np.abs(D)) instead.",
            stacklevel=2,
        )
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError(
                "Max-filtering cannot be applied to 1-dimensional input"
            )
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError(
                        f"Max-filtering a {S.ndim:d}-dimensional spectrogram "
                        "requires you to specify max_axis"
                    )
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    if zi is None:
        # Make sure zi matches dimension to input
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]

    # Temporal integration
    S_smooth: np.ndarray
    zf: np.ndarray
    S_smooth, zf = scipy.signal.lfilter([b], [1, b - 1], ref, zi=zi, axis=axis)

    # Adaptive gain control
    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))

    # Dynamic range compression
    S_out: np.ndarray
    if power == 0:
        S_out = np.log1p(S * smooth)
    elif bias == 0:
        S_out = np.exp(power * (np.log(S) + np.log(smooth)))
    else:
        S_out = (bias**power) * np.expm1(power * np.log1p(S * smooth / bias))

    if return_zf:
        return S_out, zf
    else:
        return S_out


def griffinlim(
    S: np.ndarray,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    pad_mode: _PadModeSTFT = "constant",
    momentum: float = 0.99,
    init: Optional[str] = "random",
    random_state: Optional[
        Union[int, np.random.RandomState, np.random.Generator]
    ] = None,
) -> np.ndarray:
    """Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.

    Given a short-time Fourier transform magnitude matrix (``S``), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Parameters
    ----------
    S : np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
        An array of short-time Fourier transform magnitudes as produced by
        `stft`.

    n_iter : int > 0
        The number of iterations to run

    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``

    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``

    n_fft : None or int > 0
        The number of samples per frame.
        By default, this will be inferred from the shape of ``S`` as an even number.
        However, if an odd frame length was used, you can explicitly set ``n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`

    center : boolean
        If ``True``, the STFT is assumed to use centered frames.
        If ``False``, the STFT is assumed to use left-aligned frames.

    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is inferred
        to match the precision of the input spectrogram.

    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``S`` is
        a magnitude spectrogram with no initial phase estimates.

        If `None`, then the phase is initialized from ``S``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, np.random.RandomState, or np.random.Generator
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` or `np.random.Generator` instance, the random number
        generator itself.

        If `None`, defaults to the `np.random.default_rng()` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``S``

    See Also
    --------
    stft
    istft
    magphase
    filters.get_window

    Examples
    --------
    A basic STFT inverse example

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> # Get the magnitude spectrogram
    >>> S = np.abs(librosa.stft(y))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim(S)
    >>> # Invert without estimating phase
    >>> y_istft = librosa.istft(S)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
    >>> ax[0].set(title='Original', xlabel=None)
    >>> ax[0].label_outer()
    >>> librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
    >>> ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    >>> ax[1].label_outer()
    >>> librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
    >>> ax[2].set_title('Magnitude-only istft reconstruction')
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)  # type: ignore
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state  # type: ignore
    else:
        raise ParameterError(f"Unsupported random_state={random_state!r}")

    if momentum > 1:
        warnings.warn(
            f"Griffin-Lim with momentum={momentum} > 1 can be unstable. "
            "Proceed with caution!",
            stacklevel=2,
        )
    elif momentum < 0:
        raise ParameterError(f"griffinlim() called with momentum={momentum} < 0")

    # Infer n_fft from the spectrogram shape
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)

    # Infer the dtype from S
    angles = np.empty(S.shape, dtype=util.dtype_r2c(S.dtype))
    eps = util.tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = util.phasor((2 * np.pi * rng.random(size=S.shape)))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")

    # Place-holders for temporary data and reconstructed buffer
    rebuilt = None
    tprev = None
    inverse = None

    # Absorb magnitudes into angles
    angles *= S
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = istft(
            angles,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            out=inverse,
        )

        # Rebuild the spectrogram
        rebuilt = stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            out=rebuilt,
        )

        # Update our phase estimates
        angles[:] = rebuilt
        if tprev is not None:
            angles -= (momentum / (1 + momentum)) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        # Store
        rebuilt, tprev = tprev, rebuilt

    # Return the final phase estimates
    return istft(
        angles,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        out=inverse,
    )


def _spectrogram(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    power: float = 1,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, int]:
    """Retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Parameters
    ----------
    y : None or np.ndarray
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
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """
    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if n_fft is None:
            raise ParameterError(f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ParameterError(
                "Input signal must be provided to compute a spectrogram"
            )
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft
