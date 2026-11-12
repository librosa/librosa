#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature inversion"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy

from .. import filters
from ..core.spectrum import db_to_power, griffinlim
from ..util import expand_to, nnls
from ..util.exceptions import ParameterError
from ..util.utils import tiny

if TYPE_CHECKING:
    from collections.abc import Collection
    from typing import Any, Literal

    from numpy.typing import DTypeLike

    from .._typing import _DCTNorm, _DCTType, _PadModeSTFT, _WindowSpec

__all__ = [
    "mel_to_stft",
    "mel_to_audio",
    "mfcc_to_mel",
    "mfcc_to_audio",
    "logfrequency_to_stft",
    "logfrequency_to_audio",
    "pitch_to_stft",
    "pitch_to_audio",
]


def mel_to_stft(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:
    """Approximate STFT magnitude from a Mel power spectrogram.

    Parameters
    ----------
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        length of the FFT frame of the resulting STFT; the output has ``1 + n_fft // 2`` frequency bins
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    **kwargs :
        additional keyword arguments for Mel filter bank parameters
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    S : np.ndarray [shape=(..., 1 + n_fft // 2, t), non-negative]
        An approximate linear magnitude spectrogram

    See Also
    --------
    librosa.feature.melspectrogram
    librosa.stft
    librosa.filters.mel
    librosa.util.nnls

    Examples
    --------
    >>> y, sr = librosa.loadx('trumpet')
    >>> S = librosa.util.abs2(librosa.stft(y))
    >>> mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    >>> S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

    Compare the results visually

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(S, vscale='dBFS[power]', top_db=None,
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Original STFT')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(S_inv, vscale='dBFS[power]', top_db=None,
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Reconstructed STFT')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(S_inv - S, vscale=f'dB[power,{S.max()}]', top_db=None,
    ...                          vmax=0, y_axis='log', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Residual error (dB)')
    >>> librosa.display.colorbar_db(img, ax=ax)
    """
    # Construct a mel basis with dtype matching the input data
    mel_basis = filters.mel(
        sr=sr, n_fft=n_fft, n_mels=M.shape[-2], dtype=M.dtype, **kwargs
    )

    # Find the non-negative least squares solution, and apply
    # the inverse exponent.
    # We'll do the exponentiation in-place.
    inverse = nnls(mel_basis, M)
    np.power(inverse, 1.0 / power, out=inverse)
    return inverse


def mel_to_audio(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: int | None = None,
    dtype: DTypeLike = np.float32,
    **kwargs: Any,
) -> np.ndarray:
    """Invert a mel power spectrogram to audio using Griffin-Lim.

    This is primarily a convenience wrapper for:

        >>> S = librosa.feature.inverse.mel_to_stft(M)
        >>> y = librosa.griffinlim(S)

    Parameters
    ----------
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        length of the FFT frame for the intermediate STFT
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``
    window : str, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : bool
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    pad_mode : str
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    n_iter : int > 0
        The number of iterations for Griffin-Lim
    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.
    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is 32-bit float.
    **kwargs :
        additional keyword arguments for Mel filter bank parameters
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0

    Returns
    -------
    y : np.ndarray [shape(..., n,)]
        time-domain signal reconstructed from ``M``

    See Also
    --------
    librosa.griffinlim
    librosa.feature.melspectrogram
    librosa.filters.mel
    librosa.feature.inverse.mel_to_stft
    """
    stft = mel_to_stft(M, sr=sr, n_fft=n_fft, power=power, **kwargs)

    return griffinlim(
        stft,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
    )


def mfcc_to_mel(
    mfcc: np.ndarray,
    *,
    n_mels: int = 128,
    dct_type: _DCTType = 2,
    norm: _DCTNorm | None = "ortho",
    ref: float = 1.0,
    lifter: float = 0,
) -> np.ndarray:
    """Invert Mel-frequency cepstral coefficients to approximate a Mel power spectrogram.

    This inversion proceeds in two steps:

        1. The inverse DCT is applied to the MFCCs
        2. `librosa.db_to_power` is applied to map the dB-scaled result to a power spectrogram

    Parameters
    ----------
    mfcc : np.ndarray [shape=(..., n_mfcc, n)]
        The Mel-frequency cepstral coefficients
    n_mels : int > 0
        The number of Mel frequencies
    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type
        By default, DCT type-2 is used.
    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an orthonormal
        DCT basis.
        Normalization is not supported for `dct_type=1`.
    ref : float
        Reference power for (inverse) decibel calculation
    lifter : number >= 0
        If ``lifter>0``, apply inverse liftering (inverse cepstral filtering)::
            M[n, :] <- M[n, :] / (1 + sin(pi * (n + 1) / lifter) * lifter / 2)

    Returns
    -------
    M : np.ndarray [shape=(..., n_mels, n)]
        An approximate Mel power spectrum recovered from ``mfcc``

    Warns
    -----
    UserWarning
        due to critical values in lifter array that invokes underflow.

    See Also
    --------
    librosa.feature.mfcc
    librosa.feature.melspectrogram
    scipy.fft.dct
    """
    if lifter > 0:
        n_mfcc = mfcc.shape[-2]
        idx = np.arange(1, 1 + n_mfcc, dtype=mfcc.dtype)
        idx = expand_to(idx, ndim=mfcc.ndim, axes=-2)
        lifter_sine = 1 + lifter * 0.5 * np.sin(np.pi * idx / lifter)

        # raise a UserWarning if lifter array includes critical values
        if np.any(np.abs(lifter_sine) < np.finfo(lifter_sine.dtype).eps):
            warnings.warn(
                message="lifter array includes critical values that may invoke underflow.",
                category=UserWarning,
                stacklevel=2,
            )

        # lifter mfcc values
        mfcc = mfcc / (lifter_sine + tiny(mfcc))

    elif lifter != 0:
        raise ParameterError("MFCC to mel lifter must be a non-negative number.")

    logmel = scipy.fft.idct(mfcc, axis=-2, type=dct_type, norm=norm, n=n_mels)
    melspec: np.ndarray = db_to_power(logmel, ref=ref)
    return melspec


def mfcc_to_audio(
    mfcc: np.ndarray,
    *,
    n_mels: int = 128,
    dct_type: _DCTType = 2,
    norm: _DCTNorm | None = "ortho",
    ref: float = 1.0,
    lifter: float = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Convert Mel-frequency cepstral coefficients to a time-domain audio signal

    This function is primarily a convenience wrapper for the following steps:

        1. Convert mfcc to Mel power spectrum (`mfcc_to_mel`)
        2. Convert Mel power spectrum to time-domain audio (`mel_to_audio`)

    Parameters
    ----------
    mfcc : np.ndarray [shape=(..., n_mfcc, n)]
        The Mel-frequency cepstral coefficients
    n_mels : int > 0
        The number of Mel frequencies
    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type
        By default, DCT type-2 is used.
    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an orthonormal
        DCT basis.
        Normalization is not supported for ``dct_type=1``.
    ref : float
        Reference power for (inverse) decibel calculation
    lifter : number >= 0
        If ``lifter>0``, apply inverse liftering (inverse cepstral filtering)::
            M[n, :] <- M[n, :] / (1 + sin(pi * (n + 1) / lifter)) * lifter / 2
    **kwargs
        additional keyword arguments to pass through to `mel_to_audio`
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        length of the FFT frame for the intermediate STFT
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``
    window : str, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : bool
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    pad_mode : str
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    n_iter : int > 0
        The number of iterations for Griffin-Lim
    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.
    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is 32-bit float.
    **kwargs
        additional keyword arguments for Mel filter bank parameters
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        A time-domain signal reconstructed from `mfcc`

    See Also
    --------
    mfcc_to_mel
    mel_to_audio
    librosa.feature.mfcc
    librosa.griffinlim
    scipy.fft.dct
    """
    mel_spec = mfcc_to_mel(
        mfcc, n_mels=n_mels, dct_type=dct_type, norm=norm, ref=ref, lifter=lifter
    )

    return mel_to_audio(mel_spec, **kwargs)


def logfrequency_to_stft(
    L: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
    n_bins: int | None = None,
    bins_per_octave: int = 12,
    fmin: float | str | None = None,
    tuning: float = 0.0,
    filter_window: _WindowSpec = "triangle",
    norm: Literal["slaney"] | float | None = "slaney",
    **kwargs: Any,
) -> np.ndarray:
    """Approximate STFT magnitude from a logarithmic frequency spectrogram.

    Parameters
    ----------
    L : np.ndarray [shape=(..., n_bins, t), non-negative]
        The spectrogram as produced by `feature.logfrequency_spectrogram`
    sr : number > 0 [scalar]
        Sampling rate of the underlying signal.
    n_fft : int > 0 [scalar]
        Length of the FFT frame of the resulting STFT; the output has ``1 + n_fft // 2`` frequency bins.
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
    n_bins : int > 0 [scalar, optional]
        Number of logarithmic frequency bands. If None, defaults to ``L.shape[-2]``.
    bins_per_octave : int > 0 [scalar], default=12
        Number of filter bands per octave.
    fmin : float > 0 or str, optional
        Lowest frequency (in Hz) or note name (e.g. `'C1'`).
        Defaults to `note_to_hz('C1')` (~32.7 Hz).
    tuning : float, default=0.0
        Tuning deviation from A440 in fractional bins.
    filter_window : str or callable, default='triangle'
        Filter shape for the log-frequency basis.
    norm : {None, 'slaney', or number}, default='slaney'
        Normalization applied to each filter row.
    **kwargs : additional keyword arguments
        Additional keyword arguments for `librosa.filters.logfrequency`.

    Returns
    -------
    S : np.ndarray [shape=(..., 1 + n_fft // 2, t), non-negative]
        An approximate linear magnitude spectrogram.

    Raises
    ------
    ParameterError
        If ``n_bins`` is provided and does not match ``L.shape[-2]``.

    See Also
    --------
    logfrequency_to_audio
    librosa.feature.logfrequency_spectrogram
    librosa.filters.logfrequency
    librosa.util.nnls
    mel_to_stft

    Examples
    --------
    >>> y, sr = librosa.loadx('trumpet')
    >>> S = librosa.util.abs2(librosa.stft(y))
    >>> log_spec = librosa.feature.logfrequency_spectrogram(
    ...     S=S, sr=sr, bins_per_octave=24, fmin='C1'
    ... )
    >>> S_inv = librosa.feature.inverse.logfrequency_to_stft(
    ...     log_spec, sr=sr, bins_per_octave=24, fmin='C1'
    ... )

    Compare the results visually:

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(
    ...     S, vscale='dBFS[power]', top_db=None,
    ...     y_axis='log', x_axis='time', ax=ax[0]
    ... )
    >>> ax[0].set(title='Original STFT')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(
    ...     S_inv, vscale='dBFS[power]', top_db=None,
    ...     y_axis='log', x_axis='time', ax=ax[1]
    ... )
    >>> ax[1].set(title='Reconstructed STFT')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(
    ...     S_inv - S, vscale=f'dB[power,{S.max()}]', top_db=None,
    ...     vmax=0, y_axis='log', x_axis='time', ax=ax[2]
    ... )
    >>> ax[2].set(title='Residual error (dB)')
    >>> librosa.display.colorbar_db(img, ax=ax)
    """
    if n_bins is not None and n_bins != L.shape[-2]:
        raise ParameterError(
            f"Number of frequency bins in L ({L.shape[-2]}) does not match n_bins ({n_bins})"
        )

    log_basis = filters.logfrequency(
        sr=sr,
        n_fft=n_fft,
        n_bins=L.shape[-2],
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        tuning=tuning,
        window=filter_window,
        norm=norm,
        dtype=L.dtype,
        **kwargs,
    )

    inverse = nnls(log_basis, L)
    np.power(inverse, 1.0 / power, out=inverse)
    return inverse


def logfrequency_to_audio(
    L: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: int | None = None,
    dtype: DTypeLike = np.float32,
    n_bins: int | None = None,
    bins_per_octave: int = 12,
    fmin: float | str | None = None,
    tuning: float = 0.0,
    filter_window: _WindowSpec = "triangle",
    norm: Literal["slaney"] | float | None = "slaney",
    **kwargs: Any,
) -> np.ndarray:
    """Invert a logarithmic frequency spectrogram to audio using Griffin-Lim.

    This is primarily a convenience wrapper for:

        >>> S = librosa.feature.inverse.logfrequency_to_stft(L)
        >>> y = librosa.griffinlim(S)

    Parameters
    ----------
    L : np.ndarray [shape=(..., n_bins, t), non-negative]
        The spectrogram as produced by `feature.logfrequency_spectrogram`
    sr : number > 0 [scalar]
        Sampling rate of the underlying signal.
    n_fft : int > 0 [scalar]
        Length of the FFT frame for the intermediate STFT.
    hop_length : None or int > 0
        The hop length of the STFT. If not provided, it will default to ``n_fft // 4``.
    win_length : None or int > 0
        The window length of the STFT. By default, it will equal ``n_fft``.
    window : str, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`.
    center : bool
        If True, the STFT is assumed to use centered frames.
        If False, the STFT is assumed to use left-aligned frames.
    pad_mode : str
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
    n_iter : int > 0
        The number of iterations for Griffin-Lim.
    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length`` samples.
    dtype : np.dtype
        Real numeric type for the time-domain signal. Default is 32-bit float.
    n_bins : int > 0 [scalar, optional]
        Number of logarithmic frequency bands. If None, defaults to ``L.shape[-2]``.
    bins_per_octave : int > 0 [scalar], default=12
        Number of filter bands per octave.
    fmin : float > 0 or str, optional
        Lowest frequency (in Hz) or note name (e.g. `'C1'`).
        Defaults to `note_to_hz('C1')` (~32.7 Hz).
    tuning : float, default=0.0
        Tuning deviation from A440 in fractional bins.
    filter_window : str or callable, default='triangle'
        Filter shape for the log-frequency basis.
    norm : {None, 'slaney', or number}, default='slaney'
        Normalization applied to each filter row.
    **kwargs : additional keyword arguments
        Additional keyword arguments for `librosa.filters.logfrequency`.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        Time-domain signal reconstructed from ``L``.

    See Also
    --------
    logfrequency_to_stft
    librosa.feature.logfrequency_spectrogram
    librosa.griffinlim
    mel_to_audio
    """
    stft = logfrequency_to_stft(
        L,
        sr=sr,
        n_fft=n_fft,
        power=power,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        tuning=tuning,
        filter_window=filter_window,
        norm=norm,
        **kwargs,
    )

    return griffinlim(
        stft,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
    )


def pitch_to_stft(
    P: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
    n_pitches: int | None = None,
    fmin: float | str | None = None,
    intervals: str | Collection[float] = "equal",
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    filter_window: _WindowSpec = "boxcar",
    norm: Literal["slaney"] | float | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Approximate STFT magnitude from a musical pitch spectrogram.

    Parameters
    ----------
    P : np.ndarray [shape=(..., n_pitches, t), non-negative]
        The pitch spectrogram as produced by `feature.pitch_spectrogram`
    sr : number > 0 [scalar]
        Sampling rate of the underlying signal.
    n_fft : int > 0 [scalar]
        Length of the FFT frame of the resulting STFT; the output has ``1 + n_fft // 2`` frequency bins.
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
    n_pitches : int > 0 [scalar, optional]
        Number of pitch bands. If None, defaults to ``P.shape[-2]``.
    fmin : float or str, optional
        Minimum pitch center frequency (or note name, e.g. `'C1'`).
        If None, defaults to MIDI 0 (~8.18 Hz) when ``n_pitches=128``, or `'C1'`.
    intervals : str or collection of float, default='equal'
        Interval system used to construct the pitch filterbank.
    bins_per_octave : int > 0 [scalar], default=12
        Number of bins per octave in the interval system.
    tuning : float, default=0.0
        Tuning deviation from A440 in fractional bins.
    filter_window : str, default='boxcar'
        Filter shape for each pitch band.
    norm : {None, 'slaney', or number}, default=None
        Normalization applied to each filter row.
    **kwargs : additional keyword arguments
        Additional keyword arguments for `librosa.filters.pitch`.

    Returns
    -------
    S : np.ndarray [shape=(..., 1 + n_fft // 2, t), non-negative]
        An approximate linear magnitude spectrogram.

    Raises
    ------
    ParameterError
        If ``n_pitches`` is provided and does not match ``P.shape[-2]``.

    See Also
    --------
    pitch_to_audio
    librosa.feature.pitch_spectrogram
    librosa.filters.pitch
    librosa.util.nnls
    logfrequency_to_stft
    mel_to_stft

    Examples
    --------
    >>> y, sr = librosa.loadx('trumpet')
    >>> S = librosa.util.abs2(librosa.stft(y))
    >>> pitch_spec = librosa.feature.pitch_spectrogram(
    ...     S=S, sr=sr, n_pitches=84, fmin='C1'
    ... )
    >>> S_inv = librosa.feature.inverse.pitch_to_stft(
    ...     pitch_spec, sr=sr, n_pitches=84, fmin='C1'
    ... )
    """
    if n_pitches is not None and n_pitches != P.shape[-2]:
        raise ParameterError(
            f"Number of pitch bands in P ({P.shape[-2]}) does not match n_pitches ({n_pitches})"
        )

    pitch_basis = filters.pitch(
        sr=sr,
        n_fft=n_fft,
        n_pitches=P.shape[-2],
        fmin=fmin,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        window=filter_window,
        norm=norm,
        dtype=P.dtype,
        **kwargs,
    )

    inverse = nnls(pitch_basis, P)
    np.power(inverse, 1.0 / power, out=inverse)
    return inverse


def pitch_to_audio(
    P: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: int | None = None,
    dtype: DTypeLike = np.float32,
    n_pitches: int | None = None,
    fmin: float | str | None = None,
    intervals: str | Collection[float] = "equal",
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    filter_window: _WindowSpec = "boxcar",
    norm: Literal["slaney"] | float | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Invert a musical pitch spectrogram to audio using Griffin-Lim.

    This is primarily a convenience wrapper for:

        >>> S = librosa.feature.inverse.pitch_to_stft(P)
        >>> y = librosa.griffinlim(S)

    Parameters
    ----------
    P : np.ndarray [shape=(..., n_pitches, t), non-negative]
        The spectrogram as produced by `feature.pitch_spectrogram`
    sr : number > 0 [scalar]
        Sampling rate of the underlying signal.
    n_fft : int > 0 [scalar]
        Length of the FFT frame for the intermediate STFT.
    hop_length : None or int > 0
        The hop length of the STFT. If not provided, it will default to ``n_fft // 4``.
    win_length : None or int > 0
        The window length of the STFT. By default, it will equal ``n_fft``.
    window : str, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`.
    center : bool
        If True, the STFT is assumed to use centered frames.
        If False, the STFT is assumed to use left-aligned frames.
    pad_mode : str
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
    n_iter : int > 0
        The number of iterations for Griffin-Lim.
    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length`` samples.
    dtype : np.dtype
        Real numeric type for the time-domain signal. Default is 32-bit float.
    n_pitches : int > 0 [scalar, optional]
        Number of pitch bands. If None, defaults to ``P.shape[-2]``.
    fmin : float or str, optional
        Minimum pitch center frequency (or note name, e.g. `'C1'`).
        If None, defaults to MIDI 0 (~8.18 Hz) when ``n_pitches=128``, or `'C1'`.
    intervals : str or collection of float, default='equal'
        Interval system used to construct the pitch filterbank.
    bins_per_octave : int > 0 [scalar], default=12
        Number of bins per octave in the interval system.
    tuning : float, default=0.0
        Tuning deviation from A440 in fractional bins.
    filter_window : str, default='boxcar'
        Filter shape for each pitch band.
    norm : {None, 'slaney', or number}, default=None
        Normalization applied to each filter row.
    **kwargs : additional keyword arguments
        Additional keyword arguments for `librosa.filters.pitch`.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        Time-domain signal reconstructed from ``P``.

    See Also
    --------
    pitch_to_stft
    librosa.feature.pitch_spectrogram
    librosa.griffinlim
    logfrequency_to_audio
    mel_to_audio
    """
    stft = pitch_to_stft(
        P,
        sr=sr,
        n_fft=n_fft,
        power=power,
        n_pitches=n_pitches,
        fmin=fmin,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_window=filter_window,
        norm=norm,
        **kwargs,
    )

    return griffinlim(
        stft,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        pad_mode=pad_mode,
    )
