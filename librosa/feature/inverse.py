#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature inversion"""

import warnings
import numpy as np

from ..core.fft import get_fftlib
from ..util.exceptions import ParameterError
from ..core.spectrum import griffinlim
from ..core.spectrum import db_to_power
from ..util.utils import tiny
from .. import filters
from ..util import nnls, expand_to
from numpy.typing import DTypeLike
from typing import Any, Optional
from .._typing import _WindowSpec, _PadModeSTFT

__all__ = ["mel_to_stft", "mel_to_audio", "mfcc_to_mel", "mfcc_to_audio"]


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
        number of FFT components in the resulting STFT
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram
    **kwargs : additional keyword arguments for Mel filter bank parameters
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
    S : np.ndarray [shape=(..., n_fft, t), non-negative]
        An approximate linear magnitude spectrogram

    See Also
    --------
    librosa.feature.melspectrogram
    librosa.stft
    librosa.filters.mel
    librosa.util.nnls

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = librosa.util.abs2(librosa.stft(y))
    >>> mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    >>> S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)

    Compare the results visually

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max, top_db=None),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Original STFT')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_inv, ref=np.max, top_db=None),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Reconstructed STFT')
    >>> ax[1].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(np.abs(S_inv - S),
    ...                                                  ref=S.max(), top_db=None),
    ...                          vmax=0, y_axis='log', x_axis='time', cmap='magma', ax=ax[2])
    >>> ax[2].set(title='Residual error (dB)')
    >>> fig.colorbar(img, ax=ax, format="%+2.f dB")
    """
    # Construct a mel basis with dtype matching the input data
    mel_basis = filters.mel(
        sr=sr, n_fft=n_fft, n_mels=M.shape[-2], dtype=M.dtype, **kwargs
    )

    # Find the non-negative least squares solution, and apply
    # the inverse exponent.
    # We'll do the exponentiation in-place.
    inverse = nnls(mel_basis, M)
    return np.power(inverse, 1.0 / power, out=inverse)


def mel_to_audio(
    M: np.ndarray,
    *,
    sr: float = 22050,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    power: float = 2.0,
    n_iter: int = 32,
    length: Optional[int] = None,
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
        number of FFT components in the resulting STFT
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : boolean
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    pad_mode : string
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
    **kwargs : additional keyword arguments for Mel filter bank parameters
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
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    ref: float = 1.0,
    lifter: float = 0,
) -> np.ndarray:
    """Invert Mel-frequency cepstral coefficients to approximate a Mel power
    spectrogram.

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

    fft = get_fftlib()
    logmel = fft.idct(mfcc, axis=-2, type=dct_type, norm=norm, n=n_mels)
    melspec: np.ndarray = db_to_power(logmel, ref=ref)
    return melspec


def mfcc_to_audio(
    mfcc: np.ndarray,
    *,
    n_mels: int = 128,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
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
    **kwargs : additional keyword arguments to pass through to `mel_to_audio`
    M : np.ndarray [shape=(..., n_mels, n), non-negative]
        The spectrogram as produced by `feature.melspectrogram`
    sr : number > 0 [scalar]
        sampling rate of the underlying signal
    n_fft : int > 0 [scalar]
        number of FFT components in the resulting STFT
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : boolean
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    pad_mode : string
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
    **kwargs : additional keyword arguments for Mel filter bank parameters
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
