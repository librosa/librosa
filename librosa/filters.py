#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
librosa.filters provides functionality for filter bank construction, window
function processing, and related utilities.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numba import jit

from . import util
from ._cache import cache
from .core.convert import (
    fft_frequencies,
    hz_to_midi,
    hz_to_octs,
    mel_frequencies,
    midi_to_hz,
    note_to_hz,
    note_to_midi,
)
from .core.intervals import interval_frequencies
from .util.decorators import deprecated
from .util.exceptions import ParameterError

if TYPE_CHECKING:
    from typing import Any, Callable, Collection, Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from ._typing import _FloatLike_co, _WindowSpec


__all__ = [
    "mel",
    "chroma",
    "cq_to_chroma",
    "pitch_to_chroma",
    "frequency_filterbank",
    "logfrequency",
    "pitch",
    "window_bandwidth",
    "get_window",
    "mr_frequencies",
    "semitone_filterbank",
    "window_sumsquare",
    "diagonal_filter",
    "wavelet",
    "wavelet_lengths",
]

# Dictionary of window function bandwidths

WINDOW_BANDWIDTHS: dict[Any, float] = {
    "bart": 1.3334961334912805,
    "barthann": 1.4560255965133932,
    "bartlett": 1.3334961334912805,
    "bkh": 2.0045975283585014,
    "black": 1.7269681554262326,
    "blackharr": 2.0045975283585014,
    "blackman": 1.7269681554262326,
    "blackmanharris": 2.0045975283585014,
    "blk": 1.7269681554262326,
    "bman": 1.7859588613860062,
    "bmn": 1.7859588613860062,
    "bohman": 1.7859588613860062,
    "box": 1.0,
    "boxcar": 1.0,
    "brt": 1.3334961334912805,
    "brthan": 1.4560255965133932,
    "bth": 1.4560255965133932,
    "cosine": 1.2337005350199792,
    "flat": 2.7762255046484143,
    "flattop": 2.7762255046484143,
    "flt": 2.7762255046484143,
    "halfcosine": 1.2337005350199792,
    "ham": 1.3629455320350348,
    "hamm": 1.3629455320350348,
    "hamming": 1.3629455320350348,
    "han": 1.50018310546875,
    "hann": 1.50018310546875,
    "nut": 1.9763500280946082,
    "nutl": 1.9763500280946082,
    "nuttall": 1.9763500280946082,
    "ones": 1.0,
    "par": 1.9174603174603191,
    "parz": 1.9174603174603191,
    "parzen": 1.9174603174603191,
    "rect": 1.0,
    "rectangular": 1.0,
    "tri": 1.3331706523555851,
    "triang": 1.3331706523555851,
    "triangle": 1.3331706523555851,
}


@cache(level=10)
def mel(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: _FloatLike_co | str = 0.0,
    fmax: _FloatLike_co | str | None = None,
    htk: bool = False,
    norm: Literal["slaney"] | float | None = "slaney",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Create a Mel filter-bank.

    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.

    Parameters
    ----------
    sr : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft : int > 0 [scalar]
        length of the FFT frame; the filter bank has ``1 + n_fft // 2`` frequency bins

    n_mels : int > 0 [scalar]
        number of Mel bands to generate

    fmin : float >= 0 or str [scalar]
        lowest frequency (in Hz) or note name (e.g., 'C1')

    fmax : float >= 0 or str [scalar]
        highest frequency (in Hz) or note name (e.g., 'C7').
        If `None`, use ``fmax = sr / 2.0``

    htk : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization).

        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values
        (including `+-np.inf`).

        Otherwise, leave all the triangles aiming for a peak value of 1.0

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    See Also
    --------
    librosa.util.normalize

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(sr=22050, n_fft=2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])

    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(sr=22050, n_fft=2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(melfb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Mel filter', title='Mel filter bank')
    >>> fig.colorbar(img, ax=ax)
    """
    fmin = util._ensure_hz(fmin)

    if fmax is None:
        fmax = float(sr) / 2
    else:
        fmax = util._ensure_hz(fmax)

    # Initialize the weights
    n_mels = int(n_mels)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    # Vectorized lower and upper slopes for all bins
    lower = -ramps[:n_mels] / fdiff[:n_mels, np.newaxis]
    upper = ramps[2 : n_mels + 2] / fdiff[1 : n_mels + 1, np.newaxis]
    weights = np.maximum(0, np.minimum(lower, upper)).astype(dtype)

    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
        else:
            raise ParameterError(f"Unsupported norm={norm}")
    else:
        weights = util.normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels.",
            stacklevel=2,
        )

    return weights


@cache(level=10)
def chroma(
    *,
    sr: float,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: float | None = 2,
    norm: float | None = 2,
    base_c: bool = True,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Create a chroma filter bank.

    This creates a linear transformation matrix to project
    FFT bins onto chroma bins (i.e. pitch classes).

    Parameters
    ----------
    sr : number > 0 [scalar]
        audio sampling rate

    n_fft : int > 0 [scalar]
        length of the FFT frame; the filter bank has ``1 + n_fft // 2`` frequency bins

    n_chroma : int > 0 [scalar]
        number of chroma bins

    tuning : float
        Tuning deviation from A440 in fractions of a chroma bin.

    ctroct : float > 0 [scalar]
        Center of the octave dominance window (in octaves, A0 = 27.5 Hz).

    octwidth : float > 0 or None [scalar]
        ``ctroct`` and ``octwidth`` specify a dominance window:
        a Gaussian weighting centered on ``ctroct`` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of ``octwidth``.

        Set ``octwidth`` to `None` to use a flat weighting.

    norm : float > 0 or np.inf
        Normalization factor for each filter

    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    wts : ndarray [shape=(n_chroma, 1 + n_fft / 2)]
        Chroma filter matrix

    See Also
    --------
    librosa.util.normalize
    librosa.feature.chroma_stft

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Build a simple chroma filter bank

    >>> chromafb = librosa.filters.chroma(sr=22050, n_fft=4096)
    array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
           [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
    ...,
           [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
           [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

    Use quarter-tones instead of semitones

    >>> librosa.filters.chroma(sr=22050, n_fft=4096, n_chroma=24)
    array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
           [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
    ...,
           [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
           [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])

    Equally weight all octaves

    >>> librosa.filters.chroma(sr=22050, n_fft=4096, octwidth=None)
    array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
           [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
    ...,
           [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
           [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(chromafb, x_axis='linear', ax=ax)
    >>> ax.set(ylabel='Chroma filter', title='Chroma filter bank')
    >>> fig.colorbar(img, ax=ax)
    """
    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)

    # normalize each column
    wts = util.normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2))[np.newaxis, :]

    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)], dtype=dtype)


@cache(level=10)
def frequency_filterbank(
    freq_in: ArrayLike,
    freq_out: ArrayLike,
    *,
    bandwidths: float | ArrayLike | None = None,
    window: _WindowSpec | Callable[[np.ndarray], np.ndarray] = "triangle",
    scale: Literal["linear", "log"] = "linear",
    norm: Literal["slaney"] | float | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    r"""Construct a linear filterbank matrix over arbitrary input and output frequency coordinates.

    Parameters
    ----------
    freq_in : np.ndarray [shape=(n_in,)]
        Coordinates of input frequency bins (e.g. from `fft_frequencies`, `cqt_frequencies`).
        Must be a 1-dimensional array and monotonically non-decreasing.
    freq_out : np.ndarray [shape=(n_out,)]
        Center coordinates of target filter bands.
        Must be a 1-dimensional non-empty array and strictly increasing.
    bandwidths : float or np.ndarray [shape=(n_out,)], optional
        Bandwidth of each filter.
        If None, bandwidths and filter boundaries are automatically derived from
        adjacent centers in ``freq_out``.
    window : str or callable, default="triangle"
        Filter window shape:
        - `'triangle'` or `'triang'`: Piecewise linear ramp from the lower boundary
          to the center and down to the upper boundary.
        - `'boxcar'` or `'rect'` or `'rectangular'`: Rectangular indicator window
          between adjacent midpoints.
        - `'hann'` or `'cosine'`: Raised-cosine bell.
        - `'gaussian'`: Gaussian bell centered at each band center.
        - Callable: Any window function :math:`w(x)` mapping normalized coordinates
          :math:`x \\in [-1, 1]` to filter weights.
    scale : {'linear', 'log'}, default='linear'
        Domain in which bandwidths and filter supports are evaluated:
        - `'linear'`: Distances and bandwidths are measured in linear Hz.
        - `'log'`: Distances and bandwidths are measured in logarithmic units (octaves).
          Geometric midpoints are used for filter boundaries, preserving constant-Q
          relationships.
    norm : {None, 'slaney', or number}, default=None
        Normalization applied to each filter row:
        - `None`: Unnormalized peak height (1.0).
        - `'slaney'`: Continuous area normalization (energy density per Hz),
          scaling each filter by :math:`2 / \\text{width}` (or :math:`1 / \\text{width}` for boxcar).
        - Numeric: Use `librosa.util.normalize` to normalize each filter to unit :math:`L_p` norm.
    dtype : np.dtype, default=np.float32
        Data type of the output basis matrix.

    Returns
    -------
    filterbank : np.ndarray [shape=(n_out, n_in)]
        Transformation matrix: ``Y = filterbank.dot(X)``

    Raises
    ------
    ParameterError
        If ``freq_in`` is not 1D or not monotonically non-decreasing.
        If ``freq_out`` is not 1D, is empty, or not strictly increasing.
        If ``scale`` is not `'linear'` or `'log'`.
        If ``scale='log'`` and any frequency in ``freq_out`` is non-positive.
        If ``bandwidths`` is negative or improperly shaped.
        If ``window`` or ``norm`` is unsupported.

    See Also
    --------
    logfrequency
    pitch
    mel
    chroma
    pitch_to_chroma

    Examples
    --------
    Construct a triangular filterbank mapping FFT bins onto linearly spaced bands:

    >>> sr, n_fft = 22050, 2048
    >>> f_in = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    >>> f_out = np.linspace(200, 4000, 32)
    >>> fb = librosa.filters.frequency_filterbank(f_in, f_out, window="triangle")
    >>> fb.shape
    (32, 1025)

    Construct a logarithmic boxcar filterbank (exact energy pooling):

    >>> f_pitches = librosa.midi_to_hz(np.arange(21, 109))
    >>> fb_pitch = librosa.filters.frequency_filterbank(f_in, f_pitches, window="boxcar", scale="log")
    >>> fb_pitch.shape
    (88, 1025)
    """
    f_in = np.asarray(freq_in, dtype=float)
    f_out = np.asarray(freq_out, dtype=float)

    if f_in.ndim != 1:
        raise ParameterError(f"freq_in must be 1D, shape given: {f_in.shape}")
    if f_out.ndim != 1 or len(f_out) == 0:
        raise ParameterError(f"freq_out must be a non-empty 1D array, shape given: {f_out.shape}")
    if len(f_in) > 1 and np.any(np.diff(f_in) < 0):
        raise ParameterError("freq_in must be monotonically non-decreasing")
    if len(f_out) > 1 and np.any(np.diff(f_out) <= 0):
        raise ParameterError("freq_out must be strictly increasing")
    if scale not in ("linear", "log"):
        raise ParameterError(f"scale must be 'linear' or 'log', given: {scale}")

    if scale == "log":
        if np.any(f_out <= 0):
            raise ParameterError("freq_out must be strictly positive when scale='log'")
        u_out = np.log2(f_out)
        u_in = np.where(f_in > 0, np.log2(np.maximum(f_in, np.finfo(float).tiny)), -np.inf)
    else:
        u_out = f_out
        u_in = f_in

    n_out = len(u_out)
    if bandwidths is None:
        if n_out == 1:
            raise ParameterError("bandwidths must be specified when len(freq_out) == 1")
        c_prev = u_out[0] - (u_out[1] - u_out[0])
        c_next = u_out[-1] + (u_out[-1] - u_out[-2])
        c_ext = np.concatenate(([c_prev], u_out, [c_next]))

        left_overlapping = c_ext[:-2]
        right_overlapping = c_ext[2:]

        left_midpoints = (c_ext[:-2] + c_ext[1:-1]) / 2.0
        right_midpoints = (c_ext[1:-1] + c_ext[2:]) / 2.0
    else:
        b = np.broadcast_to(np.asarray(bandwidths, dtype=float), (n_out,))
        if np.any(b <= 0):
            raise ParameterError("bandwidths must be strictly positive")
        h = b / 2.0
        left_overlapping = u_out - h
        right_overlapping = u_out + h
        left_midpoints = left_overlapping
        right_midpoints = right_overlapping
        c_ext = None

    if window in ("triangle", "triang"):
        if bandwidths is None:
            assert c_ext is not None
            fdiff = np.diff(c_ext)
            ramps = np.subtract.outer(c_ext, u_in)
            lower = -ramps[:n_out] / fdiff[:n_out, np.newaxis]
            upper = ramps[2 : n_out + 2] / fdiff[1 : n_out + 1, np.newaxis]
            weights = np.maximum(0.0, np.minimum(lower, upper)).astype(dtype)
        else:
            diff = np.abs(u_in - u_out[:, np.newaxis])
            weights = np.maximum(0.0, 1.0 - diff / h[:, np.newaxis]).astype(dtype)
    elif window in ("boxcar", "rect", "rectangular"):
        in_band = (u_in >= left_midpoints[:, np.newaxis]) & (u_in < right_midpoints[:, np.newaxis])
        in_band[-1] |= (u_in == right_midpoints[-1])
        weights = in_band.astype(dtype)
    elif window in ("hann", "cosine"):
        l_edge = left_overlapping[:, np.newaxis]
        r_edge = right_overlapping[:, np.newaxis]
        c_pt = u_out[:, np.newaxis]
        weights = np.zeros((n_out, len(u_in)), dtype=dtype)
        left_mask = (u_in >= l_edge) & (u_in <= c_pt)
        right_mask = (u_in > c_pt) & (u_in <= r_edge)
        d_left = (u_in - l_edge) / (c_pt - l_edge)
        d_right = (u_in - c_pt) / (r_edge - c_pt)
        weights[left_mask] = np.sin(0.5 * np.pi * d_left[left_mask]) ** 2
        weights[right_mask] = np.cos(0.5 * np.pi * d_right[right_mask]) ** 2
    elif window == "gaussian":
        sigma = (right_overlapping - left_overlapping) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        diff = (u_in - u_out[:, np.newaxis]) / sigma[:, np.newaxis]
        weights = np.exp(-0.5 * (diff ** 2)).astype(dtype)
    elif callable(window):
        l_edge = left_overlapping[:, np.newaxis]
        r_edge = right_overlapping[:, np.newaxis]
        c_pt = u_out[:, np.newaxis]
        norm_coord = 2.0 * (u_in - c_pt) / (r_edge - l_edge)
        in_band = (norm_coord >= -1.0) & (norm_coord <= 1.0)
        weights = np.zeros((n_out, len(u_in)), dtype=dtype)
        weights[in_band] = window(norm_coord[in_band])  # type: ignore[arg-type]
    else:
        raise ParameterError(f"Unsupported window={window}")

    if norm == "slaney":
        if scale == "log":
            width_hz = 2.0 ** right_overlapping - 2.0 ** left_overlapping
        else:
            width_hz = right_overlapping - left_overlapping
        scale_factor = 1.0 if window in ("boxcar", "rect", "rectangular") else 2.0
        enorm = scale_factor / width_hz
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        weights = util.normalize(weights, norm=norm, axis=-1)

    return weights


@cache(level=10)
def logfrequency(
    *,
    sr: float | None = None,
    n_fft: int | None = None,
    freq_in: ArrayLike | None = None,
    n_bins: int | None = None,
    bins_per_octave: int = 12,
    fmin: float | str | None = None,
    tuning: float = 0.0,
    window: _WindowSpec = "triangle",
    norm: Literal["slaney"] | float | None = "slaney",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Create a logarithmic frequency filterbank.

    Produces a linear transformation matrix to project frequency bins
    onto geometrically spaced logarithmic frequency bands.

    Parameters
    ----------
    sr : number > 0 [scalar, optional]
        Audio sampling rate. Required if ``freq_in`` is not specified.
    n_fft : int > 0 [scalar, optional]
        Length of the FFT frame. Required if ``freq_in`` is not specified.
    freq_in : np.ndarray [shape=(n_in,)], optional
        Coordinates of input frequency bins. If not provided, computed
        as ``fft_frequencies(sr=sr, n_fft=n_fft)``.
    n_bins : int > 0 [scalar, optional]
        Number of logarithmic frequency bands to generate.
        If None, automatically spans up to the Nyquist frequency (or ``freq_in[-1]``).
    bins_per_octave : int > 0 [scalar], default=12
        Number of filter bands per octave. Common values:
        - 12: Semitone resolution.
        - 24: Quarter-tone resolution (as used in SuperFlux and madmom).
    fmin : float > 0 or str, optional
        Lowest frequency (in Hz) or note name (e.g. `'C1'`).
        Defaults to `note_to_hz('C1')` (~32.7 Hz).
    tuning : float, default=0.0
        Tuning deviation from A440 in fractional bins.
    window : str, default='triangle'
        Filter shape for each band. Choices include `'triangle'`, `'boxcar'`,
        `'hann'`, `'gaussian'`.
    norm : {None, 'slaney', or number}, default='slaney'
        Normalization applied to each filter row.
    dtype : np.dtype, default=np.float32
        Data type of the output basis matrix.

    Returns
    -------
    filterbank : np.ndarray [shape=(n_bins, n_in)]
        Log-frequency filter matrix: ``Y = filterbank.dot(X)``

    See Also
    --------
    frequency_filterbank
    pitch
    mel
    librosa.feature.logfrequency_spectrogram

    Examples
    --------
    Construct a semitone logarithmic filterbank covering 7 octaves (84 bins):

    >>> fb = librosa.filters.logfrequency(
    ...     sr=22050, n_fft=2048, n_bins=84, bins_per_octave=12, fmin="C1"
    ... )
    >>> fb.shape
    (84, 1025)

    Construct a quarter-tone filterbank (24 bins/octave) extending to Nyquist:

    >>> fb_nyq = librosa.filters.logfrequency(
    ...     sr=22050, n_fft=2048, bins_per_octave=24, fmin=30.0
    ... )
    >>> fb_nyq.shape
    (205, 1025)
    """
    if freq_in is None:
        if sr is None or n_fft is None:
            raise ParameterError("Either freq_in or both sr and n_fft must be provided")
        f_in = fft_frequencies(sr=sr, n_fft=n_fft)
    else:
        f_in = np.asarray(freq_in, dtype=float)

    if fmin is None:
        f_min = float(note_to_hz("C1"))
    elif isinstance(fmin, str):
        f_min = float(note_to_hz(fmin))
    else:
        f_min = float(fmin)

    if f_min <= 0:
        raise ParameterError(f"fmin={fmin} must be strictly positive")

    if tuning != 0.0:
        f_min = f_min * (2.0 ** (tuning / float(bins_per_octave)))

    f_top = float(sr / 2.0) if sr is not None else float(f_in[-1])
    if f_min >= f_top:
        raise ParameterError(f"fmin={f_min} must be less than maximum frequency {f_top}")

    if n_bins is None:
        n_octaves = np.log2(f_top / f_min)
        n_bins = int(np.floor(n_octaves * bins_per_octave)) + 1
        f_out = f_min * (2.0 ** (np.arange(n_bins) / float(bins_per_octave)))
        f_out = f_out[f_out <= f_top]
    else:
        if n_bins <= 0:
            raise ParameterError(f"n_bins={n_bins} must be strictly positive")
        f_out = f_min * (2.0 ** (np.arange(n_bins) / float(bins_per_octave)))

    if len(f_out) == 0:
        raise ParameterError(f"No filters fall within range below {f_top}.")

    return frequency_filterbank(
        freq_in=f_in,
        freq_out=f_out,
        window=window,
        scale="log",
        norm=norm,
        dtype=dtype,
    )


@cache(level=10)
def pitch(
    *,
    sr: float | None = None,
    n_fft: int | None = None,
    freq_in: ArrayLike | None = None,
    n_pitches: int = 128,
    fmin: float | str | None = None,
    intervals: str | Collection[float] = "equal",
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    window: _WindowSpec = "boxcar",
    norm: Literal["slaney"] | float | None = None,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Create a musical pitch-anchored filterbank.

    Produces a linear transformation matrix mapping input frequency bins
    to discrete musical pitch bands under specified intonation assumptions.

    Parameters
    ----------
    sr : number > 0 [scalar, optional]
        Audio sampling rate. Required if ``freq_in`` is not specified.
    n_fft : int > 0 [scalar, optional]
        Length of the FFT frame. Required if ``freq_in`` is not specified.
    freq_in : np.ndarray [shape=(n_in,)], optional
        Coordinates of input frequency bins. If not provided, computed
        as ``fft_frequencies(sr=sr, n_fft=n_fft)``.
    n_pitches : int > 0 [scalar], default=128
        Number of pitch bands to generate. Default 128 covers MIDI notes 0 to 127.
    fmin : float or str, optional
        Minimum pitch center frequency (or note name, e.g. `'C1'`).
        If None, defaults to MIDI 0 (~8.18 Hz) when ``n_pitches=128``, or `'C1'` (~32.7 Hz).
    intervals : str or collection of float, default='equal'
        Tuning / interval system to use:
        - `'equal'`: Equal temperament.
        - `'pythagorean'`: Pythagorean intervals.
        - `'ji3'`, `'ji5'`, `'ji7'`: 3-limit, 5-limit, or 7-limit Just Intonation.
        - Collection of floats in [1, 2): Custom interval set.
    bins_per_octave : int > 0 [scalar], default=12
        Number of bins per octave in the interval system.
    tuning : float [scalar], default=0.0
        Tuning deviation from A440 in fractions of a bin.
    window : str, default='boxcar'
        Filter shape for each pitch band:
        - `'boxcar'`: Rectangular indicator window $[f(p - 0.5), f(p + 0.5))$.
          With ``norm=None``, this identically reproduces `libfmp.pool_pitch` energy pooling.
        - `'triangle'`: Triangular center-weighted filter.
        - `'hann'`: Raised-cosine bell.
        - `'gaussian'`: Gaussian bump.
    norm : {None, 'slaney', or number}, default=None
        Normalization applied to each filter row. Default None performs unnormalized
        energy summation.
    dtype : np.dtype, default=np.float32
        Data type of the output basis matrix.

    Returns
    -------
    filterbank : np.ndarray [shape=(n_pitches, n_in)]
        Pitch filter matrix: ``Y = filterbank.dot(X)``

    See Also
    --------
    frequency_filterbank
    logfrequency
    pitch_to_chroma
    librosa.feature.pitch_spectrogram

    Examples
    --------
    Construct a 128-pitch boxcar filterbank (libfmp pool_pitch equivalent):

    >>> fb = librosa.filters.pitch(sr=22050, n_fft=2048)
    >>> fb.shape
    (128, 1025)

    Construct a Just Intonation pitch filterbank:

    >>> fb_ji = librosa.filters.pitch(sr=22050, n_fft=2048, intervals="ji5", fmin="C1", n_pitches=84)
    >>> fb_ji.shape
    (84, 1025)
    """
    if freq_in is None:
        if sr is None or n_fft is None:
            raise ParameterError("Either freq_in or both sr and n_fft must be provided")
        f_in = fft_frequencies(sr=sr, n_fft=n_fft)
    else:
        f_in = np.asarray(freq_in, dtype=float)

    if fmin is None:
        if intervals == "equal" and n_pitches == 128 and bins_per_octave == 12:
            fmin_ = float(midi_to_hz(0))
        else:
            fmin_ = float(note_to_hz("C1"))
    elif isinstance(fmin, str):
        fmin_ = float(note_to_hz(fmin))
    else:
        fmin_ = float(fmin)

    f_out = interval_frequencies(
        n_bins=n_pitches,
        fmin=fmin_,
        intervals=intervals,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
    )

    return frequency_filterbank(
        freq_in=f_in,
        freq_out=f_out,
        window=window,
        scale="log",
        norm=norm,
        dtype=dtype,
    )


def __float_window(window_spec):
    """Decorate a window function to support fractional input lengths.

    This function guarantees that for fractional ``x``, the following hold:

    1. ``__float_window(window_function)(x)`` has length ``np.ceil(x)``
    2. all values from ``np.floor(x)`` are set to 0.

    For integer-valued ``x``, there should be no change in behavior.
    """
    def _wrap(n, **kwargs):
        """Wrap the window"""
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min, **kwargs)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))], mode="constant")

        window[n_min:] = 0.0

        return window

    return _wrap


@cache(level=10)
def wavelet_lengths(
    *,
    freqs: ArrayLike,
    sr: float = 22050,
    window: _WindowSpec = "hann",
    filter_scale: float = 1,
    gamma: float | None = 0,
    alpha: float | np.ndarray | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Return length of each filter in a wavelet basis.

    Parameters
    ----------
    freqs : np.ndarray (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.

    sr : number > 0 [scalar]
        Audio sampling rate

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    gamma : number >= 0 [scalar, optional]
        Bandwidth offset for determining filter lengths, as used in
        Variable-Q transforms.

        Bandwidth for the k'th filter is determined by::

            B[k] = alpha[k] * freqs[k] + gamma

        ``alpha[k]`` is twice the relative difference between ``freqs[k+1]`` and ``freqs[k-1]``::

            alpha[k] = (freqs[k+1]-freqs[k-1]) / (freqs[k+1]+freqs[k-1])

        If ``freqs`` follows a geometric progression (as in CQT and VQT), the vector
        ``alpha`` is constant and such that::

            (1 + alpha) * freqs[k-1] = (1 - alpha) * freqs[k+1]

        Furthermore, if ``gamma=0`` (default), ``alpha`` is such that even-``k`` and
        odd-``k`` filters are interleaved::

            freqs[k-1] + B[k-1] = freqs[k+1] - B[k+1]

        If ``gamma=None`` is specified, then ``gamma`` is computed such
        that each filter has bandwidth proportional to the equivalent
        rectangular bandwidth (ERB) at frequency ``freqs[k]``::

            gamma[k] = 24.7 * alpha[k] / 0.108

        as derived by [#]_.

        .. [#] Glasberg, Brian R., and Brian CJ Moore.
            "Derivation of auditory filter shapes from notched-noise data."
            Hearing research 47.1-2 (1990): 103-138.

    alpha : number > 0 [optional]
        Optional pre-computed relative bandwidth parameter.
        Note that this must be provided if ``len(freqs)==1`` because bandwidth cannot be
        inferred from a single frequency.
        Otherwise, if left unspecified, it will be automatically derived by the rules
        specified above.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.
    f_cutoff : float
        The lowest frequency at which all filters' main lobes have decayed by
        at least 3dB.

        This second output serves in cqt and vqt to ensure that all wavelet
        bands remain below the Nyquist frequency.

    Raises
    ------
    ParameterError
        - If ``filter_scale`` is not strictly positive

        - If ``gamma`` is a negative number

        - If any frequencies are <= 0

        - If the frequency array is not sorted in ascending order

    Notes
    -----
    This function caches at level 10.
    """
    freqs = np.asarray(freqs)
    if filter_scale <= 0:
        raise ParameterError(f"filter_scale={filter_scale} must be positive")

    if gamma is not None and gamma < 0:
        raise ParameterError(f"gamma={gamma} must be non-negative")

    if np.any(freqs <= 0):
        raise ParameterError("frequencies must be strictly positive")

    if len(freqs) > 1 and np.any(freqs[:-1] > freqs[1:]):
        raise ParameterError(
            f"Frequency array={freqs} must be in strictly ascending order"
        )

    if alpha is None:
        alpha = _relative_bandwidth(freqs=freqs)
    else:
        alpha = np.asarray(alpha)

    gamma_: _FloatLike_co | np.ndarray
    if gamma is None:
        gamma_ = alpha * 24.7 / 0.108
    else:
        gamma_ = gamma
    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / alpha

    # How far up does our highest frequency reach?
    f_cutoff = max(freqs * (1 + 0.5 * window_bandwidth(window) / Q) + 0.5 * gamma_)

    # Convert frequencies to filter lengths
    lengths = Q * sr / (freqs + gamma_ / alpha)

    return lengths, f_cutoff


def _relative_bandwidth(*, freqs: np.ndarray) -> np.ndarray:
    """Compute the relative bandwidth for each of a set of specified frequencies.

    This function is used as a helper in wavelet basis construction.

    Parameters
    ----------
    freqs : np.ndarray
        The array of frequencies

    Returns
    -------
    alpha : np.ndarray
        Relative bandwidth
    """
    if len(freqs) <= 1:
        raise ParameterError(f"2 or more frequencies are required to compute bandwidths. Given freqs={freqs}")

    # Approximate the local octave resolution around each frequency
    bpo = np.empty_like(freqs)
    logf = np.log2(freqs)
    # Reflect at the lowest and highest frequencies
    bpo[0] = 1 / (logf[1] - logf[0])
    bpo[-1] = 1 / (logf[-1] - logf[-2])

    # For everything else, do a centered difference
    bpo[1:-1] = 2 / (logf[2:] - logf[:-2])

    # Compute relative bandwidths
    alpha = (2.0 ** (2 / bpo) - 1) / (2.0 ** (2 / bpo) + 1)
    return alpha


@cache(level=10)
def wavelet(
    *,
    freqs: np.ndarray,
    sr: float = 22050,
    window: _WindowSpec = "hann",
    filter_scale: float = 1,
    pad_fft: bool = True,
    norm: float | None = 1,
    dtype: DTypeLike = np.complex64,
    gamma: float = 0,
    alpha: float | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a wavelet basis using windowed complex sinusoids.

    This function constructs a wavelet filterbank at a specified set of center
    frequencies.

    Parameters
    ----------
    freqs : np.ndarray (positive)
        Center frequencies of the filters (in Hz).
        Must be in ascending order.

    sr : number > 0 [scalar]
        Audio sampling rate

    window : str, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : bool
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the ``mode=`` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    dtype : np.dtype
        The data type of the output basis.
        By default, uses 64-bit (single precision) complex floating point.

    gamma : number >= 0
        Bandwidth offset for variable-Q transforms.

    alpha : number > 0 [optional]
        Optional pre-computed relative bandwidth parameter.
        Note that this must be provided if ``len(freqs)==1`` because bandwidth cannot be
        inferred from a single frequency.
        Otherwise, if left unspecified, it will be automatically derived by the rules
        specified above.

    **kwargs : additional keyword arguments
        Arguments to `np.pad()` when ``pad==True``.

    Returns
    -------
    filters : np.ndarray, ``len(filters) == n_bins``
        each ``filters[i]`` is a (complex) time-domain filter
    lengths : np.ndarray, ``len(lengths) == n_bins``
        The (fractional) length of each filter in samples

    See Also
    --------
    wavelet_lengths
    librosa.cqt
    librosa.vqt
    librosa.util.normalize

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Create a constant-Q basis

    >>> freqs = librosa.cqt_frequencies(n_bins=84, fmin='C1')
    >>> basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=22050)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.wavelet(freqs=freqs, sr=22050)
    >>> fig, ax = plt.subplots(nrows=2, figsize=(10, 6))
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     ax[0].plot(i + f_scale.real)
    ...     ax[0].plot(i + f_scale.imag, linestyle=':')
    >>> ax[0].set(yticks=np.arange(len(notes[:12])), yticklabels=notes[:12],
    ...           ylabel='CQ filters',
    ...           title='CQ filters (one octave, time domain)',
    ...           xlabel='Time (samples at 22050 Hz)')
    >>> ax[0].legend(['Real', 'Imaginary'])
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear', y_axis='cqt_note', ax=ax[1])
    >>> ax[1].set(ylabel='CQ filters', title='CQ filter magnitudes (frequency domain)')
    """
    # Pass-through parameters to get the filter lengths
    lengths, _ = wavelet_lengths(
        freqs=freqs,
        sr=sr,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
        alpha=alpha,
    )

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs, strict=True):
        # Build the filter: note, length will be ceil(ilen)
        sig = util.phasor(
            np.arange(-ilen // 2, ilen // 2, dtype=float) * 2 * np.pi * freq / sr
        )

        # Apply the windowing function
        sig *= __float_window(window)(len(sig))

        # Normalize
        sig = util.normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [util.pad_center(filt, size=max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, lengths


@cache(level=10)
def pitch_to_chroma(
    n_input: int,
    *,
    bins_per_octave: int = 12,
    n_chroma: int = 12,
    fmin: _FloatLike_co | str | None = None,
    window: np.ndarray | None = None,
    base: str | int = "C",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Construct a linear transformation matrix to map pitch/Constant-Q bins onto chroma bins.

    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (pitch or CQT bins)
    bins_per_octave : int > 0 [scalar], default=12
        How many bins per octave in the input representation.
    n_chroma : int > 0 [scalar], default=12
        Number of output bins (per octave) in the chroma.
    fmin : None or float > 0, optional
        Center frequency of the first constant-Q / pitch channel.
        Default: `note_to_hz('C1')` (~32.7 Hz).
    window : None or np.ndarray, optional
        If provided, the filter bank will be convolved with ``window``.
    base : str or int, default='C'
        The base note name (e.g. `'C'`, `'A'`, `'F#'`) or integer pitch class (0 to 11)
        for the first chroma bin.
    dtype : np.dtype, default=np.float32
        The data type of the output basis matrix.

    Returns
    -------
    pitch_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: ``Chroma = pitch_to_chroma.dot(X)``

    Raises
    ------
    ParameterError
        If ``n_input`` is not an integer multiple of ``n_chroma``.
        If ``base`` is invalid.

    See Also
    --------
    pitch
    logfrequency
    chroma
    cq_to_chroma

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.loadx('trumpet')
    >>> CQT = np.abs(librosa.cqt(y, sr=sr))
    >>> chroma_map = librosa.filters.pitch_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)
    """
    n_merge = float(bins_per_octave) / n_chroma

    fmin_: _FloatLike_co
    if fmin is None:
        fmin_ = note_to_hz("C1")
    else:
        fmin_ = util._ensure_hz(fmin)

    if np.mod(n_merge, 1) != 0:
        raise ParameterError(
            "Incompatible CQ merge: "
            "input bins must be an "
            "integer multiple of output bins."
        )

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), int(n_merge), axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, -int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin?
    midi_0 = np.mod(hz_to_midi(fmin_), 12)

    if isinstance(base, str):
        base_midi = np.mod(note_to_midi(base), 12)
    elif isinstance(base, (int, np.integer)):
        base_midi = int(base) % 12
    else:
        raise ParameterError(f"base={base} must be a string or integer")

    roll = int(np.round((midi_0 - base_midi) * (n_chroma / 12.0)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(dtype)

    if window is not None:
        import scipy.signal

        cq_to_ch = scipy.signal.convolve(cq_to_ch, np.atleast_2d(window), mode="same")

    return cq_to_ch


@deprecated(version="1.1.0", version_removed="2.0")
@cache(level=10)
def cq_to_chroma(
    n_input: int,
    *,
    bins_per_octave: int = 12,
    n_chroma: int = 12,
    fmin: _FloatLike_co | str | None = None,
    window: np.ndarray | None = None,
    base_c: bool = True,
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Construct a linear transformation matrix to map Constant-Q bins onto chroma bins.

    .. deprecated:: 1.1.0
        This function is deprecated and will be removed in version 2.0.
        Use `librosa.filters.pitch_to_chroma` instead.

    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (CQT bins)
    bins_per_octave : int > 0 [scalar]
        How many bins per octave in the CQT
    n_chroma : int > 0 [scalar]
        Number of output bins (per octave) in the chroma
    fmin : None, float > 0, or str
        Center frequency of the first constant-Q channel.
        Default: 'C1' ~= 32.7 Hz
    window : None or np.ndarray
        If provided, the cq_to_chroma filter bank will be
        convolved with ``window``.
    base_c : bool
        If True, the first chroma bin will start at 'C'
        If False, the first chroma bin will start at 'A'
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    cq_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: ``Chroma = np.dot(cq_to_chroma, CQT)``

    Raises
    ------
    ParameterError
        If ``n_input`` is not an integer multiple of ``n_chroma``

    See Also
    --------
    pitch_to_chroma

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.loadx('trumpet')
    >>> CQT = np.abs(librosa.cqt(y, sr=sr))
    >>> chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> # Max-normalize each time step
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)
    """
    base = "C" if base_c else "A"
    return pitch_to_chroma(
        n_input,
        bins_per_octave=bins_per_octave,
        n_chroma=n_chroma,
        fmin=fmin,
        window=window,
        base=base,
        dtype=dtype,
    )


@cache(level=10)
def window_bandwidth(window: _WindowSpec, n: int = 1000) -> float:
    """Get the equivalent noise bandwidth (ENBW) of a window function.

    The ENBW of a window is defined by [#]_ (equation 11) as the normalized
    ratio of the sum of squares to the square of sums::

        enbw = n * sum(window**2) / sum(window)**2

    .. [#] Harris, F. J.
        "On the use of windows for harmonic analysis with the discrete Fourier transform."
        Proceedings of the IEEE, 66(1), 51-83.  1978.

    Parameters
    ----------
    window : callable or str
        A window function, or the name of a window function,
        e.g.: `scipy.signal.hann` or `'boxcar'`
    n : int > 0
        The number of coefficients to use in estimating the
        window bandwidth

    Returns
    -------
    bandwidth : float
        The equivalent noise bandwidth (in FFT bins) of the
        given window function

    See Also
    --------
    get_window

    Notes
    -----
    This function caches at level 10.
    """
    if hasattr(window, "__name__"):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = (
            n * np.sum(win**2) / (np.sum(win) ** 2 + util.tiny(win))
        )

    return WINDOW_BANDWIDTHS[key]


@cache(level=10)
def get_window(window: _WindowSpec, Nx: int, *, fftbins: bool = True) -> np.ndarray:
    """Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : str, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.

    See Also
    --------
    scipy.signal.get_window

    Notes
    -----
    This function caches at level 10.
    """
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        import scipy.signal
        win: np.ndarray = scipy.signal.get_window(window, Nx, fftbins=fftbins)  # type: ignore
        return win

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError(f"Window size mismatch: {len(window):d} != {Nx:d}")
    else:
        raise ParameterError(f"Invalid window specification: {window!r}")


@cache(level=10)
def _multirate_fb(
    center_freqs: np.ndarray | None = None,
    sample_rates: np.ndarray | None = None,
    Q: float = 25.0,
    passband_ripple: float = 1,
    stopband_attenuation: float = 50,
    ftype: Literal["butter", "cheby1", "cheby2", "ellip"] = "ellip",
    flayout: Literal["ba", "sos", "zpk"] = "sos",
) -> tuple[list[Any], np.ndarray]:
    r"""Construct a multirate filterbank.

     A filter bank consists of multiple band-pass filters which divide the input signal
     into subbands. In the case of a multirate filter bank, the band-pass filters
     operate with resampled versions of the input signal, e.g. to keep the length
     of a filter constant while shifting its center frequency.

     This implementation uses `scipy.signal.iirdesign` to design the filters.

    Parameters
    ----------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.

    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter (used for multirate filterbank).

    Q : float
        Q factor (influences the filter bandwidth).

    passband_ripple : float
        The maximum loss in the passband (dB)
        See `scipy.signal.iirdesign` for details.

    stopband_attenuation : float
        The minimum attenuation in the stopband (dB)
        See `scipy.signal.iirdesign` for details.

    ftype : str
        The type of IIR filter to design
        See `scipy.signal.iirdesign` for details.

    flayout : str
        Valid `output` argument for `scipy.signal.iirdesign`.

        - If `ba`, returns numerators/denominators of the transfer functions,
          used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.

        - If `sos`, returns a series of second-order filters,
          used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.

        - If `zpk`, returns zeros, poles, and system gains of the transfer functions.

    Returns
    -------
    filterbank : list [shape=(n,), dtype=float]
        Each list entry comprises the filter coefficients for a single filter.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Samplerate for each filter.

    Raises
    ------
    ParameterError
        If ``center_freqs`` is ``None``.
        If ``sample_rates`` is ``None``.
        If ``center_freqs.shape`` does not match ``sample_rates.shape``.

    See Also
    --------
    scipy.signal.iirdesign

    Notes
    -----
    This function caches at level 10.
    """
    if center_freqs is None:
        raise ParameterError("center_freqs must be provided.")

    if sample_rates is None:
        raise ParameterError("sample_rates must be provided.")

    if center_freqs.shape != sample_rates.shape:
        raise ParameterError(
            "Number of provided center_freqs and sample_rates must be equal."
        )

    nyquist = 0.5 * sample_rates
    filter_bandwidths = center_freqs / float(Q)

    filterbank = []

    for cur_center_freq, cur_nyquist, cur_bw in zip(
        center_freqs, nyquist, filter_bandwidths, strict=True
    ):
        passband_freqs = [
            cur_center_freq - 0.5 * cur_bw,
            cur_center_freq + 0.5 * cur_bw,
        ] / cur_nyquist
        stopband_freqs = [
            cur_center_freq - cur_bw,
            cur_center_freq + cur_bw,
        ] / cur_nyquist

        import scipy.signal
        cur_filter = scipy.signal.iirdesign(
            passband_freqs,
            stopband_freqs,
            passband_ripple,
            stopband_attenuation,
            analog=False,
            ftype=ftype,
            output=flayout,
        )

        filterbank.append(cur_filter)

    return filterbank, sample_rates


@cache(level=10)
def mr_frequencies(tuning: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Generate center frequencies and sample rate pairs.

    This function will return center frequency and corresponding sample rates
    to obtain similar pitch filterbank settings as described in [#]_.
    Instead of starting with MIDI pitch `A0`, we start with `C0`.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    tuning : float [scalar]
        Tuning deviation from A440, measure as a fraction of the equally
        tempered semitone (1/12 of an octave).

    Returns
    -------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter, used for multirate filterbank.

    See Also
    --------
    librosa.filters.semitone_filterbank

    Notes
    -----
    This function caches at level 10.
    """
    center_freqs = midi_to_hz(np.arange(24 + tuning, 109 + tuning))

    sample_rates = np.asarray(
        len(np.arange(0, 36))
        * [
            882.0,
        ]
        + len(np.arange(36, 70))
        * [
            4410.0,
        ]
        + len(np.arange(70, 85))
        * [
            22050.0,
        ]
    )

    return center_freqs, sample_rates


def semitone_filterbank(
    *,
    center_freqs: np.ndarray | None = None,
    tuning: float = 0.0,
    sample_rates: np.ndarray | None = None,
    flayout: Literal["ba", "sos"] = "ba",
    **kwargs: Any,
) -> tuple[list[Any], np.ndarray]:
    r"""Construct a multi-rate IIR band-pass filter bank at specified center frequencies and sample rates.

    By default, these center frequencies are set equal to the 88 fundamental
    frequencies of the grand piano keyboard, according to a pitch tuning standard
    of A440, that is, note A above middle C set to 440 Hz. The center frequencies
    are tuned to the twelve-tone equal temperament, which means that they grow
    exponentially at a rate of 2**(1/12), that is, twelve notes per octave.

    The A440 tuning can be changed by the user while keeping twelve-tone equal
    temperament. While A440 is currently the international standard in the music
    industry (ISO 16), some orchestras tune to A441-A445, whereas baroque musicians
    tune to A415.

    See [#]_ for details.

    .. [#] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.
    tuning : float [scalar]
        Tuning deviation from A440 as a fraction of a semitone (1/12 of an octave
        in equal temperament).
    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rates of each filter in the multirate filterbank.
    flayout : str
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
        - If `sos`, a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.
    **kwargs : additional keyword arguments
        Additional arguments to the private function `_multirate_fb()`.

    Returns
    -------
    filterbank : list [shape=(n,), dtype=float]
        Each list entry contains the filter coefficients for a single filter.
    fb_sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter.

    See Also
    --------
    librosa.cqt
    librosa.iirt
    librosa.filters.mr_frequencies
    scipy.signal.iirdesign

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import scipy.signal
    >>> semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank(
    ...     center_freqs=librosa.midi_to_hz(np.arange(60, 72)),
    ...     sample_rates=np.repeat(4410.0, 12),
    ...     flayout='sos'
    ...     )
    >>> magnitudes = []
    >>> for cur_sr, cur_filter in zip(sample_rates, semitone_filterbank):
    ...     w, h = scipy.signal.sosfreqz(cur_filter,fs=cur_sr, worN=1025)
    ...     magnitudes.append(20 * np.log10(np.abs(h)))
    >>> fig, ax = plt.subplots(figsize=(12,6))
    >>> img = librosa.display.specshow(
    ...     np.array(magnitudes),
    ...     x_axis="hz",
    ...     sr=4410,
    ...     y_coords=librosa.midi_to_hz(np.arange(60, 72)),
    ...     vmin=-60,
    ...     vmax=3,
    ...     ax=ax
    ...     )
    >>> librosa.display.colorbar_db(img)
    >>> ax.set(
    ...     xlim=[200, 600],
    ...     yticks=librosa.midi_to_hz(np.arange(60, 72)),
    ...     title='Magnitude Responses of the Pitch Filterbank',
    ...     xlabel='Frequency (Hz)',
    ...     ylabel='Semitone filter center frequency (Hz)'
    ... )
    """
    if (center_freqs is None) and (sample_rates is None):
        center_freqs, sample_rates = mr_frequencies(tuning)

    filterbank, fb_sample_rates = _multirate_fb(
        center_freqs=center_freqs, sample_rates=sample_rates, flayout=flayout, **kwargs
    )

    return filterbank, fb_sample_rates


@jit(nopython=True, cache=True)
def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Compute the sum-square envelope of a window."""
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


def window_sumsquare(
    *,
    window: _WindowSpec,
    n_frames: int,
    hop_length: int = 512,
    win_length: int | None = None,
    n_fft: int = 2048,
    dtype: DTypeLike = np.float32,
    norm: float | None = None,
) -> np.ndarray:
    """Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing observations
    in short-time Fourier transforms.

    Parameters
    ----------
    window : str, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches ``n_fft``.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    norm : {np.inf, -np.inf, 0, float > 0, None}
        Normalization mode used in window construction.
        Note that this does not affect the squaring operation.

    Returns
    -------
    wss : np.ndarray, shape=``(n_fft + hop_length * (n_frames - 1))``
        The sum-squared envelope of the window function

    Examples
    --------
    For a fixed frame length (2048), compare modulation effects for a Hann window
    at different hop lengths:

    >>> n_frames = 50
    >>> wss_256 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=256)
    >>> wss_512 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=512)
    >>> wss_1024 = librosa.filters.window_sumsquare(window='hann', n_frames=n_frames, hop_length=1024)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharey=True)
    >>> ax[0].plot(wss_256)
    >>> ax[0].set(title='hop_length=256')
    >>> ax[1].plot(wss_512)
    >>> ax[1].set(title='hop_length=512')
    >>> ax[2].plot(wss_1024)
    >>> ax[2].set(title='hop_length=1024')
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm) ** 2
    win_sq = util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x


@cache(level=10)
def diagonal_filter(
    window: _WindowSpec,
    n: int,
    *,
    slope: float = 1.0,
    angle: float | None = None,
    zero_mean: bool = False,
) -> np.ndarray:
    """Build a two-dimensional diagonal filter.

    This is primarily used for smoothing recurrence or self-similarity matrices.

    Parameters
    ----------
    window : str, tuple, number, callable, or list-like
        The window function to use for the filter.

        See `get_window` for details.

        Note that the window used here should be non-negative.

    n : int > 0
        the length of the filter

    slope : float
        The slope of the diagonal filter to produce

    angle : float or None
        If given, the slope parameter is ignored,
        and angle directly sets the orientation of the filter (in radians).
        Otherwise, angle is inferred as `arctan(slope)`.

    zero_mean : bool
        If True, a zero-mean filter is used.
        Otherwise, a non-negative averaging filter is used.

        This should be enabled if you want to enhance paths and suppress
        blocks.

    Returns
    -------
    kernel : np.ndarray, shape=[(m, m)]
        The 2-dimensional filter kernel

    Notes
    -----
    This function caches at level 10.
    """
    if angle is None:
        angle = np.arctan(slope)

    win: np.ndarray = np.diag(get_window(window, n, fftbins=False))

    if not np.isclose(angle, np.pi / 4):
        import scipy.ndimage
        win = scipy.ndimage.rotate(
            win, 45 - angle * 180 / np.pi, order=5, prefilter=False
        )

    np.clip(win, 0, None, out=win)
    win /= win.sum()

    if zero_mean:
        win -= win.mean()

    return win
