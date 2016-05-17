#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filters
=======

Filter bank construction
------------------------
.. autosummary::
    :toctree: generated/

    dct
    mel
    chroma
    constant_q

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/

    constant_q_lengths
    cq_to_chroma
    window_bandwidth

Deprecated
----------
.. autosummary::
    :toctree: generated/

    logfrequency
"""

import warnings

import numpy as np
import scipy
import scipy.signal

from . import cache
from . import util
from .util.exceptions import ParameterError

from .core.time_frequency import note_to_hz, hz_to_midi, hz_to_octs
from .core.time_frequency import fft_frequencies, mel_frequencies

# Dictionary of window function bandwidths
WINDOW_BANDWIDTHS = dict(hann=0.725)

__all__ = ['dct',
           'mel',
           'chroma',
           'constant_q',
           'constant_q_lengths',
           'cq_to_chroma',
           'window_bandwidth',
           # Deprecated
           'logfrequency']


@cache
def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    Examples
    --------
    >>> n_fft = 2048
    >>> dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
    >>> dct_filters
    array([[ 0.031,  0.031, ...,  0.031,  0.031],
           [ 0.044,  0.044, ..., -0.044, -0.044],
           ...,
           [ 0.044,  0.044, ..., -0.044, -0.044],
           [ 0.044,  0.044, ...,  0.044,  0.044]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(dct_filters, x_axis='linear')
    >>> plt.ylabel('DCT function')
    >>> plt.title('DCT filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis


@cache
def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    freqs = mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax,
                            htk=htk)

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = (fftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - fftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


@cache
def chroma(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0,
           octwidth=2, norm=2, base_c=True):
    """Create a Filterbank matrix to convert STFT to chroma


    Parameters
    ----------
    sr        : number > 0 [scalar]
        audio sampling rate

    n_fft     : int > 0 [scalar]
        number of FFT bins

    n_chroma  : int > 0 [scalar]
        number of chroma bins

    A440      : float > 0 [scalar]
        Reference frequency for A440

    ctroct    : float > 0 [scalar]

    octwidth  : float > 0 or None [scalar]
        `ctroct` and `octwidth` specify a dominance window -
        a Gaussian weighting centered on `ctroct` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of `octwidth`.
        Set `octwidth` to `None` to use a flat weighting.

    norm : float > 0 or np.inf
        Normalization factor for each filter

    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.
    Returns
    -------
    wts : ndarray [shape=(n_chroma, 1 + n_fft / 2)]
        Chroma filter matrix

    See Also
    --------
    util.normalize
    feature.chroma_stft

    Examples
    --------
    Build a simple chroma filter bank

    >>> chromafb = librosa.filters.chroma(22050, 4096)
    array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
           [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
    ...,
           [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
           [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

    Use quarter-tones instead of semitones

    >>> librosa.filters.chroma(22050, 4096, n_chroma=24)
    array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
           [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
    ...,
           [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
           [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])


    Equally weight all octaves

    >>> librosa.filters.chroma(22050, 4096, octwidth=None)
    array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
           [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
    ...,
           [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
           [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(chromafb, x_axis='linear')
    >>> plt.ylabel('Chroma filter')
    >>> plt.title('Chroma filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1],
                                              1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype='d')).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10*n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (n_chroma, 1)))**2)

    # normalize each column
    wts = util.normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins/n_chroma - ctroct)/octwidth)**2)),
            (n_chroma, 1))

    if base_c:
        wts = np.roll(wts, -3, axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, :int(1 + n_fft/2)])


@util.decorators.deprecated('0.4', '0.5')
def logfrequency(sr, n_fft, n_bins=84, bins_per_octave=12, tuning=0.0,
                 fmin=None, spread=0.125):  # pragma: no cover
    '''Approximate a constant-Q filter bank for a fixed-window STFT.

    Each filter is a log-normal window centered at the corresponding frequency.

    .. warning:: Deprecated in librosa 0.4

    Parameters
    ----------
    sr : number > 0 [scalar]
        audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size

    n_bins : int > 0 [scalar]
        Number of bins.  Defaults to 84 (7 octaves).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave. Defaults to 12 (semitones).

    tuning : None or float in `[-0.5, +0.5]` [scalar]
        Tuning correction parameter, in fractions of a bin.

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    spread : float > 0 [scalar]
        Spread of each filter, as a fraction of a bin.

    Returns
    -------
    C : np.ndarray [shape=(n_bins, 1 + n_fft/2)]
        log-frequency filter bank.

    Examples
    --------
    Simple log frequency filters

    >>> logfb = librosa.filters.logfrequency(22050, 4096)
    >>> logfb
    array([[ 0.,  0., ...,  0.,  0.],
           [ 0.,  0., ...,  0.,  0.],
    ...,
           [ 0.,  0., ...,  0.,  0.],
           [ 0.,  0., ...,  0.,  0.]])


    Use a narrower frequency range

    >>> librosa.filters.logfrequency(22050, 4096, n_bins=48, fmin=110)
    array([[ 0.,  0., ...,  0.,  0.],
           [ 0.,  0., ...,  0.,  0.],
    ...,
           [ 0.,  0., ...,  0.,  0.],
           [ 0.,  0., ...,  0.,  0.]])


    Use narrower filters for sparser response: 5% of a semitone

    >>> librosa.filters.logfrequency(22050, 4096, spread=0.05)

    Or wider: 50% of a semitone

    >>> librosa.filters.logfrequency(22050, 4096, spread=0.5)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(logfb, x_axis='linear')
    >>> plt.ylabel('Logarithmic filters')
    >>> plt.title('Log-frequency filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    '''

    if fmin is None:
        fmin = note_to_hz('C1')

    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)

    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave

    # Construct the output matrix
    basis = np.zeros((n_bins, int(1 + n_fft/2)))

    # Get log frequencies of bins
    log_freqs = np.log2(fft_frequencies(sr, n_fft)[1:])

    for i in range(n_bins):
        # What's the center (median) frequency of this filter?
        c_freq = correction * fmin * (2.0**(float(i) / bins_per_octave))

        # Place a log-normal window around c_freq
        basis[i, 1:] = np.exp(-0.5 * ((log_freqs - np.log2(c_freq)) / sigma)**2
                              - np.log2(sigma) - log_freqs)

    # Normalize the filters
    basis = util.normalize(basis, norm=1, axis=1)

    return basis


def __float_window(window_function):
    '''Decorator function for windows with fractional input.

    This function guarantees that for fractional `x`, the following hold:

    1. `__float_window(window_function)(x)` has length `np.ceil(x)`
    2. all values from `np.floor(x)` are set to 0.

    For integer-valued `x`, there should be no change in behavior.
    '''

    def _wrap(n, *args, **kwargs):
        '''The wrapped window'''
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = window_function(n, *args, **kwargs)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))],
                            mode='constant')

        window[n_min:] = 0.0

        return window

    return _wrap


@cache
def constant_q(sr, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0,
               window=None, filter_scale=1, pad_fft=True, norm=1,
               resolution=util.Deprecated(), **kwargs):
    r'''Construct a constant-Q basis.

    This uses the filter bank described by [1]_.

    .. [1] McVicar, Matthew.
            "A machine learning approach to automatic chord extraction."
            Dissertation, University of Bristol. 2013.


    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : function or `None`
        Windowing function to apply to filters.
        Default: `scipy.signal.hann`

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the `mode=` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    kwargs : additional keyword arguments
        Arguments to `np.pad()` when `pad==True`.

    resolution : float
        .. warning:: This parameter name was in librosa 0.4.2
            Use the `filter_scale` parameter instead.
            The `resolution` parameter will be removed in librosa 0.5.0.

    Returns
    -------
    filters : np.ndarray, `len(filters) == n_bins`
        `filters[i]` is `i`\ th time-domain CQT basis filter

    lengths : np.ndarray, `len(lengths) == n_bins`
        The (fractional) length of each filter

    See Also
    --------
    constant_q_lengths
    librosa.core.cqt
    librosa.util.normalize


    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> basis, lengths = librosa.filters.constant_q(22050)
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2, 1, 1)
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes)[:12]):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     plt.plot(i + f_scale.real)
    ...     plt.plot(i + f_scale.imag, linestyle=':')
    >>> plt.axis('tight')
    >>> plt.yticks(range(len(notes[:12])), notes[:12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filters (one octave, time domain)')
    >>> plt.xlabel('Time (samples at 22050 Hz)')
    >>> plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)
    >>> plt.subplot(2, 1, 2)
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear')
    >>> plt.yticks(range(len(notes))[::12], notes[::12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filter magnitudes (frequency domain)')
    >>> plt.tight_layout()
    '''

    if fmin is None:
        fmin = note_to_hz('C1')

    if window is None:
        window = scipy.signal.hann

    filter_scale = util.rename_kw('resolution', resolution,
                                  'filter_scale', filter_scale,
                                  '0.4.2', '0.5.0')

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(sr, fmin,
                                 n_bins=n_bins,
                                 bins_per_octave=bins_per_octave,
                                 tuning=tuning,
                                 window=window,
                                 filter_scale=filter_scale)

    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Convert lengths back to frequencies
    freqs = Q * sr / lengths

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(np.arange(ilen, dtype=float) * 1j * 2 * np.pi * freq / sr)

        # Apply the windowing function
        sig = sig * __float_window(window)(ilen)

        # Normalize
        sig = util.normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0**(np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray([util.pad_center(filt, max_len, **kwargs)
                          for filt in filters])

    return filters, np.asarray(lengths)


@cache
def constant_q_lengths(sr, fmin, n_bins=84, bins_per_octave=12,
                       tuning=0.0, window='hann', filter_scale=1,
                       resolution=util.Deprecated()):
    r'''Return length of each filter in a constant-Q basis.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin.

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    resolution : float
        .. warning:: This parameter name was in librosa 0.4.2
            Use the `filter_scale` parameter instead.
            The `resolution` parameter will be removed in librosa 0.5.0.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    See Also
    --------
    constant_q
    librosa.core.cqt
    '''

    filter_scale = util.rename_kw('resolution', resolution,
                                  'filter_scale', filter_scale,
                                  '0.4.2', '0.5.0')

    if fmin <= 0:
        raise ParameterError('fmin must be positive')

    if bins_per_octave <= 0:
        raise ParameterError('bins_per_octave must be positive')

    if filter_scale <= 0:
        raise ParameterError('filter_scale must be positive')

    if n_bins <= 0 or not isinstance(n_bins, int):
        raise ParameterError('n_bins must be a positive integer')

    correction = 2.0**(float(tuning) / bins_per_octave)

    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    if np.any(freq * (1 + window_bandwidth(window) / Q) > sr / 2.0):
        raise ParameterError('Filter pass-band lies beyond Nyquist')

    # Convert frequencies to filter lengths
    lengths = Q * sr / freq

    return lengths


@cache
def cq_to_chroma(n_input, bins_per_octave=12, n_chroma=12,
                 fmin=None, window=None, base_c=True):
    '''Convert a Constant-Q basis to Chroma.


    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (CQT bins)

    bins_per_octave : int > 0 [scalar]
        How many bins per octave in the CQT

    n_chroma : int > 0 [scalar]
        Number of output bins (per octave) in the chroma

    fmin : None or float > 0
        Center frequency of the first constant-Q channel.
        Default: 'C1' ~= 32.7 Hz

    window : None or np.ndarray
        If provided, the cq_to_chroma filter bank will be
        convolved with `window`.

    base_c : bool
        If True, the first chroma bin will start at 'C'
        If False, the first chroma bin will start at 'A'

    Returns
    -------
    cq_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: `Chroma = np.dot(cq_to_chroma, CQT)`

    Raises
    ------
    ParameterError
        If `n_input` is not an integer multiple of `n_chroma`

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = librosa.cqt(y, sr=sr)
    >>> chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> # Max-normalize each time step
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(CQT**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='cqt_note', x_axis='time')
    >>> plt.title('CQT Power')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    >>> plt.title('Chroma (wrapped CQT)')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 3)
    >>> chroma = librosa.feature.chromagram(y=y, sr=sr)
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    >>> plt.title('librosa.feature.chroma')
    >>> plt.colorbar()
    >>> plt.tight_layout()

    '''

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz('C1')

    if np.mod(n_merge, 1) != 0:
        raise ParameterError('Incompatible CQ merge: '
                             'input bins must be an '
                             'integer multiple of output bins.')

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, - int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(np.float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin in the CQT?
    # midi uses 12 bins per octave here
    midi_0 = np.mod(hz_to_midi(fmin), 12)

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(np.round(roll * (n_chroma / 12.)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(float)

    if window is not None:
        cq_to_ch = scipy.signal.convolve(cq_to_ch,
                                         np.atleast_2d(window),
                                         mode='same')

    return cq_to_ch


def window_bandwidth(window, default=1.0):
    '''Get the bandwidth of a window function.

    If the window function is unknown, return a default value.

    Parameters
    ----------
    window : callable or string
        A window function, or the name of a window function.
        Examples:
        - scipy.signal.hann
        - 'boxcar'

    default : float >= 0
        The default value, if `window` is unknown.

    Returns
    -------
    bandwidth : float
        The bandwidth of the given window function

    See Also
    --------
    scipy.signal.get_window
    '''

    if hasattr(window, '__name__'):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        warnings.warn("Unknown window function '{:s}'.".format(key))

    return WINDOW_BANDWIDTHS.get(key, default)
