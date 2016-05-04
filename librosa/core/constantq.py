#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pitch-tracking and tuning estimation'''
from __future__ import division

from warnings import warn

import numpy as np
import scipy.fftpack as fft

from . import audio
from .time_frequency import cqt_frequencies, note_to_hz
from .spectrum import stft
from .pitch import estimate_tuning
from .. import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError

__all__ = ['cqt', 'hybrid_cqt', 'pseudo_cqt']


@cache
def cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, tuning=None, filter_scale=1,
        aggregate=util.Deprecated(),
        norm=1, sparsity=0.01, real=None,
        resolution=util.Deprecated()):
    '''Compute the constant-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [1]_.

    .. [1] Schoerkhuber, Christian, and Anssi Klapuri.
        "Constant-Q transform toolbox for music processing."
        7th Sound and Music Computing Conference, Barcelona, Spain. 2010.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C1 ~= 32.70 Hz

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at `fmin`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float in `[-0.5, 0.5)`
        Tuning offset in fractions of a bin (cents).

        If `None`, tuning will be automatically estimated.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    aggregate : [DEPRECATED]
        .. warning:: This parameter was is deprecated in librosa 0.4.3.
            It will be removed in librosa 0.5.0.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    real : bool
        If true, return only the magnitude of the CQT.

    resolution : float
        .. warning:: This parameter name was deprecated in librosa 0.4.2
            Use the `filter_scale` parameter instead.
            The `resolution` parameter will be removed in librosa 0.5.0.


    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.complex or np.float]
        Constant-Q value each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length` is not an integer multiple of
        `2**(n_bins / bins_per_octave)`

        Or if `y` is too short to support the frequency range of the CQT.

    See Also
    --------
    librosa.core.resample
    librosa.util.normalize

    Examples
    --------
    Generate and plot a constant-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = librosa.cqt(y, sr=sr)
    >>> librosa.display.specshow(librosa.logamplitude(C**2, ref_power=np.max),
    ...                          sr=sr, x_axis='time', y_axis='cqt_note')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Constant-Q power spectrum')
    >>> plt.tight_layout()


    Limit the frequency range

    >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60)
    >>> C
    array([[  8.827e-04,   9.293e-04, ...,   3.133e-07,   2.942e-07],
           [  1.076e-03,   1.068e-03, ...,   1.153e-06,   1.148e-06],
           ...,
           [  1.042e-07,   4.087e-07, ...,   1.612e-07,   1.928e-07],
           [  2.363e-07,   5.329e-07, ...,   1.294e-07,   1.611e-07]])


    Using a higher frequency resolution

    >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60 * 2, bins_per_octave=12 * 2)
    >>> C
    array([[  1.536e-05,   5.848e-05, ...,   3.241e-07,   2.453e-07],
           [  1.856e-03,   1.854e-03, ...,   2.397e-08,   3.549e-08],
           ...,
           [  2.034e-07,   4.245e-07, ...,   6.213e-08,   1.463e-07],
           [  4.896e-08,   5.407e-07, ...,   9.176e-08,   1.051e-07]])
    '''

    filter_scale = util.rename_kw('resolution', resolution,
                                  'filter_scale', filter_scale,
                                  '0.4.2', '0.5.0')

    if real is None:
        warn('Real-valued CQT (real=True) is deprecated in 0.4.2. '
             'Complex-valued CQT will become the default in 0.5.0. '
             'Consider using np.abs(librosa.cqt(..., real=False)) '
             'instead of real=True to maintain forward compatibility.',
             DeprecationWarning)
        real = True

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    len_orig = len(y)

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t * (1 + filters.window_bandwidth('hann') / Q)
    nyquist = sr / 2.0
    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'kaiser_fast'
    else:
        res_type = 'kaiser_best'

    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff)

    cqt_resp = []

    if res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               tuning,
                                               filter_scale,
                                               norm,
                                               sparsity)

        # Compute the CQT filter response and append it to the stack
        cqt_resp.append(__cqt_response(y, n_fft, hop_length, fft_basis))

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + filters.window_bandwidth('hann') / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise ParameterError('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))

    # Now do the recursive bit
    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           tuning,
                                           filter_scale,
                                           norm,
                                           sparsity)

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            if len(my_y) < 2:
                raise ParameterError('Input signal length={} is too short for '
                                     '{:d}-octave CQT'.format(len_orig, n_octaves))

            # The additional scaling of sqrt(2) here is to implicitly rescale the filters
            my_y = np.sqrt(2) * audio.resample(my_y, my_sr, my_sr/2.0,
                                               res_type=res_type,
                                               scale=True)
            my_sr /= 2.0
            my_hop //= 2

        # Compute the cqt filter response and append to the stack
        cqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis))


    return __trim_stack(cqt_resp, n_bins, real)


@cache
def hybrid_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=None, filter_scale=1,
               norm=1, sparsity=0.01,
               resolution=util.Deprecated()):
    '''Compute the hybrid constant-Q transform of an audio signal.

    Here, the hybrid CQT uses the pseudo CQT for higher frequencies where
    the hop_length is longer than half the filter length and the full CQT
    for lower frequencies.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C1 ~= 32.70 Hz

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at `fmin`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float in `[-0.5, 0.5)`
        Tuning offset in fractions of a bin (cents).

        If `None`, tuning will be automatically estimated.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    resolution : float
        .. warning:: This parameter name was deprecated in librosa 0.4.2
            Use the `filter_scale` parameter instead.
            The `resolution` parameter will be removed in librosa 0.5.0.


    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
        Constant-Q energy for each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length` is not an integer multiple of
        `2**(n_bins / bins_per_octave)`

        Or if `y` is too short to support the frequency range of the CQT.

    See Also
    --------
    cqt
    pseudo_cqt
    '''

    filter_scale = util.rename_kw('resolution', resolution,
                                  'filter_scale', filter_scale,
                                  '0.4.2', '0.5.0')

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    # Get all CQT frequencies
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave,
                            tuning=tuning)

    # Compute the length of each constant-Q basis function
    lengths = filters.constant_q_lengths(sr, fmin,
                                         n_bins=n_bins,
                                         bins_per_octave=bins_per_octave,
                                         tuning=tuning,
                                         filter_scale=filter_scale)

    # Determine which filters to use with Pseudo CQT
    # These are the ones that fit within 2 hop lengths after padding
    pseudo_filters = 2.0**np.ceil(np.log2(lengths)) < 2 * hop_length

    n_bins_pseudo = int(np.sum(pseudo_filters))

    n_bins_full = n_bins - n_bins_pseudo
    cqt_resp = []

    if n_bins_pseudo > 0:
        fmin_pseudo = np.min(freqs[pseudo_filters])
        cqt_resp.append(pseudo_cqt(y, sr,
                                   hop_length=hop_length,
                                   fmin=fmin_pseudo,
                                   n_bins=n_bins_pseudo,
                                   bins_per_octave=bins_per_octave,
                                   tuning=tuning,
                                   filter_scale=filter_scale,
                                   norm=norm,
                                   sparsity=sparsity))

    if n_bins_full > 0:
        cqt_resp.append(np.abs(cqt(y, sr,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   n_bins=n_bins_full,
                                   bins_per_octave=bins_per_octave,
                                   tuning=tuning,
                                   filter_scale=filter_scale,
                                   norm=norm,
                                   sparsity=sparsity,
                                   real=False)))

    return __trim_stack(cqt_resp, n_bins, True)


@cache
def pseudo_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=None, filter_scale=1,
               norm=1, sparsity=0.01,
               resolution=util.Deprecated()):
    '''Compute the pseudo constant-Q transform of an audio signal.

    This uses a single fft size that is the smallest power of 2 that is greater
    than or equal to the max of:

        1. The longest CQT filter
        2. 2x the hop_length

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C1 ~= 32.70 Hz

    n_bins : int > 0 [scalar]
        Number of frequency bins, starting at `fmin`

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : None or float in `[-0.5, 0.5)`
        Tuning offset in fractions of a bin (cents).

        If `None`, tuning will be automatically estimated.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    resolution : float
        .. warning:: This parameter name was deprecated in librosa 0.4.2
            Use the `filter_scale` parameter instead.
            The `resolution` parameter will be removed in librosa 0.5.0.


    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
        Pseudo Constant-Q energy for each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length` is not an integer multiple of
        `2**(n_bins / bins_per_octave)`

        Or if `y` is too short to support the frequency range of the CQT.
    '''

    filter_scale = util.rename_kw('resolution', resolution,
                                  'filter_scale', filter_scale,
                                  '0.4.2', '0.5.0')

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin, n_bins,
                                           bins_per_octave,
                                           tuning, filter_scale,
                                           norm, sparsity,
                                           hop_length=hop_length)

    fft_basis = np.abs(fft_basis)

    # Compute the magnitude STFT with Hann window
    D = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))

    # Project onto the pseudo-cqt basis
    return fft_basis.dot(D)


def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, tuning,
                     filter_scale, norm, sparsity, hop_length=None):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        filter_scale=filter_scale,
                                        norm=norm,
                                        pad_fft=True)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0**(1 + np.ceil(np.log2(hop_length))):
        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins, real):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    max_col = min([x.shape[1] for x in cqt_resp])

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity

    C = np.ascontiguousarray(cqt_resp[-n_bins:].T).T
    if real:
        C = np.abs(C)
    return C


def __cqt_response(y, n_fft, hop_length, fft_basis):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=hop_length, window=np.ones)

    # And filter response energy
    return fft_basis.dot(D)


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = int(np.ceil(np.log2(audio.BW_FASTEST * nyquist /
                                            filter_cutoff)) - 1)

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2.0**(downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        # The additional scaling of sqrt(downsample_factor) here is to implicitly
        # rescale the filters
        y = np.sqrt(downsample_factor) * audio.resample(y, sr, sr / downsample_factor,
                                                        res_type=res_type, scale=True)

        sr /= downsample_factor

    return y, sr, hop_length


def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos
