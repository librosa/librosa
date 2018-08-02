#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Constant-Q transforms'''
from __future__ import division

import warnings
import numpy as np
import scipy.fftpack as fft
from numba import jit

from . import audio
from .time_frequency import cqt_frequencies, note_to_hz
from .spectrum import stft
from .pitch import estimate_tuning
from .. import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError

__all__ = ['cqt', 'hybrid_cqt', 'pseudo_cqt', 'icqt']


@cache(level=20)
def cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, tuning=0.0, filter_scale=1,
        norm=1, sparsity=0.01, window='hann',
        scale=True,
        pad_mode='reflect'):
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

        If `None`, tuning will be automatically estimated from the signal.

    filter_scale : float > 0
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If `True`, scale the CQT response by square-root the length of
        each channel's filter.  This is analogous to `norm='ortho'` in FFT.

        If `False`, do not scale the CQT. This is analogous to
        `norm=None` in FFT.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.core.stft` and `np.pad`.

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

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Generate and plot a constant-Q power spectrum

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = np.abs(librosa.cqt(y, sr=sr))
    >>> librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                          sr=sr, x_axis='time', y_axis='cqt_note')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Constant-Q power spectrum')
    >>> plt.tight_layout()


    Limit the frequency range

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60))
    >>> C
    array([[  8.827e-04,   9.293e-04, ...,   3.133e-07,   2.942e-07],
           [  1.076e-03,   1.068e-03, ...,   1.153e-06,   1.148e-06],
           ...,
           [  1.042e-07,   4.087e-07, ...,   1.612e-07,   1.928e-07],
           [  2.363e-07,   5.329e-07, ...,   1.294e-07,   1.611e-07]])


    Using a higher frequency resolution

    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60 * 2, bins_per_octave=12 * 2))
    >>> C
    array([[  1.536e-05,   5.848e-05, ...,   3.241e-07,   2.453e-07],
           [  1.856e-03,   1.854e-03, ...,   2.397e-08,   3.549e-08],
           ...,
           [  2.034e-07,   4.245e-07, ...,   6.213e-08,   1.463e-07],
           [  4.896e-08,   5.407e-07, ...,   9.176e-08,   1.051e-07]])
    '''

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
    filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)
    nyquist = sr / 2.0
    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'kaiser_fast'
    else:
        res_type = 'kaiser_best'

    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale)

    cqt_resp = []

    if res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               tuning,
                                               filter_scale,
                                               norm,
                                               sparsity,
                                               window=window)

        # Compute the CQT filter response and append it to the stack
        cqt_resp.append(__cqt_response(y, n_fft, hop_length, fft_basis, pad_mode))

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)

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
                                           sparsity,
                                           window=window)

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            if len(my_y) < 2:
                raise ParameterError('Input signal length={} is too short for '
                                     '{:d}-octave CQT'.format(len_orig,
                                                              n_octaves))

            my_y = audio.resample(my_y, my_sr, my_sr/2.0,
                                  res_type=res_type,
                                  scale=True)
            # The re-scale the filters to compensate for downsampling
            fft_basis[:] *= np.sqrt(2)

            my_sr /= 2.0
            my_hop //= 2

        # Compute the cqt filter response and append to the stack
        cqt_resp.append(__cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode))

    C = __trim_stack(cqt_resp, n_bins)

    if scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)
        C /= np.sqrt(lengths[:, np.newaxis])

    return C


@cache(level=20)
def hybrid_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):
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

        If `None`, tuning will be automatically estimated from the signal.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.core.stft` and `np.pad`.

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

    Notes
    -----
    This function caches at level 20.

    '''

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
                                         filter_scale=filter_scale,
                                         window=window)

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
                                   sparsity=sparsity,
                                   window=window,
                                   scale=scale,
                                   pad_mode=pad_mode))

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
                                   window=window,
                                   scale=scale,
                                   pad_mode=pad_mode)))

    return __trim_stack(cqt_resp, n_bins)


@cache(level=20)
def pseudo_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):
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

        If `None`, tuning will be automatically estimated from the signal.

    filter_scale : float > 0
        Filter filter_scale factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    pad_mode : string
        Padding mode for centered frame analysis.

        See also: `librosa.core.stft` and `np.pad`.

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

    Notes
    -----
    This function caches at level 20.

    '''

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin, n_bins,
                                           bins_per_octave,
                                           tuning, filter_scale,
                                           norm, sparsity,
                                           hop_length=hop_length,
                                           window=window)

    fft_basis = np.abs(fft_basis)

    # Compute the magnitude STFT with Hann window
    D = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length, pad_mode=pad_mode))

    # Project onto the pseudo-cqt basis
    C = fft_basis.dot(D)

    if scale:
        C /= np.sqrt(n_fft)
    else:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)

        C *= np.sqrt(lengths[:, np.newaxis] / n_fft)

    return C


@cache(level=40)
def icqt(C, sr=22050, hop_length=512, fmin=None,
         bins_per_octave=12,
         tuning=0.0,
         filter_scale=1,
         norm=1,
         sparsity=0.01,
         window='hann',
         scale=True,
         amin=1e-6):
    '''Compute the inverse constant-Q transform.

    Given a constant-Q transform representation `C` of an audio signal `y`,
    this function produces an approximation `y_hat`.

    .. warning:: This implementation is unstable, and subject to change in
                 future versions of librosa.  We recommend that its use be
                 limited to sonification and diagnostic applications.


    Parameters
    ----------
    C : np.ndarray, [shape=(n_bins, n_frames)]
        Constant-Q representation as produced by `core.cqt`

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C1 ~= 32.70 Hz

    tuning : float in `[-0.5, 0.5)` [scalar]
        Tuning offset in fractions of a bin (cents).

    filter_scale : float > 0 [scalar]
        Filter scale factor. Small values (<1) use shorter windows
        for improved time resolution.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    window : str, tuple, number, or function
        Window specification for the basis filters.
        See `filters.get_window` for details.

    scale : bool
        If `True`, scale the CQT response by square-root the length
        of each channel's filter. This is analogous to `norm='ortho'` in FFT.

        If `False`, do not scale the CQT. This is analogous to `norm=None`
        in FFT.

    amin : float or None
        When applying squared window normalization, sample positions with
        coefficients below `amin` will left as is.

        If `None`, then `amin` is inferred as the smallest valid floating
        point value.

    Returns
    -------
    y : np.ndarray, [shape=(n_samples), dtype=np.float]
        Audio time-series reconstructed from the CQT representation.

    See Also
    --------
    cqt

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Using default parameters

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> C = librosa.cqt(y=y, sr=sr)
    >>> y_hat = librosa.icqt(C=C, sr=sr)

    Or with a different hop length and frequency resolution:

    >>> hop_length = 256
    >>> bins_per_octave = 12 * 3
    >>> C = librosa.cqt(y=y, sr=sr, hop_length=256, n_bins=7*bins_per_octave,
    ...                 bins_per_octave=bins_per_octave)
    >>> y_hat = librosa.icqt(C=C, sr=sr, hop_length=hop_length,
    ...                 bins_per_octave=bins_per_octave)
    '''
    warnings.warn('librosa.icqt is unstable, and subject to change in future versions. '
                  'Please use with caution.')

    n_bins, n_frames = C.shape
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    if amin is None:
        amin = util.tiny(C)

    if fmin is None:
        fmin = note_to_hz('C1')

    freqs = cqt_frequencies(n_bins,
                            fmin,
                            bins_per_octave=bins_per_octave,
                            tuning=tuning)[-bins_per_octave:]

    fmin_t = np.min(freqs)

    # Make the filter bank
    basis, lengths = filters.constant_q(sr=sr,
                                        fmin=fmin_t,
                                        n_bins=bins_per_octave,
                                        bins_per_octave=bins_per_octave,
                                        filter_scale=filter_scale,
                                        tuning=tuning,
                                        norm=norm,
                                        window=window,
                                        pad_fft=True)
    n_fft = basis.shape[1]

    # The extra factor of lengths**0.5 corrects for within-octave tapering
    basis = basis * np.sqrt(lengths[:, np.newaxis])

    # Estimate the gain per filter
    bdot = basis.conj().dot(basis.T)
    bscale = np.sum(np.abs(bdot), axis=1)

    n_trim = basis.shape[1] // 2

    if scale:
        Cnorm = np.ones(n_bins)[:, np.newaxis]
    else:
        Cnorm = filters.constant_q_lengths(sr=sr,
                                           fmin=fmin,
                                           n_bins=n_bins,
                                           bins_per_octave=bins_per_octave,
                                           filter_scale=filter_scale,
                                           tuning=tuning,
                                           window=window)[:, np.newaxis]**0.5

    y = None

    # Revised algorithm:
    #   for each octave
    #      upsample old octave
    #      @--numba accelerate this loop?
    #      for each basis
    #         convolve with activation (valid-mode)
    #         divide by window sumsquare
    #         trim and add to total

    for octave in range(n_octaves - 1, -1, -1):
        # Compute the slice index for the current octave
        slice_ = slice(-(octave+1) * bins_per_octave - 1,
                       -(octave) * bins_per_octave - 1)

        # Project onto the basis
        C_oct = C[slice_] / Cnorm[slice_]
        basis_oct = basis[-C_oct.shape[0]:]

        y_oct = None

        # Make a dummy activation
        oct_hop = hop_length // 2**octave
        n = n_fft + (C_oct.shape[1] - 1) * oct_hop

        for i in range(basis_oct.shape[0]-1, -1, -1):
            wss = filters.window_sumsquare(window,
                                           n_frames,
                                           hop_length=oct_hop,
                                           win_length=int(lengths[i]),
                                           n_fft=n_fft,
                                           norm=norm)

            wss *= lengths[i]**2

            # Construct the response for this filter
            y_oct_i = np.zeros(n, dtype=C_oct.dtype)
            __activation_fill(y_oct_i, basis_oct[i], C_oct[i], oct_hop)
            # Retain only the real part
            # Only do window normalization for sufficiently large window
            # coefficients
            y_oct_i = y_oct_i.real / np.maximum(amin, wss)

            if y_oct is None:
                y_oct = y_oct_i
            else:
                y_oct += y_oct_i

        # Remove the effects of zero-padding
        y_oct = y_oct[n_trim:-n_trim] * bscale[i]

        if y is None:
            y = y_oct
        else:
            # Up-sample the previous buffer and add in the new one
            # Scipy-resampling is fast here, since it's a power-of-two relation
            y = audio.resample(y, 1, 2, scale=True, res_type='scipy') + y_oct

    return y


@cache(level=10)
def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, tuning,
                     filter_scale, norm, sparsity, hop_length=None,
                     window='hann'):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        filter_scale=filter_scale,
                                        norm=norm,
                                        pad_fft=True,
                                        window=window)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if (hop_length is not None and
            n_fft < 2.0**(1 + np.ceil(np.log2(hop_length)))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    max_col = min(x.shape[1] for x in cqt_resp)

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity
    return np.ascontiguousarray(cqt_resp[-n_bins:].T).T


def __cqt_response(y, n_fft, hop_length, fft_basis, mode):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=hop_length,
             window='ones',
             pad_mode=mode)

    # And filter response energy
    return fft_basis.dot(D)


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = max(0, int(np.ceil(np.log2(audio.BW_FASTEST * nyquist /
                                                   filter_cutoff)) - 1) - 1)

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff, scale):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        new_sr = sr / float(downsample_factor)
        y = audio.resample(y, sr, new_sr,
                           res_type=res_type,
                           scale=True)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

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


@jit(nopython=True)
def __activation_fill(x, basis, activation, hop_length):  # pragma: no cover
    '''Helper function for icqt time-domain reconstruction'''

    n = len(x)
    n_fft = len(basis)
    n_frames = len(activation)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += activation[i] * basis[:max(0, min(n_fft, n - sample))]
