#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pitch-tracking and tuning estimation'''
from __future__ import division

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
        bins_per_octave=12, tuning=None, resolution=2,
        aggregate=None, norm=1, sparsity=0.01):
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

    sr : int > 0 [scalar]
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

    resolution : float > 0
        Filter resolution factor. Larger values use longer windows.

    aggregate : None or function
        Aggregation function for time-oversampling energy aggregation.
        By default, `np.mean`.  See `librosa.feature.sync`.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See `librosa.util.normalize`.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
        Constant-Q energy for each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length < 2**(n_bins / bins_per_octave)`

    See Also
    --------
    librosa.core.resample
    librosa.feature.sync
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


    Using a higher resolution

    >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
    ...                 n_bins=60 * 2, bins_per_octave=12 * 2)
    >>> C
    array([[  1.536e-05,   5.848e-05, ...,   3.241e-07,   2.453e-07],
           [  1.856e-03,   1.854e-03, ...,   2.397e-08,   3.549e-08],
           ..., 
           [  2.034e-07,   4.245e-07, ...,   6.213e-08,   1.463e-07],
           [  4.896e-08,   5.407e-07, ...,   9.176e-08,   1.051e-07]])
    '''

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    # Make sure our hop is long enough to support the bottom octave
    if hop_length < 2**n_octaves:
        raise ParameterError('Insufficient hop_length {:d} '
                             'for {:d} octaves'.format(hop_length, n_octaves))

    if fmin is None:
        # C2 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(resolution) / (2.0**(1. / bins_per_octave) - 1)

    filter_cutoff = fmax_t * (1 + filters.window_bandwidth('hann') / Q)

    nyquist = sr / 2.0

    if filter_cutoff < audio.BW_FASTEST * nyquist:
        res_type = 'sinc_fastest'
    elif filter_cutoff < audio.BW_MEDIUM * nyquist:
        res_type = 'sinc_medium'
    elif filter_cutoff < audio.BW_BEST * nyquist:
        res_type = 'sinc_best'
    else:
        res_type = 'sinc_best'

    cqt_resp = []

    y, sr, hop_length = __early_downsample(y, sr, hop_length,
                                           res_type, n_octaves,
                                           nyquist, filter_cutoff)

    n_filters = min(bins_per_octave, n_bins)

    if res_type != 'sinc_fastest' and audio._HAS_SAMPLERATE:

        # Do two octaves before resampling to allow for usage of sinc_fastest
        fft_basis, n_fft, filter_lengths = __fft_filters(sr, fmin_t,
                                                         n_filters,
                                                         bins_per_octave,
                                                         tuning,
                                                         resolution,
                                                         norm,
                                                         sparsity)
        min_filter_length = np.min(filter_lengths)

        # Compute a dynamic hop based on n_fft
        my_cqt = __variable_hop_response(y, n_fft,
                                         hop_length,
                                         min_filter_length,
                                         fft_basis,
                                         aggregate)

        # Convolve
        cqt_resp.append(my_cqt)

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + filters.window_bandwidth('hann') / Q)
        assert filter_cutoff < audio.BW_FASTEST*nyquist

        res_type = 'sinc_fastest'

    # Now do the recursive bit
    fft_basis, n_fft, filter_lengths = __fft_filters(sr, fmin_t,
                                                     n_filters,
                                                     bins_per_octave,
                                                     tuning,
                                                     resolution,
                                                     norm,
                                                     sparsity)

    min_filter_length = np.min(filter_lengths)

    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            my_y = audio.resample(my_y, my_sr, my_sr/2.0, res_type=res_type)
            my_sr = my_sr / 2.0
            my_hop = int(my_hop / 2.0)

        # Compute a dynamic hop based on n_fft
        my_cqt = __variable_hop_response(my_y, n_fft,
                                         my_hop,
                                         min_filter_length,
                                         fft_basis,
                                         aggregate)

        # Convolve
        cqt_resp.append(my_cqt)

    return __trim_stack(cqt_resp, n_bins)


@cache
def hybrid_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=None, resolution=2,
               norm=1, sparsity=0.01):
    '''Compute the hybrid constant-Q transform of an audio signal.

    Here, the hybrid CQT uses the pseudo CQT for higher frequencies where
    the hop_length is longer than half the filter length and the full CQT
    for lower frequencies.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : int > 0 [scalar]
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

    resolution : float > 0
        Filter resolution factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
        Constant-Q energy for each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length < 2**(n_bins / bins_per_octave)`

    See Also
    --------
    cqt
    pseudo_cqt
    '''

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    # Make sure our hop is long enough to support the bottom octave
    if hop_length < 2**n_octaves:
        raise ParameterError('Insufficient hop_length {:d} '
                             'for {:d} octaves'.format(hop_length, n_octaves))

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
                                         resolution=resolution)

    # Determine which filters to use with Pseudo CQT
    pseudo_filters = lengths < 2*hop_length
    n_bins_pseudo = int(np.sum(pseudo_filters))

    cqt_resp = []

    if n_bins_pseudo > 0:
        fmin_pseudo = np.min(freqs[pseudo_filters])
        my_pseudo_cqt = pseudo_cqt(y, sr,
                                   hop_length=hop_length,
                                   fmin=fmin_pseudo,
                                   n_bins=n_bins_pseudo,
                                   bins_per_octave=bins_per_octave,
                                   tuning=tuning,
                                   resolution=resolution,
                                   norm=norm,
                                   sparsity=sparsity)
        cqt_resp.append(my_pseudo_cqt)

    n_bins_full = int(np.sum(~pseudo_filters))

    if n_bins_full > 0:

        fmin_full = np.min(freqs[~pseudo_filters])

        my_cqt = cqt(y, sr,
                     hop_length=hop_length,
                     fmin=fmin_full,
                     n_bins=n_bins_full,
                     bins_per_octave=bins_per_octave,
                     tuning=tuning,
                     resolution=resolution,
                     norm=norm,
                     sparsity=sparsity)

        cqt_resp.append(my_cqt)

    return __trim_stack(cqt_resp, n_bins)


@cache
def pseudo_cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=None, resolution=2,
               norm=1, sparsity=0.01):
    '''Compute the pseudo constant-Q transform of an audio signal.

    This uses a single fft size that is the smallest power of 2 that is greater
    than or equal to the max of:

        1. The longest CQT filter
        2. 2x the hop_length

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : int > 0 [scalar]
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

    resolution : float > 0
        Filter resolution factor. Larger values use longer windows.

    sparsity : float in [0, 1)
        Sparsify the CQT basis by discarding up to `sparsity`
        fraction of the energy in each basis.

        Set `sparsity=0` to disable sparsification.

    Returns
    -------
    CQT : np.ndarray [shape=(n_bins, t), dtype=np.float]
        Pseudo Constant-Q energy for each frequency at each time.

    Raises
    ------
    ParameterError
        If `hop_length < 2**(n_bins / bins_per_octave)`

    '''

    if fmin is None:
        # C1 by default
        fmin = note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    fft_basis, n_fft, _ = __fft_filters(sr,
                                        fmin,
                                        n_bins,
                                        bins_per_octave,
                                        tuning,
                                        resolution,
                                        norm,
                                        sparsity,
                                        hop_length=hop_length)

    # Remove phase for Pseudo CQT
    fft_basis = np.abs(fft_basis)

    # Compute the magnitude STFT with Hann window
    D = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))

    # Project onto the pseudo-cqt basis
    return fft_basis.dot(D)


def __fft_filters(sr, fmin, n_bins, bins_per_octave, tuning,
                  resolution, norm, sparsity, hop_length=None):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        resolution=resolution,
                                        norm=norm,
                                        pad_fft=True)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2 * hop_length:
        n_fft = int(2.0 ** (np.ceil(np.log2(2 * hop_length))))

    # normalize by inverse length to compensate for phase invariance
    basis *= lengths.reshape((-1, 1)) / n_fft

    # FFT and retain only the non-negative frequencies
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # normalize as in Parseval's relation, and sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis / n_fft, quantile=sparsity)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    max_col = min([x.shape[1] for x in cqt_resp])

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity
    return np.ascontiguousarray(cqt_resp[-n_bins:].T).T


def __variable_hop_response(y, n_fft, hop_length, min_filter_length,
                            fft_basis, aggregate):
    '''Compute the filter response with a target STFT hop.
    If the hop is too large (more than half the frame length),
    then over-sample at a smaller hop, and aggregate the results
    to the desired resolution.
    '''

    from ..feature.utils import sync
    # If target_hop <= n_fft / 2:
    #   my_hop = target_hop
    # else:
    #   my_hop = target_hop * 2**(-k)

    zoom_factor = np.ceil(np.log2(hop_length) - np.log2(min_filter_length))

    zoom_factor = 2**int(np.maximum(0, 1 + zoom_factor))

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=int(hop_length / zoom_factor),
             window=np.ones)

    # And filter response energy
    my_cqt = np.abs(fft_basis.dot(D))

    if zoom_factor > 1:
        # We need to aggregate.  Generate the boundary frames
        bounds = np.arange(0, my_cqt.shape[1], zoom_factor, dtype=int)
        my_cqt = sync(my_cqt, bounds, aggregate=aggregate)

    return my_cqt


def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff):
    '''Perform early downsampling on an audio signal, if it applies.'''

    if not (res_type == 'sinc_fastest' and audio._HAS_SAMPLERATE):
        return y, sr, hop_length

    downsample_count1 = int(np.ceil(np.log2(audio.BW_FASTEST * nyquist
                                            / filter_cutoff)) - 1)

    downsample_count2 = int(np.ceil(np.log2(hop_length) - n_octaves) - 1)

    downsample_count = min(downsample_count1, downsample_count2)

    if downsample_count > 0:
        downsample_factor = 2**(downsample_count)

        hop_length = hop_length // downsample_factor

        y = audio.resample(y, sr, sr / downsample_factor, res_type=res_type)

        sr = sr // downsample_factor

    return y, sr, hop_length
