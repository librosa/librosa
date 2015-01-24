#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Pitch-tracking and tuning estimation'''

import numpy as np
import warnings

from . import audio
from .time_frequency import cqt_frequencies, note_to_hz
from .spectrum import stft
from .pitch import estimate_tuning
from .. import cache
from .. import filters
from .. import util
from ..feature.utils import sync




@cache
def cqt(y, sr=22050, hop_length=512, fmin=None, n_bins=84,
        bins_per_octave=12, tuning=None, resolution=2,
        aggregate=None, norm=2, sparsity=0.01):
    '''Compute the constant-Q transform of an audio signal.

    This implementation is based on the recursive sub-sampling method
    described by [1]_.

    .. [1] Schoerkhuber, Christian, and Anssi Klapuri.
        "Constant-Q transform toolbox for music processing."
        7th Sound and Music Computing Conference, Barcelona, Spain. 2010.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = librosa.cqt(y, sr=sr)
    >>> C
    array([[  3.985e-01,   4.696e-01, ...,   7.009e-04,   8.497e-04],
           [  1.135e+00,   1.220e+00, ...,   1.669e-03,   1.691e-03],
           ...,
           [  6.036e-04,   3.765e-02, ...,   3.100e-14,   0.000e+00],
           [  4.690e-04,   8.762e-02, ...,   1.995e-14,   0.000e+00]])

    >>> # Limit the frequency range
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C3'),
                        n_bins=60)
    >>> C
    array([[  8.936e+01,   9.573e+01, ...,   1.333e-02,   1.443e-02],
           [  8.181e+01,   8.231e+01, ...,   2.552e-02,   2.147e-02],
           ...,
           [  2.791e-03,   2.463e-02, ...,   9.306e-04,   0.000e+00],
           [  2.687e-03,   1.446e-02, ...,   8.878e-04,   0.000e+00]])

    >>> # Use higher resolution
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C3'),
                        n_bins=60 * 2, bins_per_octave=12 * 2)
    >>> C
    array([[  1.000e+02,   1.094e+02, ...,   8.701e-02,   8.098e-02],
           [  2.222e+02,   2.346e+02, ...,   5.625e-02,   4.770e-02],
           ...,
           [  5.177e-02,   1.710e-02, ...,   4.670e-03,   7.403e-12],
           [  1.981e-02,   2.721e-03, ...,   1.943e-03,   7.246e-12]])

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : int > 0 [scalar]
        sampling rate of `y`

    hop_length : int > 0 [scalar]
        number of samples between successive CQT columns.

    fmin : float > 0 [scalar]
        Minimum frequency. Defaults to C2 ~= 32.70 Hz

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
    ValueError
        If `hop_length < 2**(n_bins / bins_per_octave)`

    See Also
    --------
    librosa.core.resample
    librosa.feature.sync
    librosa.util.normalize
    '''

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))

    # Make sure our hop is long enough to support the bottom octave
    if hop_length < 2**n_octaves:
        raise ValueError('Insufficient hop_length {:d} '
                         'for {:d} octaves'.format(hop_length, n_octaves))

    if fmin is None:
        # C2 by default
        fmin = note_to_hz('C2')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)[-bins_per_octave:]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(resolution) / (2.0**(1. / bins_per_octave) - 1)
    filter_cutoff = fmax_t*(1 + filters.HANN_BW / Q)
    nyquist = sr / 2.0

    if filter_cutoff < audio.BW_FASTEST*nyquist:
        res_type = 'sinc_fastest'
    elif filter_cutoff < audio.BW_MEDIUM*nyquist:
        res_type = 'sinc_medium'
    elif filter_cutoff < audio.BW_BEST*nyquist:
        res_type = 'sinc_best'
    else:
        res_type = 'sinc_best'


    cqt_resp = []



    if res_type == 'sinc_fastest' and audio._HAS_SAMPLERATE:

        # How many times can we downsample by 2 before filtering?
        # Requirements:
        # filter_cutoff < BW*nyquist  # (BW is resampling bandwidth fraction)
        # hop_length > 2**n_octaves

        downsample_count1 = int(np.ceil(np.log2(audio.BW_FASTEST * nyquist
                                                / filter_cutoff)) - 1)
        downsample_count2 = int(np.ceil(np.log2(hop_length) - n_octaves) - 1)
        downsample_count = min(downsample_count1, downsample_count2)
        if downsample_count > 0:
            downsample_factor = 2**downsample_count
            hop_length = int(hop_length/downsample_factor)
            y = audio.resample(y, sr, sr/downsample_factor, res_type=res_type)
            sr = sr/downsample_factor

    if res_type != 'sinc_fastest' and audio._HAS_SAMPLERATE:

        # Do two octaves before resampling to allow for usage of sinc_fastest

        # Generate the basis filters
        basis, lengths = filters.constant_q(sr,
                                            fmin=fmin_t,
                                            n_bins=bins_per_octave,
                                            bins_per_octave=bins_per_octave,
                                            tuning=tuning,
                                            resolution=resolution,
                                            norm=norm,
                                            pad_fft=True,
                                            return_lengths=True)

        # FFT the filters
        min_filter_length = np.min(lengths)

        # Filters are padded up to the nearest integral power of 2
        n_fft = basis.shape[1]

        # FFT and retain only the non-negative frequencies
        fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft / 2)+1]

        # normalize as in Parseval's relation, and sparsify the basis
        fft_basis = util.sparsify_rows(fft_basis / n_fft, quantile=sparsity)

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

        filter_cutoff = fmax_t*(1 + filters.HANN_BW / Q)
        assert filter_cutoff < audio.BW_FASTEST*nyquist

        res_type = 'sinc_fastest'


    # Generate the basis filters
    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin_t,
                                        n_bins=bins_per_octave,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        resolution=resolution,
                                        norm=norm,
                                        pad_fft=True,
                                        return_lengths=True)

    # FFT the filters
    min_filter_length = np.min(lengths)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    # FFT and retain only the non-negative frequencies
    fft_basis = np.fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft / 2)+1]

    # normalize as in Parseval's relation, and sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis / n_fft, quantile=sparsity)

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
