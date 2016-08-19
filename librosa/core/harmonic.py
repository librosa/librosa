#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Harmonic calculations for frequency representations'''

import numpy as np
import scipy.interpolate
from ..util.exceptions import ParameterError

__all__ = ['harmonics']


def harmonics(x, freqs, h_range, kind='linear', fill_value=0, axis=0):
    '''Built a harmonic tensor from a time-frequency representation.

    Parameters
    ----------
    X : np.ndarray, real-valued
        The input energy

    freqs : np.ndarray, shape=(X.shape[axis])
        The frequency values corresponding to X's elements along the
        chosen axis.

    h_range : list-like, non-negative
        Harmonics to compute.  The first harmonic (1) corresponds to `X`
        itself.
        Values less than one (e.g., 1/2) correspond to sub-harmonics.

    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.

    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.

    axis : int
        The axis along which to compute harmonics

    Returns
    -------
    X_harm : np.ndarray, shape=(len(h_range), [X.shape])
        `X_harm[i]` will have the same shape as `X`, and measure
        the energy at the `h_range[i]` harmonic of each frequency.

    See Also
    --------
    scipy.interpolate.interp1d


    Examples
    --------
    Estimate the harmonics of a time-averaged tempogram

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=15, offset=30)
    >>> # Compute the time-varying tempogram and average over time
    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    >>> # We'll measure the first five harmonics
    >>> h_range = [1, 2, 3, 4, 5]
    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
    >>> # Build the harmonic tensor
    >>> t_harmonics = librosa.harmonics(tempi, f_tempo, h_range)
    >>> print(t_harmonics.shape)
    (5, 384)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(t_harmonics, x_axis='tempo', sr=sr)
    >>> plt.yticks(0.5 + np.arange(len(h_range)),
    ...            ['{:.3g}'.format(_) for _ in h_range])
    >>> plt.ylabel('Harmonic')
    >>> plt.xlabel('Tempo (BPM)')
    >>> plt.tight_layout()

    We can also compute frequency harmonics for spectrograms.
    To calculate subharmonic energy, use values < 1.

    >>> h_range = [1./3, 1./2, 1, 2, 3, 4]
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> S_harm = librosa.harmonics(S, fft_freqs, h_range, axis=0)
    >>> print(S_harm.shape)
    (6, 1025, 646)

    >>> plt.figure()
    >>> for i, _sh in enumerate(S_harm, 1):
    ...     plt.subplot(3,2,i)
    ...     librosa.display.specshow(librosa.logamplitude(_sh**2,
    ...                                                   ref_power=S.max()**2),
    ...                              sr=sr, y_axis='log')
    ...     plt.title('h={:.3g}'.format(h_range[i-1]))
    ...     plt.yticks([])
    >>> plt.tight_layout()
    '''

    # X_out will be the same shape as X, plus a leading
    # axis that has length = len(h_range)
    out_shape = [len(h_range)]
    out_shape.extend(x.shape)

    harmonic_out = np.zeros(out_shape, dtype=x.dtype)

    if freqs.ndim == 1 and len(freqs) == x.shape[axis]:
        harmonics_1d(harmonic_out, x, freqs, h_range,
                     kind=kind, fill_value=fill_value,
                     axis=axis)

    elif freqs.ndim == 2 and freqs.shape == x.shape:
        harmonics_2d(harmonic_out, x, freqs, h_range,
                     kind=kind, fill_value=fill_value,
                     axis=axis)
    else:
        raise ParameterError('freqs.shape={} does not match '
                             'input shape={}'.format(freqs.shape, x.shape))

    return harmonic_out


def harmonics_1d(harmonic_out, x, freqs, h_range, kind='linear',
                 fill_value=0, axis=0):
    '''Built a harmonic tensor from a time-frequency representation.

    Parameters
    ----------
    harmonic_out : np.ndarray, shape=(len(h_range), X.shape)
        The output array to store harmonics

    X : np.ndarray, real-valued
        The input energy

    freqs : np.ndarray, shape=(X.shape[axis])
        The frequency values corresponding to X's elements along the
        chosen axis.

    h_range : list-like, non-negative
        Harmonics to compute.  The first harmonic (1) corresponds to `X`
        itself.
        Values less than one (e.g., 1/2) correspond to sub-harmonics.

    kind : str
        Interpolation type.  See `scipy.interpolate.interp1d`.

    fill_value : float
        The value to fill when extrapolating beyond the observed
        frequency range.

    axis : int
        The axis along which to compute harmonics

    See Also
    --------
    scipy.interpolate.interp1d


    Examples
    --------
    Estimate the harmonics of a time-averaged tempogram

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      duration=15, offset=30)
    >>> # Compute the time-varying tempogram and average over time
    >>> tempi = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)
    >>> # We'll measure the first five harmonics
    >>> h_range = [1, 2, 3, 4, 5]
    >>> f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
    >>> # Build the harmonic tensor
    >>> t_harmonics = librosa.harmonics(tempi, f_tempo, h_range)
    >>> print(t_harmonics.shape)
    (5, 384)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(t_harmonics, x_axis='tempo', sr=sr)
    >>> plt.yticks(0.5 + np.arange(len(h_range)),
    ...            ['{:.3g}'.format(_) for _ in h_range])
    >>> plt.ylabel('Harmonic')
    >>> plt.xlabel('Tempo (BPM)')
    >>> plt.tight_layout()

    We can also compute frequency harmonics for spectrograms.
    To calculate subharmonic energy, use values < 1.

    >>> h_range = [1./3, 1./2, 1, 2, 3, 4]
    >>> S = np.abs(librosa.stft(y))
    >>> fft_freqs = librosa.fft_frequencies(sr=sr)
    >>> S_harm = librosa.harmonics(S, fft_freqs, h_range, axis=0)
    >>> print(S_harm.shape)
    (6, 1025, 646)

    >>> plt.figure()
    >>> for i, _sh in enumerate(S_harm, 1):
    ...     plt.subplot(3,2,i)
    ...     librosa.display.specshow(librosa.logamplitude(_sh**2,
    ...                                                   ref_power=S.max()**2),
    ...                              sr=sr, y_axis='log')
    ...     plt.title('h={:.3g}'.format(h_range[i-1]))
    ...     plt.yticks([])
    >>> plt.tight_layout()
    '''

    # Note: this only works for fixed-grid, 1d interpolation
    f_interp = scipy.interpolate.interp1d(freqs, x,
                                          kind=kind,
                                          axis=axis,
                                          copy=False,
                                          bounds_error=False,
                                          fill_value=fill_value)

    idx_out = [slice(None)] * harmonic_out.ndim

    # Compute the output index of the interpolated values
    interp_axis = 1 + (axis % x.ndim)

    # Iterate over the harmonics range
    for h_index, harmonic in enumerate(h_range):
        idx_out[0] = h_index

        # Iterate over frequencies
        for f_index, frequency in enumerate(freqs):
            # Offset the output axis by 1 to account for the harmonic index
            idx_out[interp_axis] = f_index

            # Estimate the harmonic energy at this frequency across time
            harmonic_out[tuple(idx_out)] = f_interp(harmonic * frequency)

    return harmonic_out


def harmonics_2d(harmonic_out, x, freqs, h_range, kind='linear', fill_value=0, axis=0):

    idx_in = [slice(None)] * x.ndim
    idx_freq = [slice(None)] * x.ndim
    idx_out = [slice(None)] * harmonic_out.ndim

    # This is the non-interpolation axis
    ni_axis = (1 + axis) % x.ndim

    # For each value in the non-interpolated axis, compute its harmonics
    for i in range(x.shape[ni_axis]):
        idx_in[ni_axis] = slice(i, i + 1)
        idx_freq[ni_axis] = i
        idx_out[1 + ni_axis] = idx_in[ni_axis]

        harmonics_1d(harmonic_out[idx_out], x[idx_in], freqs[idx_freq],
                     h_range, kind=kind, fill_value=fill_value,
                     axis=axis)
