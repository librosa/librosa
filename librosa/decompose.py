#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectrogram decomposition"""

import numpy as np
import scipy

import sklearn.decomposition

from . import core
from . import cache
from . import util


def decompose(S, n_components=None, transformer=None, sort=False):
    """Decompose a feature matrix.

    Given a spectrogram `S`, produce a decomposition into `components`
    and `activations` such that `S ~= components.dot(activations)`.

    By default, this is done with with non-negative matrix factorization (NMF),
    but any `sklearn.decomposition`-type object will work.

    Examples
    --------
    >>> # Decompose a magnitude spectrogram into 32 components with NMF
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> comps, acts = librosa.decompose.decompose(S, n_components=8)
    >>> comps
    array([[  9.826e-02,   6.439e-02, ...,   1.194e-01,   1.790e-02],
           [  2.565e-01,   1.600e-01, ...,   2.181e-01,   7.890e-02],
           ...,
           [  8.500e-08,   5.685e-08, ...,   3.240e-08,   3.534e-08],
           [  8.421e-08,   4.543e-08, ...,   2.183e-08,   2.353e-08]])
    >>> acts
    array([[  3.629e-02,   1.766e-01, ...,   3.379e-05,   5.473e-06],
           [  1.225e-02,   1.294e-01, ...,   3.544e-05,   3.386e-06],
           ...,
           [  4.268e-02,   4.184e-02, ...,   1.240e-05,   5.790e-06],
           [  6.748e-03,   1.720e-01, ...,   3.043e-05,  -0.000e+00]])

    >>> # Sort components by ascending peak frequency
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> comps, acts = librosa.decompose.decompose(S, n_components=8,
                                                  sort=True)

    >>> # Or with sparse dictionary learning
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> T = sklearn.decomposition.DictionaryLearning(n_components=8)
    >>> comps, acts = librosa.decompose.decompose(S, transformer=T)

    Parameters
    ----------
    S : np.ndarray [shape=(n_features, n_samples), dtype=float]
        The input feature matrix (e.g., magnitude spectrogram)

    n_components : int > 0 [scalar] or None
        number of desired components

        if None, then `n_features` components are used

    transformer : None or object
        If None, use `sklearn.decomposition.NMF`

        Otherwise, any object with a similar interface to NMF should work.
        `transformer` must follow the scikit-learn convention, where
        input data is `(n_samples, n_features)`.

        `transformer.fit_transform()` will be run on `S.T` (not `S`),
        the return value of which is stored (transposed) as `activations`

        The components will be retrieved as `transformer.components_.T`

        `S ~= np.dot(activations, transformer.components_).T`

        or equivalently:
        `S ~= np.dot(transformer.components_.T, activations.T)`

    sort : bool
        If `True`, components are sorted by ascending peak frequency.

        .. note:: If used with `transformer`, sorting is applied to copies
            of the decomposition parameters, and not to `transformer`'s
            internal parameters.

    Returns
    -------
    components: np.ndarray [shape=(n_features, n_components)]
        matrix of components (basis elements).

    activations: np.ndarray [shape=(n_components, n_samples)]
        transformed matrix/activation matrix

    See Also
    --------
    sklearn.decomposition : SciKit-Learn matrix decomposition modules
    """

    if transformer is None:
        transformer = sklearn.decomposition.NMF(n_components=n_components)

    activations = transformer.fit_transform(S.T).T

    components = transformer.components_.T

    if sort:
        components, idx = util.axis_sort(components, index=True)
        activations = activations[idx]

    return components, activations


@cache
def hpss(S, kernel_size=31, power=2.0, mask=False):
    """Median-filtering harmonic percussive source separation (HPSS).

    Decomposes an input spectrogram `S = H + P`
    where `H` contains the harmonic components,
    and `P` contains the percussive components.

    This implementation is based upon the algorithm described by [1]_.

    .. [1] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.


    Examples
    --------
    >>> # Separate into harmonic and percussive
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D)
    >>> H
    array([[  1.103e-02 +0.000e+00j,   2.519e-03 +0.000e+00j, ...,
              6.132e-05 +0.000e+00j,   1.439e-04 +0.000e+00j],
           [  3.373e-03 +1.423e-18j,   1.178e-03 +4.197e-04j, ...,
             -4.451e-05 +2.202e-05j,   8.756e-05 -9.594e-07j],
           ...,
           [ -2.175e-09 +1.902e-16j,   9.658e-10 +4.394e-10j, ...,
              0.000e+00 +0.000e+00j,  -1.984e-11 -2.210e-13j],
           [ -3.468e-09 +3.032e-16j,  -3.086e-10 +2.698e-17j, ...,
             -0.000e+00 +0.000e+00j,  -3.002e-11 +2.624e-18j]],
          dtype=complex64)
    >>> P
    array([[  1.412e-02 +0.000e+00j,   7.064e-02 +0.000e+00j, ...,
              1.903e-04 +0.000e+00j,   1.298e-06 +0.000e+00j],
           [  5.560e-02 +2.346e-17j,   4.777e-02 +1.702e-02j, ...,
             -1.669e-04 +8.257e-05j,   4.820e-06 -5.281e-08j],
           ...,
           [ -2.175e-09 +1.902e-16j,   1.681e-08 +7.649e-09j, ...,
              1.227e-10 +5.685e-11j,  -1.984e-11 -2.210e-13j],
           [ -1.458e-08 +1.275e-15j,  -1.258e-08 +1.100e-15j, ...,
             -1.181e-10 +1.033e-17j,  -3.002e-11 +2.624e-18j]],
          dtype=complex64)

    >>> # Or with a narrower horizontal filter
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    >>> # Just get harmonic/percussive masks, not the spectra
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    >>> mask_H
    array([[ 0.,  0., ...,  0.,  1.],
           [ 0.,  0., ...,  0.,  1.],
           ...,
           [ 0.,  0., ...,  0.,  1.],
           [ 0.,  0., ...,  0.,  1.]])
    >>> mask_P
    array([[ 1.,  1., ...,  1.,  0.],
           [ 1.,  1., ...,  1.,  0.],
           ...,
           [ 1.,  1., ...,  1.,  0.],
           [ 1.,  1., ...,  1.,  0.]])

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        input spectrogram. May be real (magnitude) or complex.

    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
        kernel size(s) for the median filters.

        - If scalar, the same size is used for both harmonic and percussive.
        - If iterable, the first value specifies the width of the
          harmonic filter, and the second value specifies the width
          of the percussive filter.


    power : float >= 0 [scalar]
        Exponent for the Wiener filter when constructing mask matrices.

        Mask matrices are defined by
        `mask_H = (r_H ** power) / (r_H ** power + r_P ** power)`
        where `r_H` and `r_P` are the median-filter responses for
        harmonic and percussive components.

    mask : bool
        Return the masking matrices instead of components

    Returns
    -------
    harmonic : np.ndarray [shape=(d, n)]
        harmonic component (or mask)

    percussive : np.ndarray [shape=(d, n)]
        percussive component (or mask)
    """

    if np.iscomplexobj(S):
        S, phase = core.magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = util.medfilt(S, kernel_size=(1, win_harm))

    perc = np.empty_like(S)
    perc[:] = util.medfilt(S, kernel_size=(win_perc, 1))

    if mask or power < util.SMALL_FLOAT:
        mask_harm = (harm > perc).astype(float)
        mask_perc = 1 - mask_harm
        if mask:
            return mask_harm, mask_perc
    else:
        perc = perc ** power
        zero_perc = (perc < util.SMALL_FLOAT)
        perc[zero_perc] = 0.0

        harm = harm ** power
        zero_harm = (harm < util.SMALL_FLOAT)
        harm[zero_harm] = 0.0

        # Find points where both are zero, equalize
        harm[zero_harm & zero_perc] = 0.5
        perc[zero_harm & zero_perc] = 0.5

        # Compute harmonic mask
        mask_harm = harm / (harm + perc)
        mask_perc = perc / (harm + perc)

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)
