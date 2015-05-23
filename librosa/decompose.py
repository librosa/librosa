#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrogram decomposition
=========================
.. autosummary::
    :toctree: generated/

    decompose
    hpss
"""

import numpy as np

from scipy.ndimage import median_filter

import sklearn.decomposition

from . import core
from . import cache
from . import util

__all__ = ['decompose', 'hpss']


def decompose(S, n_components=None, transformer=None, sort=False, **kwargs):
    """Decompose a feature matrix.

    Given a spectrogram `S`, produce a decomposition into `components`
    and `activations` such that `S ~= components.dot(activations)`.

    By default, this is done with with non-negative matrix factorization (NMF),
    but any `sklearn.decomposition`-type object will work.


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

    kwargs : Additional keyword arguments to the default transformer
        `sklearn.decomposition.NMF`


    Returns
    -------
    components: np.ndarray [shape=(n_features, n_components)]
        matrix of components (basis elements).

    activations: np.ndarray [shape=(n_components, n_samples)]
        transformed matrix/activation matrix


    See Also
    --------
    sklearn.decomposition : SciKit-Learn matrix decomposition modules


    Examples
    --------
    Decompose a magnitude spectrogram into 32 components with NMF

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> comps, acts = librosa.decompose.decompose(S, n_components=8)
    >>> comps
    array([[  1.876e-01,   5.559e-02, ...,   1.687e-01,   4.907e-02],
           [  3.148e-01,   1.719e-01, ...,   2.314e-01,   9.493e-02],
           ..., 
           [  1.561e-07,   8.564e-08, ...,   7.167e-08,   4.997e-08],
           [  1.531e-07,   7.880e-08, ...,   5.632e-08,   4.028e-08]])
    >>> acts
    array([[  4.197e-05,   8.512e-03, ...,   3.056e-05,   9.159e-06],
           [  9.568e-06,   1.718e-02, ...,   3.322e-05,   7.869e-06],
           ..., 
           [  5.982e-05,   1.311e-02, ...,  -0.000e+00,   6.323e-06],
           [  3.782e-05,   7.056e-03, ...,   3.290e-05,  -0.000e+00]])


    Sort components by ascending peak frequency

    >>> comps, acts = librosa.decompose.decompose(S, n_components=8,
    ...                                           sort=True)


    Or with sparse dictionary learning

    >>> import sklearn.decomposition
    >>> T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=8)
    >>> comps, acts = librosa.decompose.decompose(S, transformer=T, sort=True)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10,8))
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(S**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Input spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.subplot(3, 2, 3)
    >>> librosa.display.specshow(comps, y_axis='log')
    >>> plt.title('Components')
    >>> plt.subplot(3, 2, 4)
    >>> librosa.display.specshow(acts, x_axis='time')
    >>> plt.ylabel('Components')
    >>> plt.title('Activations')
    >>> plt.subplot(3, 1, 3)
    >>> S_approx = comps.dot(acts)
    >>> librosa.display.specshow(librosa.logamplitude(S_approx**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Reconstructed spectrogram')
    >>> plt.tight_layout()
    """

    if transformer is None:
        transformer = sklearn.decomposition.NMF(n_components=n_components,
                                                **kwargs)

    if n_components is None:
        n_components = S.shape[0]

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


    Examples
    --------
    Separate into harmonic and percussive

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Full power spectrogram')
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(H)**2,
    ...                          ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Harmonic power spectrogram')
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(librosa.logamplitude(np.abs(P)**2,
    ...                          ref_power=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Percussive power spectrogram')
    >>> plt.tight_layout()


    Or with a narrower horizontal filter

    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    Just get harmonic/percussive masks, not the spectra

    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    >>> mask_H
    array([[ 1.,  0., ...,  0.,  0.],
           [ 1.,  0., ...,  0.,  0.],
           ..., 
           [ 0.,  0., ...,  0.,  0.],
           [ 0.,  0., ...,  0.,  0.]])
    >>> mask_P
    array([[ 0.,  1., ...,  1.,  1.],
           [ 0.,  1., ...,  1.,  1.],
           ..., 
           [ 1.,  1., ...,  1.,  1.],
           [ 1.,  1., ...,  1.,  1.]])

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
    harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

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
