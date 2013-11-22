#!/usr/bin/env python
"""Spectrogram decomposition"""

import numpy as np
import scipy
import scipy.signal

import sklearn.decomposition

import librosa.core


def decompose(S, n_components=None, transformer=None):
    """Decompose the feature matrix with non-negative matrix factorization

    :parameters:
        - S : np.ndarray
            feature matrix (d-by-t)
        - n_components : int > 0 or None
            number of components, if None then all d components are used
        - transformer : any instance which implements fit_transform()
            If None, use sklearn.decomposition.NMF by default
            Otherwise, because of scikit-learn convention where the input data
            is (n_samples, n_features), NMF.fit_transform() should take S.T as
            input, and returns transformed S_new, where:
            ``S.T ~= S_new.dot(transformer.components_)``
            or equivalently:
            ``S ~= transformer.components_.T.dot(S_new.T)``

    :returns:
        - components: np.ndarray
            dictionary matrix (d-by-n_components)
        - activations: np.ndarray
            transformed matrix/activation matrix (n_components-by-t)

    """

    if transformer is None:
        transformer = sklearn.decomposition.NMF(n_components=n_components)
    activations = transformer.fit_transform(S.T)
    return (transformer.components_.T, activations.T)

def hpss(S, kernel_size=19, power=1.0, mask=False):
    """Median-filtering harmonic percussive separation

    :parameters:
      - S : np.ndarray
          input spectrogram. May be real (magnitude) or complex.

      - kernel_size : int or array_like (kernel_harmonic, kernel_percussive)
          kernel size for the median filters.
          If scalar, the same size is used for both harmonic and percussive.
          If array_like, the first value specifies the width of the harmonic filter,
          and the second value specifies the width of the percussive filter.

      - power : float
          Exponent for the Wiener filter

      - mask : boolean
          Return the masking matrices instead of components

    :returns:
      - harmonic : np.ndarray
          harmonic component (or mask)

      - percussive : np.ndarray
          percussive component (or mask)

      .. note:: harmonic + percussive = S

    .. note::
      - Fitzgerald, D. (2010). 
        Harmonic/percussive separation using median filtering.

    """

    if np.iscomplex(S).any():
        S, phase = librosa.core.magphase(S)
    else:
        phase = 1

    if np.isscalar(kernel_size):
        win_harm = kernel_size
        win_perc = kernel_size
    else:
        win_harm = kernel_size[0]
        win_perc = kernel_size[1]

    # Compute median filters
    harm = scipy.signal.medfilt2d(S, [1, win_harm])
    perc = scipy.signal.medfilt2d(S, [win_perc, 1])

    if mask or power == 0:
        mask_harm = (harm > perc).astype(float)
        mask_perc = 1 - mask_harm
        if mask: 
            return mask_harm, mask_perc
    else:
        zero_perc = (perc == 0)
        perc = perc ** power
        perc[zero_perc] = 0.0
    
        zero_harm = (harm == 0)
        harm = harm ** power
        harm[zero_harm] = 0.0

        # Find points where both are zero, equalize
        harm[zero_harm & zero_perc] = 0.5
        perc[zero_harm & zero_perc] = 0.5

        # Compute harmonic mask
        mask_harm = harm / (harm + perc)
        mask_perc = perc / (harm + perc)

    return (mask_harm * S * phase, mask_perc * S * phase)

