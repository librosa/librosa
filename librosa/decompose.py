#!/usr/bin/env python
"""Spectrogram decomposition"""

import numpy as np
import scipy
import scipy.signal

import sklearn.decomposition

import librosa.core


def decompose(S, n_components=None, transformer=None):
    """Decompose a feature matrix.
    
    By default, this is done with with non-negative matrix factorization,
    but any ``sklearn.decomposition``-type object will work.

    :usage:
        >>> # Decompose a magnitude spectrogram into 32 components with NMF
        >>> S = np.abs(librosa.stft(y))
        >>> components, activations = librosa.decompose.decompose(S, n_components=32)

        >>> # Or with sparse dictionary learning
        >>> T = sklearn.decomposition.DictionaryLearning(n_components=32)
        >>> components, activations = librosa.decompose.decompose(S, transformer=T)

    :parameters:
        - S : np.ndarray, shape=(n_features, n_samples)
            feature matrix

        - n_components : int > 0 or None
            number of desired components
            if None, then ``n_features`` components are used

        - transformer : None or object
            If None, use ``sklearn.decomposition.NMF``

            Otherwise, any object with a similar interface to NMF should work.
            ``transformer`` must follow the scikit-learn convention, where input data
            is (n_samples, n_features). 

            ``transformer.fit_transform()`` will be run on ``S.T`` (not ``S``),
            the return value of which is stored (transposed) as ``activations``.

            The components will be retrieved as ``transformer.components_.T``

            ``S ~= np.dot(activations, transformer.components_).T``
            
            or equivalently:
            ``S ~= np.dot(transformer.components_.T, activations.T)``

    :returns:
        - components: np.ndarray, shape=(n_features, n_components)
            dictionary matrix

        - activations: np.ndarray, shape=(n_components, n_samples)
            transformed matrix/activation matrix

    """

    if transformer is None:
        transformer = sklearn.decomposition.NMF(n_components=n_components)

    activations = transformer.fit_transform(S.T)

    return (transformer.components_.T, activations.T)

def hpss(S, kernel_size=31, power=1.0, mask=False):
    """Median-filtering harmonic percussive separation

    Decomposes an input spectrogram ``S = H + P``
    where ``H`` contains the harmonic components, 
    and ``P`` contains the percussive components.

    :usage:
        >>> D = librosa.stft(y)
        >>> H, P = librosa.decompose.hpss(D)
        >>> y_harmonic = librosa.istft(H)

        >>> # Or with a narrower horizontal filter
        >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    :parameters:
      - S : np.ndarray
          input spectrogram. May be real (magnitude) or complex.

      - kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
          kernel size for the median filters.
          If scalar, the same size is used for both harmonic and percussive.
          If array_like, the first value specifies the width of the harmonic filter,
          and the second value specifies the width of the percussive filter.

      - power : float > 0
          Exponent for the Wiener filter

      - mask : bool
          Return the masking matrices instead of components

    :returns:
      - harmonic : np.ndarray
          harmonic component (or mask)

      - percussive : np.ndarray
          percussive component (or mask)

      .. note:: harmonic + percussive = S

    .. note::
      @article{fitzgerald2010harmonic,
        title={Harmonic/percussive separation using median filtering},
        author={Fitzgerald, Derry},
        year={2010},
        publisher={Dublin Institute of Technology}}
    
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

