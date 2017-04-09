#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spectrogram decomposition
=========================
.. autosummary::
    :toctree: generated/

    decompose
    hpss
    nn_filter
"""

import numpy as np

import scipy.sparse
from scipy.ndimage import median_filter

import sklearn.decomposition

from . import core
from . import cache
from . import segment
from . import util
from .util.exceptions import ParameterError

__all__ = ['decompose', 'hpss', 'nn_filter']


def decompose(S, n_components=None, transformer=None, sort=False, fit=True, **kwargs):
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

    fit : bool
        If `True`, components are estimated from the input ``S``.

        If `False`, components are assumed to be pre-computed and stored
        in ``transformer``, and are not changed.

    kwargs : Additional keyword arguments to the default transformer
        `sklearn.decomposition.NMF`


    Returns
    -------
    components: np.ndarray [shape=(n_features, n_components)]
        matrix of components (basis elements).

    activations: np.ndarray [shape=(n_components, n_samples)]
        transformed matrix/activation matrix


    Raises
    ------
    ParameterError
        if `fit` is False and no `transformer` object is provided.


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

    >>> comps, acts = librosa.decompose.decompose(S, n_components=16,
    ...                                           sort=True)


    Or with sparse dictionary learning

    >>> import sklearn.decomposition
    >>> T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)
    >>> scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10,8))
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Input spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.subplot(3, 2, 3)
    >>> librosa.display.specshow(librosa.amplitude_to_db(comps,
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Components')
    >>> plt.subplot(3, 2, 4)
    >>> librosa.display.specshow(acts, x_axis='time')
    >>> plt.ylabel('Components')
    >>> plt.title('Activations')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 3)
    >>> S_approx = comps.dot(acts)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S_approx,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Reconstructed spectrogram')
    >>> plt.tight_layout()
    """

    if transformer is None:
        if fit is False:
            raise ParameterError('fit must be True if transformer is None')

        transformer = sklearn.decomposition.NMF(n_components=n_components,
                                                **kwargs)

    if n_components is None:
        n_components = S.shape[0]

    if fit:
        activations = transformer.fit_transform(S.T).T
    else:
        activations = transformer.transform(S.T).T

    components = transformer.components_.T

    if sort:
        components, idx = util.axis_sort(components, index=True)
        activations = activations[idx]

    return components, activations


@cache(level=30)
def hpss(S, kernel_size=31, power=2.0, mask=False, margin=1.0):
    """Median-filtering harmonic percussive source separation (HPSS).

    If `margin = 1.0`, decomposes an input spectrogram `S = H + P`
    where `H` contains the harmonic components,
    and `P` contains the percussive components.

    If `margin > 1.0`, decomposes an input spectrogram `S = H + P + R`
    where `R` contains residual components not included in `H` or `P`.

    This implementation is based upon the algorithm described by [1]_ and [2]_.

    .. [1] Fitzgerald, Derry.
        "Harmonic/percussive separation using median filtering."
        13th International Conference on Digital Audio Effects (DAFX10),
        Graz, Austria, 2010.

    .. [2] Driedger, Müller, Disch.
        "Extending harmonic-percussive separation of audio."
        15th International Society for Music Information Retrieval Conference (ISMIR 2014),
        Taipei, Taiwan, 2014.

    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        input spectrogram. May be real (magnitude) or complex.

    kernel_size : int or tuple (kernel_harmonic, kernel_percussive)
        kernel size(s) for the median filters.

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the width of the
          harmonic filter, and the second value specifies the width
          of the percussive filter.

    power : float > 0 [scalar]
        Exponent for the Wiener filter when constructing soft mask matrices.

    mask : bool
        Return the masking matrices instead of components.

        Masking matrices contain non-negative real values that
        can be used to measure the assignment of energy from `S`
        into harmonic or percussive components.

        Components can be recovered by multiplying `S * mask_H`
        or `S * mask_P`.


    margin : float or tuple (margin_harmonic, margin_percussive)
        margin size(s) for the masks (as described in [2]_)

        - If scalar, the same size is used for both harmonic and percussive.
        - If tuple, the first value specifies the margin of the
          harmonic mask, and the second value specifies the margin
          of the percussive mask.

    Returns
    -------
    harmonic : np.ndarray [shape=(d, n)]
        harmonic component (or mask)

    percussive : np.ndarray [shape=(d, n)]
        percussive component (or mask)


    See Also
    --------
    util.softmask

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Separate into harmonic and percussive

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> D = librosa.stft(y)
    >>> H, P = librosa.decompose.hpss(D)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D,
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Full power spectrogram')
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(H,
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Harmonic power spectrogram')
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(librosa.amplitude_to_db(P,
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Percussive power spectrogram')
    >>> plt.tight_layout()


    Or with a narrower horizontal filter

    >>> H, P = librosa.decompose.hpss(D, kernel_size=(13, 31))

    Just get harmonic/percussive masks, not the spectra

    >>> mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    >>> mask_H
    array([[  1.000e+00,   1.469e-01, ...,   2.648e-03,   2.164e-03],
           [  1.000e+00,   2.368e-01, ...,   9.413e-03,   7.703e-03],
           ...,
           [  8.869e-01,   5.673e-02, ...,   4.603e-02,   1.247e-05],
           [  7.068e-01,   2.194e-02, ...,   4.453e-02,   1.205e-05]], dtype=float32)
    >>> mask_P
    array([[  2.858e-05,   8.531e-01, ...,   9.974e-01,   9.978e-01],
           [  1.586e-05,   7.632e-01, ...,   9.906e-01,   9.923e-01],
           ...,
           [  1.131e-01,   9.433e-01, ...,   9.540e-01,   1.000e+00],
           [  2.932e-01,   9.781e-01, ...,   9.555e-01,   1.000e+00]], dtype=float32)

    Separate into harmonic/percussive/residual components by using a margin > 1.0

    >>> H, P = librosa.decompose.hpss(D, margin=3.0)
    >>> R = D - (H+P)
    >>> y_harm = librosa.core.istft(H)
    >>> y_perc = librosa.core.istft(P)
    >>> y_resi = librosa.core.istft(R)


    Get a more isolated percussive component by widening its margin

    >>> H, P = librosa.decompose.hpss(D, margin=(1.0,5.0))

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

    if np.isscalar(margin):
        margin_harm = margin
        margin_perc = margin
    else:
        margin_harm = margin[0]
        margin_perc = margin[1]

    # margin minimum is 1.0
    if margin_harm < 1 or margin_perc < 1:
        raise ParameterError("Margins must be >= 1.0. "
                             "A typical range is between 1 and 10.")

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(S)
    harm[:] = median_filter(S, size=(1, win_harm), mode='reflect')

    perc = np.empty_like(S)
    perc[:] = median_filter(S, size=(win_perc, 1), mode='reflect')

    split_zeros = (margin_harm == 1 and margin_perc == 1)

    mask_harm = util.softmask(harm, perc * margin_harm,
                              power=power,
                              split_zeros=split_zeros)

    mask_perc = util.softmask(perc, harm * margin_perc,
                              power=power,
                              split_zeros=split_zeros)

    if mask:
        return mask_harm, mask_perc

    return ((S * mask_harm) * phase, (S * mask_perc) * phase)


@cache(level=30)
def nn_filter(S, rec=None, aggregate=None, axis=-1, **kwargs):
    '''Filtering by nearest-neighbors.

    Each data point (e.g, spectrogram column) is replaced
    by aggregating its nearest neighbors in feature space.

    This can be useful for de-noising a spectrogram or feature matrix.

    The non-local means method [1]_ can be recovered by providing a
    weighted recurrence matrix as input and specifying `aggregate=np.average`.

    Similarly, setting `aggregate=np.median` produces sparse de-noising
    as in REPET-SIM [2]_.

    .. [1] Buades, A., Coll, B., & Morel, J. M.
        (2005, June). A non-local algorithm for image denoising.
        In Computer Vision and Pattern Recognition, 2005.
        CVPR 2005. IEEE Computer Society Conference on (Vol. 2, pp. 60-65). IEEE.

    .. [2] Rafii, Z., & Pardo, B.
        (2012, October).  "Music/Voice Separation Using the Similarity Matrix."
        International Society for Music Information Retrieval Conference, 2012.

    Parameters
    ----------
    S : np.ndarray
        The input data (spectrogram) to filter

    rec : (optional) scipy.sparse.spmatrix or np.ndarray
        Optionally, a pre-computed nearest-neighbor matrix
        as provided by `librosa.segment.recurrence_matrix`

    aggregate : function
        aggregation function (default: `np.mean`)

        If `aggregate=np.average`, then a weighted average is
        computed according to the (per-row) weights in `rec`.

        For all other aggregation functions, all neighbors
        are treated equally.


    axis : int
        The axis along which to filter (by default, columns)

    kwargs
        Additional keyword arguments provided to
        `librosa.segment.recurrence_matrix` if `rec` is not provided

    Returns
    -------
    S_filtered : np.ndarray
        The filtered data

    Raises
    ------
    ParameterError
        if `rec` is provided and its shape is incompatible with `S`.

    See also
    --------
    decompose
    hpss
    librosa.segment.recurrence_matrix


    Notes
    -----
    This function caches at level 30.


    Examples
    --------

    De-noise a chromagram by non-local median filtering.
    By default this would use euclidean distance to select neighbors,
    but this can be overridden directly by setting the `metric` parameter.

    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=30, duration=10)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_med = librosa.decompose.nn_filter(chroma,
    ...                                          aggregate=np.median,
    ...                                          metric='cosine')

    To use non-local means, provide an affinity matrix and `aggregate=np.average`.

    >>> rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
    ...                                         metric='cosine', sparse=True)
    >>> chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
    ...                                          aggregate=np.average)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 8))
    >>> plt.subplot(5, 1, 1)
    >>> librosa.display.specshow(chroma, y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Unfiltered')
    >>> plt.subplot(5, 1, 2)
    >>> librosa.display.specshow(chroma_med, y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Median-filtered')
    >>> plt.subplot(5, 1, 3)
    >>> librosa.display.specshow(chroma_nlm, y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Non-local means')
    >>> plt.subplot(5, 1, 4)
    >>> librosa.display.specshow(chroma - chroma_med,
    ...                          y_axis='chroma')
    >>> plt.colorbar()
    >>> plt.title('Original - median')
    >>> plt.subplot(5, 1, 5)
    >>> librosa.display.specshow(chroma - chroma_nlm,
    ...                          y_axis='chroma', x_axis='time')
    >>> plt.colorbar()
    >>> plt.title('Original - NLM')
    >>> plt.tight_layout()
    '''
    if aggregate is None:
        aggregate = np.mean

    if rec is None:
        kwargs = dict(kwargs)
        kwargs['sparse'] = True
        rec = segment.recurrence_matrix(S, axis=axis, **kwargs)
    elif not scipy.sparse.issparse(rec):
        rec = scipy.sparse.csr_matrix(rec)

    if rec.shape[0] != S.shape[axis] or rec.shape[0] != rec.shape[1]:
        raise ParameterError('Invalid self-similarity matrix shape '
                             'rec.shape={} for S.shape={}'.format(rec.shape,
                                                                  S.shape))

    s_out = S.copy()

    index = [slice(None)] * s_out.ndim

    for i in range(rec.shape[0]):
        index[axis] = i

        # Get the non-zeros out of the recurrence matrix
        targets = rec[i].nonzero()[-1]
        if not len(targets):
            continue

        neighbors = np.take(S, targets, axis=axis)

        if aggregate is np.average:
            weights = rec[i, targets].toarray().squeeze()
            s_out[tuple(index)] = aggregate(neighbors, axis=axis, weights=weights)
        else:
            s_out[tuple(index)] = aggregate(neighbors, axis=axis)

    return s_out
