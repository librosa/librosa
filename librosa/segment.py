#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal segmentation
=====================

Recurrence and self-similarity
------------------------------
.. autosummary::
    :toctree: generated/

    recurrence_matrix
    recurrence_to_lag
    lag_to_recurrence
    timelag_filter

Temporal clustering
-------------------
.. autosummary::
    :toctree: generated/

    agglomerative
    subsegment

Deprecated
----------
.. autosummary::
    :toctree: generated/

    structure_feature
"""

from decorator import decorator

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

from . import cache
from . import util
from .util.exceptions import ParameterError

__all__ = ['recurrence_matrix',
           'recurrence_to_lag',
           'lag_to_recurrence',
           'timelag_filter',
           'agglomerative',
           'subsegment',
           # Deprecated functions
           'structure_feature']


@cache
def __band_infinite(n, width, v_in=0.0, v_out=np.inf, dtype=np.float32):
    '''Construct a square, banded matrix `X` where
    `X[i, j] == v_in` if `|i - j| <= width`
    `X[i, j] == v_out` if `|i - j| > width`

    This is used to suppress nearby links in `recurrence_matrix`.
    '''

    if width > n:
        raise ParameterError('width cannot exceed n')

    # Instantiate the matrix
    band = np.empty((n, n), dtype=dtype)

    # Fill the out-of-band values
    band.fill(v_out)

    # Fill the in-band values
    band[np.triu_indices_from(band, width)] = v_in
    band[np.tril_indices_from(band, -width)] = v_in

    return band


@cache
def recurrence_matrix(data, k=None, width=1, metric='sqeuclidean', sym=False):
    '''Compute the binary recurrence matrix from a time-series.

    `rec[i,j] == True` if (and only if) (`data[:,i]`, `data[:,j]`) are
    k-nearest-neighbors and `|i-j| >= width`


    Parameters
    ----------
    data : np.ndarray [shape=(d, t)]
        A feature matrix

    k : int > 0 [scalar] or None
        the number of nearest-neighbors for each sample

        Default: `k = 2 * ceil(sqrt(t - 2 * width + 1))`,
        or `k = 2` if `t <= 2 * width + 1`

    width : int >= 1 [scalar]
        only link neighbors `(data[:, i], data[:, j])`
        if `|i-j| >= width`

    metric : str
        Distance metric to use for nearest-neighbor calculation.

        See `scipy.spatial.distance.cdist()` for details.

    sym : bool [scalar]
        set `sym=True` to only link mutual nearest-neighbors

    Returns
    -------
    rec : np.ndarray [shape=(t,t), dtype=bool]
        Binary recurrence matrix

    See Also
    --------
    scipy.spatial.distance.cdist
    librosa.feature.stack_memory
    structure_feature

    Examples
    --------
    Find nearest neighbors in MFCC space

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> R = librosa.segment.recurrence_matrix(mfcc)

    Or fix the number of nearest neighbors to 5

    >>> R = librosa.segment.recurrence_matrix(mfcc, k=5)

    Suppress neighbors within +- 7 samples

    >>> R = librosa.segment.recurrence_matrix(mfcc, width=7)

    Use cosine similarity instead of Euclidean distance

    >>> R = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

    Require mutual nearest neighbors

    >>> R = librosa.segment.recurrence_matrix(mfcc, sym=True)

    Plot the feature and recurrence matrices

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(1, 2, 1)
    >>> librosa.display.specshow(mfcc, x_axis='time')
    >>> plt.title('MFCC')
    >>> plt.subplot(1, 2, 2)
    >>> librosa.display.specshow(R, x_axis='time', y_axis='time',
    ...                          aspect='equal')
    >>> plt.title('MFCC recurrence (symmetric)')
    >>> plt.tight_layout()

    '''

    data = np.atleast_2d(data)

    t = data.shape[1]

    if width < 1:
        raise ParameterError('width must be at least 1')

    if k is None:
        if t > 2 * width + 1:
            k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
        else:
            k = 2

    k = int(k)

    # Build the distance matrix
    D = scipy.spatial.distance.cdist(data.T, data.T, metric=metric)

    # Max out the diagonal band
    D = D + __band_infinite(t, width)

    # build the recurrence plot
    rec = np.zeros((t, t), dtype=bool)

    # get the k nearest neighbors for each point
    for i in range(t):
        for j in np.argsort(D[i])[:k]:
            rec[i, j] = True

    # symmetrize
    if sym:
        rec = rec * rec.T

    return rec


def recurrence_to_lag(rec, pad=True):
    '''Convert a recurrence matrix into a lag matrix.

        `lag[i, j] == rec[i+j, j]`

    Parameters
    ----------
    rec : np.ndarray, [shape=(n, n)]
        A (binary) recurrence matrix, as returned by `recurrence_matrix`

    pad : bool
        If False, `lag` matrix is square, which is equivalent to
        assuming that the signal repeats itself indefinitely.

        If True, `lag` is padded with `n` zeros, which eliminates
        the assumption of repetition.

    Returns
    -------
    lag : np.ndarray [shape=(2*n, n) or (n, n)]
        The recurrence matrix in (lag, time) coordinates

    Raises
    ------
    ParameterError : if `rec` is non-square

    See Also
    --------
    recurrence_matrix
    lag_to_recurrence

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)
    >>> recurrence = librosa.segment.recurrence_matrix(mfccs)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> librosa.display.specshow(lag_pad, x_axis='time', y_axis='lag')
    >>> plt.title('Lag (zero-padded)')
    >>> plt.subplot(1, 2, 2)
    >>> librosa.display.specshow(lag_nopad, x_axis='time', y_axis='time')
    >>> plt.title('Lag (no padding)')
    >>> plt.tight_layout()
    '''

    if rec.ndim != 2 or rec.shape[0] != rec.shape[1]:
        raise ParameterError('non-square recurrence matrix shape: '
                         '{}'.format(rec.shape))

    t = rec.shape[1]
    if pad:
        lag = np.pad(rec, [(0, t), (0, 0)], mode='constant')
    else:
        lag = rec.copy()

    for i in range(1, t):
        lag[:, i] = np.roll(lag[:, i], -i)

    return np.ascontiguousarray(lag.T).T


def lag_to_recurrence(lag):
    '''Convert a lag matrix into a recurrence matrix.

    Parameters
    ----------
    lag : np.ndarray [shape=(2*n, n) or (n, n)]
        A lag matrix, as produced by `recurrence_to_lag`

    Returns
    -------
    rec : np.ndarray [shape=(n, n)]
        A recurrence matrix in (time, time) coordinates

    Raises
    ------
    ParameterError : if `lag` does not have the correct shape

    See Also
    --------
    recurrence_to_lag

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)
    >>> recurrence = librosa.segment.recurrence_matrix(mfccs)
    >>> lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=True)
    >>> lag_nopad = librosa.segment.recurrence_to_lag(recurrence, pad=False)
    >>> rec_pad = librosa.segment.lag_to_recurrence(lag_pad)
    >>> rec_nopad = librosa.segment.lag_to_recurrence(lag_nopad)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(2, 2, 1)
    >>> librosa.display.specshow(lag_pad, x_axis='time', y_axis='lag')
    >>> plt.title('Lag (zero-padded)')
    >>> plt.subplot(2, 2, 2)
    >>> librosa.display.specshow(lag_nopad, x_axis='time', y_axis='time')
    >>> plt.title('Lag (no padding)')
    >>> plt.subplot(2, 2, 3)
    >>> librosa.display.specshow(rec_pad, x_axis='time', y_axis='time')
    >>> plt.title('Recurrence (with padding)')
    >>> plt.subplot(2, 2, 4)
    >>> librosa.display.specshow(rec_nopad, x_axis='time', y_axis='time')
    >>> plt.title('Recurrence (without padding)')
    >>> plt.tight_layout()

    '''

    pad = (lag.shape[0] == 2 * lag.shape[-1])

    if lag.ndim != 2 or (lag.shape[0] != lag.shape[1] and not pad):
        raise ParameterError('Invalid lag matrix shape: {}'.format(lag.shape))

    t = lag.shape[1]
    lag = lag.copy()

    for i in range(1, t):
        lag[:, i] = np.roll(lag[:, i], i)

    return np.ascontiguousarray(lag[:t].T).T


def timelag_filter(function, pad=True, index=0):
    '''Filtering in the time-lag domain.

    This is primarily useful for adapting image filters to operate on
    `structure_feature` output.

    Using `timelag_filter` is equivalent to the following sequence of
    operations:

    >>> data_tl = librosa.segment.recurrence_to_lag(data)
    >>> data_filtered_tl = function(data_tl)
    >>> data_filtered = librosa.segment.lag_to_recurrence(data_filtered_tl)

    Parameters
    ----------
    function : callable
        The filtering function to wrap, e.g., `scipy.ndimage.median_filter`

    pad : bool
        Whether to zero-pad the structure feature matrix

    index : int >= 0
        If `function` accepts input data as a positional argument, it should be
        indexed by `index`


    Returns
    -------
    wrapped_function : callable
        A new filter function which applies in time-lag space rather than
        time-time space.


    See Also
    --------
    structure_feature


    Examples
    --------

    Apply a 5-bin median filter to the diagonal of a recurrence matrix

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> rec = librosa.segment.recurrence_matrix(mfcc, sym=True)
    >>> from scipy.ndimage import median_filter
    >>> diagonal_median = librosa.segment.timelag_filter(median_filter)
    >>> rec_filtered = diagonal_median(rec, size=(1, 5), mode='mirror')

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(1, 2, 1)
    >>> librosa.display.specshow(rec, x_axis='time', y_axis='time',
    ...                          aspect='equal')
    >>> plt.title('Raw recurrence matrix')
    >>> plt.subplot(1, 2, 2)
    >>> librosa.display.specshow(rec_filtered, x_axis='time', y_axis='time',
    ...                          aspect='equal')
    >>> plt.title('Filtered recurrence matrix')
    >>> plt.tight_layout()
    '''

    @cache
    def __my_filter(wrapped_f, *args, **kwargs):
        '''Decorator to wrap the filter'''
        # Map the input data into time-lag space
        args = list(args)

        args[index] = recurrence_to_lag(args[index], pad=pad)

        # Apply the filtering function
        result = wrapped_f(*args, **kwargs)

        # Map back into time-time and return
        return lag_to_recurrence(result)

    return decorator(__my_filter, function)


@cache
def subsegment(data, frames, n_segments=4):
    '''Sub-divide a segmentation by feature clustering.

    Given a set of frame boundaries (`frames`), and a data matrix (`data`),
    each successive interval defined by `frames` is partitioned into
    `n_segments` by constrained agglomerative clustering.

    .. note::
        If an interval spans fewer than `n_segments` frames, then each
        frame becomes a sub-segment.

    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        Data matrix to use in clustering

    frames : np.ndarray [shape=(n_boundaries,)], dtype=int, non-negative]
        Array of beat or segment boundaries, as provided by
        `librosa.beat.beat_track`,
        `librosa.onset.onset_detect`,
        or `agglomerative`.

    n_segments : int > 0
        Maximum number of frames to sub-divide each interval.

    Returns
    -------
    boundaries : np.ndarray [shape=(n_subboundaries,)]
        List of sub-divided segment boundaries

    See Also
    --------
    agglomerative : Temporal segmentation
    librosa.onset.onset_detect : Onset detection
    librosa.beat.beat_track : Beat tracking

    Examples
    --------
    Load audio, detect beat frames, and subdivide in twos by CQT

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> cqt = librosa.cqt(y, sr=sr, hop_length=512)
    >>> subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
    >>> subseg
    array([  0,   2,   4,  21,  23,  26,  43,  55,  63,  72,  83,
            97, 102, 111, 122, 137, 142, 153, 162, 180, 182, 185,
           202, 210, 221, 231, 241, 256, 261, 271, 281, 296, 301,
           310, 320, 339, 341, 344, 361, 368, 382, 389, 401, 416,
           420, 430, 436, 451, 456, 465, 476, 489, 496, 503, 515,
           527, 535, 544, 553, 558, 571, 578, 590, 607, 609, 638])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(librosa.logamplitude(cqt**2,
    ...                                               ref_power=np.max),
    ...                          y_axis='cqt_hz', x_axis='time')
    >>> plt.vlines(beats, 0, cqt.shape[0], color='r', alpha=0.5,
    ...            label='Beats')
    >>> plt.vlines(subseg, 0, cqt.shape[0], color='b', linestyle='--',
    ...            alpha=0.5, label='Sub-beats')
    >>> plt.legend(frameon=True, shadow=True)
    >>> plt.title('CQT + Beat and sub-beat markers')
    >>> plt.tight_layout()

    '''

    data = np.atleast_2d(data)
    frames = util.fix_frames(frames, x_min=0, x_max=data.shape[1], pad=True)

    if n_segments < 1:
        raise ParameterError('n_segments must be a positive integer')

    boundaries = []
    for seg_start, seg_end in zip(frames[:-1], frames[1:]):
        boundaries.extend(seg_start + agglomerative(data[:, seg_start:seg_end],
                                                    min(seg_end - seg_start,
                                                        n_segments)))

    return np.ascontiguousarray(boundaries)


def agglomerative(data, k, clusterer=None):
    """Bottom-up temporal segmentation.

    Use a temporally-constrained agglomerative clustering routine to partition
    `data` into `k` contiguous segments.

    Parameters
    ----------
    data     : np.ndarray [shape=(d, t)]
        feature matrix

    k        : int > 0 [scalar]
        number of segments to produce

    clusterer : sklearn.cluster.AgglomerativeClustering, optional
        An optional AgglomerativeClustering object.
        If `None`, a constrained Ward object is instantiated.

    Returns
    -------
    boundaries : np.ndarray [shape=(k,)]
        left-boundaries (frame numbers) of detected segments. This
        will always include `0` as the first left-boundary.

    See Also
    --------
    sklearn.cluster.AgglomerativeClustering

    Examples
    --------
    Cluster by chroma similarity, break into 20 segments

    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> boundary_frames = librosa.segment.agglomerative(chroma, 20)
    >>> librosa.frames_to_time(boundary_frames, sr=sr)
    array([  0.   ,   1.672,   2.322,   2.624,   3.251,   3.506,
             4.18 ,   5.387,   6.014,   6.293,   6.943,   7.198,
             7.848,   9.033,   9.706,   9.961,  10.635,  10.89 ,
            11.54 ,  12.539])

    Plot the segmentation against the spectrogram

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> S = np.abs(librosa.stft(y))**2
    >>> librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.vlines(boundary_frames, 0, S.shape[0], color='r', alpha=0.9,
    ...            label='Segment boundaries')
    >>> plt.legend(frameon=True, shadow=True)
    >>> plt.title('Power spectrogram')
    >>> plt.tight_layout()

    """

    data = np.atleast_2d(data)

    if clusterer is None:
        # Connect the temporal connectivity graph
        n = data.shape[1]
        grid = sklearn.feature_extraction.image.grid_to_graph(n_x=n,
                                                              n_y=1, n_z=1)

        # Instantiate the clustering object
        clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=k,
                                                            connectivity=grid,
                                                            memory=cache)

    # Fit the model
    clusterer.fit(data.T)

    # Find the change points from the labels
    boundaries = [0]
    boundaries.extend(
        list(1 + np.nonzero(np.diff(clusterer.labels_))[0].astype(int)))
    return np.asarray(boundaries)


# Deprecated functions

@util.decorators.deprecated('0.4', '0.5')
@cache
def structure_feature(rec, pad=True, inverse=False):
    '''Compute the structure feature from a recurrence matrix.

    The i'th column of the recurrence matrix is shifted up by i.
    The resulting matrix is indexed horizontally by time,
    and vertically by lag.

    .. warning:: Deprected in librosa 0.4
                 Functionality is superseded by
                 `librosa.segment.recurrence_to_lag` and
                 `librosa.segment.lag_to_recurrence`.

    Parameters
    ----------
    rec   : np.ndarray [shape=(t,t) or shape=(2*t, t)]
        recurrence matrix or pre-computed structure feature

    pad : bool [scalar]
        Pad the matrix with `t` rows of zeros to avoid looping.

    inverse : bool [scalar]
        Unroll the opposite direction. This is useful for converting
        structure features back into recurrence plots.

        .. note: Reversing with `pad==True` will truncate the
            inferred padding.

    Returns
    -------
    struct : np.ndarray [shape=(2*t, t) or shape=(t, t)]
        `struct[i, t]` = the recurrence at time `t` with lag `i`.

        .. note:: negative lag values are supported by wrapping to the
            end of the array.

    See Also
    --------
    recurrence_matrix : build a recurrence matrix from feature vectors

    Examples
    --------
    Build the structure feature over mfcc similarity

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)
    >>> recurrence = librosa.segment.recurrence_matrix(mfccs)
    >>> struct = librosa.segment.structure_feature(recurrence)


    Invert the structure feature to get a recurrence matrix

    >>> recurrence_2 = librosa.segment.structure_feature(struct,
    ...                                                  inverse=True)

    Display recurrence in time-time and time-lag space

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 5))
    >>> plt.subplot(1, 2, 1)
    >>> librosa.display.specshow(recurrence, aspect='equal', x_axis='time',
    ...                          y_axis='time')
    >>> plt.ylabel('Time')
    >>> plt.title('Recurrence (time-time)')
    >>> plt.subplot(1, 2, 2)
    >>> librosa.display.specshow(struct, aspect='auto', x_axis='time')
    >>> plt.ylabel('Lag')
    >>> plt.title('Structure feature')
    >>> plt.tight_layout()

    '''
    if inverse:
        return lag_to_recurrence(rec)
    else:
        return recurrence_to_lag(rec, pad=pad)
