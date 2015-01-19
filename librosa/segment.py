#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Temporal segmentation utilities"""

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

from . import cache
from . import util


@cache
def __band_infinite(n, width, v_in=0.0, v_out=np.inf, dtype=np.float32):
    '''Construct a square, banded matrix `X` where
    `X[i, j] == v_in` if `|i - j| <= width`
    `X[i, j] == v_out` if `|i - j| > width`

    This is used to suppress nearby links in `recurrence_matrix`.
    '''

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

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfcc = librosa.feature.mfcc(y=y, sr=sr)
    >>> R = librosa.segment.recurrence_matrix(mfcc)
    >>> # Or fix the number of nearest neighbors to 5
    >>> R = librosa.segment.recurrence_matrix(mfcc, k=5)
    >>> # Suppress neighbors within +- 7 samples
    >>> R = librosa.segment.recurrence_matrix(mfcc, width=7)
    >>> # Use cosine similarity instead of Euclidean distance
    >>> R = librosa.segment.recurrence_matrix(mfcc, metric='cosine')
    >>> # Require mutual nearest neighbors
    >>> R = librosa.segment.recurrence_matrix(mfcc, sym=True)

    Parameters
    ----------
    data : np.ndarray [shape=(d, t)]
        A feature matrix

    k : int > 0 [scalar] or None
        the number of nearest-neighbors for each sample

        Default: `k = 2 * ceil(sqrt(t - 2 * width + 1))`,
        or `k = 2` if `t <= 2 * width + 1`

    width : int > 0 [scalar]
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
    '''

    t = data.shape[1]

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


@cache
def structure_feature(rec, pad=True, inverse=False):
    '''Compute the structure feature from a recurrence matrix.

    The i'th column of the recurrence matrix is shifted up by i.
    The resulting matrix is indexed horizontally by time,
    and vertically by lag.

    Examples
    --------
    >>> # Build the structure feature over mfcc similarity
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr)
    >>> recurrence = librosa.feature.recurrence_matrix(mfccs)
    >>> struct = librosa.feature.structure_feature(recurrence)
    >>> # Invert the structure feature to get a recurrence matrix
    >>> recurrence_2 = librosa.feature.structure_feature(struct,
                                                         inverse=True)

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
    '''

    t = rec.shape[1]

    if pad and not inverse:
        # If we don't assume that the signal loops,
        # stack zeros underneath in the recurrence plot.
        struct = np.pad(rec, [(0, t), (0, 0)], mode='constant')
    else:
        struct = rec.copy()

    if inverse:
        direction = +1
    else:
        direction = -1

    for i in range(1, t):
        struct[:, i] = np.roll(struct[:, i], direction * i, axis=-1)

    if inverse and pad:
        struct = struct[:t]

    # Make column-contiguous
    return np.ascontiguousarray(struct.T).T


@cache
def subsegment(data, frames, n_segments=4, pad=True):
    '''Sub-divide a segmentation by feature clustering.

    Given a set of frame boundaries (`frames`), and a data matrix (`data`),
    each successive interval defined by `frames` is partitioned into
    `n_segments` by constrained agglomerative clustering.

    .. note::
        If an interval spans fewer than `n_segments` frames, then each
        frame becomes a sub-segment.

    Examples
    --------
    >>> # Load audio, detect beat frames, and compute a CQT
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    >>> cqt = librosa.cqt(y, sr=sr, hop_length=512)
    >>> beats
    array([ 186,  211,  236,  261,  286,  311,  336,  361,  386,  411,
            436,  461,  486,  510,  535,  560,  585,  609,  634,  658,
            684,  710,  737,  763,  789,  817,  843,  869,  896,  922,
            948,  976, 1001, 1026, 1051, 1076, 1101, 1126, 1150, 1175,
           1202, 1229, 1254, 1279, 1304, 1329, 1354, 1379, 1404, 1429,
           1454, 1479, 1503, 1528, 1553, 1578, 1603, 1627, 1652, 1676,
           1700, 1724, 1748, 1772, 1797, 1822, 1846, 1871, 1896, 1921,
           1946, 1971, 1995, 2020, 2045, 2070, 2095, 2120, 2145, 2169,
           2194, 2220, 2248, 2273, 2298, 2323, 2348, 2373, 2398, 2423,
           2448, 2472, 2497, 2522])
    >>> # Sub-divide the beats into (up to) 4 sub-segments each
    >>> librosa.segment.subsegment(cqt, beats, n_segments=4)
    array([   0,   74, ..., 2548, 2582])

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

    pad : bool
        If `True`, then `frames` is expanded to cover the full
        range `[0, n]`

    Returns
    -------
    boundaries : np.ndarray [shape=(n_subboundaries,)]
        List of sub-divided segment boundaries

    See Also
    --------
    agglomerative : Temporal segmentation
    librosa.onset.onset_detect : Onset detection
    librosa.beat.beat_track : Beat tracking
    '''

    frames = util.fix_frames(frames, 0, data.shape[1], pad=pad)

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

    Examples
    --------
    >>> # Cluster by Mel spectrogram similarity
    >>> # Break into 32 segments
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr)
    >>> boundary_frames = librosa.segment.agglomerative(S, 32)
    >>> librosa.frames_to_time(boundary_frames, sr=sr)
    array([  0.   ,  18.297,  18.367,  21.989,  22.059,  23.382,  23.452,
            25.681,  25.751,  27.074,  27.144,  33.065,  33.135,  34.458,
            34.528,  36.757,  36.827,  38.22 ,  41.842,  41.912,  44.373,
            44.536,  47.833,  47.949,  51.525,  51.641,  52.918,  52.988,
            55.217,  55.287,  56.61 ,  56.68 ])

    Parameters
    ----------
    data     : np.ndarray [shape=(d, t)]
        feature matrix

    k        : int > 0 [scalar]
        number of segments to produce

    clusterer : sklearn.cluster.AgglomerativeClustering or `None`
        An optional AgglomerativeClustering object.
        If `None`, a constrained Ward object is instantiated.

    Returns
    -------
    boundaries : np.ndarray [shape=(k,)]
        left-boundaries (frame numbers) of detected segments. This
        will always include `0` as the first left-boundary.
    """

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
