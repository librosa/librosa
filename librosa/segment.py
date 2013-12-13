#!/usr/bin/env python
"""Temporal segmentation utilities"""

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

def stack_memory(data, n_steps=2, delay=1, trim=True):
    """Short-term history embedding.

    Each column ``data[:, i]`` is mapped to

    ``data[:,i] ->  [   data[:, i].T, data[:, i - delay].T ...  data[:, i - (n_steps-1)*delay].T ].T``


    :usage:
        >>> mfccs       = librosa.feature.mfcc(y=y, sr=sr)
        >>> mfcc_stack  = librosa.segment.stack_memory(mfccs)

    :parameters:
      - data : np.ndarray
          feature matrix (d-by-t)
      - n_steps : int > 0
          embedding dimension, the number of steps back in time to stack
      - delay : int > 0
          the number of columns to step
      - trim : boolean
          Crop dimension to original number of columns

    :returns:
      - data_history : np.ndarray, shape=(d*m, t)
          data augmented with lagged copies of itself.
          
      .. note:: zeros are padded for the initial columns

    """

    t = data.shape[1]
    # Pad the end with zeros, which will roll to the front below
    data = np.pad(data, [(0, 0), (0, (n_steps-1) * delay)], mode='constant')

    history = data

    for i in range(1, n_steps):
        history = np.vstack([history, np.roll(data, i * delay, axis=1)])

    # Trim to original width
    if trim:
        history = history[:, :t]

    return history

def recurrence_matrix(data, k=None, width=1, metric='sqeuclidean', sym=False):
    '''Compute the binary recurrence matrix from a time-series.

    ``rec[i,j] == True`` <=> (``data[:,i]``, ``data[:,j]``) are k-nearest-neighbors and ``|i-j| >= width``

    :usage:
        >>> mfcc    = librosa.feature.mfcc(y=y, sr=sr)
        >>> R       = librosa.segment.recurrence_matrix(mfcc)

        >>> # Or fix the number of nearest neighbors to 5
        >>> R       = librosa.segment.recurrence_matrix(mfcc, k=5)

        >>> # Suppress neighbors within +- 7 samples
        >>> R       = librosa.segment.recurrence_matrix(mfcc, width=7)

        >>> # Use cosine similarity instead of Euclidean distance
        >>> R       = librosa.segment.recurrence_matrix(mfcc, metric='cosine')

        >>> # Require mutual nearest neighbors
        >>> R       = librosa.segment.recurrence_matrix(mfcc, sym=True)

    :parameters:
      - data : np.ndarray
          feature matrix (d-by-t)
      - k : int > 0 or None
          the number of nearest-neighbors for each sample
          Default: ``k = ceil(sqrt(t - 2 * width + 1))``
      - width : int > 0
          only link neighbors ``(data[:, i], data[:, j])`` if ``|i-j| >= width`` 
      - metric : see ``scipy.spatial.distance.pdist()``
          distance metric to use for nearest-neighbor calculation
      - sym : bool
          set ``sym=True`` to only link mutual nearest-neighbors

    :returns:
      - rec : np.ndarray, shape=(t,t), dtype=bool
          Binary recurrence matrix
    '''

    t = data.shape[1]

    if k is None:
        k = np.ceil(np.sqrt(t - 2 * width + 1))

    def _band_infinite():
        '''Suppress the diagonal+- of a distance matrix'''
        band       = np.empty( (t, t) )
        band[:]    = np.inf
        band[np.triu_indices_from(band, width)] = 0
        band[np.tril_indices_from(band, -width)] = 0

        return band

    # Build the distance matrix
    D = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(data.T, metric=metric))

    # Max out the diagonal band
    D = D + _band_infinite()

    # build the recurrence plot

    rec = np.zeros( (t, t), dtype=bool)

    # get the k nearest neighbors for each point
    for i in range(t):
        for j in np.argsort(D[i])[:k]:
            rec[i, j] = True

    # symmetrize
    if sym:
        rec = rec * rec.T

    return rec

def structure_feature(rec, pad=True, inverse=False):
    '''Compute the structure feature from a recurrence matrix.

    The i'th column of the recurrence matrix is shifted up by i.
    The resulting matrix is indexed horizontally by time,
    and vertically by lag.

    :usage:
        >>> # Build the structure feature over mfcc similarity
        >>> mfccs   = librosa.feature.mfcc(y=y, sr=sr)
        >>> R       = librosa.feature.recurrence_matrix(mfccs)
        >>> S       = librosa.feature.structure_feature(R)

        >>> # Invert the structure feature to get a recurrence matrix
        >>> R_hat   = librosa.feature.structure_feature(S, inverse=True)

    :parameters:
      - rec   : np.ndarray, shape=(t,t)
          recurrence matrix (see `librosa.segment.recurrence_matrix`)
      
      - pad : boolean
          Pad the matrix with t rows of zeros to avoid looping.

      - inverse : boolean
          Unroll the opposite direction. This is useful for converting
          structure features back into recurrence plots.

          .. note: Reversing with ``pad==True`` will truncate the inferred padding.

    :returns:
      - struct : np.ndarray
          ``struct[i, t]`` = the recurrence at time ``t`` with lag ``i``.

      .. note:: negative lag values are supported by wrapping to the end of the array.
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

    return struct

def agglomerative(data, k):
    """Bottom-up temporal segmentation.

    Use a temporally-constrained agglomerative clustering routine to partition
    ``data`` into ``k`` contiguous segments.

    :usage:
        >>> # Cluster by Mel spectrogram similarity
        >>> # Break into 32 segments
        >>> S                   = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        >>> boundary_frames     = librosa.segment.agglomerative(S, 32)
        >>> boundary_times      = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=512)

    :parameters:
      - data     : np.ndarray    
          feature matrix (d-by-t)

      - k        : int > 0
          number of segments to produce

    :returns:
      - boundaries : np.ndarray, shape=(k,1)  
          left-boundaries (frame numbers) of detected segments

    """

    # Connect the temporal connectivity graph
    grid = sklearn.feature_extraction.image.grid_to_graph(  n_x=data.shape[1], 
                                                            n_y=1, 
                                                            n_z=1)

    # Instantiate the clustering object
    ward = sklearn.cluster.Ward(n_clusters=k, connectivity=grid)

    # Fit the model
    ward.fit(data.T)

    # Find the change points from the labels
    boundaries = [0]
    boundaries.extend(
        list(1 + np.nonzero(np.diff(ward.labels_))[0].astype(int)))
    return boundaries

