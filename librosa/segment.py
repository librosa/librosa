#!/usr/bin/env python
"""Temporal segmentation"""

import numpy as np
import scipy
import scipy.signal

import sklearn
import sklearn.cluster
import sklearn.feature_extraction

def stack_memory(data, n_steps=2, delay=1):
    """Short-term history embedding.

    Each column ``data[:, i]`` is mapped to

    ``data[:,i] ->  [   data[:, i].T, data[:, i - delay].T ...  data[:, i - (n_steps-1)*delay].T ].T``


    :parameters:
      - data : np.ndarray
          feature matrix (d-by-t)
      - n_steps : int > 0
          embedding dimension, the number of steps back in time to stack
      - delay : int > 0
          the number of columns to step

    :returns:
      - data_history : np.ndarray, shape=(d*m, t)
          data augmented with lagged copies of itself.
          
      .. note:: zeros are padded for the initial columns

    """

    # Pad the end with zeros, which will roll to the front below
    data = np.pad(data, [(0, 0), (0, n_steps * delay)], mode='constant')

    history = data

    for i in range(1, n_steps):
        history = np.vstack([history, np.roll(data, i * delay, axis=1)])

    # Trim to original width
    return history[:, :data.shape[1]]

def recurrence_matrix(data, k=None, width=1, metric='sqeuclidean', sym=False):
    '''Compute the binary recurrence matrix from a time-series.

    rec[i,j] == True <=> (data[:,i], data[:,j]) are k-nearest-neighbors and ||i-j|| >= width

    :parameters:
      - data : np.ndarray
          feature matrix (d-by-t)
      - k : int > 0, float in (0, 1)
          if integer, the number of nearest-neighbors.
          Default: ceil(sqrt(t))
      - width : int > 0
          do not link columns within `width` of each-other
      - metric : see scipy.spatial.distance.pdist()
          distance metric to use for nearest-neighbor calculation
      - sym : bool
          set sym=True to only link mutual nearest-neighbors

    :returns:
      - rec : np.ndarray, shape=(t,t), dtype=bool
          Binary recurrence matrix
    :raises:
      - ValueError
          if k is a float outside the range (0,1)
          or if mode is not one of {'knn', 'gaussian'}
    '''

    t = data.shape[1]

    if k is None:
        k = np.ceil(np.sqrt(t))

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

def structure_feature(rec, pad=True):
    '''Compute the structure feature from a recurrence matrix.

    The i'th column of the recurrence matrix is shifted up by i.
    The resulting matrix is indexed horizontally by time,
    and vertically by lag.

    :parameters:
      - rec   : np.ndarray, shape=(t,t)
          recurrence matrix (see `librosa.segment.recurrence_matrix`)
      
      - pad : boolean

    :returns:
      - struct : np.ndarray
          ``struct[i, t]`` = the recurrence at time ``t`` with lag ``i``.

      .. note:: negative lag values are supported by wrapping to the end of the array.

    :raises:
      - ValueError
          if rec is not square
    '''

    t = rec.shape[0]
    if t != rec.shape[1]:
        raise ValueError('rec must be a square matrix')

    if pad:
        # If we don't assume that the signal loops,
        # stack zeros underneath in the recurrence plot.
        struct = np.pad(rec, [(0, t), (0, 0)], mode='constant')
    else:
        struct = rec.copy()

    for i in range(1, t):
        struct[:, i] = np.roll(struct[:, i], -i, axis=-1)

    return struct

def agglomerative(data, k):
    """Bottom-up temporal segmentation

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

